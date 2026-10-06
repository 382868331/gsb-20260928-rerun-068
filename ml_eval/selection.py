"""多重保护组约束下的策略选择研究（冻结留出集上的只读比较）。

本模块在**已经冻结的留出资料**与**已经定版的决策策略**之上做一次性的
选择研究，不重新拟合任何策略，也不调整留出集、样本权重或判定阈值：

* 输入的每个 ``AbstentionPolicy`` 都是先前在各自选择资料上定版的冻结
  策略；研究层只调用 ``evaluate_policy`` 在同一份只读留出集上重放其
  逐样本决策，比较层只汇总这些重放结果；
* 一个样本可以同时属于多个保护组：各组覆盖率把该样本分别计入其每个
  归属组，但总体收益与总体成本每个样本只计一次（直接采用策略重放的
  总体账目，重叠样本不重复计费）；
* 组标签缺失的样本保留为 ``__unassigned__`` 未知归属，计入总体成本与
  总体分母，但绝不默认归入任何保护组、绝不用来凑够某组覆盖；受约束
  组在留出集上完全没有成员时也不能默认其满足约束，而按无法核对报告；
* 约束为各组最低覆盖、组间最大覆盖差异（受约束已知组的最大覆盖率减
  最小覆盖率）以及总体成本上限，三者必须**同时**满足才算可行；
* 没有任何候选策略同时满足全部约束时，研究明确给出不可行结论，列出
  始终无法满足的组与限制以及全体候选能达到的最好水平，绝不通过修改
  留出集、阈值或权重制造可行结果；
* 研究在可行候选上比较两种目标：成本最小（总体加权代价最小）与最坏
  组改善（受约束组中最低覆盖率最大化），两者的选择都随研究冻结；
* 新策略到达时只能用扩大的候选集合形成**新研究**（study_version 改变），
  旧策略、样本权重、逐样本判定与旧研究结果保持冻结、仍可追溯；
* ``to_dict/from_dict`` 保存重开后按内容重算 study_version 并校验一致；
  ``recompute_study`` 用冻结策略与冻结留出集重新重放，结果必须与原研究
  逐项一致（含可行性、违约明细与两种目标的选择）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .policy import (
    _ALL_GROUPS,
    _EPS,
    AbstentionPolicy,
    DECISION_ACCEPTED_CORRECT,
    evaluate_policy,
)
from .predictions import PredictionBatch
from .versioning import content_version

UNKNOWN_GROUP = "__unassigned__"

OBJECTIVE_MIN_COST = "min_cost"
OBJECTIVE_WORST_GROUP = "worst_group_improvement"
OBJECTIVES = (OBJECTIVE_MIN_COST, OBJECTIVE_WORST_GROUP)

VIOLATION_MIN_COVERAGE = "min_coverage"
VIOLATION_DISPARITY = "max_disparity"
VIOLATION_COST_CAP = "total_cost_cap"
VIOLATION_GROUP_ABSENT = "group_absent_on_holdout"
VIOLATION_GROUP_ABSENT = "group_absent_on_holdout"


@dataclass(frozen=True)
class SelectionConstraints:
    """策略选择的多重约束：各组最低覆盖、组间最大差异与总体成本上限。

    min_coverage 接受单一数值（作用于留出集上出现的每个已知组）或
    ``{组: 要求}``（``"*"`` 为缺省）。max_disparity 限制受约束已知组
    之间覆盖率的最大差值；None 表示不限制。未知归属样本不参与任何组
    约束，只计入总体账目。
    """

    min_coverage: float | Mapping[str, float] = 0.0
    max_disparity: float | None = None
    max_total_cost: float | None = None

    def normalized(self) -> tuple[float, tuple[tuple[str, float], ...]]:
        def check(value: float) -> float:
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"最低覆盖要求必须在 [0, 1] 内: {value!r}")
            return float(value)

        if isinstance(self.min_coverage, Mapping):
            table = {str(k): check(v) for k, v in self.min_coverage.items()}
            default = table.pop(_ALL_GROUPS, 0.0)
            return default, tuple(sorted(table.items()))
        return check(self.min_coverage), ()

    def __post_init__(self) -> None:
        self.normalized()
        if self.max_disparity is not None and (
            not isinstance(self.max_disparity, (int, float))
            or not math.isfinite(self.max_disparity)
            or self.max_disparity < 0
        ):
            raise ValueError("max_disparity 必须是非负有限数值或 None")
        if self.max_total_cost is not None and (
            not isinstance(self.max_total_cost, (int, float))
            or not math.isfinite(self.max_total_cost)
            or self.max_total_cost < 0
        ):
            raise ValueError("max_total_cost 必须是非负有限数值或 None")

    def to_dict(self) -> dict[str, Any]:
        default, table = self.normalized()
        raw = (
            dict(self.min_coverage)
            if isinstance(self.min_coverage, Mapping)
            else float(self.min_coverage)
        )
        return {
            "min_coverage": raw,
            "min_coverage_default": default,
            "min_coverage_by_group": dict(table),
            "max_disparity": self.max_disparity,
            "max_total_cost": self.max_total_cost,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "SelectionConstraints":
        raw = data["min_coverage"]
        if isinstance(raw, Mapping):
            min_coverage: float | Mapping[str, float] = dict(raw)
        else:
            min_coverage = float(raw)
        return SelectionConstraints(
            min_coverage=min_coverage,
            max_disparity=(
                None if data["max_disparity"] is None
                else float(data["max_disparity"])
            ),
            max_total_cost=(
                None if data["max_total_cost"] is None
                else float(data["max_total_cost"])
            ),
        )


@dataclass(frozen=True)
class CandidateOutcome:
    """单个冻结策略在冻结留出集上重放后的选择口径结果。

    benefit_weight 为正确接受样本的权重和（总体收益，重叠样本只计一次）；
    worst_group_coverage/slack 只在受约束的已知组上取最小值；unknown_*
    单独保留缺失组标签样本的权重去向，绝不并入任何受约束组。
    """

    policy_version: str
    strategy: str
    feasible: bool
    total_cost: float
    total_weight: float
    accepted_weight: float
    benefit_weight: float
    worst_group_coverage: float | None
    worst_group_slack: float | None
    observed_disparity: float | None
    group_coverage: tuple[tuple[str, float | None], ...]
    group_total_weight: tuple[tuple[str, float], ...]
    unknown_weight: float
    unknown_accepted_weight: float
    violations: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "strategy": self.strategy,
            "feasible": self.feasible,
            "total_cost": self.total_cost,
            "total_weight": self.total_weight,
            "accepted_weight": self.accepted_weight,
            "benefit_weight": self.benefit_weight,
            "worst_group_coverage": self.worst_group_coverage,
            "worst_group_slack": self.worst_group_slack,
            "observed_disparity": self.observed_disparity,
            "group_coverage": dict(self.group_coverage),
            "group_total_weight": dict(self.group_total_weight),
            "unknown_weight": self.unknown_weight,
            "unknown_accepted_weight": self.unknown_accepted_weight,
            "violations": list(self.violations),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "CandidateOutcome":
        return CandidateOutcome(
            policy_version=str(data["policy_version"]),
            strategy=str(data["strategy"]),
            feasible=bool(data["feasible"]),
            total_cost=float(data["total_cost"]),
            total_weight=float(data["total_weight"]),
            accepted_weight=float(data["accepted_weight"]),
            benefit_weight=float(data["benefit_weight"]),
            worst_group_coverage=data["worst_group_coverage"],
            worst_group_slack=data["worst_group_slack"],
            observed_disparity=data["observed_disparity"],
            group_coverage=tuple(
                (str(g), c) for g, c in data["group_coverage"].items()
            ),
            group_total_weight=tuple(
                (str(g), float(w))
                for g, w in data["group_total_weight"].items()
            ),
            unknown_weight=float(data["unknown_weight"]),
            unknown_accepted_weight=float(data["unknown_accepted_weight"]),
            violations=tuple(dict(v) for v in data["violations"]),
        )


@dataclass(frozen=True)
class ObjectiveChoice:
    """某种目标在可行候选上做出的冻结选择；不可行时 policy_version 为 None。"""

    objective: str
    policy_version: str | None
    reason: str
    objective_value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "policy_version": self.policy_version,
            "reason": self.reason,
            "objective_value": self.objective_value,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ObjectiveChoice":
        return ObjectiveChoice(
            objective=str(data["objective"]),
            policy_version=data["policy_version"],
            reason=str(data["reason"]),
            objective_value=data["objective_value"],
        )


@dataclass(frozen=True)
class StrategySelectionStudy:
    """多重保护组约束下的一次性策略选择研究（全部输入均冻结）。

    outcomes 保留每个候选策略在冻结留出集上的重放汇总；choices 给出
    成本最小与最坏组改善两种目标各自的可行选择；没有候选可行时
    feasible=False，infeasible_constraints 列出无法同时满足的组与限制
    及全体候选能达到的最好水平。evaluations 为重放明细（含逐样本
    判定），不参与序列化，重开后用 recompute_study 复算并逐项比对。
    """

    study_version: str
    sample_set_version: str
    prediction_batch_version: str
    group_attrs: tuple[str, ...]
    constraints: SelectionConstraints
    candidate_policy_versions: tuple[str, ...]
    outcomes: tuple[CandidateOutcome, ...]
    choices: tuple[ObjectiveChoice, ...]
    feasible: bool
    infeasible_constraints: tuple[dict[str, Any], ...]
    evaluations: tuple[Any, ...] = ()

    def outcome_for(self, policy_version: str) -> CandidateOutcome:
        for outcome in self.outcomes:
            if outcome.policy_version == policy_version:
                return outcome
        raise KeyError(policy_version)

    def choice_for(self, objective: str) -> ObjectiveChoice:
        for choice in self.choices:
            if choice.objective == objective:
                return choice
        raise KeyError(objective)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_version": self.study_version,
            "sample_set_version": self.sample_set_version,
            "prediction_batch_version": self.prediction_batch_version,
            "group_attrs": list(self.group_attrs),
            "constraints": self.constraints.to_dict(),
            "candidate_policy_versions": list(self.candidate_policy_versions),
            "outcomes": [o.to_dict() for o in self.outcomes],
            "choices": [c.to_dict() for c in self.choices],
            "feasible": self.feasible,
            "infeasible_constraints": list(self.infeasible_constraints),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "StrategySelectionStudy":
        """从 to_dict 结果还原研究（保存后重开），并按内容重算版本校验。

        重开不携带逐样本重放明细；调用方应随后用 recompute_study 与
        冻结策略、冻结留出集复算，复算结果与研究内容必须完全一致。
        """
        study = StrategySelectionStudy(
            study_version=str(data["study_version"]),
            sample_set_version=str(data["sample_set_version"]),
            prediction_batch_version=str(data["prediction_batch_version"]),
            group_attrs=tuple(str(a) for a in data["group_attrs"]),
            constraints=SelectionConstraints.from_dict(data["constraints"]),
            candidate_policy_versions=tuple(
                str(v) for v in data["candidate_policy_versions"]
            ),
            outcomes=tuple(
                CandidateOutcome.from_dict(o) for o in data["outcomes"]
            ),
            choices=tuple(
                ObjectiveChoice.from_dict(c) for c in data["choices"]
            ),
            feasible=bool(data["feasible"]),
            infeasible_constraints=tuple(
                dict(item) for item in data["infeasible_constraints"]
            ),
        )
        recomputed = _study_version(study)
        if recomputed != study.study_version:
            raise ValueError("研究内容与其 study_version 不一致，拒绝载入")
        return study
        if recomputed != study.study_version:
            raise ValueError("研究内容与其 study_version 不一致，拒绝载入")
        return study


def _is_unknown_key(group: str) -> bool:
    return group == UNKNOWN_GROUP or group.endswith(f"={UNKNOWN_GROUP}")


def _study_version(study: StrategySelectionStudy) -> str:
    return content_version(
        [
            ("sample_set_version", study.sample_set_version),
            ("prediction_batch_version", study.prediction_batch_version),
            ("group_attrs", list(study.group_attrs)),
            ("constraints", study.constraints.to_dict()),
            (
                "candidate_policy_versions",
                list(study.candidate_policy_versions),
            ),
            ("outcomes", [o.to_dict() for o in study.outcomes]),
            ("choices", [c.to_dict() for c in study.choices]),
            ("feasible", study.feasible),
            (
                "infeasible_constraints",
                list(study.infeasible_constraints),
            ),
        ]
    )


def _evaluate_candidate(
    policy: AbstentionPolicy,
    evaluation: Any,
    constrained_groups: tuple[str, ...],
    requirement: Mapping[str, float],
    constraints: SelectionConstraints,
    holdout_known_groups: frozenset[str],
) -> CandidateOutcome:
    """把一份冻结策略的重放结果按选择口径汇总并逐条列出违约。"""

    group_results = {g.group: g for g in evaluation.groups}
    violations: list[dict[str, Any]] = []
    coverages: dict[str, float] = {}
    group_totals: dict[str, float] = {}

    for group in constrained_groups:
        req = requirement[group]
        result = group_results.get(group)
        if result is None or result.total_weight <= 0 or group not in holdout_known_groups:
            violations.append(
                {
                    "type": VIOLATION_GROUP_ABSENT,
                    "group": group,
                    "min_coverage": req,
                    "actual_coverage": None,
                    "reason": "受约束组在冻结留出集上没有任何已知成员，无法核对",
                }
            )
            continue
        coverage = result.coverage
        coverages[group] = coverage if coverage is not None else 0.0
        group_totals[group] = result.total_weight
        if coverages[group] + _EPS < req:
            violations.append(
                {
                    "type": VIOLATION_MIN_COVERAGE,
                    "group": group,
                    "min_coverage": req,
                    "actual_coverage": coverages[group],
                    "shortfall": req - coverages[group],
                }
            )

    observed_disparity = (
        max(coverages.values()) - min(coverages.values())
        if coverages
        else None
    )
    if (
        constraints.max_disparity is not None
        and observed_disparity is not None
        and observed_disparity > constraints.max_disparity + _EPS
    ):
        violations.append(
            {
                "type": VIOLATION_DISPARITY,
                "max_disparity": constraints.max_disparity,
                "observed_disparity": observed_disparity,
                "excess": observed_disparity - constraints.max_disparity,
                "min_coverage_group": min(coverages, key=coverages.get),
                "max_coverage_group": max(coverages, key=coverages.get),
            }
        )

    overall = evaluation.overall
    if (
        constraints.max_total_cost is not None
        and overall.total_cost > constraints.max_total_cost + _EPS
    ):
        violations.append(
            {
                "type": VIOLATION_COST_CAP,
                "max_total_cost": constraints.max_total_cost,
                "actual_total_cost": overall.total_cost,
                "excess": overall.total_cost - constraints.max_total_cost,
            }
        )

    benefit_weight = float(
        sum(
            row.weight
            for row in evaluation.decisions
            if row.status == DECISION_ACCEPTED_CORRECT
        )
    )
    # 缺失组标签的样本按样本只计一次：只有该样本在研究使用的全部属性上
    # 都缺失时才算未知归属；仅缺某个属性但属于其他保护组的样本仍计入其
    # 已知组，绝不通过未知桶重复或挪用权重。
    unknown_weight = 0.0
    unknown_accepted = 0.0
    for row in evaluation.decisions:
        if row.groups and all(_is_unknown_key(g) for g in row.groups):
            unknown_weight += row.weight
            if row.accepted:
                unknown_accepted += row.weight
    unknown_weight = float(unknown_weight)
    unknown_accepted = float(unknown_accepted)

    if coverages and not any(
        v["type"] == VIOLATION_GROUP_ABSENT for v in violations
    ):
        worst_group = min(coverages, key=coverages.get)
        worst_cov: float | None = coverages[worst_group]
        worst_slack: float | None = worst_cov - requirement[worst_group]
    else:
        worst_cov = None
        worst_slack = None

    return CandidateOutcome(
        policy_version=policy.policy_version,
        strategy=policy.strategy,
        feasible=not violations,
        total_cost=float(overall.total_cost),
        total_weight=float(overall.total_weight),
        accepted_weight=float(overall.accepted_weight),
        benefit_weight=benefit_weight,
        worst_group_coverage=worst_cov,
        worst_group_slack=worst_slack,
        observed_disparity=observed_disparity,
        group_coverage=tuple(sorted(coverages.items())),
        group_total_weight=tuple(sorted(group_totals.items())),
        unknown_weight=unknown_weight,
        unknown_accepted_weight=unknown_accepted,
        violations=tuple(violations),
    )


def _summarize_infeasible(
    outcomes: tuple[CandidateOutcome, ...],
    constrained_groups: tuple[str, ...],
    requirement: Mapping[str, float],
    constraints: SelectionConstraints,
) -> tuple[dict[str, Any], ...]:
    """汇总全体候选都无法同时满足的组与限制（不修改任何输入）。"""

    items: list[dict[str, Any]] = []
    for group in constrained_groups:
        present = [
            dict(o.group_coverage).get(group)
            for o in outcomes
            if dict(o.group_coverage).get(group) is not None
        ]
        absent = sum(
            1
            for o in outcomes
            if any(
                v["type"] == VIOLATION_GROUP_ABSENT and v["group"] == group
                for v in o.violations
            )
        )
        always_below = all(
            any(
                v["type"] == VIOLATION_MIN_COVERAGE and v["group"] == group
                for v in o.violations
            )
            for o in outcomes
        )
        if always_below or absent == len(outcomes):
            items.append(
                {
                    "constraint": VIOLATION_MIN_COVERAGE,
                    "group": group,
                    "required": requirement[group],
                    "best_achievable_coverage": max(present) if present else None,
                    "reason": (
                        "受约束组在留出集上没有已知成员，不能默认满足约束"
                        if absent == len(outcomes)
                        else "所有候选策略的该组覆盖率都低于最低要求"
                    ),
                }
            )
    disparities = [
        o.observed_disparity
        for o in outcomes
        if o.observed_disparity is not None
    ]
    if (
        constraints.max_disparity is not None
        and disparities
        and min(disparities) > constraints.max_disparity + _EPS
    ):
        items.append(
            {
                "constraint": VIOLATION_DISPARITY,
                "required_max": constraints.max_disparity,
                "best_achievable_disparity": min(disparities),
                "reason": "所有候选策略的组间覆盖差异都高于最大差异限制",
            }
        )
    costs = [o.total_cost for o in outcomes]
    if (
        constraints.max_total_cost is not None
        and min(costs) > constraints.max_total_cost + _EPS
    ):
        items.append(
            {
                "constraint": VIOLATION_COST_CAP,
                "required_max": constraints.max_total_cost,
                "best_achievable_cost": min(costs),
                "reason": "所有候选策略的总体成本都高于成本上限",
            }
        )
    return tuple(items)
    return tuple(items)


def run_selection_study(
    policies: "list[AbstentionPolicy] | tuple[AbstentionPolicy, ...]",
    holdout_batch: PredictionBatch,
    constraints: SelectionConstraints,
    seed: int = 20260929,
    n_replicates: int = 1000,
    confidence_level: float = 0.95,
    bootstrap: bool = False,
) -> StrategySelectionStudy:
    """在冻结留出集上对若干冻结策略做多重约束选择研究。

    只重放各策略的冻结阈值（evaluate_policy），绝不重新拟合、不调整
    留出集/权重/判定。每个候选策略按同一组多重约束核对可行性；在
    可行候选上分别给出成本最小与最坏组改善两种目标的选择。
    """

    policies = tuple(policies)
    if not policies:
        raise ValueError("选择研究至少需要一个候选策略")
    attrs_sets = {tuple(p.group_attrs) for p in policies}
    if len(attrs_sets) != 1:
        raise ValueError("候选策略必须使用相同的分组属性定义")
    versions = [p.policy_version for p in policies]
    if len(set(versions)) != len(versions):
        raise ValueError("候选策略版本必须互不相同")

    group_attrs = policies[0].group_attrs
    default_req, req_pairs = constraints.normalized()
    explicit = dict(req_pairs)

    evaluations = tuple(
        evaluate_policy(
            policy,
            holdout_batch,
            n_replicates=n_replicates,
            seed=seed,
            confidence_level=confidence_level,
            bootstrap=bootstrap,
        )
        for policy in policies
    )
    holdout_group_set = frozenset(
        gg.group for ev in evaluations for gg in ev.groups if gg.group is not None
    )
    holdout_known_groups = frozenset(
        g for g in holdout_group_set if not _is_unknown_key(g)
    )
    constrained_set = (
        set(explicit) | set(holdout_known_groups) if explicit
        else set(holdout_known_groups)
    )
    constrained_groups = tuple(sorted(constrained_set))
    requirement = {
        group: explicit.get(group, default_req) for group in constrained_groups
    }

    outcomes = tuple(
        _evaluate_candidate(
            policy,
            evaluation,
            constrained_groups,
            requirement,
            constraints,
            holdout_known_groups,
        )
        for policy, evaluation in zip(policies, evaluations)
    )
    feasible_outcomes = [o for o in outcomes if o.feasible]
    feasible = bool(feasible_outcomes)
    feasible = bool(feasible_outcomes)

    choices: list[ObjectiveChoice] = []
    infeasible_constraints: tuple[dict[str, Any], ...] = ()
    if feasible:
        min_cost_outcome = min(
            feasible_outcomes,
            key=lambda o: (o.total_cost, o.policy_version),
        )
        choices.append(
            ObjectiveChoice(
                objective=OBJECTIVE_MIN_COST,
                policy_version=min_cost_outcome.policy_version,
                reason="在同时满足全部约束的候选中总体加权代价最小",
                objective_value=min_cost_outcome.total_cost,
            )
        )
        worst_outcome = min(
            feasible_outcomes,
            key=lambda o: (
                -(o.worst_group_coverage if o.worst_group_coverage is not None else 0.0),
                o.total_cost,
                o.policy_version,
            ),
        )
        choices.append(
            ObjectiveChoice(
                objective=OBJECTIVE_WORST_GROUP,
                policy_version=worst_outcome.policy_version,
                reason="在同时满足全部约束的候选中最大化受约束组的最低覆盖率",
                objective_value=worst_outcome.worst_group_coverage,
            )
        )
    else:
        for objective in OBJECTIVES:
            choices.append(
                ObjectiveChoice(
                    objective=objective,
                    policy_version=None,
                    reason="没有任何候选策略同时满足全部约束，不做选择",
                    objective_value=None,
                )
            )
        infeasible_constraints = _summarize_infeasible(
            outcomes, constrained_groups, requirement, constraints
        )

    study = StrategySelectionStudy(
        study_version="",
        sample_set_version=holdout_batch.sample_set.version,
        prediction_batch_version=holdout_batch.version,
        group_attrs=group_attrs,
        constraints=constraints,
        candidate_policy_versions=tuple(versions),
        outcomes=outcomes,
        choices=tuple(choices),
        feasible=feasible,
        infeasible_constraints=infeasible_constraints,
        evaluations=evaluations,
    )
    object.__setattr__(study, "study_version", _study_version(study))
    return study


def recompute_study(
    study: StrategySelectionStudy,
    policies: "list[AbstentionPolicy] | tuple[AbstentionPolicy, ...]",
    holdout_batch: PredictionBatch,
    seed: int = 20260929,
    n_replicates: int = 1000,
    confidence_level: float = 0.95,
    bootstrap: bool = False,
) -> StrategySelectionStudy:
    """用冻结策略与冻结留出集重新重放一份已保存研究并逐项比对。

    重开后复算必须得到相同的可行性、违约明细、组覆盖、两种目标选择与
    study_version；候选集合、权重或判定与保存时不一致即抛错，绝不静默
    采用重算结果覆盖原研究。
    """

    policies = tuple(policies)
    recomputed = run_selection_study(
        policies,
        holdout_batch,
        study.constraints,
        seed=seed,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        bootstrap=bootstrap,
    )
    if recomputed.to_dict() != study.to_dict():
        raise ValueError("重开后复算结果与冻结研究不一致，拒绝接受")
    return recomputed
