"""多重保护组约束下的策略选择研究。

给定**冻结的核对资料**、一组**已定版（冻结）的决策策略**与总体成本上限，
寻找同时满足各保护组最低覆盖与组间最大差异约束的可行策略，并比较两种
目标：成本最小（min_cost）与最坏组改善（worst_group，最大化最差组覆盖）。

流程约束：

* 研究只在核对资料上**重放**冻结策略（evaluate_policy），绝不根据核对
  结果重调阈值、不重开留出集、不改动样本权重与逐样本判定；
* 一个样本可以同时属于多个保护组：总体收益（覆盖率）与总体代价对每个
  样本只计一次，各组覆盖率分别把该样本计入其每个归属组，重叠组不拆成
  互斥组；
* 缺失分组标签的样本保留未知归属（组键为 __unassigned__），作为独立
  组参与最低覆盖与差异约束的核对，不会被默认归入满足约束的一侧，也不
  会被悄悄排除；
* 约束冲突时逐条列出无法同时满足的组与限制（最低覆盖缺口、差异超限、
  成本超限），不通过调整留出集或阈值制造可行结果；
* 新策略到达只能形成新研究：把新策略与旧策略一起重新运行得到新的
  study_version，旧研究、旧策略、样本权重与判定保持冻结可追溯；
* study_version 由内容重算校验，保存重开后结果完全一致，篡改即拒绝载入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .policy import (
    AbstentionPolicy,
    PolicyEvaluation,
    evaluate_policy,
    _normalize_min_coverage,
)
from .predictions import PredictionBatch
from .versioning import content_version

OBJECTIVE_MIN_COST = "min_cost"
OBJECTIVE_WORST_GROUP = "worst_group"
OBJECTIVES = (OBJECTIVE_MIN_COST, OBJECTIVE_WORST_GROUP)

_EPS = 1e-12


@dataclass(frozen=True)
class ConstraintViolation:
    """单条未满足的约束：类型、涉及组、限制值、实际值与缺口。"""

    constraint: str  # "min_coverage" | "max_disparity" | "cost_cap"
    group: str | None
    limit: float
    actual: float
    shortfall: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint,
            "group": self.group,
            "limit": self.limit,
            "actual": self.actual,
            "shortfall": self.shortfall,
            "detail": self.detail,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ConstraintViolation":
        return ConstraintViolation(
            constraint=str(data["constraint"]),
            group=data["group"],
            limit=float(data["limit"]),
            actual=float(data["actual"]),
            shortfall=float(data["shortfall"]),
            detail=str(data["detail"]),
        )


@dataclass(frozen=True)
class PolicyAssessment:
    """单个冻结策略在核对资料上的约束核对结果（只重放，不重调）。"""

    policy_version: str
    strategy: str
    feasible: bool
    total_cost: float
    coverage: float | None
    worst_group_coverage: float | None
    disparity: float | None
    group_coverages: tuple[tuple[str, float | None], ...]
    violations: tuple[ConstraintViolation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "strategy": self.strategy,
            "feasible": self.feasible,
            "total_cost": self.total_cost,
            "coverage": self.coverage,
            "worst_group_coverage": self.worst_group_coverage,
            "disparity": self.disparity,
            "group_coverages": dict(self.group_coverages),
            "violations": [v.to_dict() for v in self.violations],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "PolicyAssessment":
        return PolicyAssessment(
            policy_version=str(data["policy_version"]),
            strategy=str(data["strategy"]),
            feasible=bool(data["feasible"]),
            total_cost=float(data["total_cost"]),
            coverage=data["coverage"],
            worst_group_coverage=data["worst_group_coverage"],
            disparity=data["disparity"],
            group_coverages=tuple(
                sorted((str(k), v) for k, v in data["group_coverages"].items())
            ),
            violations=tuple(
                ConstraintViolation.from_dict(v) for v in data["violations"]
            ),
        )


@dataclass(frozen=True)
class PolicyStudy:
    """一次策略选择研究：约束、逐策略核对、两种目标的选择与并排比较。"""

    study_version: str
    sample_set_version: str
    prediction_batch_version: str
    cost_cap: float
    max_disparity: float
    min_coverage_default: float
    min_coverage_by_group: tuple[tuple[str, float], ...]
    assessments: tuple[PolicyAssessment, ...]
    selections: tuple[tuple[str, str | None], ...]
    objective_comparison: tuple[dict[str, Any], ...]
    evaluations: tuple[PolicyEvaluation, ...]

    def selection_for(self, objective: str) -> str | None:
        return dict(self.selections).get(objective)

    def assessment_for(self, policy_version: str) -> PolicyAssessment | None:
        for assessment in self.assessments:
            if assessment.policy_version == policy_version:
                return assessment
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_version": self.study_version,
            "sample_set_version": self.sample_set_version,
            "prediction_batch_version": self.prediction_batch_version,
            "cost_cap": self.cost_cap,
            "max_disparity": self.max_disparity,
            "min_coverage_default": self.min_coverage_default,
            "min_coverage_by_group": dict(self.min_coverage_by_group),
            "assessments": [a.to_dict() for a in self.assessments],
            "selections": dict(self.selections),
            "objective_comparison": list(self.objective_comparison),
            "evaluations": [e.to_dict() for e in self.evaluations],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "PolicyStudy":
        """从 to_dict 结果还原研究（保存后重开），并校验内容版本未被篡改。

        还原出的研究与原研究完全一致；study_version 由内容重算，
        与保存值不一致时拒绝载入。
        """
        study = PolicyStudy(
            study_version=str(data["study_version"]),
            sample_set_version=str(data["sample_set_version"]),
            prediction_batch_version=str(data["prediction_batch_version"]),
            cost_cap=float(data["cost_cap"]),
            max_disparity=float(data["max_disparity"]),
            min_coverage_default=float(data["min_coverage_default"]),
            min_coverage_by_group=tuple(
                sorted(
                    (str(k), float(v))
                    for k, v in data["min_coverage_by_group"].items()
                )
            ),
            assessments=tuple(
                PolicyAssessment.from_dict(a) for a in data["assessments"]
            ),
            selections=tuple(
                sorted((str(k), v) for k, v in data["selections"].items())
            ),
            objective_comparison=tuple(data["objective_comparison"]),
            evaluations=tuple(
                PolicyEvaluation.from_dict(e) for e in data["evaluations"]
            ),
        )
        if _study_version(study) != study.study_version:
            raise ValueError("研究内容与其 study_version 不一致，拒绝载入")
        return study


def _study_version(study: PolicyStudy) -> str:
    return content_version(
        [
            ("holdout_sample_set_version", study.sample_set_version),
            ("holdout_prediction_batch_version", study.prediction_batch_version),
            ("cost_cap", study.cost_cap),
            ("max_disparity", study.max_disparity),
            ("min_coverage_default", study.min_coverage_default),
            ("min_coverage_by_group", list(study.min_coverage_by_group)),
            ("assessments", [a.to_dict() for a in study.assessments]),
            ("selections", list(study.selections)),
            ("objective_comparison", list(study.objective_comparison)),
            ("evaluations", [e.evaluation_version for e in study.evaluations]),
        ]
    )


def _assess(
    evaluation: PolicyEvaluation,
    cost_cap: float,
    max_disparity: float,
    default_req: float,
    req_table: Mapping[str, float],
) -> PolicyAssessment:
    """核对一个冻结策略在核对资料上是否同时满足全部约束。"""
    violations: list[ConstraintViolation] = []
    group_coverages: dict[str, float | None] = {}
    for group in evaluation.groups:
        coverage = group.coverage
        group_coverages[group.group] = coverage
        required = req_table.get(group.group, default_req)
        if coverage is None:
            violations.append(
                ConstraintViolation(
                    constraint="min_coverage",
                    group=group.group,
                    limit=required,
                    actual=0.0,
                    shortfall=required,
                    detail="该组在核对资料中没有可评估样本，覆盖率未知",
                )
            )
        elif coverage + _EPS < required:
            violations.append(
                ConstraintViolation(
                    constraint="min_coverage",
                    group=group.group,
                    limit=required,
                    actual=coverage,
                    shortfall=required - coverage,
                    detail=(
                        f"组 {group.group} 覆盖率 {coverage:.6f} "
                        f"低于最低要求 {required:.6f}"
                    ),
                )
            )
    known = [c for c in group_coverages.values() if c is not None]
    disparity = (max(known) - min(known)) if len(known) >= 2 else 0.0
    if disparity > max_disparity + _EPS:
        worst = min(
            (g for g, c in group_coverages.items() if c is not None),
            key=lambda g: group_coverages[g],
        )
        best = max(
            (g for g, c in group_coverages.items() if c is not None),
            key=lambda g: group_coverages[g],
        )
        violations.append(
            ConstraintViolation(
                constraint="max_disparity",
                group=None,
                limit=max_disparity,
                actual=disparity,
                shortfall=disparity - max_disparity,
                detail=(
                    f"组间覆盖率差异 {disparity:.6f} 超过上限 "
                    f"{max_disparity:.6f}（最低 {worst}，最高 {best}）"
                ),
            )
        )
    total_cost = evaluation.overall.total_cost
    if total_cost > cost_cap + _EPS:
        violations.append(
            ConstraintViolation(
                constraint="cost_cap",
                group=None,
                limit=cost_cap,
                actual=total_cost,
                shortfall=total_cost - cost_cap,
                detail=f"总体加权代价 {total_cost:.6f} 超过上限 {cost_cap:.6f}",
            )
        )
    worst_group = min(known) if known else None
    return PolicyAssessment(
        policy_version=evaluation.policy_version,
        strategy=evaluation.strategy,
        feasible=not violations,
        total_cost=total_cost,
        coverage=evaluation.overall.coverage,
        worst_group_coverage=worst_group,
        disparity=disparity,
        group_coverages=tuple(sorted(group_coverages.items())),
        violations=tuple(violations),
    )


def run_policy_study(
    policies: list[AbstentionPolicy] | tuple[AbstentionPolicy, ...],
    holdout_batch: PredictionBatch,
    cost_cap: float,
    min_coverage: float | Mapping[str, float] = 0.0,
    max_disparity: float = 1.0,
) -> PolicyStudy:
    """在冻结核对资料上运行策略选择研究。

    "cost_cap" 为总体加权代价上限（每个样本只计一次）；"min_coverage"
    为各保护组最低覆盖要求（单一数值或 {组: 要求}，"*" 为缺省）；
    "max_disparity" 为组间覆盖率最大允许差异。候选策略必须是已定版
    （冻结）的 AbstentionPolicy；研究只重放其决策，绝不重调阈值或
    改动留出集。约束冲突时每个策略的 violations 逐条列出无法满足的
    组与限制；没有任何可行策略时两种目标的选择均为 None。
    新策略到达时把它加入 policies 重新运行即得到新研究（新
    study_version），旧研究保持不变。
    """
    if not isinstance(cost_cap, (int, float)) or cost_cap < 0:
        raise ValueError(f"总体成本上限必须是非负数值: {cost_cap!r}")
    if not isinstance(max_disparity, (int, float)) or not 0.0 <= max_disparity <= 1.0:
        raise ValueError(f"最大差异约束必须在 [0, 1] 内: {max_disparity!r}")
    policies = tuple(policies)
    if not policies:
        raise ValueError("研究至少需要一个候选策略")
    versions = [p.policy_version for p in policies]
    if len(set(versions)) != len(versions):
        raise ValueError("参与研究的策略版本必须互不相同")
    default_req, group_reqs = _normalize_min_coverage(min_coverage)
    req_table = dict(group_reqs)

    evaluations = tuple(
        evaluate_policy(policy, holdout_batch, bootstrap=False)
        for policy in policies
    )
    assessments = tuple(
        _assess(evaluation, cost_cap, max_disparity, default_req, req_table)
        for evaluation in evaluations
    )
    feasible = [a for a in assessments if a.feasible]

    def select(objective: str) -> str | None:
        if not feasible:
            return None
        if objective == OBJECTIVE_MIN_COST:
            best = min(feasible, key=lambda a: (a.total_cost, a.policy_version))
        else:
            best = min(
                feasible,
                key=lambda a: (
                    -(a.worst_group_coverage or 0.0),
                    a.total_cost,
                    a.policy_version,
                ),
            )
        return best.policy_version

    selections = tuple((objective, select(objective)) for objective in OBJECTIVES)
    comparison: list[dict[str, Any]] = []
    for objective, chosen in selections:
        if chosen is None:
            comparison.append(
                {
                    "objective": objective,
                    "policy_version": None,
                    "total_cost": None,
                    "coverage": None,
                    "worst_group_coverage": None,
                    "note": "没有同时满足全部约束的可行策略",
                }
            )
            continue
        chosen_assessment = next(
            a for a in assessments if a.policy_version == chosen
        )
        comparison.append(
            {
                "objective": objective,
                "policy_version": chosen,
                "total_cost": chosen_assessment.total_cost,
                "coverage": chosen_assessment.coverage,
                "worst_group_coverage": chosen_assessment.worst_group_coverage,
                "note": None,
            }
        )
    draft = PolicyStudy(
        study_version="",
        sample_set_version=holdout_batch.sample_set.version,
        prediction_batch_version=holdout_batch.version,
        cost_cap=float(cost_cap),
        max_disparity=float(max_disparity),
        min_coverage_default=default_req,
        min_coverage_by_group=group_reqs,
        assessments=assessments,
        selections=selections,
        objective_comparison=tuple(comparison),
        evaluations=evaluations,
    )
    return PolicyStudy(
        study_version=_study_version(draft),
        sample_set_version=draft.sample_set_version,
        prediction_batch_version=draft.prediction_batch_version,
        cost_cap=draft.cost_cap,
        max_disparity=draft.max_disparity,
        min_coverage_default=draft.min_coverage_default,
        min_coverage_by_group=draft.min_coverage_by_group,
        assessments=draft.assessments,
        selections=draft.selections,
        objective_comparison=draft.objective_comparison,
        evaluations=draft.evaluations,
    )

