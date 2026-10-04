"""拒答策略：按分组约束在选择资料上定版，再在独立核对资料上报告。

流程约束：

* 阈值与分组选择只使用 fit_abstention_policy 收到的选择资料；
  evaluate_policy 只把已定版策略应用到核对资料上，绝不根据核对结果
  重新调整阈值或分组；
* 一个样本可以同时属于多个保护组（例如同时属于某个小样本地区和某个
  风险等级）：总体账目中每个样本只计一次，各组账目分别核对该样本对
  本组覆盖约束的贡献，重叠组的账目加总因此可能超过总体，不会把重叠
  组拆成互斥组来改变原始约束；
* 同时属于多个组的样本必须通过自己的**所有**组的阈值（取最严者）才会
  被接受；任一组未确定或不可行时该样本按拒答处理；
* 代价（错误接受/正确接受/拒答）与各组最低覆盖要求是策略内容的一部分：
  修改任一者都会得到新的 policy_version，旧策略与旧结果仍可追溯；
* 资料不足（选择资料中无可用打分预测）或约束不可行（可达覆盖率低于最低
  要求）的组保留 undetermined/infeasible 状态与原因；核对时这些组整组
  按拒答处理并留在总体账目里，不会被悄悄排除来美化总体成绩；
* 不同组的最低覆盖与代价要求可能互相冲突（尤其存在高权重重叠样本时）：
  不可行的约束组合在 group_statuses 中明确标记并给出可达覆盖率，
  selection_results 同时保留当前策略实际达到的各组结果；
* 缺预测、冲突预测、未知预测类别的样本一律按拒答计入人数、覆盖率、
  代价与区间；组间无重叠时分组账目加总等于总体账目。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .predictions import PredictionBatch
from .resample import BootstrapResult, bootstrap_weighted_stat
from .versioning import content_version

STRATEGY_UNIFORM = "uniform"
STRATEGY_PER_GROUP = "per_group"
STRATEGIES = (STRATEGY_UNIFORM, STRATEGY_PER_GROUP)

STATUS_OK = "ok"
STATUS_UNDETERMINED = "undetermined"
STATUS_INFEASIBLE = "infeasible"
STATUS_PARTIAL = "partial"

DECISION_ACCEPTED_CORRECT = "accepted_correct"
DECISION_ACCEPTED_WRONG = "accepted_wrong"
DECISION_ABSTAIN_LOW_SCORE = "abstain_low_score"
DECISION_ABSTAIN_MISSING_SCORE = "abstain_missing_score"
DECISION_ABSTAIN_NO_THRESHOLD = "abstain_no_threshold"
DECISION_MISSING = "missing_prediction"
DECISION_CONFLICTING = "conflicting_prediction"
DECISION_UNKNOWN_LABEL = "unknown_predicted_label"

FORCED_ABSTAIN_STATUSES = (
    DECISION_MISSING,
    DECISION_CONFLICTING,
    DECISION_UNKNOWN_LABEL,
)

_ALL_GROUPS = "*"
_EPS = 1e-12


@dataclass(frozen=True)
class CostModel:
    """单次决策代价：错误接受、正确接受与拒答。"""

    false_accept: float
    correct_accept: float
    reject: float

    def __post_init__(self) -> None:
        for name in ("false_accept", "correct_accept", "reject"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"代价 {name} 必须是有限数值")
            if value < 0:
                raise ValueError(f"代价 {name} 不能为负")

    def to_dict(self) -> dict[str, float]:
        return {
            "false_accept": self.false_accept,
            "correct_accept": self.correct_accept,
            "reject": self.reject,
        }


@dataclass(frozen=True)
class GroupStatus:
    """单个组（或统一阈值整体）在选择资料上的定版状态。"""

    group: str
    status: str
    reason: str | None
    threshold: float | None
    min_coverage: float
    max_coverage: float | None
    n_scored: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "status": self.status,
            "reason": self.reason,
            "threshold": self.threshold,
            "min_coverage": self.min_coverage,
            "max_coverage": self.max_coverage,
            "n_scored": self.n_scored,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "GroupStatus":
        return GroupStatus(
            group=str(data["group"]),
            status=str(data["status"]),
            reason=data["reason"],
            threshold=data["threshold"],
            min_coverage=float(data["min_coverage"]),
            max_coverage=data["max_coverage"],
            n_scored=int(data["n_scored"]),
        )


@dataclass(frozen=True)
class DecisionRow:
    """单个样本在策略下的决策结果，携带权重、状态、所用阈值与全部归属组。"""

    sample_id: str
    groups: tuple[str, ...]
    weight: float
    status: str
    threshold: float | None
    cost: float
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "groups": list(self.groups),
            "weight": self.weight,
            "status": self.status,
            "threshold": self.threshold,
            "cost": self.cost,
            "accepted": self.accepted,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "DecisionRow":
        return DecisionRow(
            sample_id=str(data["sample_id"]),
            groups=tuple(str(g) for g in data["groups"]),
            weight=float(data["weight"]),
            status=str(data["status"]),
            threshold=data["threshold"],
            cost=float(data["cost"]),
            accepted=bool(data["accepted"]),
        )


@dataclass(frozen=True)
class GroupPolicyResult:
    """单组（或总体）的策略结果：人数、覆盖率、代价与区间。"""

    group: str | None
    status: str
    reason: str | None
    threshold: float | None
    min_coverage: float
    n_samples: int
    total_weight: float
    accepted_weight: float
    abstained_weight: float
    forced_abstain_weight: float
    coverage: float | None
    mean_cost: float | None
    total_cost: float
    status_counts: tuple[tuple[str, int], ...]
    coverage_ci: BootstrapResult | None = None
    cost_ci: BootstrapResult | None = None

    def to_dict(self) -> dict[str, Any]:
        def ci_dict(ci: BootstrapResult | None) -> dict[str, Any] | None:
            if ci is None:
                return None
            return {
                "metric": ci.metric,
                "point": ci.point,
                "lower": ci.lower,
                "upper": ci.upper,
                "n_replicates": ci.n_replicates,
                "seed": ci.seed,
                "confidence_level": ci.confidence_level,
                "replicate_version": ci.replicate_version,
            }

        return {
            "group": self.group,
            "status": self.status,
            "reason": self.reason,
            "threshold": self.threshold,
            "min_coverage": self.min_coverage,
            "n_samples": self.n_samples,
            "total_weight": self.total_weight,
            "accepted_weight": self.accepted_weight,
            "abstained_weight": self.abstained_weight,
            "forced_abstain_weight": self.forced_abstain_weight,
            "coverage": self.coverage,
            "mean_cost": self.mean_cost,
            "total_cost": self.total_cost,
            "status_counts": dict(self.status_counts),
            "coverage_ci": ci_dict(self.coverage_ci),
            "cost_ci": ci_dict(self.cost_ci),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "GroupPolicyResult":
        """从 to_dict 结果还原；置信区间随内容一并还原，保证保存重载后一致。"""

        def ci_from(raw: Mapping[str, Any] | None) -> BootstrapResult | None:
            if raw is None:
                return None
            return BootstrapResult(
                metric=str(raw["metric"]),
                point=float(raw["point"]),
                lower=float(raw["lower"]),
                upper=float(raw["upper"]),
                n_replicates=int(raw["n_replicates"]),
                seed=int(raw["seed"]),
                confidence_level=float(raw["confidence_level"]),
                replicate_version=str(raw["replicate_version"]),
            )

        return GroupPolicyResult(
            group=data["group"],
            status=str(data["status"]),
            reason=data["reason"],
            threshold=data["threshold"],
            min_coverage=float(data["min_coverage"]),
            n_samples=int(data["n_samples"]),
            total_weight=float(data["total_weight"]),
            accepted_weight=float(data["accepted_weight"]),
            abstained_weight=float(data["abstained_weight"]),
            forced_abstain_weight=float(data["forced_abstain_weight"]),
            coverage=data["coverage"],
            mean_cost=data["mean_cost"],
            total_cost=float(data["total_cost"]),
            status_counts=tuple(
                sorted((str(k), int(v)) for k, v in data["status_counts"].items())
            ),
            coverage_ci=ci_from(data.get("coverage_ci")),
            cost_ci=ci_from(data.get("cost_ci")),
        )


@dataclass(frozen=True)
class AbstentionPolicy:
    """定版后的拒答策略：阈值、代价、覆盖要求与选择资料版本。"""

    strategy: str
    group_attrs: tuple[str, ...]
    costs: CostModel
    min_coverage_default: float
    min_coverage_by_group: tuple[tuple[str, float], ...]
    thresholds: tuple[tuple[str, float | None], ...]
    group_statuses: tuple[GroupStatus, ...]
    selection_sample_set_version: str
    selection_prediction_batch_version: str
    selection_results: tuple[GroupPolicyResult, ...]
    policy_version: str

    def threshold_for(self, group: str) -> float | None:
        table = dict(self.thresholds)
        if self.strategy == STRATEGY_UNIFORM:
            return table.get(_ALL_GROUPS)
        return table.get(group)

    def min_coverage_for(self, group: str) -> float:
        return dict(self.min_coverage_by_group).get(
            group, self.min_coverage_default
        )

    def status_for(self, group: str) -> GroupStatus | None:
        for status in self.group_statuses:
            if status.group == group:
                return status
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "group_attrs": list(self.group_attrs),
            "costs": self.costs.to_dict(),
            "min_coverage_default": self.min_coverage_default,
            "min_coverage_by_group": dict(self.min_coverage_by_group),
            "thresholds": dict(self.thresholds),
            "group_statuses": [s.to_dict() for s in self.group_statuses],
            "selection_sample_set_version": self.selection_sample_set_version,
            "selection_prediction_batch_version": (
                self.selection_prediction_batch_version
            ),
            "selection_results": [r.to_dict() for r in self.selection_results],
            "policy_version": self.policy_version,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "AbstentionPolicy":
        """从 to_dict 结果还原策略（保存后重开），并校验内容版本未被篡改。

        还原出的策略与原策略产生完全相同的逐样本决策；policy_version
        由内容重算，与保存值不一致时拒绝载入。
        """
        statuses = tuple(
            GroupStatus.from_dict(s) for s in data["group_statuses"]
        )
        costs = CostModel(**data["costs"])
        group_reqs = tuple(
            sorted((str(k), float(v)) for k, v in data["min_coverage_by_group"].items())
        )
        threshold_table = {
            str(k): v for k, v in data["thresholds"].items()
        }
        policy = AbstentionPolicy(
            strategy=str(data["strategy"]),
            group_attrs=tuple(str(a) for a in data["group_attrs"]),
            costs=costs,
            min_coverage_default=float(data["min_coverage_default"]),
            min_coverage_by_group=group_reqs,
            thresholds=tuple(sorted(threshold_table.items())),
            group_statuses=statuses,
            selection_sample_set_version=str(data["selection_sample_set_version"]),
            selection_prediction_batch_version=str(
                data["selection_prediction_batch_version"]
            ),
            selection_results=tuple(
                GroupPolicyResult.from_dict(r) for r in data["selection_results"]
            ),
            policy_version=str(data["policy_version"]),
        )
        recomputed = content_version(
            [
                ("strategy", policy.strategy),
                ("group_attrs", list(policy.group_attrs)),
                ("costs", policy.costs.to_dict()),
                ("min_coverage_default", policy.min_coverage_default),
                ("min_coverage_by_group", list(policy.min_coverage_by_group)),
                (
                    "thresholds",
                    sorted(
                        (g, None if t is None else float(t))
                        for g, t in dict(policy.thresholds).items()
                    ),
                ),
                (
                    "group_statuses",
                    [
                        [s.group, s.status, s.reason, s.threshold, s.min_coverage]
                        for s in policy.group_statuses
                    ],
                ),
                (
                    "selection_sample_set_version",
                    policy.selection_sample_set_version,
                ),
                (
                    "selection_prediction_batch_version",
                    policy.selection_prediction_batch_version,
                ),
            ]
        )
        if recomputed != policy.policy_version:
            raise ValueError(
                "策略内容与其 policy_version 不一致，拒绝载入"
            )
        return policy


@dataclass(frozen=True)
class PolicyEvaluation:
    """定版策略在独立核对资料上的结果。"""

    policy_version: str
    evaluation_version: str
    strategy: str
    sample_set_version: str
    prediction_batch_version: str
    selection_sample_set_version: str
    selection_prediction_batch_version: str
    thresholds_source: str
    overall: GroupPolicyResult
    groups: tuple[GroupPolicyResult, ...]
    decisions: tuple[DecisionRow, ...] = ()
    n_replicates: int = 1000
    seed: int = 20260929
    confidence_level: float = 0.95
    bootstrap_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "evaluation_version": self.evaluation_version,
            "strategy": self.strategy,
            "sample_set_version": self.sample_set_version,
            "prediction_batch_version": self.prediction_batch_version,
            "selection_sample_set_version": self.selection_sample_set_version,
            "selection_prediction_batch_version": (
                self.selection_prediction_batch_version
            ),
            "thresholds_source": self.thresholds_source,
            "overall": self.overall.to_dict(),
            "groups": [g.to_dict() for g in self.groups],
            "decisions": [d.to_dict() for d in self.decisions],
            "bootstrap": {
                "enabled": self.bootstrap_enabled,
                "n_replicates": self.n_replicates,
                "seed": self.seed,
                "confidence_level": self.confidence_level,
            },
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "PolicyEvaluation":
        """从 to_dict 结果还原核对结果（保存后重载），并校验内容版本。

        还原出的结果与原结果完全一致（含逐样本决策与置信区间）；
        evaluation_version 由内容重算，与保存值不一致时拒绝载入。
        """
        bootstrap = data.get("bootstrap") or {}
        evaluation = PolicyEvaluation(
            policy_version=str(data["policy_version"]),
            evaluation_version=str(data["evaluation_version"]),
            strategy=str(data["strategy"]),
            sample_set_version=str(data["sample_set_version"]),
            prediction_batch_version=str(data["prediction_batch_version"]),
            selection_sample_set_version=str(
                data["selection_sample_set_version"]
            ),
            selection_prediction_batch_version=str(
                data["selection_prediction_batch_version"]
            ),
            thresholds_source=str(data["thresholds_source"]),
            overall=GroupPolicyResult.from_dict(data["overall"]),
            groups=tuple(
                GroupPolicyResult.from_dict(g) for g in data["groups"]
            ),
            decisions=tuple(
                DecisionRow.from_dict(d) for d in data.get("decisions", ())
            ),
            n_replicates=int(bootstrap.get("n_replicates", 1000)),
            seed=int(bootstrap.get("seed", 20260929)),
            confidence_level=float(bootstrap.get("confidence_level", 0.95)),
            bootstrap_enabled=bool(bootstrap.get("enabled", True)),
        )
        recomputed = content_version(
            [
                ("policy_version", evaluation.policy_version),
                ("holdout_sample_set_version", evaluation.sample_set_version),
                (
                    "holdout_prediction_batch_version",
                    evaluation.prediction_batch_version,
                ),
                (
                    "bootstrap",
                    [
                        evaluation.n_replicates,
                        evaluation.seed,
                        evaluation.confidence_level,
                        evaluation.bootstrap_enabled,
                    ],
                ),
                ("overall", evaluation.overall.to_dict()),
                ("groups", [g.to_dict() for g in evaluation.groups]),
                ("decisions", [d.to_dict() for d in evaluation.decisions]),
            ]
        )
        if recomputed != evaluation.evaluation_version:
            raise ValueError(
                "核对结果内容与其 evaluation_version 不一致，拒绝载入"
            )
        return evaluation


@dataclass(frozen=True)
class PolicyComparison:
    """多个定版策略（如统一阈值 vs 分组阈值）在同一核对资料上的并排比较。

    比较层只汇总各策略独立核对的结果，绝不在核对资料上重选阈值：
    summary 并排列出各策略的总体代价与覆盖率；unmet_constraints 列出
    核对资料上实际覆盖率低于最低要求的组；decision_changes 逐样本列出
    各策略决策不一致的位置。重叠组样本在总体账目中只计一次，缺预测
    保留在分母；to_dict/from_dict 保证保存重载后结果一致。
    """

    comparison_version: str
    sample_set_version: str
    evaluations: tuple[PolicyEvaluation, ...]
    summary: tuple[dict[str, Any], ...] = ()
    unmet_constraints: tuple[dict[str, Any], ...] = ()
    decision_changes: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_version": self.comparison_version,
            "sample_set_version": self.sample_set_version,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "summary": list(self.summary),
            "unmet_constraints": list(self.unmet_constraints),
            "decision_changes": list(self.decision_changes),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "PolicyComparison":
        """从 to_dict 结果还原比较结果（保存后重载），并校验内容版本。

        重载后的比较结果与原结果完全一致；comparison_version 由内容
        重算，与保存值不一致时拒绝载入。
        """
        comparison = PolicyComparison(
            comparison_version=str(data["comparison_version"]),
            sample_set_version=str(data["sample_set_version"]),
            evaluations=tuple(
                PolicyEvaluation.from_dict(e) for e in data["evaluations"]
            ),
            summary=tuple(data.get("summary", ())),
            unmet_constraints=tuple(data.get("unmet_constraints", ())),
            decision_changes=tuple(data.get("decision_changes", ())),
        )
        recomputed = content_version(
            [
                ("holdout_sample_set_version", comparison.sample_set_version),
                (
                    "evaluations",
                    [e.evaluation_version for e in comparison.evaluations],
                ),
                ("summary", list(comparison.summary)),
                ("unmet_constraints", list(comparison.unmet_constraints)),
                ("decision_changes", list(comparison.decision_changes)),
            ]
        )
        if recomputed != comparison.comparison_version:
            raise ValueError(
                "比较结果内容与其 comparison_version 不一致，拒绝载入"
            )
        return comparison


def _normalize_group_attrs(
    group_attr: "str | list[str] | tuple[str, ...] | None",
) -> tuple[str, ...]:
    """分组属性可以是一个属性名、属性名列表（样本同时属于多个保护组）或 None。"""
    if group_attr is None:
        return ()
    if isinstance(group_attr, str):
        return (group_attr,)
    attrs = tuple(str(a) for a in group_attr)
    if len(set(attrs)) != len(attrs):
        raise ValueError(f"分组属性存在重复: {attrs}")
    return attrs


def _group_keys(sample, group_attrs: tuple[str, ...]) -> tuple[str, ...]:
    """样本所属的全部保护组键（一个样本可同时属于多个组）。

    单属性时保持裸取值键以兼容既有报告；多属性时使用 属性=取值
    以避免不同属性的同名取值互相混淆。
    """
    if not group_attrs:
        return (_ALL_GROUPS,)
    keys: list[str] = []
    for attr in group_attrs:
        value = sample.groups.get(attr)
        token = "__unassigned__" if value is None else str(value)
        keys.append(token if len(group_attrs) == 1 else f"{attr}={token}")
    return tuple(keys)


def _normalize_min_coverage(
    min_coverage: float | Mapping[str, float],
) -> tuple[float, tuple[tuple[str, float], ...]]:
    """接受单一数值或 {组: 要求}（可用 "*" 作为缺省），返回 (缺省, 分组表)。"""

    def check(value: float) -> float:
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"最低覆盖要求必须在 [0, 1] 内: {value!r}")
        return float(value)

    if isinstance(min_coverage, Mapping):
        table = {str(k): check(v) for k, v in min_coverage.items()}
        default = table.pop(_ALL_GROUPS, 0.0)
        return default, tuple(sorted(table.items()))
    return check(min_coverage), ()


def _decisions(
    batch: PredictionBatch,
    group_attrs: tuple[str, ...],
    costs: CostModel,
    threshold_for,
) -> list[DecisionRow]:
    """对样本集里每一个样本给出决策；异常样本一律按拒答保留在账上。

    同时属于多个组的样本必须通过自己的所有组的阈值（取最严者）才会被
    接受；任一组未确定阈值（None）时该样本按拒答处理。每个样本只产生
    一条决策行，因此总体账目中每个样本只计一次。
    """
    quality = batch.summary
    sample_set = batch.sample_set
    eval_index = {r.sample_id: r for r in batch.evaluable_rows}
    missing_ids = set(quality.missing_sample_ids)
    conflicting_ids = {sid for sid, _ in quality.conflicting_rows}
    unknown_label_ids = {sid for sid, _ in quality.unknown_label_rows}

    rows: list[DecisionRow] = []
    for sample in sample_set.samples:
        sid = sample.sample_id
        keys = _group_keys(sample, group_attrs)
        applicable = [threshold_for(key) for key in keys]
        threshold = (
            None
            if any(t is None for t in applicable)
            else max(applicable)
        )
        accepted = False
        if sid in missing_ids:
            status = DECISION_MISSING
        elif sid in conflicting_ids:
            status = DECISION_CONFLICTING
        elif sid in unknown_label_ids:
            status = DECISION_UNKNOWN_LABEL
        elif threshold is None:
            status = DECISION_ABSTAIN_NO_THRESHOLD
        else:
            ev = eval_index[sid]
            if ev.score is None:
                status = DECISION_ABSTAIN_MISSING_SCORE
            elif ev.score >= threshold:
                accepted = True
                status = (
                    DECISION_ACCEPTED_CORRECT
                    if ev.predicted == ev.label
                    else DECISION_ACCEPTED_WRONG
                )
            else:
                status = DECISION_ABSTAIN_LOW_SCORE
        if status == DECISION_ACCEPTED_CORRECT:
            cost = costs.correct_accept
        elif status == DECISION_ACCEPTED_WRONG:
            cost = costs.false_accept
        else:
            cost = costs.reject
        rows.append(
            DecisionRow(
                sample_id=sid,
                groups=keys,
                weight=float(sample.weight),
                status=status,
                threshold=threshold,
                cost=float(cost),
                accepted=accepted,
            )
        )
    return rows


def _summarize(
    group: str | None,
    rows: list[DecisionRow],
    status: str,
    reason: str | None,
    threshold: float | None,
    min_coverage: float,
    with_ci: bool,
    n_replicates: int,
    seed: int,
    confidence_level: float,
) -> GroupPolicyResult:
    total_weight = float(sum(r.weight for r in rows))
    accepted_weight = float(sum(r.weight for r in rows if r.accepted))
    forced_weight = float(
        sum(r.weight for r in rows if r.status in FORCED_ABSTAIN_STATUSES)
    )
    total_cost = float(sum(r.cost * r.weight for r in rows))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    coverage = accepted_weight / total_weight if total_weight > 0 else None
    mean_cost = total_cost / total_weight if total_weight > 0 else None
    coverage_ci = None
    cost_ci = None
    if with_ci and rows:
        coverage_ci = bootstrap_weighted_stat(
            rows,
            lambda r: 1.0 if r.accepted else 0.0,
            "coverage",
            n_replicates=n_replicates,
            seed=seed,
            confidence_level=confidence_level,
        )
        cost_ci = bootstrap_weighted_stat(
            rows,
            lambda r: r.cost,
            "mean_cost",
            n_replicates=n_replicates,
            seed=seed,
            confidence_level=confidence_level,
        )
    return GroupPolicyResult(
        group=group,
        status=status,
        reason=reason,
        threshold=threshold,
        min_coverage=min_coverage,
        n_samples=len(rows),
        total_weight=total_weight,
        accepted_weight=accepted_weight,
        abstained_weight=total_weight - accepted_weight,
        forced_abstain_weight=forced_weight,
        coverage=coverage,
        mean_cost=mean_cost,
        total_cost=total_cost,
        status_counts=tuple(sorted(counts.items())),
        coverage_ci=coverage_ci,
        cost_ci=cost_ci,
    )


def fit_abstention_policy(
    selection_batch: PredictionBatch,
    group_attr: "str | list[str] | tuple[str, ...] | None" = None,
    costs: CostModel | None = None,
    min_coverage: float | Mapping[str, float] = 0.0,
    strategy: str = STRATEGY_PER_GROUP,
) -> AbstentionPolicy:
    """在选择资料上定版拒答策略。

    "group_attr" 可以是单个属性名或属性名列表；给出多个属性时一个样本
    同时属于多个保护组（如地区与风险等级），总体账目仍每个样本只计一次，
    各组分别核对该样本对本组覆盖约束的贡献，重叠组不会被拆成互斥组。
    "strategy=\"uniform\"" 选一个同时满足各组最低覆盖要求的统一阈值；
    "strategy=\"per_group\"" 为每组选阈值。候选阈值取选择资料中出现的
    打分值（外加“全部拒答”），在满足最低覆盖要求的候选中取代价最小者。
    组间存在重叠样本时各组约束互相耦合：per_group 先以“全部接受”确定
    各组可达覆盖率上限并标记不可行组，再以坐标下降逐组降低总体代价，
    最后做联合约束终检；仍不可行的约束组合明确标记 infeasible 并给出
    可达覆盖率，当前策略实际达到的各组结果保留在 selection_results 中。
    """
    if costs is None:
        raise ValueError("必须显式给出 CostModel（错误接受/正确接受/拒答代价）")
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略 {strategy!r}，可选: {STRATEGIES}")
    attrs = _normalize_group_attrs(group_attr)
    default_req, group_reqs = _normalize_min_coverage(min_coverage)
    req_table = dict(group_reqs)

    def req_for(group: str) -> float:
        return req_table.get(group, default_req)

    sample_set = selection_batch.sample_set
    groups = sorted(
        {
            key
            for sample in sample_set.samples
            for key in _group_keys(sample, attrs)
        }
    )
    scored_by_group: dict[str, list[float]] = {g: [] for g in groups}
    for row in selection_batch.evaluable_rows:
        if row.score is not None:
            for key in _group_keys(sample_set.get(row.sample_id), attrs):
                scored_by_group.setdefault(key, []).append(row.score)

    def coverage_and_cost(
        threshold_map: Mapping[str, "float | None"],
    ) -> tuple[dict[str, tuple[float, float]], list[DecisionRow], float]:
        """给定各组阈值，返回 ({组: (覆盖率, 总权重)}, 决策行, 总体加权代价)。

        总体代价按样本计一次；组覆盖率把重叠样本分别计入其每个归属组。
        """
        decisions = _decisions(
            selection_batch, attrs, costs, lambda g: threshold_map.get(g)
        )
        table: dict[str, tuple[float, float]] = {}
        for group in groups:
            sub = [r for r in decisions if group in r.groups]
            total = sum(r.weight for r in sub)
            accepted = sum(r.weight for r in sub if r.accepted)
            table[group] = (accepted / total if total > 0 else 0.0, total)
        total_cost = sum(r.cost * r.weight for r in decisions)
        return table, decisions, total_cost

    statuses: list[GroupStatus] = []
    thresholds: dict[str, float | None] = {}

    if strategy == STRATEGY_PER_GROUP:
        candidates = {
            g: sorted(set(scored_by_group.get(g, []))) + [math.inf]
            for g in groups
        }
        undetermined = {g for g in groups if not scored_by_group.get(g)}
        threshold_map: dict[str, float | None] = {
            g: (None if g in undetermined else min(scored_by_group[g]))
            for g in groups
        }
        # 全部接受（各自最低候选阈值）时各组覆盖率达到可达上限
        max_cov_table, _, _ = coverage_and_cost(threshold_map)
        infeasible: dict[str, float] = {}
        # 不可行组整组拒答会降低与其重叠组的覆盖率，迭代到不动点
        changed = True
        while changed:
            changed = False
            cov_table, _, _ = coverage_and_cost(threshold_map)
            for group in groups:
                if group in undetermined or group in infeasible:
                    continue
                if cov_table[group][0] + _EPS < req_for(group):
                    infeasible[group] = max_cov_table[group][0]
                    threshold_map[group] = None
                    changed = True
        ok_groups = [
            g for g in groups if g not in undetermined and g not in infeasible
        ]
        # 坐标下降：每次只调整一个组，在满足本组覆盖要求的前提下降低总体代价
        for _ in range(100):
            stable = True
            for group in ok_groups:
                best: tuple[float, float] | None = None
                for threshold in candidates[group]:
                    trial = dict(threshold_map)
                    trial[group] = threshold
                    cov_table, _, cost = coverage_and_cost(trial)
                    if cov_table[group][0] + _EPS < req_for(group):
                        continue
                    if best is None or cost < best[1] - _EPS:
                        best = (threshold, cost)
                if best is not None and threshold_map[group] != best[0]:
                    threshold_map[group] = best[0]
                    stable = False
            if stable:
                break
        # 联合约束终检：重叠样本使组间耦合，他组阈值上调可能破坏本组约束
        changed = True
        while changed:
            changed = False
            cov_table, _, _ = coverage_and_cost(threshold_map)
            for group in list(ok_groups):
                if cov_table[group][0] + _EPS < req_for(group):
                    infeasible[group] = cov_table[group][0]
                    threshold_map[group] = None
                    ok_groups.remove(group)
                    changed = True
        for group in groups:
            req = req_for(group)
            n_scored = len(scored_by_group.get(group, []))
            if group in undetermined:
                statuses.append(
                    GroupStatus(
                        group=group,
                        status=STATUS_UNDETERMINED,
                        reason="选择资料中该组没有可用打分预测，无法定阈值",
                        threshold=None,
                        min_coverage=req,
                        max_coverage=None,
                        n_scored=0,
                    )
                )
            elif group in infeasible:
                statuses.append(
                    GroupStatus(
                        group=group,
                        status=STATUS_INFEASIBLE,
                        reason=(
                            f"可达覆盖率 {infeasible[group]:.6f} "
                            f"低于最低要求 {req:.6f}"
                        ),
                        threshold=None,
                        min_coverage=req,
                        max_coverage=infeasible[group],
                        n_scored=n_scored,
                    )
                )
            else:
                statuses.append(
                    GroupStatus(
                        group=group,
                        status=STATUS_OK,
                        reason=None,
                        threshold=threshold_map[group],
                        min_coverage=req,
                        max_coverage=max_cov_table[group][0],
                        n_scored=n_scored,
                    )
                )
            thresholds[group] = threshold_map[group]
    else:
        all_scores = sorted(
            {s for scores in scored_by_group.values() for s in scores}
        )
        best_uniform: tuple[float, float] | None = None
        for threshold in all_scores + [math.inf]:
            tmap = {g: threshold for g in groups}
            cov_table, _, cost = coverage_and_cost(tmap)
            if all(
                cov_table[g][0] + _EPS >= req_for(g) for g in groups
            ) and (best_uniform is None or cost < best_uniform[1] - _EPS):
                best_uniform = (threshold, cost)
        min_map: dict[str, float | None] = {
            g: (min(all_scores) if all_scores else None) for g in groups
        }
        max_cov_table, _, _ = coverage_and_cost(min_map)
        if best_uniform is None:
            for group in groups:
                req = req_for(group)
                scores = scored_by_group.get(group, [])
                if not scores:
                    statuses.append(
                        GroupStatus(
                            group=group,
                            status=STATUS_UNDETERMINED,
                            reason="选择资料中该组没有可用打分预测，无法定阈值",
                            threshold=None,
                            min_coverage=req,
                            max_coverage=None,
                            n_scored=0,
                        )
                    )
                    continue
                max_cov = max_cov_table[group][0]
                statuses.append(
                    GroupStatus(
                        group=group,
                        status=(
                            STATUS_INFEASIBLE
                            if max_cov + _EPS < req
                            else STATUS_OK
                        ),
                        reason=(
                            None
                            if max_cov + _EPS >= req
                            else (
                                f"可达覆盖率 {max_cov:.6f} "
                                f"低于最低要求 {req:.6f}"
                            )
                        ),
                        threshold=None,
                        min_coverage=req,
                        max_coverage=max_cov,
                        n_scored=len(scores),
                    )
                )
            statuses.append(
                GroupStatus(
                    group=_ALL_GROUPS,
                    status=STATUS_INFEASIBLE,
                    reason="不存在同时满足各组最低覆盖要求的统一阈值",
                    threshold=None,
                    min_coverage=default_req,
                    max_coverage=None,
                    n_scored=sum(len(s) for s in scored_by_group.values()),
                )
            )
            thresholds[_ALL_GROUPS] = None
        else:
            for group in groups:
                req = req_for(group)
                scores = scored_by_group.get(group, [])
                statuses.append(
                    GroupStatus(
                        group=group,
                        status=STATUS_OK,
                        reason=None,
                        threshold=best_uniform[0],
                        min_coverage=req,
                        max_coverage=max_cov_table[group][0],
                        n_scored=len(scores),
                    )
                )
            thresholds[_ALL_GROUPS] = best_uniform[0]

    threshold_table = (
        thresholds
        if strategy == STRATEGY_PER_GROUP
        else {_ALL_GROUPS: thresholds[_ALL_GROUPS]}
    )
    selection_decisions = _decisions(
        selection_batch,
        attrs,
        costs,
        lambda g: (
            threshold_table[_ALL_GROUPS]
            if strategy == STRATEGY_UNIFORM
            else threshold_table.get(g)
        ),
    )
    selection_results: list[GroupPolicyResult] = []
    for group in groups:
        sub = [r for r in selection_decisions if group in r.groups]
        st = next(s for s in statuses if s.group == group)
        selection_results.append(
            _summarize(
                group, sub, st.status, st.reason, st.threshold,
                req_for(group), False, 0, 0, 0.95,
            )
        )
    non_ok = [s for s in statuses if s.group != _ALL_GROUPS and s.status != STATUS_OK]
    overall_status = STATUS_OK if not non_ok else STATUS_PARTIAL
    overall_reason = (
        None
        if not non_ok
        else "存在未确定或不可行的组: " + ", ".join(s.group for s in non_ok)
    )
    selection_results.append(
        _summarize(
            None, selection_decisions, overall_status, overall_reason,
            threshold_table.get(_ALL_GROUPS), default_req, False, 0, 0, 0.95,
        )
    )

    policy_version = content_version(
        [
            ("strategy", strategy),
            ("group_attrs", list(attrs)),
            ("costs", costs.to_dict()),
            ("min_coverage_default", default_req),
            ("min_coverage_by_group", list(group_reqs)),
            (
                "thresholds",
                sorted(
                    (g, None if t is None else float(t))
                    for g, t in threshold_table.items()
                ),
            ),
            (
                "group_statuses",
                [
                    [s.group, s.status, s.reason, s.threshold, s.min_coverage]
                    for s in statuses
                ],
            ),
            ("selection_sample_set_version", sample_set.version),
            ("selection_prediction_batch_version", selection_batch.version),
        ]
    )
    return AbstentionPolicy(
        strategy=strategy,
        group_attrs=attrs,
        costs=costs,
        min_coverage_default=default_req,
        min_coverage_by_group=group_reqs,
        thresholds=tuple(sorted(threshold_table.items())),
        group_statuses=tuple(statuses),
        selection_sample_set_version=sample_set.version,
        selection_prediction_batch_version=selection_batch.version,
        selection_results=tuple(selection_results),
        policy_version=policy_version,
    )


def evaluate_policy(
    policy: AbstentionPolicy,
    holdout_batch: PredictionBatch,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
    bootstrap: bool = True,
) -> PolicyEvaluation:
    """把已定版策略应用到独立核对资料上。

    阈值与分组完全来自 `policy`（即选择资料），这里只重放决策并汇总，
    不因核对结果不理想而重新调阈值。策略未覆盖的组、未确定或不可行的组
    整组按拒答处理并保留在总体账目中。同时属于多个组的样本在总体账目
    中只计一次，并分别计入其每个归属组的组级账目。
    """
    decisions = _decisions(
        holdout_batch, policy.group_attrs, policy.costs, policy.threshold_for
    )
    policy_groups = {
        s.group for s in policy.group_statuses if s.group != _ALL_GROUPS
    }
    holdout_groups = {key for r in decisions for key in r.groups}
    groups = sorted(policy_groups | holdout_groups)

    group_results: list[GroupPolicyResult] = []
    for group in groups:
        sub = [r for r in decisions if group in r.groups]
        status = policy.status_for(group)
        if status is None:
            g_status = STATUS_UNDETERMINED
            reason = "策略未覆盖该组（选择资料中不存在），整组按拒答处理"
            threshold = None
        else:
            g_status = status.status
            reason = status.reason
            threshold = status.threshold
        group_results.append(
            _summarize(
                group,
                sub,
                g_status,
                reason,
                threshold,
                policy.min_coverage_for(group),
                bootstrap,
                n_replicates,
                seed,
                confidence_level,
            )
        )

    non_ok = [g for g in group_results if g.status != STATUS_OK]
    overall_status = STATUS_OK if not non_ok else STATUS_PARTIAL
    overall_reason = (
        None
        if not non_ok
        else "存在未确定或不可行的组: " + ", ".join(g.group for g in non_ok)
    )
    overall = _summarize(
        None,
        decisions,
        overall_status,
        overall_reason,
        policy.threshold_for(_ALL_GROUPS),
        policy.min_coverage_default,
        bootstrap,
        n_replicates,
        seed,
        confidence_level,
    )
    evaluation_version = content_version(
        [
            ("policy_version", policy.policy_version),
            ("holdout_sample_set_version", holdout_batch.sample_set.version),
            ("holdout_prediction_batch_version", holdout_batch.version),
            ("bootstrap", [n_replicates, seed, confidence_level, bootstrap]),
            ("overall", overall.to_dict()),
            ("groups", [g.to_dict() for g in group_results]),
            ("decisions", [d.to_dict() for d in decisions]),
        ]
    )
    return PolicyEvaluation(
        policy_version=policy.policy_version,
        evaluation_version=evaluation_version,
        strategy=policy.strategy,
        sample_set_version=holdout_batch.sample_set.version,
        prediction_batch_version=holdout_batch.version,
        selection_sample_set_version=policy.selection_sample_set_version,
        selection_prediction_batch_version=(
            policy.selection_prediction_batch_version
        ),
        thresholds_source="selection",
        overall=overall,
        groups=tuple(group_results),
        decisions=tuple(decisions),
        n_replicates=n_replicates,
        seed=seed,
        confidence_level=confidence_level,
        bootstrap_enabled=bootstrap,
    )


def compare_policies(
    policies: list[AbstentionPolicy] | tuple[AbstentionPolicy, ...],
    holdout_batch: PredictionBatch,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
) -> PolicyComparison:
    """在同一核对资料上并排比较多个定版策略（如统一阈值 vs 分组阈值）。

    每个策略独立走 evaluate_policy（只重放冻结阈值，绝不在核对资料上
    重选阈值），比较层只汇总不重算：summary 并排列出各策略总体代价、
    覆盖率与各保护组实际覆盖率；unmet_constraints 列出核对资料上实际
    覆盖率低于最低要求的组；decision_changes 逐样本列出各策略决策不
    一致的位置。重叠组样本在总体账目中只计一次，缺预测保留在分母；
    to_dict/from_dict 保存重载后结果完全一致（版本号由内容重算校验）。
    """
    policies = tuple(policies)
    if not policies:
        raise ValueError("比较至少需要一个策略")
    versions = [p.policy_version for p in policies]
    if len(set(versions)) != len(versions):
        raise ValueError("参与比较的策略版本必须互不相同")
    evaluations = tuple(
        evaluate_policy(
            policy,
            holdout_batch,
            n_replicates=n_replicates,
            seed=seed,
            confidence_level=confidence_level,
        )
        for policy in policies
    )
    summary: list[dict[str, Any]] = []
    unmet: list[dict[str, Any]] = []
    for evaluation in evaluations:
        overall = evaluation.overall
        summary.append(
            {
                "policy_version": evaluation.policy_version,
                "strategy": evaluation.strategy,
                "total_cost": overall.total_cost,
                "mean_cost": overall.mean_cost,
                "coverage": overall.coverage,
                "accepted_weight": overall.accepted_weight,
                "total_weight": overall.total_weight,
                "status": overall.status,
                "group_coverage": {
                    g.group: g.coverage for g in evaluation.groups
                },
            }
        )
        for group in evaluation.groups:
            if (
                group.coverage is not None
                and group.coverage + _EPS < group.min_coverage
            ):
                unmet.append(
                    {
                        "policy_version": evaluation.policy_version,
                        "strategy": evaluation.strategy,
                        "group": group.group,
                        "min_coverage": group.min_coverage,
                        "actual_coverage": group.coverage,
                        "shortfall": group.min_coverage - group.coverage,
                    }
                )
    by_sample: dict[str, dict[str, DecisionRow]] = {}
    for evaluation in evaluations:
        for row in evaluation.decisions:
            by_sample.setdefault(row.sample_id, {})[evaluation.policy_version] = row
    changes: list[dict[str, Any]] = []
    for sample_id in sorted(by_sample):
        per_policy = by_sample[sample_id]
        signatures = {
            (row.status, row.accepted) for row in per_policy.values()
        }
        if len(signatures) <= 1:
            continue
        changes.append(
            {
                "sample_id": sample_id,
                "decisions": {
                    version: {
                        "status": row.status,
                        "accepted": row.accepted,
                        "threshold": row.threshold,
                        "cost": row.cost,
                    }
                    for version, row in sorted(per_policy.items())
                },
            }
        )
    comparison_version = content_version(
        [
            ("holdout_sample_set_version", holdout_batch.sample_set.version),
            ("evaluations", [e.evaluation_version for e in evaluations]),
            ("summary", summary),
            ("unmet_constraints", unmet),
            ("decision_changes", changes),
        ]
    )
    return PolicyComparison(
        comparison_version=comparison_version,
        sample_set_version=holdout_batch.sample_set.version,
        evaluations=evaluations,
        summary=tuple(summary),
        unmet_constraints=tuple(unmet),
        decision_changes=tuple(changes),
    )
