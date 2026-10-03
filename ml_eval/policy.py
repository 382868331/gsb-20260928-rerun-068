"""拒答策略：按分组约束在选择资料上定版，再在独立核对资料上报告。

流程约束：

* 阈值与分组选择只使用 fit_abstention_policy 收到的选择资料；
  evaluate_policy 只把已定版策略应用到核对资料上，绝不根据核对结果
  重新调整阈值或分组；
* 代价（错误接受/正确接受/拒答）与各组最低覆盖要求是策略内容的一部分：
  修改任一者都会得到新的 policy_version，旧策略与旧结果仍可追溯；
* 资料不足（选择资料中无可用打分预测）或约束不可行（可达覆盖率低于最低
  要求）的组保留 undetermined/infeasible 状态与原因；核对时这些组整组
  按拒答处理并留在总体账目里，不会被悄悄排除来美化总体成绩；
* 缺预测、冲突预测、未知预测类别的样本一律按拒答计入人数、覆盖率、
  代价与区间，分组账目加总等于总体账目。
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


@dataclass(frozen=True)
class DecisionRow:
    """单个样本在策略下的决策结果，携带权重、状态与所用阈值。"""

    sample_id: str
    group: str
    weight: float
    status: str
    threshold: float | None
    cost: float
    accepted: bool


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


@dataclass(frozen=True)
class AbstentionPolicy:
    """定版后的拒答策略：阈值、代价、覆盖要求与选择资料版本。"""

    strategy: str
    group_attr: str | None
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
            "group_attr": self.group_attr,
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
        }


@dataclass(frozen=True)
class PolicyComparison:
    """多个定版策略（如统一阈值 vs 分组阈值）在同一核对资料上的并排比较。"""

    comparison_version: str
    sample_set_version: str
    evaluations: tuple[PolicyEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_version": self.comparison_version,
            "sample_set_version": self.sample_set_version,
            "evaluations": [e.to_dict() for e in self.evaluations],
        }


def _group_key(sample, group_attr: str | None) -> str:
    if group_attr is None:
        return _ALL_GROUPS
    value = sample.groups.get(group_attr)
    return "__unassigned__" if value is None else str(value)


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
    group_attr: str | None,
    costs: CostModel,
    threshold_for,
) -> list[DecisionRow]:
    """对样本集里每一个样本给出决策；异常样本一律按拒答保留在账上。"""
    quality = batch.summary
    sample_set = batch.sample_set
    eval_index = {r.sample_id: r for r in batch.evaluable_rows}
    missing_ids = set(quality.missing_sample_ids)
    conflicting_ids = {sid for sid, _ in quality.conflicting_rows}
    unknown_label_ids = {sid for sid, _ in quality.unknown_label_rows}

    rows: list[DecisionRow] = []
    for sample in sample_set.samples:
        sid = sample.sample_id
        group = _group_key(sample, group_attr)
        threshold = threshold_for(group)
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
                group=group,
                weight=sample.weight,
                status=status,
                threshold=threshold,
                cost=cost,
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
    total_weight = sum(r.weight for r in rows)
    accepted_weight = sum(r.weight for r in rows if r.accepted)
    forced_weight = sum(
        r.weight for r in rows if r.status in FORCED_ABSTAIN_STATUSES
    )
    total_cost = sum(r.cost * r.weight for r in rows)
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
    group_attr: str | None = None,
    costs: CostModel | None = None,
    min_coverage: float | Mapping[str, float] = 0.0,
    strategy: str = STRATEGY_PER_GROUP,
) -> AbstentionPolicy:
    """在选择资料上定版拒答策略。

    `strategy="uniform"` 选一个同时满足各组最低覆盖要求的统一阈值；
    `strategy="per_group"` 为每组独立选阈值。候选阈值取选择资料中出现的
    打分值（外加“全部拒答”），在满足最低覆盖要求的候选中取代价最小者。
    """
    if costs is None:
        raise ValueError("必须显式给出 CostModel（错误接受/正确接受/拒答代价）")
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略 {strategy!r}，可选: {STRATEGIES}")
    default_req, group_reqs = _normalize_min_coverage(min_coverage)
    req_table = dict(group_reqs)

    def req_for(group: str) -> float:
        return req_table.get(group, default_req)

    sample_set = selection_batch.sample_set
    groups = sorted(
        {_group_key(sample, group_attr) for sample in sample_set.samples}
    )
    eval_rows = list(selection_batch.evaluable_rows)
    scored_by_group: dict[str, list[float]] = {g: [] for g in groups}
    for row in eval_rows:
        if row.score is not None:
            group = _group_key(sample_set.get(row.sample_id), group_attr)
            scored_by_group.setdefault(group, []).append(row.score)

    def group_stats(group: str, threshold: float | None):
        decisions = _decisions(
            selection_batch, group_attr, costs, lambda g: threshold
        )
        sub = [r for r in decisions if r.group == group]
        total = sum(r.weight for r in sub)
        accepted = sum(r.weight for r in sub if r.accepted)
        cost = sum(r.cost * r.weight for r in sub)
        return (
            accepted / total if total > 0 else 0.0,
            cost / total if total > 0 else 0.0,
            total,
        )

    statuses: list[GroupStatus] = []
    thresholds: dict[str, float | None] = {}

    if strategy == STRATEGY_PER_GROUP:
        for group in groups:
            req = req_for(group)
            scores = sorted(set(scored_by_group.get(group, [])))
            n_scored = len(scored_by_group.get(group, []))
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
                thresholds[group] = None
                continue
            max_cov, _, _ = group_stats(group, min(scores))
            best: tuple[float, float, float] | None = None
            for threshold in scores + [math.inf]:
                cov, cost, _ = group_stats(group, threshold)
                if cov + _EPS >= req and (
                    best is None or cost < best[1] - _EPS
                ):
                    best = (threshold, cost, cov)
            if best is None:
                statuses.append(
                    GroupStatus(
                        group=group,
                        status=STATUS_INFEASIBLE,
                        reason=(
                            f"可达覆盖率 {max_cov:.6f} 低于最低要求 {req:.6f}"
                        ),
                        threshold=None,
                        min_coverage=req,
                        max_coverage=max_cov,
                        n_scored=n_scored,
                    )
                )
                thresholds[group] = None
            else:
                statuses.append(
                    GroupStatus(
                        group=group,
                        status=STATUS_OK,
                        reason=None,
                        threshold=best[0],
                        min_coverage=req,
                        max_coverage=max_cov,
                        n_scored=n_scored,
                    )
                )
                thresholds[group] = best[0]
    else:
        all_scores = sorted(
            {s for scores in scored_by_group.values() for s in scores}
        )
        best_uniform: tuple[float, float] | None = None
        for threshold in all_scores + [math.inf]:
            feasible = True
            total_cost_w = 0.0
            total_w = 0.0
            for group in groups:
                cov, cost, w = group_stats(group, threshold)
                if cov + _EPS < req_for(group):
                    feasible = False
                total_cost_w += cost * w
                total_w += w
            mean_cost = total_cost_w / total_w if total_w > 0 else 0.0
            if feasible and (
                best_uniform is None or mean_cost < best_uniform[1] - _EPS
            ):
                best_uniform = (threshold, mean_cost)
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
                max_cov, _, _ = group_stats(group, min(scores))
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
                max_cov = (
                    group_stats(group, min(scores))[0] if scores else None
                )
                statuses.append(
                    GroupStatus(
                        group=group,
                        status=STATUS_OK,
                        reason=None,
                        threshold=best_uniform[0],
                        min_coverage=req,
                        max_coverage=max_cov,
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
        group_attr,
        costs,
        lambda g: (
            threshold_table[_ALL_GROUPS]
            if strategy == STRATEGY_UNIFORM
            else threshold_table.get(g)
        ),
    )
    selection_results: list[GroupPolicyResult] = []
    for group in groups:
        sub = [r for r in selection_decisions if r.group == group]
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
            ("group_attr", group_attr),
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
        group_attr=group_attr,
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
    整组按拒答处理并保留在总体账目中。
    """
    decisions = _decisions(
        holdout_batch, policy.group_attr, policy.costs, policy.threshold_for
    )
    policy_groups = {
        s.group for s in policy.group_statuses if s.group != _ALL_GROUPS
    }
    holdout_groups = {r.group for r in decisions}
    groups = sorted(policy_groups | holdout_groups)

    group_results: list[GroupPolicyResult] = []
    for group in groups:
        sub = [r for r in decisions if r.group == group]
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
    )


def compare_policies(
    policies: list[AbstentionPolicy] | tuple[AbstentionPolicy, ...],
    holdout_batch: PredictionBatch,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
) -> PolicyComparison:
    """在同一核对资料上并排比较多个定版策略（如统一阈值 vs 分组阈值）。"""
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
    comparison_version = content_version(
        [
            ("holdout_sample_set_version", holdout_batch.sample_set.version),
            ("evaluations", [e.evaluation_version for e in evaluations]),
        ]
    )
    return PolicyComparison(
        comparison_version=comparison_version,
        sample_set_version=holdout_batch.sample_set.version,
        evaluations=evaluations,
    )
