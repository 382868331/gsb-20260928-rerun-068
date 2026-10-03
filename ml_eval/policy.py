"""分组拒答策略：选择资料定版，独立核对资料复核。

研究者为每个分组规定最低覆盖率和三种决策代价（错误接受、正确接受、
拒答）。阈值只能在选择批次上扫描；定版后应用到独立核对批次，即使核对
结果不满足约束也不会重新挑选阈值。资料不足或约束不可行的分组保留
``undetermined``/``infeasible`` 状态与原因，仍然进入总体对账。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .predictions import PredictionBatch
from .resample import bootstrap_weighted_stat
from .scope import (
    STATUS_EVALUABLE,
    build_scope_rows,
)
from .versioning import content_version

STATUS_NO_SCORE = "missing_score"
STATUS_INFEASIBLE_GROUP = "infeasible_group_fallback_abstain"

POLICY_UNIFORM = "uniform_threshold"
POLICY_GROUPED = "grouped_threshold"

GROUP_FEASIBLE = "feasible"
GROUP_UNDETERMINED = "undetermined"
GROUP_INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class GroupConstraint:
    min_coverage: float = 0.0
    min_samples: int = 0

    def validate(self, group: str) -> None:
        if not math.isfinite(self.min_coverage) or not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError(f"分组 {group} 的最低覆盖率必须在 [0, 1] 内")
        if self.min_samples < 0:
            raise ValueError(f"分组 {group} 的最低样本数不能为负")

    def to_dict(self) -> dict[str, Any]:
        return {"min_coverage": self.min_coverage, "min_samples": self.min_samples}


@dataclass(frozen=True)
class Costs:
    false_accept: float
    correct_accept: float
    abstain: float

    def validate(self) -> None:
        for name, value in (
            ("false_accept", self.false_accept),
            ("correct_accept", self.correct_accept),
            ("abstain", self.abstain),
        ):
            if not math.isfinite(value):
                raise ValueError(f"代价 {name} 必须是有限数值")

    def to_dict(self) -> dict[str, Any]:
        return {
            "false_accept": self.false_accept,
            "correct_accept": self.correct_accept,
            "abstain": self.abstain,
        }


@dataclass(frozen=True)
class PolicySpec:
    group_attr: str | None
    mode: str = POLICY_GROUPED
    costs: Costs = field(default_factory=lambda: Costs(1.0, 0.0, 0.5))
    group_constraints: dict[str, GroupConstraint] = field(default_factory=dict)
    policy_name: str = "abstention-policy"

    def validate(self) -> None:
        if self.mode not in (POLICY_UNIFORM, POLICY_GROUPED):
            raise ValueError(f"未知策略模式: {self.mode}")
        self.costs.validate()
        for group, constraint in self.group_constraints.items():
            constraint.validate(group)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "group_attr": self.group_attr,
            "mode": self.mode,
            "costs": self.costs.to_dict(),
            "group_constraints": {
                group: constraint.to_dict()
                for group, constraint in sorted(self.group_constraints.items())
            },
        }


@dataclass(frozen=True)
class DecisionRow:
    sample_id: str
    group: str | None
    label: str
    predicted: str
    weight: float
    score: float | None
    threshold: float | None
    accepted: bool
    correct: bool
    reason: str


@dataclass(frozen=True)
class GroupOutcome:
    group: str | None
    status: str
    threshold: float | None
    reason: str | None
    n_samples: int
    total_weight: float
    accepted_weight: float
    abstained_weight: float
    correct_accepted_weight: float
    false_accepted_weight: float
    coverage: float
    total_cost: float
    mean_cost: float | None
    min_coverage: float | None
    constraint_satisfied: bool | None
    max_feasible_coverage: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "status": self.status,
            "threshold": _threshold_to_json(self.threshold),
            "reason": self.reason,
            "n_samples": self.n_samples,
            "total_weight": self.total_weight,
            "accepted_weight": self.accepted_weight,
            "abstained_weight": self.abstained_weight,
            "correct_accepted_weight": self.correct_accepted_weight,
            "false_accepted_weight": self.false_accepted_weight,
            "coverage": self.coverage,
            "total_cost": self.total_cost,
            "mean_cost": self.mean_cost,
            "min_coverage": self.min_coverage,
            "constraint_satisfied": self.constraint_satisfied,
            "max_feasible_coverage": self.max_feasible_coverage,
        }


@dataclass(frozen=True)
class ConfidenceSummary:
    coverage_lower: float
    coverage_upper: float
    mean_cost_lower: float
    mean_cost_upper: float
    coverage_replicate_version: str
    cost_replicate_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": [self.coverage_lower, self.coverage_upper],
            "mean_cost": [self.mean_cost_lower, self.mean_cost_upper],
            "coverage_replicate_version": self.coverage_replicate_version,
            "cost_replicate_version": self.cost_replicate_version,
        }


@dataclass(frozen=True)
class PolicyResult:
    role: str
    policy_version: str
    result_version: str
    sample_set_version: str
    prediction_batch_version: str
    spec: PolicySpec
    status: str
    uniform_threshold: float | None
    groups: tuple[GroupOutcome, ...]
    decisions: tuple[DecisionRow, ...]
    overall: GroupOutcome
    confidence: ConfidenceSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "policy_version": self.policy_version,
            "result_version": self.result_version,
            "sample_set_version": self.sample_set_version,
            "prediction_batch_version": self.prediction_batch_version,
            "spec": self.spec.to_dict(),
            "status": self.status,
            "uniform_threshold": _threshold_to_json(self.uniform_threshold),
            "overall": self.overall.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
            "confidence": None if self.confidence is None else self.confidence.to_dict(),
        }


def _threshold_to_json(value: float | None) -> float | None:
    if value is None:
        return None
    if math.isinf(value):
        return None
    return value


def _group_of(sample, spec: PolicySpec) -> str:
    if spec.group_attr is None:
        return "__all__"
    value = sample.groups.get(spec.group_attr)
    return "__unassigned__" if value is None else str(value)


def _candidate_thresholds(rows: list) -> list[float]:
    scores = sorted({
        row.score for row in rows if row.status == STATUS_EVALUABLE and row.score is not None
    })
    # +inf：全部拒答；每个观测分数：低于阈值则拒答。-inf 不产生额外候选，
    # 因为最低分数候选已等价于接受全部有分数行。
    return scores + [math.inf]


def _accept(row, threshold: float | None) -> tuple[bool, str]:
    if row.status != STATUS_EVALUABLE:
        return False, row.status
    if row.score is None:
        return False, STATUS_NO_SCORE
    if threshold is None or math.isinf(threshold):
        return False, STATUS_INFEASIBLE_GROUP
    return row.score >= threshold, STATUS_EVALUABLE


def _decision_cost(row: DecisionRow, costs: Costs) -> float:
    if not row.accepted:
        return costs.abstain
    return costs.correct_accept if row.correct else costs.false_accept


# 选择和应用阶段都从全样本 ScopeRow 出发，避免缺预测被扫描前丢弃。
def _decisions_for(
    batch: PredictionBatch,
    thresholds: dict[str, float | None],
    unknown_group_threshold: float | None,
    spec: PolicySpec,
) -> list[DecisionRow]:
    rows = build_scope_rows(batch, "all_samples")
    decisions: list[DecisionRow] = []
    for row in rows:
        sample = batch.sample_set.get(row.sample_id)
        group = _group_of(sample, spec)
        threshold = thresholds.get(group, unknown_group_threshold)
        accepted, reason = _accept(row, threshold)
        decisions.append(
            DecisionRow(
                sample_id=row.sample_id,
                group=None if spec.group_attr is None else group,
                label=row.label,
                predicted=row.predicted if accepted else "__abstain__",
                weight=row.weight,
                score=row.score,
                threshold=threshold,
                accepted=accepted,
                correct=accepted and row.predicted == row.label,
                reason=reason,
            )
        )
    return decisions


def _summarize_group(
    group: str | None,
    decisions: list[DecisionRow],
    spec: PolicySpec,
    status: str,
    reason: str | None,
    threshold: float | None,
    max_feasible_coverage: float | None = None,
) -> GroupOutcome:
    total = sum(row.weight for row in decisions)
    accepted = sum(row.weight for row in decisions if row.accepted)
    abstained = total - accepted
    correct = sum(
        row.weight for row in decisions if row.accepted and row.correct
    )
    false = accepted - correct
    total_cost = sum(_decision_cost(row, spec.costs) * row.weight for row in decisions)
    coverage = accepted / total if total > 0 else 0.0
    constraint = spec.group_constraints.get(group if group is not None else "__all__")
    min_coverage = None if constraint is None else constraint.min_coverage
    satisfied = None if min_coverage is None else coverage + 1e-12 >= min_coverage
    return GroupOutcome(
        group=group,
        status=status,
        threshold=threshold,
        reason=reason,
        n_samples=len(decisions),
        total_weight=total,
        accepted_weight=accepted,
        abstained_weight=abstained,
        correct_accepted_weight=correct,
        false_accepted_weight=false,
        coverage=coverage,
        total_cost=total_cost,
        mean_cost=total_cost / total if total > 0 else None,
        min_coverage=min_coverage,
        constraint_satisfied=satisfied,
        max_feasible_coverage=max_feasible_coverage,
    )


def _evaluate_threshold(
    rows: list, threshold: float, costs: Costs
) -> tuple[float, float]:
    total = sum(row.weight for row in rows)
    accepted = 0.0
    cost = 0.0
    for row in rows:
        accepted_flag, _ = _accept(row, threshold)
        if accepted_flag:
            accepted += row.weight
            unit = costs.correct_accept if row.predicted == row.label else costs.false_accept
        else:
            unit = costs.abstain
        cost += unit * row.weight
    coverage = accepted / total if total > 0 else 0.0
    return coverage, cost


def _policy_version(spec: PolicySpec, batch: PredictionBatch, role: str) -> str:
    return content_version(
        [
            ("role", role),
            ("spec", spec.to_dict()),
            ("sample_set_version", batch.sample_set.version),
            ("prediction_batch_version", batch.version),
        ]
    )


def _result_version(
    spec: PolicySpec,
    batch: PredictionBatch,
    thresholds: dict[str, float | None],
    statuses: dict[str, tuple[str, str | None]],
) -> str:
    return content_version(
        [
            ("spec", spec.to_dict()),
            ("sample_set_version", batch.sample_set.version),
            ("prediction_batch_version", batch.version),
            (
                "frozen_thresholds",
                [
                    [group, _threshold_to_json(threshold), math.isinf(threshold) if threshold is not None else None]
                    for group, threshold in sorted(
                        thresholds.items(), key=lambda item: ("" if item[0] is None else item[0])
                    )
                ],
            ),
            (
                "group_statuses",
                [
                    [group, status, reason]
                    for group, (status, reason) in sorted(
                        statuses.items(), key=lambda item: ("" if item[0] is None else item[0])
                    )
                ],
            ),
        ]
    )


def select_policy(
    batch: PredictionBatch,
    spec: PolicySpec,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
) -> PolicyResult:
    """只使用选择批次扫描阈值并定版。"""
    spec.validate()
    rows = build_scope_rows(batch, "all_samples")
    by_group: dict[str, list] = {}
    for row in rows:
        sample = batch.sample_set.get(row.sample_id)
        by_group.setdefault(_group_of(sample, spec), []).append(row)

    group_info: dict[str, tuple[float | None, str, str | None, float | None]] = {}
    for group in sorted(by_group):
        group_rows = by_group[group]
        constraint = spec.group_constraints.get(group)
        min_coverage = 0.0 if constraint is None else constraint.min_coverage
        min_samples = 0 if constraint is None else constraint.min_samples
        if constraint is not None and len(group_rows) < min_samples:
            group_info[group] = (
                math.inf,
                GROUP_UNDETERMINED,
                f"selection_group_below_min_samples:{len(group_rows)}<{min_samples}",
                None,
            )
            continue
        candidates = _candidate_thresholds(group_rows)
        scored_rows = [
            row
            for row in group_rows
            if row.status == STATUS_EVALUABLE and row.score is not None
        ]
        if not scored_rows:
            group_info[group] = (
                math.inf,
                GROUP_UNDETERMINED,
                "selection_group_has_no_scored_predictions",
                0.0,
            )
            continue
        scored = []
        for threshold in candidates:
            coverage, cost = _evaluate_threshold(group_rows, threshold, spec.costs)
            scored.append((cost, -coverage, threshold, coverage))
        feasible = [item for item in scored if -item[1] + 1e-12 >= min_coverage]
        max_coverage = max((-item[1] for item in scored), default=0.0)
        if not feasible:
            # 不把任何仍违规的阈值包装成定版策略；采用全拒答保守后备，
            # 组仍然进入总体代价/覆盖率账目。
            group_info[group] = (
                math.inf,
                GROUP_INFEASIBLE,
                f"no_threshold_reaches_min_coverage:{max_coverage:.12g}<{min_coverage:.12g}",
                max_coverage,
            )
        else:
            best = min(feasible, key=lambda item: (item[0], item[1], item[2]))
            group_info[group] = (best[2], GROUP_FEASIBLE, None, max_coverage)

    if spec.mode == POLICY_UNIFORM:
        constrained_groups = set(spec.group_constraints)
        union_rows = [row for group_rows in by_group.values() for row in group_rows]
        all_scored = []
        for threshold in _candidate_thresholds(union_rows):
            coverage, cost = _evaluate_threshold(union_rows, threshold, spec.costs)
            group_coverage = {}
            for group, group_rows in by_group.items():
                group_coverage[group], _ = _evaluate_threshold(
                    group_rows, threshold, spec.costs
                )
            violations = [
                group
                for group in constrained_groups
                if group in by_group
                and group_coverage[group] + 1e-12
                < spec.group_constraints[group].min_coverage
            ]
            all_scored.append((cost, -coverage, threshold, tuple(sorted(violations))))
        feasible_uniform = [item for item in all_scored if not item[3]]
        if feasible_uniform:
            _, _, uniform, _ = min(feasible_uniform, key=lambda item: (item[0], item[1], item[2]))
            uniform_reason = None
        else:
            uniform = math.inf
            uniform_reason = "uniform_threshold_violates_group_constraints"
        thresholds = {group: uniform for group in by_group}
        chosen_violations = next(
            (violations for _, _, threshold_value, violations in all_scored
             if threshold_value == uniform),
            (),
        )
        violated_groups = set(chosen_violations)
        statuses = {
            group: (
                info[1],
                info[2],
            )
            if info[1] != GROUP_FEASIBLE
            else (
                (GROUP_INFEASIBLE, uniform_reason)
                if group in violated_groups
                else (GROUP_FEASIBLE, None)
            )
            for group, info in group_info.items()
        }
        frozen_uniform = uniform
    else:
        thresholds = {group: info[0] for group, info in group_info.items()}
        statuses = {group: (info[1], info[2]) for group, info in group_info.items()}
        frozen_uniform = None

    decisions = _decisions_for(batch, thresholds, math.inf, spec)
    groups = _build_group_outcomes(
        decisions,
        spec,
        thresholds,
        group_info if spec.mode == POLICY_GROUPED else None,
        statuses,
    )
    overall = _summarize_group(
        None, decisions, spec, _overall_status(statuses), None, frozen_uniform
    )
    confidence = _confidence(
        decisions, spec.costs, n_replicates, seed, confidence_level
    )
    overall_status = _overall_status(statuses)
    return PolicyResult(
        role="selection",
        policy_version=_policy_version(spec, batch, "selection"),
        result_version=_result_version(spec, batch, thresholds, statuses),
        sample_set_version=batch.sample_set.version,
        prediction_batch_version=batch.version,
        spec=spec,
        status=overall_status,
        uniform_threshold=frozen_uniform,
        groups=tuple(groups),
        decisions=tuple(decisions),
        overall=overall,
        confidence=confidence,
    )


def _build_group_outcomes(decisions, spec, thresholds, group_info, statuses):
    by_decisions: dict[str, list[DecisionRow]] = {}
    for row in decisions:
        group = "__all__" if row.group is None else row.group
        by_decisions.setdefault(group, []).append(row)
    outcomes = []
    for group in sorted(by_decisions):
        status, reason = statuses[group]
        max_cov = None if group_info is None else group_info[group][3]
        outcomes.append(
            _summarize_group(
                None if spec.group_attr is None else group,
                by_decisions[group],
                spec,
                status,
                reason,
                thresholds[group],
                max_cov,
            )
        )
    return outcomes


def _overall_status(statuses: dict[str, tuple[str, str | None]]) -> str:
    if any(status == GROUP_INFEASIBLE for status, _ in statuses.values()):
        return GROUP_INFEASIBLE
    if any(status == GROUP_UNDETERMINED for status, _ in statuses.values()):
        return GROUP_UNDETERMINED
    return GROUP_FEASIBLE


def _confidence(decisions, costs, n_replicates, seed, confidence_level):
    if not decisions:
        return None
    coverage_boot = bootstrap_weighted_stat(
        decisions,
        lambda row: 1.0 if row.accepted else 0.0,
        "coverage",
        n_replicates=n_replicates,
        seed=seed,
        confidence_level=confidence_level,
    )
    cost_boot = bootstrap_weighted_stat(
        decisions,
        lambda row: _decision_cost(row, costs),
        "mean_weighted_cost",
        n_replicates=n_replicates,
        seed=seed + 1,
        confidence_level=confidence_level,
    )
    return ConfidenceSummary(
        coverage_lower=coverage_boot.lower,
        coverage_upper=coverage_boot.upper,
        mean_cost_lower=cost_boot.lower,
        mean_cost_upper=cost_boot.upper,
        coverage_replicate_version=coverage_boot.replicate_version,
        cost_replicate_version=cost_boot.replicate_version,
    )


def apply_policy(
    batch: PredictionBatch,
    selected: PolicyResult,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
) -> PolicyResult:
    """把定版阈值冻结应用到独立核对批次；不根据结果重新选择。"""
    if batch.sample_set.version != selected.sample_set_version:
        raise ValueError("核对批次必须与选择批次对齐同一样本集版本")
    spec = selected.spec
    spec.validate()
    known_groups = {group.group for group in selected.groups if group.group is not None}
    if spec.group_attr is None:
        thresholds = {"__all__": selected.uniform_threshold}
    else:
        thresholds = {
            group.group: group.threshold
            for group in selected.groups
            if group.group is not None and group.threshold is not None
        }
    decisions = _decisions_for(batch, thresholds, math.inf, spec)
    statuses: dict[str, tuple[str, str | None]] = {}
    if spec.group_attr is not None:
        source_by_group = {group.group: group for group in selected.groups}
        for group in sorted({row.group for row in decisions}):
            if group not in source_by_group:
                statuses[group] = (
                    GROUP_UNDETERMINED,
                    "audit_group_absent_from_selection",
                )
                continue
            source = source_by_group[group]
            group_decisions = [item for item in decisions if item.group == group]
            group_total = sum(item.weight for item in group_decisions)
            coverage = (
                sum(item.weight for item in group_decisions if item.accepted)
                / group_total
                if group_total > 0
                else 0.0
            )
            constraint = spec.group_constraints.get(group)
            if constraint is not None and coverage + 1e-12 < constraint.min_coverage:
                statuses[group] = (
                    GROUP_INFEASIBLE,
                    "audit_below_frozen_min_coverage",
                )
            else:
                statuses[group] = (source.status, source.reason)
    else:
        statuses["__all__"] = (selected.status, "frozen_selection_policy")
    groups = _build_group_outcomes(decisions, spec, thresholds, None, statuses)
    overall = _summarize_group(
        None,
        decisions,
        spec,
        _overall_status(statuses),
        "frozen_thresholds_not_reselected",
        selected.uniform_threshold,
    )
    return PolicyResult(
        role="audit",
        policy_version=selected.policy_version,
        result_version=content_version(
            [
                ("policy_version", selected.policy_version),
                ("sample_set_version", batch.sample_set.version),
                ("prediction_batch_version", batch.version),
                ("frozen_result_version", selected.result_version),
            ]
        ),
        sample_set_version=batch.sample_set.version,
        prediction_batch_version=batch.version,
        spec=spec,
        status=_overall_status(statuses),
        uniform_threshold=selected.uniform_threshold,
        groups=tuple(groups),
        decisions=tuple(decisions),
        overall=overall,
        confidence=_confidence(
            decisions, spec.costs, n_replicates, seed, confidence_level
        ),
    )
