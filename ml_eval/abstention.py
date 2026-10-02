"""带拒答（abstain）策略的阈值扫描与模型比较。

用户对错误接受、正确接受与主动拒答分别设定代价；每个阈值下统计覆盖率、
损失与受影响分组。缺预测、冲突预测、未知类别与无分数预测一律视为拒答，
与低于阈值的拒答分开计数，人数、权重、代价与区间在同一本账上对账。

阈值选择（selection）与最终核对（final）使用两份分别留存、分别版本化的
资料：最优阈值只在 selection 资料上确定，final 资料只做核对，不允许用
final 的结果反向挑选阈值。每个模型在每个数据集上的**全部**阈值结果都
不可变地保留在报告中，后续阈值评估不会替换旧策略的结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .predictions import PredictionBatch
from .resample import BootstrapResult
from .versioning import content_version

DECISIONS = ("correct_accept", "wrong_accept", "abstain")
ABSTAIN_REASONS = (
    "below_threshold",
    "no_score",
    "missing",
    "conflicting",
    "unknown_label",
)


@dataclass(frozen=True)
class CostModel:
    """三类决策的代价（按样本权重计）。必须非负。"""

    correct_accept: float = 0.0
    wrong_accept: float = 1.0
    abstain: float = 1.0

    def __post_init__(self) -> None:
        for name in ("correct_accept", "wrong_accept", "abstain"):
            if getattr(self, name) < 0:
                raise ValueError(f"代价 {name} 不能为负")

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct_accept": self.correct_accept,
            "wrong_accept": self.wrong_accept,
            "abstain": self.abstain,
        }


@dataclass(frozen=True)
class SampleDecision:
    sample_id: str
    weight: float
    decision: str  # correct_accept / wrong_accept / abstain
    reason: str  # accepted / below_threshold / no_score / missing / conflicting / unknown_label
    unit_cost: float


def _sample_decisions(
    batch: PredictionBatch, threshold: float, costs: CostModel
) -> list[SampleDecision]:
    quality = batch.summary
    rows_by_id = {r.sample_id: r for r in batch.evaluable_rows}
    missing_ids = set(quality.missing_sample_ids)
    conflicting_ids = {sid for sid, _ in quality.conflicting_rows}
    unknown_label_ids = {sid for sid, _ in quality.unknown_label_rows}

    decisions: list[SampleDecision] = []
    for sample in batch.sample_set.samples:
        sid = sample.sample_id
        row = rows_by_id.get(sid)
        if row is None:
            if sid in missing_ids:
                reason = "missing"
            elif sid in conflicting_ids:
                reason = "conflicting"
            else:
                reason = "unknown_label"
            decisions.append(
                SampleDecision(sid, sample.weight, "abstain", reason, costs.abstain)
            )
        elif row.score is None:
            decisions.append(
                SampleDecision(sid, sample.weight, "abstain", "no_score", costs.abstain)
            )
        elif row.score < threshold:
            decisions.append(
                SampleDecision(
                    sid, sample.weight, "abstain", "below_threshold", costs.abstain
                )
            )
        elif row.predicted == row.label:
            decisions.append(
                SampleDecision(
                    sid, sample.weight, "correct_accept", "accepted", costs.correct_accept
                )
            )
        else:
            decisions.append(
                SampleDecision(
                    sid, sample.weight, "wrong_accept", "accepted", costs.wrong_accept
                )
            )
    return decisions


def _aggregate(decisions: list[SampleDecision]) -> dict[str, Any]:
    total_weight = sum(d.weight for d in decisions)
    by_decision = {name: 0.0 for name in DECISIONS}
    by_reason = {name: 0.0 for name in ABSTAIN_REASONS}
    n_by_decision = {name: 0 for name in DECISIONS}
    for d in decisions:
        by_decision[d.decision] += d.weight
        n_by_decision[d.decision] += 1
        if d.decision == "abstain":
            by_reason[d.reason] += d.weight
    accepted_weight = by_decision["correct_accept"] + by_decision["wrong_accept"]
    loss = sum(d.weight * d.unit_cost for d in decisions)
    return {
        "n_samples": len(decisions),
        "total_weight": total_weight,
        "accepted_weight": accepted_weight,
        "abstained_weight": by_decision["abstain"],
        "correct_accept_weight": by_decision["correct_accept"],
        "wrong_accept_weight": by_decision["wrong_accept"],
        "n_accepted": n_by_decision["correct_accept"] + n_by_decision["wrong_accept"],
        "n_abstained": n_by_decision["abstain"],
        "abstain_reasons_weight": by_reason,
        "coverage": accepted_weight / total_weight if total_weight > 0 else 0.0,
        "loss": loss,
        "mean_loss": loss / total_weight if total_weight > 0 else 0.0,
    }


@dataclass(frozen=True)
class GroupCostReport:
    group: str
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"group": self.group, **self.stats}


@dataclass(frozen=True)
class ThresholdResult:
    """单个阈值下的完整结果；报告保留全部阈值，旧策略不会被替换。"""

    threshold: float
    stats: dict[str, Any]
    groups: tuple[GroupCostReport, ...]
    confidence: BootstrapResult | None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "threshold": self.threshold,
            **self.stats,
            "groups": [g.to_dict() for g in self.groups],
        }
        if self.confidence is not None:
            out["confidence"] = {
                "metric": self.confidence.metric,
                "point": self.confidence.point,
                "lower": self.confidence.lower,
                "upper": self.confidence.upper,
                "n_replicates": self.confidence.n_replicates,
                "seed": self.confidence.seed,
                "confidence_level": self.confidence.confidence_level,
                "replicate_version": self.confidence.replicate_version,
            }
        return out


def bootstrap_mean_loss_ci(
    decisions: list[SampleDecision],
    threshold: float,
    costs: CostModel,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """对平均损失（总代价 / 总权重）做固定种子的加权 bootstrap 区间。

    重采样单位是样本身份，与计数、权重、代价使用同一份决策清单，
    保证区间与账目同源可对账。
    """
    if not decisions:
        raise ValueError("没有可重采样的样本")
    weights = np.array([d.weight for d in decisions], dtype=float)
    unit_costs = np.array([d.unit_cost for d in decisions], dtype=float)
    total_weight = float(weights.sum())
    point = float((weights * unit_costs).sum() / total_weight) if total_weight > 0 else 0.0

    rng = np.random.default_rng(seed)
    n = len(decisions)
    values = np.empty(n_replicates, dtype=float)
    for i in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        w = weights[idx]
        w_sum = w.sum()
        values[i] = (w * unit_costs[idx]).sum() / w_sum if w_sum > 0 else 0.0

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(values, alpha))
    upper = float(np.quantile(values, 1.0 - alpha))
    replicate_version = content_version(
        [
            (
                "abstain_bootstrap",
                [
                    threshold,
                    [costs.correct_accept, costs.wrong_accept, costs.abstain],
                    n_replicates,
                    seed,
                    confidence_level,
                    [
                        [d.sample_id, d.weight, d.decision, d.unit_cost]
                        for d in sorted(decisions, key=lambda d: d.sample_id)
                    ],
                ],
            )
        ]
    )
    return BootstrapResult(
        metric="mean_loss",
        point=point,
        lower=lower,
        upper=upper,
        n_replicates=n_replicates,
        seed=seed,
        confidence_level=confidence_level,
        replicate_version=replicate_version,
    )


@dataclass(frozen=True)
class AbstentionReport:
    """一个模型在一份资料（selection 或 final）上的全部阈值结果。"""

    batch_name: str
    role: str
    sample_set_version: str
    prediction_batch_version: str
    costs: CostModel
    results: tuple[ThresholdResult, ...]
    report_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "role": self.role,
            "sample_set_version": self.sample_set_version,
            "prediction_batch_version": self.prediction_batch_version,
            "costs": self.costs.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "report_version": self.report_version,
        }


def evaluate_abstention(
    batch: PredictionBatch,
    thresholds: list[float] | tuple[float, ...],
    costs: CostModel,
    group_attr: str | None = None,
    role: str = "final",
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
    bootstrap: bool = True,
) -> AbstentionReport:
    if role not in ("selection", "final"):
        raise ValueError("role 必须是 selection 或 final")
    thresholds = sorted({float(t) for t in thresholds})
    if not thresholds:
        raise ValueError("至少需要一个阈值")

    def group_value(sample) -> str:
        value = sample.groups.get(group_attr) if group_attr else None
        return "__unassigned__" if value is None else str(value)

    results: list[ThresholdResult] = []
    for threshold in thresholds:
        decisions = _sample_decisions(batch, threshold, costs)
        stats = _aggregate(decisions)
        groups: tuple[GroupCostReport, ...] = ()
        if group_attr is not None:
            buckets: dict[str, list[SampleDecision]] = {}
            for d in decisions:
                buckets.setdefault(
                    group_value(batch.sample_set.get(d.sample_id)), []
                ).append(d)
            groups = tuple(
                GroupCostReport(group=name, stats=_aggregate(buckets[name]))
                for name in sorted(buckets)
            )
        ci = (
            bootstrap_mean_loss_ci(
                decisions, threshold, costs, n_replicates, seed, confidence_level
            )
            if bootstrap
            else None
        )
        results.append(
            ThresholdResult(
                threshold=threshold, stats=stats, groups=groups, confidence=ci
            )
        )

    report_version = content_version(
        [
            ("batch_name", batch.batch_name),
            ("role", role),
            ("sample_set_version", batch.sample_set.version),
            ("prediction_batch_version", batch.version),
            ("costs", [costs.correct_accept, costs.wrong_accept, costs.abstain]),
            ("group_attr", group_attr),
            (
                "threshold_results",
                [[r.threshold, r.stats["loss"], r.stats["coverage"]] for r in results],
            ),
        ]
    )
    return AbstentionReport(
        batch_name=batch.batch_name,
        role=role,
        sample_set_version=batch.sample_set.version,
        prediction_batch_version=batch.version,
        costs=costs,
        results=tuple(results),
        report_version=report_version,
    )


def choose_threshold(report: AbstentionReport) -> ThresholdResult:
    """在 selection 报告上选阈值：损失最小，并列时取覆盖率更高、阈值更低者。"""
    return min(
        report.results,
        key=lambda r: (r.stats["mean_loss"], -r.stats["coverage"], r.threshold),
    )


@dataclass(frozen=True)
class ChosenThreshold:
    batch_name: str
    threshold: float
    selection_mean_loss: float
    selection_coverage: float
    selection_report_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "threshold": self.threshold,
            "selection_mean_loss": self.selection_mean_loss,
            "selection_coverage": self.selection_coverage,
            "selection_report_version": self.selection_report_version,
        }


@dataclass(frozen=True)
class AbstentionStudy:
    """阈值选择与最终核对分别留存的研究报告。

    chosen 只由 selection 报告决定；final 报告包含同一批阈值的完整
    结果用于核对，结构上没有从 final 反向挑阈值的入口。
    """

    costs: CostModel
    group_attr: str | None
    selection: tuple[AbstentionReport, ...]
    final: tuple[AbstentionReport, ...]
    chosen: tuple[ChosenThreshold, ...]
    study_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "costs": self.costs.to_dict(),
            "group_attr": self.group_attr,
            "selection": [r.to_dict() for r in self.selection],
            "final": [r.to_dict() for r in self.final],
            "chosen": [c.to_dict() for c in self.chosen],
            "study_version": self.study_version,
        }


def run_abstention_study(
    selection_batches: list[PredictionBatch] | tuple[PredictionBatch, ...],
    final_batches: list[PredictionBatch] | tuple[PredictionBatch, ...],
    thresholds: list[float] | tuple[float, ...],
    costs: CostModel,
    group_attr: str | None = None,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
    bootstrap: bool = True,
) -> AbstentionStudy:
    selection_batches = list(selection_batches)
    final_batches = list(final_batches)
    if not selection_batches or not final_batches:
        raise ValueError("selection 与 final 都至少需要一个预测批次")
    sel_names = [b.batch_name for b in selection_batches]
    fin_names = [b.batch_name for b in final_batches]
    if len(set(sel_names)) != len(sel_names) or len(set(fin_names)) != len(fin_names):
        raise ValueError("批次名必须唯一")
    if set(sel_names) != set(fin_names):
        raise ValueError(
            f"selection 与 final 的模型必须一一对应: {sorted(sel_names)} vs {sorted(fin_names)}"
        )

    kwargs = dict(
        thresholds=thresholds,
        costs=costs,
        group_attr=group_attr,
        n_replicates=n_replicates,
        seed=seed,
        confidence_level=confidence_level,
        bootstrap=bootstrap,
    )
    selection_reports = tuple(
        evaluate_abstention(batch, role="selection", **kwargs)
        for batch in selection_batches
    )
    final_reports = tuple(
        evaluate_abstention(batch, role="final", **kwargs) for batch in final_batches
    )

    chosen: list[ChosenThreshold] = []
    for report in selection_reports:
        best = choose_threshold(report)
        chosen.append(
            ChosenThreshold(
                batch_name=report.batch_name,
                threshold=best.threshold,
                selection_mean_loss=best.stats["mean_loss"],
                selection_coverage=best.stats["coverage"],
                selection_report_version=report.report_version,
            )
        )

    study_version = content_version(
        [
            ("selection", [r.report_version for r in selection_reports]),
            ("final", [r.report_version for r in final_reports]),
            ("chosen", [[c.batch_name, c.threshold] for c in chosen]),
        ]
    )
    return AbstentionStudy(
        costs=costs,
        group_attr=group_attr,
        selection=selection_reports,
        final=final_reports,
        chosen=tuple(chosen),
        study_version=study_version,
    )
