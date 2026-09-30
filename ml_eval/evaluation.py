"""总体指标与分组指标的组装入口。

分组指标总是基于各自组内的样本与权重独立计算（macro/weighted 平均
只发生在类别层面），不会对组指标再做平均来冒充总体指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .metrics import MetricSet, multiclass_metrics
from .predictions import AlignmentSummary, EvaluableRow, PredictionBatch
from .resample import BootstrapResult, bootstrap_ci
from .samples import SampleSet
from .versioning import content_version


@dataclass(frozen=True)
class GroupReport:
    group: str | None
    n_samples: int
    metrics: MetricSet | None
    confidence: BootstrapResult | None = None
    accounting: "Accounting | None" = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "group": self.group,
            "n_samples": self.n_samples,
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
        }
        if self.accounting is not None:
            out["accounting"] = self.accounting.to_dict()
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


@dataclass(frozen=True)
class Accounting:
    """全部样本的对账统计，避免剔除异常后只留下更漂亮的指标。

    缺预测、冲突预测、未知预测类别的样本在全样本准确率里一律按错误处理；
    未知样本身份的预测不对应任何样本，单独计数但不进任何分母。
    """

    n_samples: int
    total_weight: float
    evaluable_weight: float
    correct_weight: float
    excluded_weight: float
    missing_weight: float
    conflicting_weight: float
    unknown_label_weight: float
    accuracy_all_samples: float
    accuracy_evaluable: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "total_weight": self.total_weight,
            "evaluable_weight": self.evaluable_weight,
            "correct_weight": self.correct_weight,
            "excluded_weight": self.excluded_weight,
            "missing_weight": self.missing_weight,
            "conflicting_weight": self.conflicting_weight,
            "unknown_label_weight": self.unknown_label_weight,
            "accuracy_all_samples": self.accuracy_all_samples,
            "accuracy_evaluable": self.accuracy_evaluable,
        }


@dataclass(frozen=True)
class EvaluationReport:
    sample_set_version: str
    prediction_batch_version: str
    participating_version: str
    sample_ids: tuple[str, ...]
    data_quality: AlignmentSummary
    accounting: Accounting
    overall: GroupReport
    groups: tuple[GroupReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_set_version": self.sample_set_version,
            "prediction_batch_version": self.prediction_batch_version,
            "participating_version": self.participating_version,
            "sample_ids": list(self.sample_ids),
            "data_quality": self.data_quality.to_dict(),
            "accounting": self.accounting.to_dict(),
            "overall": self.overall.to_dict(),
            "groups": [g.to_dict() for g in self.groups],
        }


def _group_value(sample, group_attr: str) -> str:
    value = sample.groups.get(group_attr)
    return "__unassigned__" if value is None else str(value)


def _accounting_for(
    n_samples: int,
    total_weight: float,
    evaluable_weight: float,
    correct_weight: float,
    missing_weight: float,
    conflicting_weight: float,
    unknown_label_weight: float,
    evaluable_accuracy: float | None,
) -> Accounting:
    return Accounting(
        n_samples=n_samples,
        total_weight=total_weight,
        evaluable_weight=evaluable_weight,
        correct_weight=correct_weight,
        excluded_weight=missing_weight + conflicting_weight + unknown_label_weight,
        missing_weight=missing_weight,
        conflicting_weight=conflicting_weight,
        unknown_label_weight=unknown_label_weight,
        accuracy_all_samples=correct_weight / total_weight if total_weight > 0 else 0.0,
        accuracy_evaluable=evaluable_accuracy,
    )


def evaluate_batch(
    batch: PredictionBatch,
    group_attr: str | None = None,
    bootstrap_metric: str | None = None,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
) -> EvaluationReport:
    rows = batch.evaluable_rows
    sample_set: SampleSet = batch.sample_set
    quality = batch.summary

    def confidence(sub_rows: list[EvaluableRow]) -> BootstrapResult | None:
        if bootstrap_metric is None or not sub_rows:
            return None
        return bootstrap_ci(
            sub_rows,
            metric=bootstrap_metric,
            n_replicates=n_replicates,
            seed=seed,
            confidence_level=confidence_level,
        )

    overall = GroupReport(
        group=None,
        n_samples=len(rows),
        metrics=multiclass_metrics(rows),
        confidence=confidence(list(rows)),
    )

    groups: list[GroupReport] = []
    if group_attr is not None:
        buckets: dict[str, list[EvaluableRow]] = {}
        for row in rows:
            buckets.setdefault(
                _group_value(sample_set.get(row.sample_id), group_attr), []
            ).append(row)
        # 分组宇宙来自样本集本身：没有任何可评估行的组也要呈现，
        # 其权重去向（缺预测/冲突/未知类别）在组级对账中列明。
        conflicting_ids = {sid for sid, _ in quality.conflicting_rows}
        unknown_label_ids = {sid for sid, _ in quality.unknown_label_rows}
        missing_ids = set(quality.missing_sample_ids)
        group_values = sorted(
            {_group_value(sample, group_attr) for sample in sample_set.samples}
        )
        for name in group_values:
            sub_rows = buckets.get(name, [])
            g_samples = [
                sample
                for sample in sample_set.samples
                if _group_value(sample, group_attr) == name
            ]
            g_total = sum(sample.weight for sample in g_samples)
            g_missing = sum(
                sample.weight for sample in g_samples if sample.sample_id in missing_ids
            )
            g_conflicting = sum(
                sample.weight
                for sample in g_samples
                if sample.sample_id in conflicting_ids
            )
            g_unknown_label = sum(
                sample.weight
                for sample in g_samples
                if sample.sample_id in unknown_label_ids
            )
            g_evaluable = sum(r.weight for r in sub_rows)
            g_correct = sum(r.weight for r in sub_rows if r.predicted == r.label)
            g_metrics = multiclass_metrics(sub_rows) if sub_rows else None
            g_accounting = _accounting_for(
                n_samples=len(g_samples),
                total_weight=g_total,
                evaluable_weight=g_evaluable,
                correct_weight=g_correct,
                missing_weight=g_missing,
                conflicting_weight=g_conflicting,
                unknown_label_weight=g_unknown_label,
                evaluable_accuracy=None if g_metrics is None else g_metrics.accuracy,
            )
            groups.append(
                GroupReport(
                    group=name,
                    n_samples=len(sub_rows),
                    metrics=g_metrics,
                    confidence=confidence(sub_rows),
                    accounting=g_accounting,
                )
            )

    correct_weight = sum(r.weight for r in rows if r.predicted == r.label)
    total_weight = quality.total_weight_samples
    accounting = _accounting_for(
        n_samples=quality.n_samples,
        total_weight=total_weight,
        evaluable_weight=quality.total_weight_evaluable,
        correct_weight=correct_weight,
        missing_weight=quality.total_weight_missing,
        conflicting_weight=quality.total_weight_conflicting,
        unknown_label_weight=quality.total_weight_unknown_label,
        evaluable_accuracy=overall.metrics.accuracy,
    )

    participating_version = content_version(
        [
            ("sample_set_version", sample_set.version),
            ("prediction_batch_version", batch.version),
            ("duplicate_policy", "identical-fold-conflicting-excluded-order-independent"),
            (
                "participating_rows",
                [
                    [r.sample_id, r.label, r.predicted, r.weight]
                    for r in sorted(rows, key=lambda r: r.sample_id)
                ],
            ),
            (
                "bootstrap",
                None
                if bootstrap_metric is None
                else [bootstrap_metric, n_replicates, seed, confidence_level],
            ),
        ]
    )
    return EvaluationReport(
        sample_set_version=sample_set.version,
        prediction_batch_version=batch.version,
        participating_version=participating_version,
        sample_ids=tuple(r.sample_id for r in rows),
        data_quality=quality,
        accounting=accounting,
        overall=overall,
        groups=tuple(groups),
    )
