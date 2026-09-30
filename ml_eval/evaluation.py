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
    metrics: MetricSet
    confidence: BootstrapResult | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "group": self.group,
            "n_samples": self.n_samples,
            "metrics": self.metrics.to_dict(),
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
    accuracy_evaluable: float

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
            value = sample_set.get(row.sample_id).groups.get(group_attr)
            if value is None:
                value = "__unassigned__"
            buckets.setdefault(str(value), []).append(row)
        for name in sorted(buckets):
            sub_rows = buckets[name]
            groups.append(
                GroupReport(
                    group=name,
                    n_samples=len(sub_rows),
                    metrics=multiclass_metrics(sub_rows),
                    confidence=confidence(sub_rows),
                )
            )

    correct_weight = sum(r.weight for r in rows if r.predicted == r.label)
    total_weight = quality.total_weight_samples
    excluded_weight = (
        quality.total_weight_missing
        + quality.total_weight_conflicting
        + quality.total_weight_unknown_label
    )
    accounting = Accounting(
        n_samples=quality.n_samples,
        total_weight=total_weight,
        evaluable_weight=quality.total_weight_evaluable,
        correct_weight=correct_weight,
        excluded_weight=excluded_weight,
        missing_weight=quality.total_weight_missing,
        conflicting_weight=quality.total_weight_conflicting,
        unknown_label_weight=quality.total_weight_unknown_label,
        accuracy_all_samples=correct_weight / total_weight if total_weight > 0 else 0.0,
        accuracy_evaluable=overall.metrics.accuracy,
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
