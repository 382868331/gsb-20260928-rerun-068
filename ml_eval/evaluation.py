"""总体指标与分组指标的组装入口。

主口径（headline）一律使用 ``all_samples`` 参与范围：样本集中的每个样本
都参与，缺预测/冲突/未知类别按错误计，总体、分组与重采样区间使用同一个
范围，最显眼的成绩不会掩盖未覆盖样本。仅在可评估样本上计算的诊断视角
单独命名为 ``diagnostic``（scope=evaluable_only），并在 ``accounting`` 中
列出被排除权重的去向。

分组指标总是基于各自组内的样本与权重独立计算（macro/weighted 平均
只发生在类别层面），不会对组指标再做平均来冒充总体指标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metrics import MetricSet, multiclass_metrics
from .predictions import AlignmentSummary, PredictionBatch, ScoredRow
from .resample import BootstrapResult, bootstrap_ci, bootstrap_scored_accuracy
from .samples import SampleSet
from .versioning import content_version

SCOPE_ALL_SAMPLES = "all_samples"
SCOPE_EVALUABLE_ONLY = "evaluable_only"


@dataclass(frozen=True)
class ScopeReport:
    """一个参与范围（总体或单个分组）的主口径报告。

    主口径字段（``accuracy``/``confidence``）基于全部样本，异常样本按错误计；
    ``diagnostic`` 是仅在可评估样本上计算的诊断指标，单独命名、不混入主口径。
    """

    group: str | None
    scope: str
    n_samples: int
    n_evaluable: int
    total_weight: float
    evaluable_weight: float
    correct_weight: float
    coverage: float
    accuracy: float
    diagnostic: MetricSet | None
    confidence: BootstrapResult | None = None
    diagnostic_confidence: BootstrapResult | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "group": self.group,
            "scope": self.scope,
            "n_samples": self.n_samples,
            "n_evaluable": self.n_evaluable,
            "total_weight": self.total_weight,
            "evaluable_weight": self.evaluable_weight,
            "correct_weight": self.correct_weight,
            "coverage": self.coverage,
            "accuracy": self.accuracy,
        }
        if self.confidence is not None:
            out["confidence"] = self.confidence.to_dict()
        if self.diagnostic is not None:
            out["diagnostic"] = {
                "scope": SCOPE_EVALUABLE_ONLY,
                **self.diagnostic.to_dict(),
            }
        if self.diagnostic_confidence is not None:
            out["diagnostic_confidence"] = self.diagnostic_confidence.to_dict()
        return out


@dataclass(frozen=True)
class Accounting:
    """全部样本的对账统计，避免剔除异常后只留下更漂亮的指标。

    缺预测、冲突预测、未知预测类别的样本在全样本准确率里一律按错误处理；
    未知样本身份的预测不对应任何样本，单独计数但不进任何分母。
    权重去向闭合：evaluable + missing + conflicting + unknown_label = total。
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
            "weight_disposition": {
                "evaluable": self.evaluable_weight,
                "missing": self.missing_weight,
                "conflicting": self.conflicting_weight,
                "unknown_label": self.unknown_label_weight,
            },
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
    overall: ScopeReport
    groups: tuple[ScopeReport, ...]

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


def _scope_report(
    group: str | None,
    scored_rows: list[ScoredRow],
    evaluable_rows,
    bootstrap_metric: str | None,
    n_replicates: int,
    seed: int,
    confidence_level: float,
) -> ScopeReport:
    total_weight = sum(r.weight for r in scored_rows)
    evaluable_weight = sum(r.weight for r in scored_rows if r.status == "evaluable")
    correct_weight = sum(r.weight for r in scored_rows if r.correct)
    evaluable_rows = list(evaluable_rows)

    confidence: BootstrapResult | None = None
    diagnostic_confidence: BootstrapResult | None = None
    if bootstrap_metric is not None and scored_rows:
        if bootstrap_metric == "accuracy":
            # 主口径区间：与总体/分组同一个 all_samples 参与范围。
            confidence = bootstrap_scored_accuracy(
                scored_rows,
                n_replicates=n_replicates,
                seed=seed,
                confidence_level=confidence_level,
            )
        elif evaluable_rows:
            # 非 accuracy 指标只在可评估行上有定义：明确标注为诊断区间。
            diagnostic_confidence = bootstrap_ci(
                evaluable_rows,
                metric=bootstrap_metric,
                n_replicates=n_replicates,
                seed=seed,
                confidence_level=confidence_level,
            )

    return ScopeReport(
        group=group,
        scope=SCOPE_ALL_SAMPLES,
        n_samples=len(scored_rows),
        n_evaluable=len(evaluable_rows),
        total_weight=total_weight,
        evaluable_weight=evaluable_weight,
        correct_weight=correct_weight,
        coverage=evaluable_weight / total_weight if total_weight > 0 else 0.0,
        accuracy=correct_weight / total_weight if total_weight > 0 else 0.0,
        diagnostic=multiclass_metrics(evaluable_rows) if evaluable_rows else None,
        confidence=confidence,
        diagnostic_confidence=diagnostic_confidence,
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
    scored_rows = list(batch.scored_rows)
    sample_set: SampleSet = batch.sample_set
    quality = batch.summary

    overall = _scope_report(
        None, scored_rows, list(rows),
        bootstrap_metric, n_replicates, seed, confidence_level,
    )

    groups: list[ScopeReport] = []
    if group_attr is not None:
        buckets: dict[str, list[ScoredRow]] = {}
        for row in scored_rows:
            value = sample_set.get(row.sample_id).groups.get(group_attr)
            if value is None:
                value = "__unassigned__"
            buckets.setdefault(str(value), []).append(row)
        evaluable_by_id = {r.sample_id: r for r in rows}
        for name in sorted(buckets):
            sub_scored = buckets[name]
            sub_evaluable = [
                evaluable_by_id[r.sample_id]
                for r in sub_scored
                if r.sample_id in evaluable_by_id
            ]
            groups.append(
                _scope_report(
                    name, sub_scored, sub_evaluable,
                    bootstrap_metric, n_replicates, seed, confidence_level,
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
        accuracy_evaluable=(
            correct_weight / quality.total_weight_evaluable
            if quality.total_weight_evaluable > 0
            else 0.0
        ),
    )

    participating_version = content_version(
        [
            ("sample_set_version", sample_set.version),
            ("prediction_batch_version", batch.version),
            ("duplicate_policy", "identical-fold-conflicting-excluded-order-independent"),
            (
                "scored_rows",
                [
                    [r.sample_id, r.label, r.predicted, r.weight, r.correct, r.status]
                    for r in sorted(scored_rows, key=lambda r: r.sample_id)
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
        sample_ids=tuple(r.sample_id for r in scored_rows),
        data_quality=quality,
        accounting=accounting,
        overall=overall,
        groups=tuple(groups),
    )

