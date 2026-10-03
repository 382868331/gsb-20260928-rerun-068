"""总体指标与分组指标的组装入口。

主成绩（`default_score`）按显式选择的参与范围（`scope`）计算：

* `all_samples`（默认）：缺预测、冲突预测、未知预测类别全部作为主动拒答
  留在分母里按错误计；
* `evaluable`：只在通过全部校验的行上计算。

诊断视角（`overall.metrics`、分组指标与 bootstrap 区间）始终基于可评估
集合，并通过 `diagnostic_accuracy` 与 `accounting.accuracy_evaluable`
显式标注，不会悄悄冒充主成绩；分组指标在组内独立计算，总体指标不是组
指标的平均。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metrics import MetricSet, multiclass_metrics
from .predictions import AlignmentSummary, EvaluableRow, PredictionBatch
from .resample import BootstrapResult, bootstrap_ci
from .samples import SampleSet
from .scope import (
    SCOPE_ALL,
    build_scope_rows,
    validate_scope,
)
from .versioning import content_version


@dataclass(frozen=True)
class GroupReport:
    group: str | None
    scope: str
    n_samples: int
    n_abstained: int
    metrics: MetricSet | None
    confidence: BootstrapResult | None = None
    accounting: "Accounting | None" = None
    diagnostic_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "group": self.group,
            "scope": self.scope,
            "n_samples": self.n_samples,
            "n_abstained": self.n_abstained,
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "diagnostics": {"accuracy_evaluable": self.diagnostic_accuracy},
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
class DefaultScore:
    """默认成绩：读取主报告时第一眼拿到的数字及其参与范围。"""

    metric: str
    value: float
    scope: str

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "value": self.value, "scope": self.scope}


@dataclass(frozen=True)
class EvaluationReport:
    sample_set_version: str
    prediction_batch_version: str
    participating_version: str
    scope: str
    sample_ids: tuple[str, ...]
    data_quality: AlignmentSummary
    accounting: Accounting
    default_score: DefaultScore
    overall: GroupReport
    groups: tuple[GroupReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_set_version": self.sample_set_version,
            "prediction_batch_version": self.prediction_batch_version,
            "participating_version": self.participating_version,
            "scope": self.scope,
            "default_score": self.default_score.to_dict(),
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
    scope: str = SCOPE_ALL,
) -> EvaluationReport:
    validate_scope(scope)
    sample_set: SampleSet = batch.sample_set
    quality = batch.summary
    labels = sample_set.labels

    # 诊断视角：指标、分组与 bootstrap 区间始终基于可评估集合，
    # 并通过 diagnostic_accuracy / accounting.accuracy_evaluable 显式标注。
    evaluable_rows = list(batch.evaluable_rows)
    evaluable_metrics = multiclass_metrics(evaluable_rows, labels=labels)
    evaluable_accuracy = evaluable_metrics.accuracy if evaluable_rows else None

    def confidence(sub_rows: list[EvaluableRow]) -> BootstrapResult | None:
        if bootstrap_metric is None or not sub_rows:
            return None
        return bootstrap_ci(
            sub_rows,
            metric=bootstrap_metric,
            n_replicates=n_replicates,
            seed=seed,
            confidence_level=confidence_level,
            labels=labels,
        )

    n_excluded = (
        quality.n_missing_predictions
        + quality.n_conflicting_samples
        + quality.n_unknown_predicted_labels
    )
    overall = GroupReport(
        group=None,
        scope=scope,
        n_samples=len(evaluable_rows),
        n_abstained=n_excluded,
        metrics=evaluable_metrics,
        confidence=confidence(list(evaluable_rows)),
        diagnostic_accuracy=evaluable_accuracy,
    )

    groups: list[GroupReport] = []
    if group_attr is not None:
        eval_buckets: dict[str, list[EvaluableRow]] = {}
        for row in evaluable_rows:
            eval_buckets.setdefault(
                _group_value(sample_set.get(row.sample_id), group_attr), []
            ).append(row)

        conflicting_ids = {sid for sid, _ in quality.conflicting_rows}
        unknown_label_ids = {sid for sid, _ in quality.unknown_label_rows}
        missing_ids = set(quality.missing_sample_ids)
        group_values = sorted(
            {_group_value(sample, group_attr) for sample in sample_set.samples}
        )
        for name in group_values:
            g_evaluable_rows = eval_buckets.get(name, [])
            g_samples = [
                sample
                for sample in sample_set.samples
                if _group_value(sample, group_attr) == name
            ]
            g_total = sum(sample.weight for sample in g_samples)
            g_missing = sum(
                sample.weight for sample in g_samples
                if sample.sample_id in missing_ids
            )
            g_conflicting = sum(
                sample.weight for sample in g_samples
                if sample.sample_id in conflicting_ids
            )
            g_unknown_label = sum(
                sample.weight for sample in g_samples
                if sample.sample_id in unknown_label_ids
            )
            g_evaluable = sum(r.weight for r in g_evaluable_rows)
            g_correct = sum(
                r.weight for r in g_evaluable_rows if r.predicted == r.label
            )
            g_eval_metrics = (
                multiclass_metrics(g_evaluable_rows, labels=labels)
                if g_evaluable_rows
                else None
            )
            g_accounting = _accounting_for(
                n_samples=len(g_samples),
                total_weight=g_total,
                evaluable_weight=g_evaluable,
                correct_weight=g_correct,
                missing_weight=g_missing,
                conflicting_weight=g_conflicting,
                unknown_label_weight=g_unknown_label,
                evaluable_accuracy=(
                    None if g_eval_metrics is None else g_eval_metrics.accuracy
                ),
            )
            groups.append(
                GroupReport(
                    group=name,
                    scope=scope,
                    n_samples=len(g_evaluable_rows),
                    n_abstained=len(g_samples) - len(g_evaluable_rows),
                    metrics=g_eval_metrics,
                    confidence=confidence(g_evaluable_rows),
                    accounting=g_accounting,
                    diagnostic_accuracy=(
                        None if g_eval_metrics is None else g_eval_metrics.accuracy
                    ),
                )
            )

    correct_weight = sum(
        r.weight for r in evaluable_rows if r.predicted == r.label
    )
    total_weight = quality.total_weight_samples
    accounting = _accounting_for(
        n_samples=quality.n_samples,
        total_weight=total_weight,
        evaluable_weight=quality.total_weight_evaluable,
        correct_weight=correct_weight,
        missing_weight=quality.total_weight_missing,
        conflicting_weight=quality.total_weight_conflicting,
        unknown_label_weight=quality.total_weight_unknown_label,
        evaluable_accuracy=evaluable_accuracy,
    )

    # 参与版本覆盖全部样本（含缺预测/冲突/未知类别的状态与权重），
    # 保证任何一类样本的变动都会改变版本号，结果可回溯。
    scope_rows = build_scope_rows(batch, SCOPE_ALL)
    participating_version = content_version(
        [
            ("sample_set_version", sample_set.version),
            ("prediction_batch_version", batch.version),
            ("scope", scope),
            ("duplicate_policy", "identical-fold-conflicting-excluded-order-independent"),
            (
                "participating_rows",
                [
                    [r.sample_id, r.label, r.predicted, r.weight, r.status, r.score]
                    for r in sorted(scope_rows, key=lambda r: r.sample_id)
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
    if scope == SCOPE_ALL:
        default_value = (
            correct_weight / total_weight if total_weight > 0 else 0.0
        )
    else:
        default_value = evaluable_accuracy if evaluable_accuracy is not None else 0.0
    return EvaluationReport(
        sample_set_version=sample_set.version,
        prediction_batch_version=batch.version,
        participating_version=participating_version,
        scope=scope,
        sample_ids=tuple(r.sample_id for r in evaluable_rows),
        data_quality=quality,
        accounting=accounting,
        default_score=DefaultScore(
            metric="accuracy", value=default_value, scope=scope
        ),
        overall=overall,
        groups=tuple(groups),
    )
