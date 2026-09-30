"""加权分类指标。

所有平均方式都显式命名：macro 为类间等权，weighted 为按真实标签支持度
加权；没有任何隐式平均，避免与分组平均混淆。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ClassMetrics:
    label: str
    precision: float | None
    recall: float | None
    f1: float | None
    support: float
    predicted_support: float
    tp: float
    fp: float
    fn: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "predicted_support": self.predicted_support,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass(frozen=True)
class MetricSet:
    labels: tuple[str, ...]
    per_class: dict[str, ClassMetrics]
    accuracy: float
    total_weight: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    confusion: dict[str, dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "accuracy": self.accuracy,
            "total_weight": self.total_weight,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
            "confusion": {k: dict(v) for k, v in self.confusion.items()},
        }


def confusion_counts(
    rows: Iterable,
    labels: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    rows = list(rows)
    if labels is None:
        labels = tuple(sorted({r.label for r in rows}))
    matrix = {true: {pred: 0.0 for pred in labels} for true in labels}
    for row in rows:
        if row.predicted in labels:
            matrix[row.label][row.predicted] += row.weight
    return matrix


def multiclass_metrics(
    rows: Iterable,
    labels: Sequence[str] | None = None,
) -> MetricSet:
    rows = list(rows)
    metric_labels = (
        tuple(labels)
        if labels is not None
        else tuple(sorted({r.label for r in rows}))
    )
    matrix = confusion_counts(rows, metric_labels)
    total = sum(r.weight for r in rows)
    correct = sum(matrix[l][l] for l in metric_labels)

    per_class: dict[str, ClassMetrics] = {}
    for label in metric_labels:
        tp = matrix[label][label]
        support = sum(r.weight for r in rows if r.label == label)
        predicted_support = sum(matrix[t][label] for t in metric_labels)
        fp = predicted_support - tp
        fn = support - tp
        precision = tp / predicted_support if predicted_support > 0 else None
        recall = tp / support if support > 0 else None
        if precision is None or recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = ClassMetrics(
            label=label,
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
            predicted_support=predicted_support,
            tp=tp,
            fp=fp,
            fn=fn,
        )

    def macro(attr: str) -> float:
        values = [getattr(c, attr) for c in per_class.values()]
        defined = [v for v in values if v is not None]
        return sum(defined) / len(defined) if defined else 0.0

    def weighted(attr: str) -> float:
        if total <= 0:
            return 0.0
        return sum(
            (getattr(c, attr) or 0.0) * c.support for c in per_class.values()
        ) / total

    return MetricSet(
        labels=metric_labels,
        per_class=per_class,
        accuracy=correct / total if total > 0 else 0.0,
        total_weight=total,
        macro_precision=macro("precision"),
        macro_recall=macro("recall"),
        macro_f1=macro("f1"),
        weighted_precision=weighted("precision"),
        weighted_recall=weighted("recall"),
        weighted_f1=weighted("f1"),
        confusion=matrix,
    )


def binary_metrics(rows: Iterable, positive_label: str) -> ClassMetrics:
    metrics = multiclass_metrics(rows)
    if positive_label not in metrics.per_class:
        raise ValueError(f"正类标签 {positive_label} 未出现在数据中")
    return metrics.per_class[positive_label]
