from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from .alignment import AlignmentStatus, EvaluatedPair


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def metric_snapshot(pairs: tuple[EvaluatedPair, ...], classes: tuple[str, ...]) -> dict:
    """Calculate weighted classification metrics."""
    total_weight = sum(pair.weight for pair in pairs)
    correct_weight = sum(pair.weight for pair in pairs if pair.correct)
    unknown_weight = sum(
        pair.weight
        for pair in pairs
        if pair.status is AlignmentStatus.UNKNOWN_LABEL
    )

    true_weight: dict[str, float] = defaultdict(float)
    pred_weight: dict[str, float] = defaultdict(float)
    true_positive: dict[str, float] = defaultdict(float)
    for pair in pairs:
        true_weight[pair.y_true] += pair.weight
        if pair.status is AlignmentStatus.KNOWN:
            pred_weight[pair.y_pred] += pair.weight
            if pair.y_pred == pair.y_true:
                true_positive[pair.y_true] += pair.weight

    per_class = {}
    f1_values = []
    for label in classes:
        precision = _safe_divide(true_positive[label], pred_weight[label])
        recall = _safe_divide(true_positive[label], true_weight[label])
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "support": true_weight[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if f1 is not None:
            f1_values.append(f1)

    macro_f1 = _safe_divide(sum(f1_values), len(f1_values))
    weighted_f1 = _safe_divide(
        sum(
            per_class[label]["f1"] * per_class[label]["support"]
            for label in classes
            if per_class[label]["f1"] is not None
        ),
        sum(
            per_class[label]["support"]
            for label in classes
            if per_class[label]["f1"] is not None
        ),
    )
    return {
        "count": len(pairs),
        "weight": total_weight,
        "accuracy": _safe_divide(correct_weight, total_weight),
        "error_rate": _safe_divide(total_weight - correct_weight, total_weight),
        "unknown_label_count": sum(
            1 for pair in pairs if pair.status is AlignmentStatus.UNKNOWN_LABEL
        ),
        "unknown_label_weight": unknown_weight,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def group_metrics(
    pairs: tuple[EvaluatedPair, ...], classes: tuple[str, ...], group_name: str
) -> dict[str, dict]:
    grouped: dict[str, list[EvaluatedPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.groups.get(group_name, "<unassigned>")].append(pair)
    return {
        value: metric_snapshot(tuple(group_pairs), classes)
        for value, group_pairs in sorted(grouped.items())
    }


def average_group_metric(metrics_by_group: Mapping[str, dict], metric_name: str) -> float | None:
    values = [snapshot[metric_name] for snapshot in metrics_by_group.values()]
    values = [value for value in values if value is not None]
    return _safe_divide(sum(values), len(values))
