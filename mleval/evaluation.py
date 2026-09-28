from __future__ import annotations

from dataclasses import dataclass

from .alignment import PredictionAlignment, align_predictions
from .metrics import average_group_metric, group_metrics, metric_snapshot
from .predictions import PredictionBatch
from .samples import SampleSet
from .versions import canonical_hash


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    sample_version: str
    prediction_version: str
    alignment_version: str
    result_version: str
    overall: dict
    groups: dict[str, dict[str, dict]]
    group_macro_accuracy: dict[str, float | None]
    data_quality: dict
    scored_sample_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "sample_version": self.sample_version,
            "prediction_version": self.prediction_version,
            "alignment_version": self.alignment_version,
            "result_version": self.result_version,
            "overall": self.overall,
            "groups": self.groups,
            "group_macro_accuracy": self.group_macro_accuracy,
            "data_quality": self.data_quality,
            "scored_sample_ids": list(self.scored_sample_ids),
        }


def evaluate_classification(
    samples: SampleSet,
    predictions: PredictionBatch,
    group_names: tuple[str, ...] | None = None,
) -> EvaluationResult:
    alignment = align_predictions(samples, predictions)
    overall = metric_snapshot(alignment.pairs, samples.classes)
    selected_groups = group_names or samples.group_names
    groups = {
        group_name: group_metrics(alignment.pairs, samples.classes, group_name)
        for group_name in selected_groups
    }
    group_macro_accuracy = {
        group_name: average_group_metric(metrics_by_group, "accuracy")
        for group_name, metrics_by_group in groups.items()
    }
    data_quality = {
        "sample_count": len(samples.samples),
        "prediction_row_count": len(predictions.predictions),
        "scored_count": len(alignment.pairs),
        "scored_weight": alignment.scored_weight,
        "missing_count": len(alignment.missing_sample_ids),
        "missing_weight": alignment.missing_weight,
        "missing_sample_ids": list(alignment.missing_sample_ids),
        "duplicate_count": len(alignment.duplicate_sample_ids),
        "duplicate_weight": alignment.duplicate_weight,
        "duplicate_sample_ids": list(alignment.duplicate_sample_ids),
        "unknown_sample_count": len(alignment.unknown_prediction_sample_ids),
        "unknown_sample_ids": list(alignment.unknown_prediction_sample_ids),
        "unknown_label_count": len(alignment.unknown_label_sample_ids),
        "unknown_label_sample_ids": list(alignment.unknown_label_sample_ids),
        "unknown_prediction_labels": alignment.unknown_prediction_labels,
    }
    result_version = canonical_hash(
        {
            "alignment_version": alignment.version,
            "overall": overall,
            "groups": groups,
            "data_quality": data_quality,
        }
    )
    return EvaluationResult(
        sample_version=samples.version,
        prediction_version=predictions.version,
        alignment_version=alignment.version,
        result_version=result_version,
        overall=overall,
        groups=groups,
        group_macro_accuracy=group_macro_accuracy,
        data_quality=data_quality,
        scored_sample_ids=tuple(pair.sample_id for pair in alignment.pairs),
    )
