from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .predictions import PredictionBatch
from .samples import SampleSet
from .versions import canonical_hash


class AlignmentStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN_LABEL = "unknown_label"


@dataclass(frozen=True, slots=True)
class EvaluatedPair:
    sample_id: str
    y_true: str
    y_pred: str
    weight: float
    groups: dict[str, str]
    status: AlignmentStatus
    correct: bool


@dataclass(frozen=True, slots=True)
class PredictionAlignment:
    pairs: tuple[EvaluatedPair, ...]
    missing_sample_ids: tuple[str, ...]
    duplicate_sample_ids: tuple[str, ...]
    unknown_label_sample_ids: tuple[str, ...]
    unknown_prediction_sample_ids: tuple[str, ...]
    unknown_prediction_labels: dict[str, str]
    duplicate_weight: float
    missing_weight: float
    scored_weight: float
    sample_version: str
    prediction_version: str
    version: str


def align_predictions(samples: SampleSet, predictions: PredictionBatch) -> PredictionAlignment:
    """Join one prediction batch to samples without silently dropping difficult cases.

    Missing and duplicate sample IDs are excluded from metric denominators but reported
    separately. Unknown predicted labels remain in denominators and count as errors.
    """
    sample_by_id = {sample.sample_id: sample for sample in samples.samples}
    first_prediction: dict[str, str] = {}
    prediction_counts: dict[str, int] = {}
    unknown_rows: list[str] = []
    unknown_labels: dict[str, str] = {}

    for prediction in predictions.predictions:
        prediction_counts[prediction.sample_id] = (
            prediction_counts.get(prediction.sample_id, 0) + 1
        )
        first_prediction.setdefault(prediction.sample_id, prediction.y_pred)
        if prediction.sample_id not in sample_by_id:
            unknown_rows.append(prediction.sample_id)
            unknown_labels.setdefault(prediction.sample_id, prediction.y_pred)

    duplicates = tuple(
        sample_id
        for sample_id, count in prediction_counts.items()
        if count > 1 and sample_id in sample_by_id
    )
    duplicate_set = set(duplicates)
    missing = tuple(
        sample.sample_id
        for sample in samples.samples
        if sample.sample_id not in prediction_counts
    )

    pairs: list[EvaluatedPair] = []
    unknown_label_sample_ids: list[str] = []
    for sample in samples.samples:
        if sample.sample_id in duplicate_set or sample.sample_id not in first_prediction:
            continue
        y_pred = first_prediction[sample.sample_id]
        status = (
            AlignmentStatus.KNOWN
            if y_pred in samples.classes
            else AlignmentStatus.UNKNOWN_LABEL
        )
        pairs.append(
            EvaluatedPair(
                sample_id=sample.sample_id,
                y_true=sample.y_true,
                y_pred=y_pred,
                weight=sample.weight,
                groups=dict(sample.groups),
                status=status,
                correct=status is AlignmentStatus.KNOWN and y_pred == sample.y_true,
            )
        )
        if status is AlignmentStatus.UNKNOWN_LABEL:
            unknown_label_sample_ids.append(sample.sample_id)

    duplicate_weight = sum(
        sample_by_id[sample_id].weight for sample_id in duplicate_set
    )
    missing_weight = sum(sample_by_id[sample_id].weight for sample_id in missing)
    scored_weight = sum(pair.weight for pair in pairs)
    version_payload = {
        "sample_version": samples.version,
        "prediction_version": predictions.version,
        "scored_sample_ids": [pair.sample_id for pair in pairs],
        "missing_sample_ids": list(missing),
        "duplicate_sample_ids": list(duplicates),
    }
    return PredictionAlignment(
        pairs=tuple(pairs),
        missing_sample_ids=missing,
        duplicate_sample_ids=duplicates,
        unknown_label_sample_ids=tuple(unknown_label_sample_ids),
        unknown_prediction_sample_ids=tuple(dict.fromkeys(unknown_rows)),
        unknown_prediction_labels=unknown_labels,
        duplicate_weight=duplicate_weight,
        missing_weight=missing_weight,
        scored_weight=scored_weight,
        sample_version=samples.version,
        prediction_version=predictions.version,
        version=canonical_hash(version_payload),
    )
