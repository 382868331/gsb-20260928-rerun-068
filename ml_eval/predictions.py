"""预测批次：校验、对齐并保留难例统计。

缺预测、重复预测、未知样本身份、未知预测类别全部单独计数并在报告中保留，
只有通过全部校验的行进入指标计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .samples import SampleSet
from .versioning import content_version


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    predicted: str | None
    score: float | None = None


@dataclass(frozen=True)
class EvaluableRow:
    sample_id: str
    label: str
    predicted: str
    weight: float


@dataclass(frozen=True)
class AlignmentSummary:
    n_samples: int
    n_predictions_received: int
    n_evaluable: int
    n_missing_predictions: int
    n_duplicate_predictions: int
    n_unknown_sample_ids: int
    n_unknown_predicted_labels: int
    total_weight_evaluable: float
    missing_sample_ids: tuple[str, ...]
    duplicate_sample_ids: tuple[str, ...]
    unknown_sample_ids: tuple[str, ...]
    unknown_label_rows: tuple[tuple[str, str], ...]
    evaluable_sample_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_predictions_received": self.n_predictions_received,
            "n_evaluable": self.n_evaluable,
            "n_missing_predictions": self.n_missing_predictions,
            "n_duplicate_predictions": self.n_duplicate_predictions,
            "n_unknown_sample_ids": self.n_unknown_sample_ids,
            "n_unknown_predicted_labels": self.n_unknown_predicted_labels,
            "total_weight_evaluable": self.total_weight_evaluable,
            "missing_sample_ids": list(self.missing_sample_ids),
            "duplicate_sample_ids": list(self.duplicate_sample_ids),
            "unknown_sample_ids": list(self.unknown_sample_ids),
            "unknown_label_rows": [list(row) for row in self.unknown_label_rows],
            "evaluable_sample_ids": list(self.evaluable_sample_ids),
        }


class PredictionBatch:
    def __init__(
        self,
        predictions: list[Prediction],
        sample_set: SampleSet,
        batch_name: str = "batch",
    ):
        self.batch_name = batch_name
        self.sample_set = sample_set

        first_seen: dict[str, Prediction] = {}
        duplicates: list[str] = []
        unknown_ids: list[str] = []
        unknown_label_rows: list[tuple[str, str]] = []

        for pred in predictions:
            if pred.sample_id not in sample_set:
                unknown_ids.append(pred.sample_id)
                continue
            if pred.sample_id in first_seen:
                duplicates.append(pred.sample_id)
                continue
            first_seen[pred.sample_id] = pred
            label = sample_set.get(pred.sample_id).label
            if pred.predicted is None or pred.predicted not in sample_set.labels:
                unknown_label_rows.append(
                    (pred.sample_id, "<missing>" if pred.predicted is None else pred.predicted)
                )

        missing = [sid for sid in sample_set.ids if sid not in first_seen]
        rows: list[EvaluableRow] = []
        for sid, pred in first_seen.items():
            sample = sample_set.get(sid)
            if pred.predicted is None or pred.predicted not in sample_set.labels:
                continue
            rows.append(EvaluableRow(sid, sample.label, pred.predicted, sample.weight))
        rows.sort(key=lambda r: r.sample_id)

        self.evaluable_rows: tuple[EvaluableRow, ...] = tuple(rows)
        self.summary = AlignmentSummary(
            n_samples=len(sample_set),
            n_predictions_received=len(predictions),
            n_evaluable=len(rows),
            n_missing_predictions=len(missing),
            n_duplicate_predictions=len(duplicates),
            n_unknown_sample_ids=len(unknown_ids),
            n_unknown_predicted_labels=len(unknown_label_rows),
            total_weight_evaluable=sum(r.weight for r in rows),
            missing_sample_ids=tuple(sorted(missing)),
            duplicate_sample_ids=tuple(sorted(set(duplicates))),
            unknown_sample_ids=tuple(unknown_ids),
            unknown_label_rows=tuple(unknown_label_rows),
            evaluable_sample_ids=tuple(r.sample_id for r in rows),
        )
        self.version = content_version(
            [
                ("batch_name", batch_name),
                ("sample_set_version", sample_set.version),
                (
                    "predictions",
                    [
                        [p.sample_id, p.predicted, p.score]
                        for p in sorted(predictions, key=lambda p: (p.sample_id, p.predicted or ""))
                    ],
                ),
            ]
        )
