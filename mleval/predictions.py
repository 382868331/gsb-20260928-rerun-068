from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .versions import canonical_hash


@dataclass(frozen=True, slots=True)
class Prediction:
    sample_id: str
    y_pred: str

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("prediction sample_id must be non-empty")
        if self.y_pred is None or self.y_pred == "":
            raise ValueError("y_pred must be non-empty; represent missing predictions by omission")


class PredictionBatch:
    """Versioned prediction rows. Duplicate rows are retained and detected."""

    def __init__(self, predictions: Iterable[Prediction | Mapping[str, Any]], batch_id: str = "batch"):
        self._predictions = tuple(
            item if isinstance(item, Prediction) else Prediction(**item) for item in predictions
        )
        self.batch_id = batch_id
        counts: dict[str, int] = {}
        for prediction in self._predictions:
            counts[prediction.sample_id] = counts.get(prediction.sample_id, 0) + 1
        self._duplicate_sample_ids = tuple(
            dict.fromkeys(
                prediction.sample_id
                for prediction in self._predictions
                if counts[prediction.sample_id] > 1
            )
        )
        self._version = canonical_hash(
            {
                "batch_id": batch_id,
                "predictions": [
                    {"sample_id": prediction.sample_id, "y_pred": prediction.y_pred}
                    for prediction in self._predictions
                ],
            }
        )

    @property
    def predictions(self) -> tuple[Prediction, ...]:
        return self._predictions

    @property
    def duplicate_sample_ids(self) -> tuple[str, ...]:
        return self._duplicate_sample_ids

    @property
    def version(self) -> str:
        return self._version

    @classmethod
    def from_records(
        cls, records: Iterable[Mapping[str, Any]], batch_id: str = "batch"
    ) -> "PredictionBatch":
        return cls((dict(record) for record in records), batch_id=batch_id)

    @classmethod
    def from_json(cls, path: str | Path, batch_id: str = "batch") -> "PredictionBatch":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_records(payload, batch_id=batch_id)

    @classmethod
    def from_csv(cls, path: str | Path, batch_id: str = "batch") -> "PredictionBatch":
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return cls.from_records(rows, batch_id=batch_id)
