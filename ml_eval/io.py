"""本地样本集与预测批次的读取（CSV / JSON）。

样本 CSV 约定列：``id,label,weight``，其余以 ``g_`` 开头的列作为分组属性；
预测 CSV 约定列：``id,predicted``。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .predictions import Prediction
from .samples import Sample, SampleSet


def _number(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def load_samples_csv(path: str | Path) -> SampleSet:
    samples: list[Sample] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            groups = {
                key[2:]: value
                for key, value in row.items()
                if key.startswith("g_") and value not in (None, "")
            }
            samples.append(
                Sample(
                    sample_id=row["id"],
                    label=row["label"],
                    weight=_number(row.get("weight"), 1.0),
                    groups=groups,
                )
            )
    return SampleSet(samples)


def load_predictions_csv(path: str | Path) -> list[Prediction]:
    preds: list[Prediction] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            score = row.get("score")
            preds.append(
                Prediction(
                    sample_id=row["id"],
                    predicted=None if row.get("predicted", "") == "" else row["predicted"],
                    score=None if score in (None, "") else float(score),
                )
            )
    return preds


def load_samples_json(path: str | Path) -> SampleSet:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    samples = [
        Sample(
            sample_id=item["id"],
            label=item["label"],
            weight=float(item.get("weight", 1.0)),
            groups=dict(item.get("groups", {})),
        )
        for item in data
    ]
    return SampleSet(samples)


def load_predictions_json(path: str | Path) -> list[Prediction]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [
        Prediction(
            sample_id=item["id"],
            predicted=item.get("predicted"),
            score=None if item.get("score") is None else float(item["score"]),
        )
        for item in data
    ]


def load_samples(path: str | Path) -> SampleSet:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_samples_json(path)
    return load_samples_csv(path)


def load_predictions(path: str | Path) -> list[Prediction]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_predictions_json(path)
    return load_predictions_csv(path)
