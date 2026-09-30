"""预测批次：校验、对齐并保留数据质量统计。

所有已知样本都进入指标计算；缺预测和未知类别预测按错误计入。重复预测必须
与同一样本的已有预测完全一致，冲突不会通过选择首条记录被静默消解。
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
    predicted: str | None
    weight: float


@dataclass(frozen=True)
class AlignmentSummary:
    n_samples: int
    n_predictions_received: int
    n_participating_samples: int
    n_valid_predictions: int
    n_missing_predictions: int
    n_duplicate_predictions: int
    n_unknown_sample_ids: int
    n_unknown_predicted_labels: int
    total_weight_participating: float
    total_weight_valid: float
    total_weight_missing: float
    total_weight_unknown_predicted: float
    missing_sample_ids: tuple[str, ...]
    duplicate_sample_ids: tuple[str, ...]
    unknown_sample_ids: tuple[str, ...]
    unknown_label_rows: tuple[tuple[str, str], ...]
    participating_sample_ids: tuple[str, ...]
    valid_sample_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_predictions_received": self.n_predictions_received,
            "n_participating_samples": self.n_participating_samples,
            "n_valid_predictions": self.n_valid_predictions,
            "n_missing_predictions": self.n_missing_predictions,
            "n_duplicate_predictions": self.n_duplicate_predictions,
            "n_unknown_sample_ids": self.n_unknown_sample_ids,
            "n_unknown_predicted_labels": self.n_unknown_predicted_labels,
            "total_weight_participating": self.total_weight_participating,
            "total_weight_valid": self.total_weight_valid,
            "total_weight_missing": self.total_weight_missing,
            "total_weight_unknown_predicted": self.total_weight_unknown_predicted,
            "missing_sample_ids": list(self.missing_sample_ids),
            "duplicate_sample_ids": list(self.duplicate_sample_ids),
            "unknown_sample_ids": list(self.unknown_sample_ids),
            "unknown_label_rows": [list(row) for row in self.unknown_label_rows],
            "participating_sample_ids": list(self.participating_sample_ids),
            "valid_sample_ids": list(self.valid_sample_ids),
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

        records_by_sample: dict[str, list[Prediction]] = {}
        unknown_ids: list[str] = []
        for pred in predictions:
            if pred.sample_id not in sample_set:
                unknown_ids.append(pred.sample_id)
                continue
            records_by_sample.setdefault(pred.sample_id, []).append(pred)

        conflicts: dict[str, set[tuple[str | None, float | None]]] = {}
        for sid, records in records_by_sample.items():
            values = {(p.predicted, p.score) for p in records}
            if len(values) > 1:
                conflicts[sid] = values
        if conflicts:
            details = "; ".join(
                f"{sid}: {sorted(
                    values,
                    key=lambda value: (value[0] or "", value[1] is None, value[1]),
                )}"
                for sid, values in sorted(conflicts.items())
            )
            raise ValueError(f"同一样本存在冲突预测: {details}")

        duplicates = [
            sid
            for sid, records in records_by_sample.items()
            for _ in records[1:]
        ]
        missing = [sid for sid in sample_set.ids if sid not in records_by_sample]
        unknown_label_rows: list[tuple[str, str]] = []
        rows: list[EvaluableRow] = []
        valid_sample_ids: list[str] = []
        for sid in sample_set.ids:
            sample = sample_set.get(sid)
            pred = records_by_sample.get(sid, [None])[0]
            if pred is None:
                rows.append(EvaluableRow(sid, sample.label, None, sample.weight))
                continue
            if pred.predicted is None or pred.predicted not in sample_set.labels:
                unknown_label_rows.append(
                    (sid, "<missing>" if pred.predicted is None else pred.predicted)
                )
                rows.append(
                    EvaluableRow(sid, sample.label, pred.predicted, sample.weight)
                )
                continue
            rows.append(
                EvaluableRow(sid, sample.label, pred.predicted, sample.weight)
            )
            valid_sample_ids.append(sid)
        rows.sort(key=lambda r: r.sample_id)
        valid_weights = {sid: sample_set.get(sid).weight for sid in valid_sample_ids}
        missing_weights = {sid: sample_set.get(sid).weight for sid in missing}
        unknown_weights = {
            sid: sample_set.get(sid).weight
            for sid, _ in unknown_label_rows
        }

        self.evaluable_rows: tuple[EvaluableRow, ...] = tuple(rows)
        self.summary = AlignmentSummary(
            n_samples=len(sample_set),
            n_predictions_received=len(predictions),
            n_participating_samples=len(rows),
            n_valid_predictions=len(valid_sample_ids),
            n_missing_predictions=len(missing),
            n_duplicate_predictions=len(duplicates),
            n_unknown_sample_ids=len(set(unknown_ids)),
            n_unknown_predicted_labels=len(unknown_label_rows),
            total_weight_participating=sum(r.weight for r in rows),
            total_weight_valid=sum(valid_weights.values()),
            total_weight_missing=sum(missing_weights.values()),
            total_weight_unknown_predicted=sum(unknown_weights.values()),
            missing_sample_ids=tuple(sorted(missing)),
            duplicate_sample_ids=tuple(sorted(set(duplicates))),
            unknown_sample_ids=tuple(sorted(set(unknown_ids))),
            unknown_label_rows=tuple(sorted(unknown_label_rows)),
            participating_sample_ids=tuple(r.sample_id for r in rows),
            valid_sample_ids=tuple(sorted(valid_sample_ids)),
        )
        self.version = content_version(
            [
                ("batch_name", batch_name),
                ("sample_set_version", sample_set.version),
                (
                    "predictions",
                    [
                        [p.sample_id, p.predicted, p.score]
                        for p in sorted(
                            predictions,
                            key=lambda p: (
                                p.sample_id,
                                p.predicted or "",
                                p.score is None,
                                p.score,
                            ),
                        )
                    ],
                ),
            ]
        )
