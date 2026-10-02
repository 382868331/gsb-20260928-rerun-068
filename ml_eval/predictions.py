"""预测批次：校验、对齐并保留难例统计。

重复处理语义与输入顺序无关：

* 同一身份的多条预测内容完全一致时视为一致重复，折叠为一条（多出的行数计入
  ``n_duplicate_predictions``）；
* 同一身份的多条预测内容冲突（预测类别不同）时不允许“先到先得”，该身份被
  明确标记为冲突样本并排除出可计算集合，其权重计入冲突权重，总体对账指标把
  它当作错误；
* 缺预测、未知样本身份、未知预测类别同样单独计数、保留身份与权重，只有
  通过全部校验的行进入指标计算。
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
    score: float | None = None


# 全样本口径下，被排除样本（缺预测/冲突/未知类别）的占位预测类别：
# 它们一律按错误计入，但不会作为真实类别进入 per-class 与平均指标。
EXCLUDED_PREDICTION = "__excluded__"


@dataclass(frozen=True)
class AlignmentSummary:
    n_samples: int
    n_predictions_received: int
    n_evaluable: int
    n_missing_predictions: int
    n_duplicate_predictions: int
    n_conflicting_samples: int
    n_unknown_sample_ids: int
    n_unknown_predicted_labels: int
    total_weight_samples: float
    total_weight_evaluable: float
    total_weight_missing: float
    total_weight_conflicting: float
    total_weight_unknown_label: float
    missing_sample_ids: tuple[str, ...]
    duplicate_sample_ids: tuple[str, ...]
    conflicting_rows: tuple[tuple[str, tuple[str, ...]], ...]
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
            "n_conflicting_samples": self.n_conflicting_samples,
            "n_unknown_sample_ids": self.n_unknown_sample_ids,
            "n_unknown_predicted_labels": self.n_unknown_predicted_labels,
            "total_weight_samples": self.total_weight_samples,
            "total_weight_evaluable": self.total_weight_evaluable,
            "total_weight_missing": self.total_weight_missing,
            "total_weight_conflicting": self.total_weight_conflicting,
            "total_weight_unknown_label": self.total_weight_unknown_label,
            "missing_sample_ids": list(self.missing_sample_ids),
            "duplicate_sample_ids": list(self.duplicate_sample_ids),
            "conflicting_rows": [[sid, list(values)] for sid, values in self.conflicting_rows],
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

        grouped: dict[str, list[Prediction]] = {}
        unknown_ids: set[str] = set()
        for pred in predictions:
            if pred.sample_id not in sample_set:
                unknown_ids.add(pred.sample_id)
                continue
            grouped.setdefault(pred.sample_id, []).append(pred)

        missing = sorted(sid for sid in sample_set.ids if sid not in grouped)
        rows: list[EvaluableRow] = []
        duplicates = 0
        duplicate_ids: set[str] = set()
        conflicting_rows: list[tuple[str, tuple[str, ...]]] = []
        unknown_label_rows: list[tuple[str, str]] = []
        weight_missing = 0.0
        weight_conflicting = 0.0
        weight_unknown_label = 0.0

        def label_token(pred: Prediction) -> str:
            return "<missing>" if pred.predicted is None else pred.predicted

        for sid in sorted(grouped):
            sample = sample_set.get(sid)
            preds = grouped[sid]
            distinct = sorted({(p.predicted, p.score) for p in preds})
            if len(preds) > 1:
                duplicates += len(preds) - 1
                duplicate_ids.add(sid)
            if len(distinct) > 1:
                values = tuple(sorted({label_token(p) for p in preds}))
                conflicting_rows.append((sid, values))
                weight_conflicting += sample.weight
                continue
            chosen = next(p for p in preds if (p.predicted, p.score) == distinct[0])
            if chosen.predicted is None or chosen.predicted not in sample_set.labels:
                unknown_label_rows.append((sid, label_token(chosen)))
                weight_unknown_label += sample.weight
                continue
            rows.append(
                EvaluableRow(sid, sample.label, chosen.predicted, sample.weight, chosen.score)
            )

        for sid in missing:
            weight_missing += sample_set.get(sid).weight

        self.evaluable_rows: tuple[EvaluableRow, ...] = tuple(rows)
        self.summary = AlignmentSummary(
            n_samples=len(sample_set),
            n_predictions_received=len(predictions),
            n_evaluable=len(rows),
            n_missing_predictions=len(missing),
            n_duplicate_predictions=duplicates,
            n_conflicting_samples=len(conflicting_rows),
            n_unknown_sample_ids=len(unknown_ids),
            n_unknown_predicted_labels=len(unknown_label_rows),
            total_weight_samples=sum(s.weight for s in sample_set.samples),
            total_weight_evaluable=sum(r.weight for r in rows),
            total_weight_missing=weight_missing,
            total_weight_conflicting=weight_conflicting,
            total_weight_unknown_label=weight_unknown_label,
            missing_sample_ids=tuple(missing),
            duplicate_sample_ids=tuple(sorted(duplicate_ids)),
            conflicting_rows=tuple(conflicting_rows),
            unknown_sample_ids=tuple(sorted(unknown_ids)),
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
                        for p in sorted(
                            predictions,
                            key=lambda p: (p.sample_id, p.predicted or "", p.score if p.score is not None else ""),
                        )
                    ],
                ),
                (
                    "duplicate_policy",
                    "identical-fold-conflicting-excluded-order-independent",
                ),
            ]
        )
