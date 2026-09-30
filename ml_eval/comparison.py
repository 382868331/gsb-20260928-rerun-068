"""多模型并排比较：覆盖率不一致时也能放在同一本账上核对。

每个模型先独立走 `evaluate_batch`（同一套参与范围、同一套对账口径），
比较层只做汇总不重算：覆盖率 = 可评估权重 / 全样本权重，分组表取各模型
分组报告的并集，某个模型没有可评估行的组也会显式出现（指标为 null，
权重去向在该模型的组级 accounting 中列明）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evaluation import EvaluationReport, evaluate_batch
from .predictions import PredictionBatch
from .versioning import content_version


@dataclass(frozen=True)
class ModelEntry:
    batch_name: str
    prediction_batch_version: str
    participating_version: str
    coverage: float
    report: EvaluationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "prediction_batch_version": self.prediction_batch_version,
            "participating_version": self.participating_version,
            "coverage": self.coverage,
            "accounting": self.report.accounting.to_dict(),
            "overall": self.report.overall.to_dict(),
            "groups": [g.to_dict() for g in self.report.groups],
        }


@dataclass(frozen=True)
class GroupComparisonRow:
    group: str
    n_samples: int
    total_weight: float
    per_model: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "n_samples": self.n_samples,
            "total_weight": self.total_weight,
            "per_model": self.per_model,
        }


@dataclass(frozen=True)
class ComparisonReport:
    sample_set_version: str
    comparison_version: str
    group_attr: str | None
    models: tuple[ModelEntry, ...]
    groups: tuple[GroupComparisonRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_set_version": self.sample_set_version,
            "comparison_version": self.comparison_version,
            "group_attr": self.group_attr,
            "models": [m.to_dict() for m in self.models],
            "groups": [g.to_dict() for g in self.groups],
        }


def compare_batches(
    batches: list[PredictionBatch] | tuple[PredictionBatch, ...],
    group_attr: str | None = None,
    bootstrap_metric: str | None = None,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
) -> ComparisonReport:
    batches = list(batches)
    if not batches:
        raise ValueError("比较至少需要一个预测批次")
    names = [b.batch_name for b in batches]
    if len(set(names)) != len(names):
        raise ValueError(f"批次名必须唯一: {sorted(names)}")
    versions = {b.sample_set.version for b in batches}
    if len(versions) != 1:
        raise ValueError("参与比较的批次必须对齐同一样本集版本")
    sample_set_version = versions.pop()

    models: list[ModelEntry] = []
    reports: dict[str, EvaluationReport] = {}
    for batch in batches:
        report = evaluate_batch(
            batch,
            group_attr=group_attr,
            bootstrap_metric=bootstrap_metric,
            n_replicates=n_replicates,
            seed=seed,
            confidence_level=confidence_level,
        )
        reports[batch.batch_name] = report
        accounting = report.accounting
        coverage = (
            accounting.evaluable_weight / accounting.total_weight
            if accounting.total_weight > 0
            else 0.0
        )
        models.append(
            ModelEntry(
                batch_name=batch.batch_name,
                prediction_batch_version=report.prediction_batch_version,
                participating_version=report.participating_version,
                coverage=coverage,
                report=report,
            )
        )

    group_rows: list[GroupComparisonRow] = []
    if group_attr is not None:
        group_names = sorted(
            {g.group for r in reports.values() for g in r.groups if g.group is not None}
        )
        for name in group_names:
            per_model: dict[str, dict[str, Any]] = {}
            n_samples = 0
            total_weight = 0.0
            for model in models:
                group_report = next(
                    (g for g in reports[model.batch_name].groups if g.group == name),
                    None,
                )
                if group_report is None or group_report.accounting is None:
                    per_model[model.batch_name] = {
                        "n_evaluable": 0,
                        "evaluable_weight": 0.0,
                        "excluded_weight": None,
                        "correct_weight": None,
                        "accuracy_evaluable": None,
                        "accuracy_all_samples": None,
                    }
                    continue
                acc = group_report.accounting
                n_samples = acc.n_samples
                total_weight = acc.total_weight
                per_model[model.batch_name] = {
                    "n_evaluable": group_report.n_samples,
                    "evaluable_weight": acc.evaluable_weight,
                    "excluded_weight": acc.excluded_weight,
                    "correct_weight": acc.correct_weight,
                    "accuracy_evaluable": acc.accuracy_evaluable,
                    "accuracy_all_samples": acc.accuracy_all_samples,
                }
            group_rows.append(
                GroupComparisonRow(
                    group=name,
                    n_samples=n_samples,
                    total_weight=total_weight,
                    per_model=per_model,
                )
            )

    comparison_version = content_version(
        [
            ("sample_set_version", sample_set_version),
            ("group_attr", group_attr),
            (
                "models",
                [[m.batch_name, m.participating_version] for m in models],
            ),
            (
                "bootstrap",
                None
                if bootstrap_metric is None
                else [bootstrap_metric, n_replicates, seed, confidence_level],
            ),
        ]
    )
    return ComparisonReport(
        sample_set_version=sample_set_version,
        comparison_version=comparison_version,
        group_attr=group_attr,
        models=tuple(models),
        groups=tuple(group_rows),
    )
