"""多模型比较：覆盖率不一致时也能对齐口径。

所有模型共享同一个样本集。比较的主口径是 ``all_samples``（异常样本按
错误计），这样覆盖率低的模型不会因为剔除难例而显得更优；可评估口径
（evaluable_only）作为诊断列一并给出。分组表取各模型分组的并集，
某模型没有样本的分组显式出现（单元格为 null），不会被悄悄省略。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .evaluation import EvaluationReport


@dataclass(frozen=True)
class ModelSummary:
    model: str
    prediction_batch_version: str
    n_samples: int
    n_evaluable: int
    total_weight: float
    evaluable_weight: float
    coverage: float
    accuracy_all_samples: float
    accuracy_evaluable: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prediction_batch_version": self.prediction_batch_version,
            "n_samples": self.n_samples,
            "n_evaluable": self.n_evaluable,
            "total_weight": self.total_weight,
            "evaluable_weight": self.evaluable_weight,
            "coverage": self.coverage,
            "accuracy_all_samples": self.accuracy_all_samples,
            "accuracy_evaluable": self.accuracy_evaluable,
        }


@dataclass(frozen=True)
class GroupCell:
    n_samples: int
    total_weight: float
    correct_weight: float
    coverage: float
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "total_weight": self.total_weight,
            "correct_weight": self.correct_weight,
            "coverage": self.coverage,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True)
class ComparisonReport:
    sample_set_version: str
    group_attr: str | None
    models: tuple[ModelSummary, ...]
    group_table: tuple[tuple[str, dict[str, GroupCell | None]], ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_set_version": self.sample_set_version,
            "group_attr": self.group_attr,
            "models": [m.to_dict() for m in self.models],
            "group_table": [
                {
                    "group": group,
                    "per_model": {
                        model: (cell.to_dict() if cell is not None else None)
                        for model, cell in cells.items()
                    },
                }
                for group, cells in self.group_table
            ],
            "warnings": list(self.warnings),
        }


def compare_reports(
    reports: dict[str, EvaluationReport],
    group_attr: str | None = None,
) -> ComparisonReport:
    """按模型名比较若干份评估报告（必须来自同一个样本集）。"""
    if not reports:
        raise ValueError("至少需要一个模型")
    versions = {r.sample_set_version for r in reports.values()}
    if len(versions) != 1:
        raise ValueError("参与比较的模型必须使用同一个样本集")

    models: list[ModelSummary] = []
    for name in sorted(reports):
        report = reports[name]
        overall = report.overall
        models.append(
            ModelSummary(
                model=name,
                prediction_batch_version=report.prediction_batch_version,
                n_samples=overall.n_samples,
                n_evaluable=overall.n_evaluable,
                total_weight=overall.total_weight,
                evaluable_weight=overall.evaluable_weight,
                coverage=overall.coverage,
                accuracy_all_samples=overall.accuracy,
                accuracy_evaluable=report.accounting.accuracy_evaluable,
            )
        )

    warnings: list[str] = []
    if len({round(m.coverage, 12) for m in models}) > 1:
        detail = ", ".join(f"{m.model}={m.coverage:.4f}" for m in models)
        warnings.append(
            "模型覆盖率不一致（" + detail + "）："
            "请使用 all_samples 口径比较，evaluable_only 口径会把未覆盖样本"
            "排除在分母之外，覆盖率低的模型会被高估。"
        )

    group_table: list[tuple[str, dict[str, GroupCell | None]]] = []
    if group_attr is not None:
        all_groups: set[str] = set()
        per_model_groups: dict[str, dict[str, GroupCell]] = {}
        for name in sorted(reports):
            cells: dict[str, GroupCell] = {}
            for group_report in reports[name].groups:
                cells[group_report.group] = GroupCell(
                    n_samples=group_report.n_samples,
                    total_weight=group_report.total_weight,
                    correct_weight=group_report.correct_weight,
                    coverage=group_report.coverage,
                    accuracy=group_report.accuracy,
                )
            per_model_groups[name] = cells
            all_groups.update(cells)
        for group in sorted(all_groups):
            row: dict[str, GroupCell | None] = {}
            for name in sorted(reports):
                cell = per_model_groups[name].get(group)
                if cell is None:
                    warnings.append(
                        f"模型 {name} 在分组 {group} 没有样本，该单元格为空。"
                    )
                row[name] = cell
            group_table.append((group, row))

    return ComparisonReport(
        sample_set_version=versions.pop(),
        group_attr=group_attr,
        models=tuple(models),
        group_table=tuple(group_table),
        warnings=tuple(dict.fromkeys(warnings)),
    )
