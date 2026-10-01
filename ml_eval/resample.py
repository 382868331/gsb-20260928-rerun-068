"""固定种子的加权 bootstrap 区间。

使用 numpy 默认 PCG64 生成器并显式传入种子；每个重采样版本记录
(seed, n_replicates, 参与身份, 指标名)，与批次版本一起可回到
参与计算的数据版本。重采样按行（样本身份）抽，抽中后携带其权重。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import MetricSet, multiclass_metrics
from .predictions import EvaluableRow, ScoredRow
from .versioning import content_version


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    scope: str
    point: float
    lower: float
    upper: float
    n_replicates: int
    n_rows: int
    seed: int
    confidence_level: float
    replicate_version: str

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "scope": self.scope,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "n_replicates": self.n_replicates,
            "n_rows": self.n_rows,
            "seed": self.seed,
            "confidence_level": self.confidence_level,
            "replicate_version": self.replicate_version,
        }


_METRIC_GETTERS = {
    "accuracy": lambda m: m.accuracy,
    "macro_f1": lambda m: m.macro_f1,
    "macro_precision": lambda m: m.macro_precision,
    "macro_recall": lambda m: m.macro_recall,
    "weighted_f1": lambda m: m.weighted_f1,
}


def metric_value(metrics: MetricSet, metric: str) -> float:
    if metric in _METRIC_GETTERS:
        return _METRIC_GETTERS[metric](metrics)
    if metric.startswith("f1:") or metric.startswith("precision:") or metric.startswith("recall:"):
        kind, label = metric.split(":", 1)
        cls = metrics.per_class.get(label)
        if cls is None:
            raise ValueError(f"指标引用了未知类别: {metric}")
        value = getattr(cls, kind)
        if value is None:
            raise ValueError(f"指标 {metric} 在该数据上无定义（分母为 0）")
        return value
    raise ValueError(f"不支持的指标: {metric}")


def bootstrap_ci(
    rows: tuple[EvaluableRow, ...] | list[EvaluableRow],
    metric: str = "accuracy",
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """可评估行上的诊断区间（scope=evaluable_only，与全样本口径分开命名）。"""
    rows = list(rows)
    if not rows:
        raise ValueError("没有可重采样的行")
    if n_replicates <= 0:
        raise ValueError("n_replicates 必须为正整数")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level 必须在 (0, 1) 内")

    point = metric_value(multiclass_metrics(rows), metric)
    rng = np.random.default_rng(seed)
    n = len(rows)
    values = np.empty(n_replicates, dtype=float)
    for i in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        replicate = [rows[j] for j in idx]
        values[i] = metric_value(multiclass_metrics(replicate), metric)

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(values, alpha))
    upper = float(np.quantile(values, 1.0 - alpha))
    # 重采样版本必须携带完整参与行：身份、真值、预测与权重缺一不可，
    # 否则同身份但不同预测/权重的数据会得到相同版本，结论无法回溯。
    rows_sorted = sorted(rows, key=lambda r: r.sample_id)
    replicate_version = content_version(
        [
            (
                "bootstrap",
                [
                    "evaluable_only",
                    metric,
                    n_replicates,
                    seed,
                    confidence_level,
                    [
                        [r.sample_id, r.label, r.predicted, r.weight]
                        for r in rows_sorted
                    ],
                ],
            )
        ]
    )
    return BootstrapResult(
        metric=metric,
        scope="evaluable_only",
        point=point,
        lower=lower,
        upper=upper,
        n_replicates=n_replicates,
        n_rows=n,
        seed=seed,
        confidence_level=confidence_level,
        replicate_version=replicate_version,
    )


def bootstrap_scored_accuracy(
    rows: tuple[ScoredRow, ...] | list[ScoredRow],
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """全样本口径（scope=all_samples）的加权准确率区间。

    参与行是样本集中的每一个样本（缺预测/冲突/未知类别按错误计），
    与总体、分组指标使用同一个参与范围；区间说明里的 ``n_rows``、
    ``scope`` 与对账计数可以直接互相核对。
    """
    rows = list(rows)
    if not rows:
        raise ValueError("没有可重采样的行")
    if n_replicates <= 0:
        raise ValueError("n_replicates 必须为正整数")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level 必须在 (0, 1) 内")

    def scored_accuracy(sample: list[ScoredRow]) -> float:
        total = sum(r.weight for r in sample)
        if total <= 0:
            return 0.0
        return sum(r.weight for r in sample if r.correct) / total

    point = scored_accuracy(rows)
    rng = np.random.default_rng(seed)
    n = len(rows)
    values = np.empty(n_replicates, dtype=float)
    for i in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        values[i] = scored_accuracy([rows[j] for j in idx])

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(values, alpha))
    upper = float(np.quantile(values, 1.0 - alpha))
    rows_sorted = sorted(rows, key=lambda r: r.sample_id)
    replicate_version = content_version(
        [
            (
                "bootstrap",
                [
                    "all_samples",
                    "accuracy",
                    n_replicates,
                    seed,
                    confidence_level,
                    [
                        [r.sample_id, r.label, r.predicted, r.weight, r.correct, r.status]
                        for r in rows_sorted
                    ],
                ],
            )
        ]
    )
    return BootstrapResult(
        metric="accuracy",
        scope="all_samples",
        point=point,
        lower=lower,
        upper=upper,
        n_replicates=n_replicates,
        n_rows=n,
        seed=seed,
        confidence_level=confidence_level,
        replicate_version=replicate_version,
    )
