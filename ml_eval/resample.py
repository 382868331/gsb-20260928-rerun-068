"""固定种子的加权 bootstrap 区间。

使用 numpy 默认 PCG64 生成器并显式传入种子；每个重采样版本记录
(seed, n_replicates, 参与身份, 指标名)，与批次版本一起可回到
参与计算的数据版本。重采样按行（样本身份）抽，抽中后携带其权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import MetricSet, multiclass_metrics
from .versioning import content_version


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    point: float
    lower: float
    upper: float
    n_replicates: int
    seed: int
    confidence_level: float
    replicate_version: str


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
    rows: tuple | list,
    metric: str = "accuracy",
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
    labels: Sequence[str] | None = None,
    abstain_label: str | None = None,
) -> BootstrapResult:
    rows = list(rows)
    if not rows:
        raise ValueError("没有可重采样的行")
    if n_replicates <= 0:
        raise ValueError("n_replicates 必须为正整数")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level 必须在 (0, 1) 内")

    point = metric_value(
        multiclass_metrics(rows, labels=labels, abstain_label=abstain_label), metric
    )
    rng = np.random.default_rng(seed)
    n = len(rows)
    values = np.empty(n_replicates, dtype=float)
    for i in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        replicate = [rows[j] for j in idx]
        values[i] = metric_value(
            multiclass_metrics(
                replicate, labels=labels, abstain_label=abstain_label
            ),
            metric,
        )

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
                    metric,
                    n_replicates,
                    seed,
                    confidence_level,
                    labels,
                    abstain_label,
                    [
                        [
                            r.sample_id,
                            r.label,
                            r.predicted,
                            r.weight,
                            getattr(r, "status", None),
                            getattr(r, "score", None),
                        ]
                        for r in rows_sorted
                    ],
                ],
            )
        ]
    )
    return BootstrapResult(
        metric=metric,
        point=point,
        lower=lower,
        upper=upper,
        n_replicates=n_replicates,
        seed=seed,
        confidence_level=confidence_level,
        replicate_version=replicate_version,
    )


def bootstrap_weighted_stat(
    rows: tuple | list,
    value_fn,
    stat_name: str,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """对任意“每行携带权重”的加权均值统计做固定种子 bootstrap。

    ``value_fn(row)`` 返回该行的单位贡献（如代价/收益）；统计量为
    ``sum(value*weight)/sum(weight)``，覆盖率、单位样本损失等都复用它，
    保证阈值扫描里所有区间使用同一套抽样与版本规则。
    """
    rows = list(rows)
    if not rows:
        raise ValueError("没有可重采样的行")
    if n_replicates <= 0:
        raise ValueError("n_replicates 必须为正整数")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level 必须在 (0, 1) 内")

    def weighted_mean(sample_rows):
        total_w = sum(r.weight for r in sample_rows)
        if total_w <= 0:
            return 0.0
        return sum(value_fn(r) * r.weight for r in sample_rows) / total_w

    point = weighted_mean(rows)
    rng = np.random.default_rng(seed)
    n = len(rows)
    values = np.empty(n_replicates, dtype=float)
    for i in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        values[i] = weighted_mean([rows[j] for j in idx])

    alpha = (1.0 - confidence_level) / 2
    rows_sorted = sorted(rows, key=lambda r: r.sample_id)
    replicate_version = content_version(
        [
            (
                "bootstrap",
                [
                    stat_name,
                    n_replicates,
                    seed,
                    confidence_level,
                    [
                        [
                            r.sample_id,
                            r.weight,
                            value_fn(r),
                            getattr(r, "status", None),
                            getattr(r, "threshold", None),
                        ]
                        for r in rows_sorted
                    ],
                ],
            )
        ]
    )
    return BootstrapResult(
        metric=stat_name,
        point=float(point),
        lower=float(np.quantile(values, alpha)),
        upper=float(np.quantile(values, 1.0 - alpha)),
        n_replicates=n_replicates,
        seed=seed,
        confidence_level=confidence_level,
        replicate_version=replicate_version,
    )
