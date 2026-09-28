from __future__ import annotations

import random
from dataclasses import dataclass

from .alignment import EvaluatedPair
from .metrics import metric_snapshot
from .versions import canonical_hash


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    metric: str
    iterations: int
    seed: int
    sample_size: int
    lower: float
    upper: float
    point_estimate: float
    replicate_values: tuple[float, ...]
    input_version: str
    result_version: str


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate percentile of no replicates")
    position = (len(sorted_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] * (1 - fraction) + sorted_values[
        upper_index
    ] * fraction


def bootstrap_interval(
    pairs: tuple[EvaluatedPair, ...],
    classes: tuple[str, ...],
    metric: str = "accuracy",
    iterations: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Fixed-seed percentile bootstrap over aligned participating data."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if not pairs:
        raise ValueError("cannot bootstrap without scored pairs")

    point_snapshot = metric_snapshot(pairs, classes)
    point_estimate = point_snapshot[metric]
    if point_estimate is None:
        raise ValueError(f"metric {metric!r} is undefined for the input data")

    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        indices = [rng.randrange(len(pairs)) for _ in pairs]
        replicate = tuple(pairs[index] for index in indices)
        value = metric_snapshot(replicate, classes)[metric]
        if value is None:
            raise ValueError(f"metric {metric!r} is undefined in a bootstrap replicate")
        values.append(value)

    ordered = sorted(values)
    tail = (1 - confidence) / 2
    lower = _percentile(ordered, tail)
    upper = _percentile(ordered, 1 - tail)
    input_version = canonical_hash(
        [
            {
                "sample_id": pair.sample_id,
                "y_true": pair.y_true,
                "y_pred": pair.y_pred,
                "weight": pair.weight,
                "groups": pair.groups,
                "status": pair.status.value,
            }
            for pair in pairs
        ]
    )
    result_version = canonical_hash(
        {
            "input_version": input_version,
            "metric": metric,
            "iterations": iterations,
            "seed": seed,
            "confidence": confidence,
            "classes": list(classes),
        }
    )
    return BootstrapResult(
        metric=metric,
        iterations=iterations,
        seed=seed,
        sample_size=len(pairs),
        lower=lower,
        upper=upper,
        point_estimate=point_estimate,
        replicate_values=tuple(values),
        input_version=input_version,
        result_version=result_version,
    )
