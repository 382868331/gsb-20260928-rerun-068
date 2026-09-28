from __future__ import annotations

import argparse
import json

from .bootstrap import bootstrap_interval
from .evaluation import evaluate_classification
from .predictions import PredictionBatch
from .samples import SampleSet


def _load_samples(path: str) -> SampleSet:
    if path.endswith(".json"):
        return SampleSet.from_json(path)
    return SampleSet.from_csv(path)


def _load_predictions(path: str, batch_id: str) -> PredictionBatch:
    if path.endswith(".json"):
        return PredictionBatch.from_json(path, batch_id=batch_id)
    return PredictionBatch.from_csv(path, batch_id=batch_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local classification prediction batch")
    parser.add_argument("samples", help="CSV or JSON sample set")
    parser.add_argument("predictions", help="CSV or JSON prediction batch")
    parser.add_argument("--batch-id", default="batch")
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)

    samples = _load_samples(args.samples)
    predictions = _load_predictions(args.predictions, args.batch_id)
    result = evaluate_classification(samples, predictions)
    payload = result.to_dict()
    if args.bootstrap_iterations:
        from .alignment import align_predictions

        alignment = align_predictions(samples, predictions)
        interval = bootstrap_interval(
            alignment.pairs,
            samples.classes,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        payload["bootstrap"] = {
            "metric": interval.metric,
            "iterations": interval.iterations,
            "seed": interval.seed,
            "lower": interval.lower,
            "upper": interval.upper,
            "point_estimate": interval.point_estimate,
            "input_version": interval.input_version,
            "result_version": interval.result_version,
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
