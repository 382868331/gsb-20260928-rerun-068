"""命令行：对本地分类预测做总体/分组评估并输出 JSON 报告。"""

from __future__ import annotations

import argparse
import json
import sys

from .evaluation import evaluate_batch
from .io import load_predictions, load_samples
from .predictions import PredictionBatch
from .compare import compare_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-eval", description="分类预测分组评估")
    parser.add_argument("--samples", required=True, help="样本集 CSV/JSON")
    parser.add_argument(
        "--predictions",
        required=True,
        action="append",
        help="预测批次 CSV/JSON；可重复给出做模型比较，"
        "形如 name=path 可显式命名（默认用文件名）",
    )
    parser.add_argument("--batch-name", default="local-batch")
    parser.add_argument("--group", help="分组属性名（对应 g_<名字> 列或 groups 键）")
    parser.add_argument(
        "--bootstrap-metric",
        default=None,
        help="如 accuracy、macro_f1、weighted_f1、f1:<类别>",
    )
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", "-o", help="报告输出路径，默认 stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_set = load_samples(args.samples)
    specs = []
    for raw in args.predictions:
        if "=" in raw:
            name, path = raw.split("=", 1)
        else:
            import os

            name, path = os.path.splitext(os.path.basename(raw))[0], raw
        specs.append((name, path))
    names = [name for name, _ in specs]
    if len(names) != len(set(names)):
        raise SystemExit("预测批次名称重复，请用 name=path 区分")

    reports = {}
    for name, path in specs:
        predictions = load_predictions(path)
        batch = PredictionBatch(
            predictions=predictions,
            sample_set=sample_set,
            batch_name=name if len(specs) > 1 else args.batch_name,
        )
        reports[name] = evaluate_batch(
            batch,
            group_attr=args.group,
            bootstrap_metric=args.bootstrap_metric,
            n_replicates=args.replicates,
            seed=args.seed,
            confidence_level=args.confidence,
        )

    if len(reports) == 1:
        payload = next(iter(reports.values())).to_dict()
    else:
        payload = {
            "models": {name: r.to_dict() for name, r in sorted(reports.items())},
            "comparison": compare_reports(reports, group_attr=args.group).to_dict(),
        }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
