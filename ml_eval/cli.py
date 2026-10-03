"""命令行：对本地分类预测做总体/分组评估并输出 JSON 报告。"""

from __future__ import annotations

import argparse
import json
import sys

from .evaluation import evaluate_batch
from .comparison import compare_batches
from .io import load_predictions, load_samples
from .predictions import PredictionBatch
from .scope import SCOPES, SCOPE_ALL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-eval", description="分类预测分组评估")
    parser.add_argument("--samples", required=True, help="样本集 CSV/JSON")
    parser.add_argument("--predictions", help="单个预测批次 CSV/JSON")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="参与比较的模型预测批次，可重复；与 --predictions 互斥",
    )
    parser.add_argument("--batch-name", default="local-batch")
    parser.add_argument("--group", help="分组属性名（对应 g_<名字> 列或 groups 键）")
    parser.add_argument(
        "--scope",
        choices=SCOPES,
        default=SCOPE_ALL,
        help="主成绩的参与范围：all_samples（默认，异常按错误计）或 evaluable（诊断口径）",
    )
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
    if args.predictions and args.model:
        raise SystemExit("--predictions 与 --model 互斥，请只选一种模式")
    if not args.predictions and not args.model:
        raise SystemExit("必须提供 --predictions 或至少一个 --model NAME=PATH")
    sample_set = load_samples(args.samples)
    if args.model:
        batches = []
        for spec in args.model:
            if "=" not in spec:
                raise SystemExit(f"--model 需要 NAME=PATH 形式: {spec}")
            name, path = spec.split("=", 1)
            batches.append(
                PredictionBatch(
                    predictions=load_predictions(path),
                    sample_set=sample_set,
                    batch_name=name,
                )
            )
        report = compare_batches(
            batches,
            group_attr=args.group,
            bootstrap_metric=args.bootstrap_metric,
            n_replicates=args.replicates,
            seed=args.seed,
            confidence_level=args.confidence,
            scope=args.scope,
        )
    else:
        batch = PredictionBatch(
            predictions=load_predictions(args.predictions),
            sample_set=sample_set,
            batch_name=args.batch_name,
        )
        report = evaluate_batch(
            batch,
            group_attr=args.group,
            bootstrap_metric=args.bootstrap_metric,
            n_replicates=args.replicates,
            seed=args.seed,
            confidence_level=args.confidence,
            scope=args.scope,
        )
    text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
