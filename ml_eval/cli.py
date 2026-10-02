"""命令行：对本地分类预测做总体/分组评估并输出 JSON 报告。"""

from __future__ import annotations

import argparse
import json
import sys

from .evaluation import evaluate_batch
from .comparison import compare_batches
from .abstention import CostModel, run_abstention_study
from .io import load_predictions, load_samples
from .predictions import PredictionBatch


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
        choices=["all_samples", "evaluable"],
        default="all_samples",
        help="参与范围：默认 all_samples（缺预测/冲突/未知类别按错误计）；"
        "evaluable 为仅在可评估样本上的诊断口径",
    )
    parser.add_argument(
        "--thresholds",
        help="拒答阈值扫描，逗号分隔（如 0.3,0.5,0.7）；需要预测带 score 列",
    )
    parser.add_argument("--cost-correct", type=float, default=0.0, help="正确接受代价")
    parser.add_argument("--cost-wrong", type=float, default=1.0, help="错误接受代价")
    parser.add_argument("--cost-abstain", type=float, default=1.0, help="主动拒答代价")
    parser.add_argument(
        "--selection-samples", help="阈值选择用样本集（与最终核对资料分别留存）"
    )
    parser.add_argument(
        "--selection-model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="阈值选择用预测批次，NAME 需与 --model 对应",
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
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]
        costs = CostModel(
            correct_accept=args.cost_correct,
            wrong_accept=args.cost_wrong,
            abstain=args.cost_abstain,
        )
        if not args.selection_samples or not args.selection_model:
            raise SystemExit(
                "阈值扫描需要 --selection-samples 与 --selection-model（选择资料与最终核对资料分别留存）"
            )
        selection_set = load_samples(args.selection_samples)
        final_specs = args.model or [(args.batch_name + "=" + args.predictions)]
        final_batches = []
        for spec in final_specs:
            name, path = spec.split("=", 1)
            final_batches.append(
                PredictionBatch(
                    predictions=load_predictions(path),
                    sample_set=sample_set,
                    batch_name=name,
                )
            )
        selection_batches = []
        for spec in args.selection_model:
            if "=" not in spec:
                raise SystemExit(f"--selection-model 需要 NAME=PATH 形式: {spec}")
            name, path = spec.split("=", 1)
            selection_batches.append(
                PredictionBatch(
                    predictions=load_predictions(path),
                    sample_set=selection_set,
                    batch_name=name,
                )
            )
        report = run_abstention_study(
            selection_batches,
            final_batches,
            thresholds,
            costs,
            group_attr=args.group,
            n_replicates=args.replicates,
            seed=args.seed,
            confidence_level=args.confidence,
        )
    elif args.model:
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
