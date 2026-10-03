"""命令行：对本地分类预测做总体/分组评估并输出 JSON 报告。"""

from __future__ import annotations

import argparse
import json
import sys

from .evaluation import evaluate_batch
from .comparison import compare_batches
from .io import load_predictions, load_samples
from .policy import (
    POLICY_GROUPED,
    POLICY_UNIFORM,
    Costs,
    GroupConstraint,
    PolicySpec,
    apply_policy,
    select_policy,
)
from .predictions import PredictionBatch
from .scope import SCOPE_ALL, SCOPE_EVALUABLE


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
    parser.add_argument(
        "--scope",
        default=SCOPE_ALL,
        choices=(SCOPE_ALL, SCOPE_EVALUABLE),
        help="主成绩参与范围；默认 all_samples，evaluable 仅为诊断口径",
    )
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
    parser.add_argument("--selection-predictions", help="拒答策略选择资料 CSV/JSON")
    parser.add_argument("--audit-predictions", help="独立核对资料 CSV/JSON；不参与选阈值")
    parser.add_argument(
        "--policy",
        choices=(POLICY_GROUPED, POLICY_UNIFORM),
        help="分组拒答策略模式；省略则不做策略选择",
    )
    parser.add_argument("--cost-false-accept", type=float, default=1.0)
    parser.add_argument("--cost-correct-accept", type=float, default=0.0)
    parser.add_argument("--cost-abstain", type=float, default=0.5)
    parser.add_argument(
        "--min-coverage",
        action="append",
        default=[],
        metavar="GROUP=VALUE",
        help="分组最低覆盖要求，可重复；组名 * 表示全体出现组",
    )
    parser.add_argument(
        "--min-samples",
        action="append",
        default=[],
        metavar="GROUP=N",
        help="选择资料中该组最少人数，不足则保留 undetermined",
    )
    return parser


def _parse_group_value(spec: str, kind: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"--{kind} 需要 GROUP=VALUE 形式: {spec}")
    return spec.split("=", 1)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_set = load_samples(args.samples)
    if args.selection_predictions:
        if not args.group and args.policy == POLICY_GROUPED:
            raise SystemExit("分组阈值策略必须通过 --group 指定分组属性")
        min_samples = {
            group: int(value)
            for group, value in (_parse_group_value(item, "min-samples") for item in args.min_samples)
        }
        constraints: dict[str, GroupConstraint] = {}
        for item in args.min_coverage:
            group, raw = _parse_group_value(item, "min-coverage")
            constraints[group] = GroupConstraint(
                min_coverage=float(raw),
                min_samples=min_samples.pop(group, 0),
            )
        for group, count in min_samples.items():
            constraints[group] = GroupConstraint(min_samples=count)
        if "*" in constraints:
            wildcard = constraints.pop("*")
            groups = sorted(
                {
                    str(sample.groups.get(args.group, "__unassigned__"))
                    for sample in sample_set.samples
                }
            )
            for group in groups:
                current = constraints.get(group, GroupConstraint())
                constraints[group] = GroupConstraint(
                    min_coverage=max(current.min_coverage, wildcard.min_coverage),
                    min_samples=max(current.min_samples, wildcard.min_samples),
                )
        spec = PolicySpec(
            group_attr=args.group,
            mode=args.policy or POLICY_GROUPED,
            costs=Costs(
                false_accept=args.cost_false_accept,
                correct_accept=args.cost_correct_accept,
                abstain=args.cost_abstain,
            ),
            group_constraints=constraints,
        )
        selection_batch = PredictionBatch(
            predictions=load_predictions(args.selection_predictions),
            sample_set=sample_set,
            batch_name="selection",
        )
        selected = select_policy(
            selection_batch,
            spec,
            n_replicates=args.replicates,
            seed=args.seed,
            confidence_level=args.confidence,
        )
        report: dict = {"selection": selected.to_dict()}
        if args.audit_predictions:
            audit_batch = PredictionBatch(
                predictions=load_predictions(args.audit_predictions),
                sample_set=sample_set,
                batch_name="audit",
            )
            report["audit"] = apply_policy(
                audit_batch,
                selected,
                n_replicates=args.replicates,
                seed=args.seed + 10000,
                confidence_level=args.confidence,
            ).to_dict()
    elif args.predictions and args.model:
        raise SystemExit("--predictions 与 --model 互斥，请只选一种模式")
    elif not args.predictions and not args.model:
        raise SystemExit(
            "必须提供 --predictions、--model，或 --selection-predictions"
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
    payload = report if isinstance(report, dict) else report.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
