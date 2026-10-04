"""多个已冻结策略版本在同一只读留出集上的并排比较。"""

import json

import pytest

from ml_eval import (
    CostModel,
    PolicyComparison,
    Prediction,
    PredictionBatch,
    Sample,
    SampleSet,
    compare_policies,
    fit_abstention_policy,
)

COSTS = CostModel(false_accept=2.0, correct_accept=0.0, reject=1.0)


def make_selection():
    samples = SampleSet(
        [
            Sample("s1", "A", 1.0, {"region": "x"}),
            Sample("s2", "A", 1.0, {"region": "x"}),
            Sample("s3", "B", 1.0, {"region": "x"}),
            Sample("s4", "A", 1.0, {"region": "y"}),
            Sample("s5", "B", 1.0, {"region": "y"}),
        ]
    )
    preds = [
        Prediction("s1", "A", 0.9),
        Prediction("s2", "B", 0.8),
        Prediction("s3", "B", 0.1),
        Prediction("s4", "A", 0.95),
    ]
    return PredictionBatch(preds, samples, "selection")


def make_holdout():
    # y1 得分 0.5：per_group 的 y 阈值 0.95 拒答，统一阈值 0.1 接受（且错误）
    # y2 得分 0.98：两版本都接受
    # x1 得分 0.05：两版本都拒答
    # miss 缺预测：两版本都强制拒答，但必须留在分母
    samples = SampleSet(
        [
            Sample("y1", "A", 1.0, {"region": "y"}),
            Sample("y2", "B", 1.0, {"region": "y"}),
            Sample("x1", "A", 1.0, {"region": "x"}),
            Sample("miss", "A", 2.0, {"region": "y"}),
        ]
    )
    preds = [
        Prediction("y1", "B", 0.5),
        Prediction("y2", "B", 0.98),
        Prediction("x1", "A", 0.05),
    ]
    return PredictionBatch(preds, samples, "holdout")


def test_side_by_side_overall_cost_group_coverage_and_unmet_constraints():
    selection = make_selection()
    per_group = fit_abstention_policy(
        selection, "region", COSTS, 0.5, strategy="per_group"
    )
    uniform = fit_abstention_policy(
        selection, "region", COSTS, 0.5, strategy="uniform"
    )
    comparison = compare_policies(
        [per_group, uniform], make_holdout(), n_replicates=20, seed=7
    )

    by_version = {row.policy_version: row for row in comparison.overall}
    pg = by_version[per_group.policy_version]
    uf = by_version[uniform.policy_version]
    # 缺预测样本权重 2 保留在总体分母：总权重 5，每个样本总体只计一次
    assert pg.total_weight == pytest.approx(5.0)
    assert uf.total_weight == pytest.approx(5.0)
    # per_group 仅接受 y2（权重 1）；uniform 还接受 y1（错误接受代价 2）
    assert pg.accepted_weight == pytest.approx(1.0)
    assert uf.accepted_weight == pytest.approx(2.0)
    assert pg.coverage == pytest.approx(0.2)
    assert uf.coverage == pytest.approx(0.4)
    # per_group 代价：4 个拒答权重（含缺预测 2）= 4；uniform：y1 错受 2 + 3 拒答
    assert pg.total_cost == pytest.approx(4.0)
    assert uf.total_cost == pytest.approx(5.0)

    groups = {row.group: row for row in comparison.groups}
    y_cells = {c["policy_version"]: c for c in groups["y"].per_policy}
    # 缺预测的 miss 留在 y 组分母：3 个样本、权重 4
    assert y_cells[per_group.policy_version]["n_samples"] == 3
    assert y_cells[per_group.policy_version]["total_weight"] == pytest.approx(4.0)
    assert y_cells[per_group.policy_version]["coverage"] == pytest.approx(0.25)
    assert y_cells[per_group.policy_version]["meets_constraint"] is False
    assert y_cells[uniform.policy_version]["coverage"] == pytest.approx(0.5)
    assert y_cells[uniform.policy_version]["meets_constraint"] is True
    x_cells = {c["policy_version"]: c for c in groups["x"].per_policy}
    # x1 得分 0.05：两版本阈值 0.1 都拒答，x 组覆盖 0 不达标
    for cell in x_cells.values():
        assert cell["coverage"] == 0.0
        assert cell["meets_constraint"] is False

    # 未达标约束按 (版本, 组) 明列；冻结阈值不随留出结果重选
    unmet = {
        (u.policy_version, u.group): u for u in comparison.unmet_constraints
    }
    assert (per_group.policy_version, "y") in unmet
    assert (uniform.policy_version, "y") not in unmet
    assert (per_group.policy_version, "x") in unmet
    assert unmet[(per_group.policy_version, "y")].gap == pytest.approx(0.25)
    assert unmet[(per_group.policy_version, "y")].actual_coverage == pytest.approx(0.25)
    pg_y_cell = y_cells[per_group.policy_version]
    assert pg_y_cell["threshold"] == pytest.approx(0.95)


def test_per_sample_decision_changes_between_versions():
    selection = make_selection()
    per_group = fit_abstention_policy(
        selection, "region", COSTS, 0.5, strategy="per_group"
    )
    uniform = fit_abstention_policy(
        selection, "region", COSTS, 0.5, strategy="uniform"
    )
    comparison = compare_policies(
        [per_group, uniform], make_holdout(), n_replicates=20, seed=7
    )
    changes = {
        (c.sample_id, c.from_policy_version, c.to_policy_version): c
        for c in comparison.decision_changes
    }
    # y1 从低分支拒答变为错误接受
    y1 = changes[("y1", per_group.policy_version, uniform.policy_version)]
    assert y1.from_status == "abstain_low_score"
    assert y1.to_status == "accepted_wrong"
    assert y1.from_accepted is False and y1.to_accepted is True
    assert y1.from_threshold == pytest.approx(0.95)
    assert y1.to_threshold == pytest.approx(0.1)
    # y2 决策（接受）不变但所用阈值不同，也要列出
    y2 = changes[("y2", per_group.policy_version, uniform.policy_version)]
    assert y2.from_status == y2.to_status == "accepted_correct"
    assert y2.from_threshold == pytest.approx(0.95)
    assert y2.to_threshold == pytest.approx(0.1)
    # 缺预测样本两版本都强制拒答：不产生变化，但分母计数闭合
    assert not any(c.sample_id == "miss" for c in comparison.decision_changes)
    assert comparison.n_changed_decisions == 2
    assert comparison.n_unchanged_decisions == 2


def test_comparison_save_reload_is_identical():
    selection = make_selection()
    per_group = fit_abstention_policy(
        selection, "region", COSTS, 0.5, strategy="per_group"
    )
    uniform = fit_abstention_policy(
        selection, "region", COSTS, 0.5, strategy="uniform"
    )
    comparison = compare_policies(
        [per_group, uniform], make_holdout(), n_replicates=20, seed=7
    )
    payload = json.dumps(comparison.to_dict(), ensure_ascii=False)
    reopened = PolicyComparison.from_dict(json.loads(payload))
    assert reopened.to_dict() == comparison.to_dict()
    assert reopened.comparison_version == comparison.comparison_version
    # 逐样本决策随比较结果一起保存并重开
    original_decisions = [
        d.to_dict() for e in comparison.evaluations for d in e.decisions
    ]
    reopened_decisions = [
        d.to_dict() for e in reopened.evaluations for d in e.decisions
    ]
    assert reopened_decisions == original_decisions


def test_overlap_groups_count_once_overall_in_comparison():
    samples = SampleSet(
        [
            Sample("a1", "A", 1.0, {"region": "x", "risk": "low"}),
            Sample("ov", "A", 4.0, {"region": "x", "risk": "high"}),
            Sample("h1", "B", 1.0, {"region": "y", "risk": "high"}),
            Sample("m1", "A", 2.0, {"region": "x", "risk": "high"}),
        ]
    )
    selection = PredictionBatch(
        [Prediction("a1", "A", 0.9), Prediction("ov", "A", 0.6)],
        samples,
        "selection",
    )
    holdout = PredictionBatch(
        [Prediction("h1", "B", 0.5)],
        samples,
        "holdout",
    )
    per_group = fit_abstention_policy(
        selection, ["region", "risk"], COSTS, 0.0, strategy="per_group"
    )
    uniform = fit_abstention_policy(
        selection, ["region", "risk"], COSTS, 0.0, strategy="uniform"
    )
    comparison = compare_policies(
        [per_group, uniform], holdout, n_replicates=20, seed=1
    )
    for row in comparison.overall:
        # 重叠样本 ov/m1 在总体只计一次，缺预测样本权重保留在分母
        assert row.n_samples == 4
        assert row.total_weight == pytest.approx(8.0)
    group_names = {g.group for g in comparison.groups}
    assert group_names == {"region=x", "region=y", "risk=high", "risk=low"}
    region_x = next(g for g in comparison.groups if g.group == "region=x")
    for cell in region_x.per_policy:
        # ov 与 m1 同时计入 region=x 与 risk=high：组账目不互斥
        assert cell["n_samples"] == 3
        assert cell["total_weight"] == pytest.approx(7.0)
