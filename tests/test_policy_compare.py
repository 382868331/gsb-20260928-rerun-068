"""冻结策略并排比较：总体代价、组覆盖、未达标约束、逐样本变化与保存重载。

手算依据（COSTS: 错误接受 2 / 正确接受 0 / 拒答 1，最低覆盖缺省 0.5）：

* 选择资料：x 组 s1(A,0.9 正确)、s2(B,0.8 错误)；y 组 s3(A,0.95 正确)、
  s4(B,0.1 正确)、s5 缺预测（权重 1，恒按拒答计 1）。
* per_group：x 阈值 0.9（代价 1，覆盖 0.5），y 阈值 0.1（代价 1，覆盖 2/3）。
* uniform：唯一同时满足两组覆盖的阈值是 0.1（x 接受 s1、s2，y 接受 s3、s4）。
* 核对资料：s1 打分掉到 0.05，其余不变，s5 仍缺预测。
  per_group 在 x 组接受 0 权重（未达标），uniform 在 x 组接受 s2（错误接受）。
"""

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


def make_samples():
    return SampleSet(
        [
            Sample("s1", "A", 1.0, {"region": "x"}),
            Sample("s2", "A", 1.0, {"region": "x"}),
            Sample("s3", "A", 1.0, {"region": "y"}),
            Sample("s4", "B", 1.0, {"region": "y"}),
            Sample("s5", "B", 1.0, {"region": "y"}),  # 缺预测
        ]
    )


def make_selection():
    preds = [
        Prediction("s1", "A", 0.9),
        Prediction("s2", "B", 0.8),
        Prediction("s3", "A", 0.95),
        Prediction("s4", "B", 0.1),
    ]
    return PredictionBatch(preds, make_samples(), "selection")


def make_holdout():
    preds = [
        Prediction("s1", "A", 0.05),  # 掉到冻结阈值以下
        Prediction("s2", "B", 0.8),
        Prediction("s3", "A", 0.95),
        Prediction("s4", "B", 0.1),
    ]
    return PredictionBatch(preds, make_samples(), "holdout")


def make_comparison():
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
    return per_group, uniform, comparison


def test_frozen_thresholds_hand_computed():
    per_group, uniform, _ = make_comparison()
    assert dict(per_group.thresholds) == {"x": 0.9, "y": 0.1}
    assert dict(uniform.thresholds) == {"*": 0.1}


def test_summary_lists_overall_cost_and_group_coverage():
    per_group, uniform, comparison = make_comparison()
    assert comparison.sample_set_version == make_holdout().sample_set.version
    by_version = {s["policy_version"]: s for s in comparison.summary}
    assert set(by_version) == {per_group.policy_version, uniform.policy_version}
    pg = by_version[per_group.policy_version]
    un = by_version[uniform.policy_version]
    # 缺预测的 s5 保留在分母；总体账目每个样本只计一次
    assert pg["total_weight"] == pytest.approx(5.0)
    assert un["total_weight"] == pytest.approx(5.0)
    # per_group：接受 s3、s4（权重 2），s1/s2/s5 拒答，代价 3
    assert pg["coverage"] == pytest.approx(0.4)
    assert pg["total_cost"] == pytest.approx(3.0)
    assert pg["group_coverage"] == {"x": 0.0, "y": pytest.approx(2 / 3)}
    # uniform：s1 同样掉阈拒答，接受 s2（错误）、s3、s4，代价 2 + s1/s5 拒答 2 = 4
    assert un["coverage"] == pytest.approx(0.6)
    assert un["total_cost"] == pytest.approx(4.0)
    assert un["group_coverage"] == {"x": 0.5, "y": pytest.approx(2 / 3)}
    # 阈值来自选择资料，核对层不重选
    for evaluation in comparison.evaluations:
        assert evaluation.thresholds_source == "selection"


def test_unmet_constraints_reported_on_holdout():
    per_group, uniform, comparison = make_comparison()
    # per_group 的 x 组冻结阈值 0.9：核对资料上 s1 掉阈、s2 也不够，
    # 实际覆盖率 0 低于要求 0.5，明确列为未达标约束
    assert len(comparison.unmet_constraints) == 1
    unmet = comparison.unmet_constraints[0]
    assert unmet["policy_version"] == per_group.policy_version
    assert unmet["group"] == "x"
    assert unmet["min_coverage"] == pytest.approx(0.5)
    assert unmet["actual_coverage"] == pytest.approx(0.0)
    assert unmet["shortfall"] == pytest.approx(0.5)
    assert all(
        u["policy_version"] != uniform.policy_version
        for u in comparison.unmet_constraints
    )


def test_per_sample_decision_changes():
    per_group, uniform, comparison = make_comparison()
    changes = {c["sample_id"]: c for c in comparison.decision_changes}
    # 仅 s2 决策不同：per_group 拒答，uniform 错误接受
    assert set(changes) == {"s2"}
    change = changes["s2"]["decisions"]
    pg = change[per_group.policy_version]
    un = change[uniform.policy_version]
    assert not pg["accepted"] and pg["status"] == "abstain_low_score"
    assert pg["threshold"] == pytest.approx(0.9)
    assert pg["cost"] == pytest.approx(COSTS.reject)
    assert un["accepted"] and un["status"] == "accepted_wrong"
    assert un["threshold"] == pytest.approx(0.1)
    assert un["cost"] == pytest.approx(COSTS.false_accept)


def test_comparison_save_reload_is_identical():
    _, _, comparison = make_comparison()
    payload = json.dumps(comparison.to_dict(), ensure_ascii=False)
    reopened = PolicyComparison.from_dict(json.loads(payload))
    assert reopened.comparison_version == comparison.comparison_version
    assert reopened.to_dict() == comparison.to_dict()
    assert reopened.summary == comparison.summary
    assert reopened.unmet_constraints == comparison.unmet_constraints
    assert reopened.decision_changes == comparison.decision_changes
    # 逐样本决策与置信区间重载后保持一致
    for original, restored in zip(comparison.evaluations, reopened.evaluations):
        assert original.decisions == restored.decisions
        assert original.overall.coverage_ci == restored.overall.coverage_ci
        assert original.overall.cost_ci == restored.overall.cost_ci
    # 篡改内容后拒绝载入
    tampered = json.loads(payload)
    tampered["evaluations"][0]["overall"]["total_cost"] = 0.0
    with pytest.raises(ValueError, match="evaluation_version"):
        PolicyComparison.from_dict(tampered)
    tampered2 = json.loads(payload)
    tampered2["sample_set_version"] = "0" * 16
    with pytest.raises(ValueError, match="comparison_version"):
        PolicyComparison.from_dict(tampered2)
