import pytest

from ml_eval import (
    CostModel,
    Prediction,
    PredictionBatch,
    Sample,
    SampleSet,
    evaluate_abstention,
    run_abstention_study,
)
from ml_eval.abstention import choose_threshold

COSTS = CostModel(correct_accept=0.0, wrong_accept=5.0, abstain=1.0)


def make_final_samples():
    return SampleSet(
        [
            Sample("s1", "A", 1.0, {"region": "x"}),
            Sample("s2", "A", 2.0, {"region": "x"}),
            Sample("s3", "B", 1.0, {"region": "y"}),
            Sample("s4", "B", 1.0, {"region": "y"}),
            Sample("s5", "A", 3.0, {"region": "x"}),  # 冲突预测
            Sample("s6", "B", 1.0, {"region": "y"}),  # 缺预测
            Sample("s7", "A", 1.0, {"region": "x"}),  # 未知类别
        ]
    )


def make_final_predictions():
    return [
        Prediction("s1", "A", 0.9),
        Prediction("s2", "B", 0.8),   # 错，高分数
        Prediction("s3", "B", 0.4),   # 对，低分数
        Prediction("s4", "A", 0.2),   # 错，低分数
        Prediction("s5", "A", 0.9),
        Prediction("s5", "B", 0.9),   # 冲突
        Prediction("s7", "Z", 0.9),   # 未知类别
        Prediction("s999", "A", 0.9),  # 未知身份
    ]


def make_final_batch():
    return PredictionBatch(make_final_predictions(), make_final_samples(), "m1")


def result_at(report, threshold):
    return next(r for r in report.results if r.threshold == threshold)


def test_threshold_sweep_hand_computed():
    report = evaluate_abstention(
        make_final_batch(), [0.1, 0.5, 0.95], COSTS, group_attr="region",
        n_replicates=50, seed=7,
    )
    assert [r.threshold for r in report.results] == [0.1, 0.5, 0.95]

    r = result_at(report, 0.5)
    # 接受 s1(对,1)、s2(错,2)；s3/s4 低于阈值，s5 冲突、s6 缺失、s7 未知类别均拒答
    assert r.stats["accepted_weight"] == pytest.approx(3.0)
    assert r.stats["coverage"] == pytest.approx(0.3)
    assert r.stats["correct_accept_weight"] == pytest.approx(1.0)
    assert r.stats["wrong_accept_weight"] == pytest.approx(2.0)
    assert r.stats["abstained_weight"] == pytest.approx(7.0)
    # 损失 = 5*2（错误接受）+ 1*7（拒答）= 17
    assert r.stats["loss"] == pytest.approx(17.0)
    assert r.stats["mean_loss"] == pytest.approx(1.7)
    # 拒答原因分开计数且与总计对账
    reasons = r.stats["abstain_reasons_weight"]
    assert reasons["below_threshold"] == pytest.approx(2.0)
    assert reasons["missing"] == pytest.approx(1.0)
    assert reasons["conflicting"] == pytest.approx(3.0)
    assert reasons["unknown_label"] == pytest.approx(1.0)
    assert sum(reasons.values()) == pytest.approx(r.stats["abstained_weight"])
    # 人数与权重同时对账
    assert r.stats["n_samples"] == 7
    assert r.stats["n_accepted"] + r.stats["n_abstained"] == 7
    assert r.stats["accepted_weight"] + r.stats["abstained_weight"] == pytest.approx(
        r.stats["total_weight"]
    )

    # 阈值 0.95：全部拒答，损失 = 总权重 10
    r_hi = result_at(report, 0.95)
    assert r_hi.stats["coverage"] == 0.0
    assert r_hi.stats["loss"] == pytest.approx(10.0)

    # 区间与账目同源：点估计就是 mean_loss
    assert r.confidence is not None
    assert r.confidence.point == pytest.approx(r.stats["mean_loss"])
    assert r.confidence.lower <= r.confidence.point <= r.confidence.upper


def test_group_breakdown_reconciles_with_overall():
    report = evaluate_abstention(
        make_final_batch(), [0.5], COSTS, group_attr="region", bootstrap=False
    )
    r = result_at(report, 0.5)
    groups = {g.group: g.stats for g in r.groups}
    assert set(groups) == {"x", "y"}
    # x 组：接受 s1(1)、s2(2)，拒答 s5(3)、s7(1) -> 损失 5*2 + 4 = 14
    assert groups["x"]["loss"] == pytest.approx(14.0)
    assert groups["x"]["coverage"] == pytest.approx(3.0 / 7.0)
    # y 组：全部拒答 -> 损失 3
    assert groups["y"]["loss"] == pytest.approx(3.0)
    for field in (
        "n_samples",
        "n_accepted",
        "n_abstained",
        "total_weight",
        "accepted_weight",
        "abstained_weight",
        "correct_accept_weight",
        "wrong_accept_weight",
        "loss",
    ):
        assert sum(g[field] for g in groups.values()) == pytest.approx(
            r.stats[field]
        )


def make_selection_batch():
    samples = SampleSet(
        [Sample("t1", "A", 1.0), Sample("t2", "B", 1.0)]
    )
    preds = [Prediction("t1", "A", 0.9), Prediction("t2", "B", 0.6)]
    return PredictionBatch(preds, samples, "m1")


def test_threshold_chosen_on_selection_not_final():
    # selection 上 0.5 最优（损失 0）；final 上 0.95 最优（损失 10 < 17）。
    # 选定阈值必须来自 selection，不能用 final 反向挑选。
    study = run_abstention_study(
        [make_selection_batch()],
        [make_final_batch()],
        [0.5, 0.95],
        COSTS,
        n_replicates=50,
        seed=7,
    )
    chosen = {c.batch_name: c for c in study.chosen}["m1"]
    assert chosen.threshold == 0.5
    assert chosen.selection_mean_loss == pytest.approx(0.0)
    # final 报告完整保留所有阈值结果用于核对，但不得影响选择
    final_report = {r.batch_name: r for r in study.final}["m1"]
    assert final_report.role == "final"
    assert [r.threshold for r in final_report.results] == [0.5, 0.95]
    assert result_at(final_report, 0.95).stats["loss"] == pytest.approx(10.0)
    # selection 与 final 分别留存、分别版本化
    sel_report = {r.batch_name: r for r in study.selection}["m1"]
    assert sel_report.role == "selection"
    assert sel_report.sample_set_version != final_report.sample_set_version
    assert sel_report.report_version != final_report.report_version
    assert chosen.selection_report_version == sel_report.report_version


def test_all_threshold_results_are_retained():
    # 旧策略结果不被后来的阈值评估替换：每个报告不可变地保留全部阈值
    batch = make_final_batch()
    r1 = evaluate_abstention(batch, [0.5], COSTS, bootstrap=False)
    r2 = evaluate_abstention(batch, [0.5, 0.9], COSTS, bootstrap=False)
    assert [r.threshold for r in r1.results] == [0.5]
    assert [r.threshold for r in r2.results] == [0.5, 0.9]
    # 同一阈值在两份报告中的结果一致（阈值集合变化不改写旧结果）
    assert result_at(r1, 0.5).stats == result_at(r2, 0.5).stats
    assert r1.report_version != r2.report_version


def test_bootstrap_is_deterministic_and_versioned():
    batch = make_final_batch()
    r1 = evaluate_abstention(batch, [0.5], COSTS, n_replicates=100, seed=11)
    r2 = evaluate_abstention(batch, [0.5], COSTS, n_replicates=100, seed=11)
    r3 = evaluate_abstention(batch, [0.5], COSTS, n_replicates=100, seed=12)
    c1 = result_at(r1, 0.5).confidence
    c2 = result_at(r2, 0.5).confidence
    c3 = result_at(r3, 0.5).confidence
    assert (c1.lower, c1.upper) == (c2.lower, c2.upper)
    assert c1.replicate_version == c2.replicate_version
    assert c1.replicate_version != c3.replicate_version


def test_validation_errors():
    with pytest.raises(ValueError, match="不能为负"):
        CostModel(correct_accept=0.0, wrong_accept=-1.0, abstain=1.0)
    with pytest.raises(ValueError, match="至少需要一个阈值"):
        evaluate_abstention(make_final_batch(), [], COSTS)
    with pytest.raises(ValueError, match="一一对应"):
        run_abstention_study(
            [make_selection_batch()],
            [PredictionBatch([], make_final_samples(), "other")],
            [0.5],
            COSTS,
        )


def test_choose_threshold_tie_breaks_by_coverage():
    samples = SampleSet([Sample("t1", "A", 1.0), Sample("t2", "B", 1.0)])
    preds = [Prediction("t1", "A", 0.9), Prediction("t2", "B", 0.6)]
    batch = PredictionBatch(preds, samples, "m")
    # 代价全 0：各阈值损失相同，选覆盖率更高者
    report = evaluate_abstention(
        batch, [0.5, 0.95], CostModel(0.0, 0.0, 0.0), bootstrap=False
    )
    assert choose_threshold(report).threshold == 0.5
