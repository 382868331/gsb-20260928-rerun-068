import pytest

from ml_eval import (
    CostModel,
    Prediction,
    PredictionBatch,
    Sample,
    SampleSet,
    compare_policies,
    evaluate_policy,
    fit_abstention_policy,
)


def make_selection():
    samples = SampleSet(
        [
            Sample("s1", "A", 1.0, {"region": "x"}),
            Sample("s2", "A", 1.0, {"region": "x"}),
            Sample("s3", "B", 1.0, {"region": "x"}),
            Sample("s4", "A", 1.0, {"region": "y"}),
            Sample("s5", "B", 1.0, {"region": "y"}),  # 缺预测
        ]
    )
    preds = [
        Prediction("s1", "A", 0.9),   # 接受则正确
        Prediction("s2", "B", 0.8),   # 接受则错误
        Prediction("s3", "B", 0.1),   # 接受则正确
        Prediction("s4", "A", 0.95),  # 接受则正确
    ]
    return PredictionBatch(preds, samples, "selection")


COSTS = CostModel(false_accept=2.0, correct_accept=0.0, reject=1.0)


def test_per_group_thresholds_hand_computed():
    policy = fit_abstention_policy(
        make_selection(), "region", COSTS, 0.5, strategy="per_group"
    )
    thresholds = dict(policy.thresholds)
    # x 组：t=0.1 接受全部（代价 2/3，覆盖 1.0）优于 t=0.8（代价 1.0）
    assert thresholds["x"] == pytest.approx(0.1)
    # y 组：s5 缺预测按拒答，可达覆盖率 0.5，t=0.95 接受 s4（代价 0.5）
    assert thresholds["y"] == pytest.approx(0.95)
    status = {s.group: s for s in policy.group_statuses}
    assert status["x"].status == "ok"
    assert status["y"].status == "ok"
    assert status["y"].max_coverage == pytest.approx(0.5)
    # 选择资料上的结果随策略保存，便于统一阈值与分组阈值对比
    sel = {r.group: r for r in policy.selection_results}
    assert sel["x"].mean_cost == pytest.approx(2.0 / 3.0)
    assert sel["y"].coverage == pytest.approx(0.5)
    assert sel[None].coverage == pytest.approx(0.8)
    assert sel[None].mean_cost == pytest.approx(0.6)


def test_uniform_threshold_hand_computed():
    policy = fit_abstention_policy(
        make_selection(), "region", COSTS, 0.5, strategy="uniform"
    )
    assert dict(policy.thresholds)["*"] == pytest.approx(0.1)
    overall = next(r for r in policy.selection_results if r.group is None)
    assert overall.mean_cost == pytest.approx(0.6)
    assert overall.coverage == pytest.approx(0.8)


def test_infeasible_group_is_kept_with_reason():
    # y 组可达覆盖率 0.5，要求 0.75 -> 不可行，但组不被排除
    policy = fit_abstention_policy(
        make_selection(), "region", COSTS, {"x": 0.5, "y": 0.75},
        strategy="per_group",
    )
    status = {s.group: s for s in policy.group_statuses}
    assert status["y"].status == "infeasible"
    assert "0.5" in status["y"].reason
    assert dict(policy.thresholds)["y"] is None

    evaluation = evaluate_policy(
        policy, make_selection(), n_replicates=20, seed=3
    )
    groups = {g.group: g for g in evaluation.groups}
    assert set(groups) == {"x", "y"}
    y = groups["y"]
    assert y.status == "infeasible"
    assert y.reason is not None
    # 不可行组整组按拒答并留在总体账目里
    assert y.coverage == 0.0
    assert y.mean_cost == pytest.approx(1.0)
    assert y.n_samples == 2
    assert evaluation.overall.status == "partial"
    # 总体仍包含 y 组：覆盖 3/5，代价 (2+1+1)/5
    assert evaluation.overall.coverage == pytest.approx(0.6)
    assert evaluation.overall.mean_cost == pytest.approx(0.8)


def test_undetermined_group_without_scored_predictions():
    samples = SampleSet(
        [
            Sample("s1", "A", 1.0, {"region": "x"}),
            Sample("s2", "B", 1.0, {"region": "z"}),  # 整组缺预测
        ]
    )
    batch = PredictionBatch([Prediction("s1", "A", 0.9)], samples, "sel")
    policy = fit_abstention_policy(batch, "region", COSTS, 0.5)
    status = {s.group: s for s in policy.group_statuses}
    assert status["z"].status == "undetermined"
    assert status["z"].reason is not None
    evaluation = evaluate_policy(policy, batch, bootstrap=False)
    z = {g.group: g for g in evaluation.groups}["z"]
    assert z.status == "undetermined"
    assert z.coverage == 0.0
    assert z.n_samples == 1


def test_policy_version_tracks_costs_and_requirements():
    batch = make_selection()
    p1 = fit_abstention_policy(batch, "region", COSTS, 0.5)
    p2 = fit_abstention_policy(batch, "region", COSTS, 0.5)
    p3 = fit_abstention_policy(
        batch, "region", CostModel(3.0, 0.0, 1.0), 0.5
    )
    p4 = fit_abstention_policy(batch, "region", COSTS, 0.6)
    assert p1.policy_version == p2.policy_version
    assert p3.policy_version != p1.policy_version
    assert p4.policy_version != p1.policy_version


def test_holdout_evaluation_never_refits_thresholds():
    selection = make_selection()
    policy = fit_abstention_policy(selection, "region", COSTS, 0.5)
    # 核对资料与选择资料相互独立（不同样本集、不同分布）
    holdout_samples = SampleSet(
        [
            Sample("h1", "A", 2.0, {"region": "x"}),
            Sample("h2", "B", 1.0, {"region": "y"}),
            Sample("h3", "A", 1.0, {"region": "w"}),  # 策略未覆盖的组
        ]
    )
    holdout = PredictionBatch(
        [
            Prediction("h1", "B", 0.99),  # 高分但错误：代价上升也不换阈值
            Prediction("h2", "B", 0.7),
            Prediction("h3", "A", 0.9),
        ],
        holdout_samples,
        "holdout",
    )
    evaluation = evaluate_policy(policy, holdout, n_replicates=20, seed=5)
    assert evaluation.thresholds_source == "selection"
    assert dict(policy.thresholds)["x"] == pytest.approx(0.1)
    assert evaluation.policy_version == policy.policy_version
    assert evaluation.sample_set_version == holdout_samples.version
    groups = {g.group: g for g in evaluation.groups}
    # 策略未覆盖的组整组拒答并保留原因
    assert groups["w"].status == "undetermined"
    assert groups["w"].coverage == 0.0
    # x 组阈值不变：h1 得分 0.99 >= 0.1 被接受，代价按错误接受计
    assert groups["x"].mean_cost == pytest.approx(2.0)
    # 人数与权重对账：总权重 4；h2 得分 0.7 < y 组阈值 0.95 被拒答，
    # 仅 h1（权重 2）被接受
    assert evaluation.overall.total_weight == pytest.approx(4.0)
    assert evaluation.overall.accepted_weight == pytest.approx(2.0)
    assert evaluation.overall.coverage == pytest.approx(0.5)


def test_reconciliation_with_hard_cases_and_weights():
    samples = SampleSet(
        [
            Sample("s1", "A", 2.0, {"region": "x"}),
            Sample("s2", "B", 1.0, {"region": "x"}),  # 冲突预测
            Sample("s3", "A", 3.0, {"region": "y"}),
            Sample("s4", "B", 1.0, {"region": "y"}),  # 未知类别
            Sample("s5", "A", 5.0, {"region": "z"}),  # 整组缺预测
        ]
    )
    preds = [
        Prediction("s1", "A", 0.9),
        Prediction("s2", "A", 0.8),
        Prediction("s2", "B", 0.8),  # 冲突
        Prediction("s3", "B", 0.4),  # 接受则错误
        Prediction("s4", "Z", 0.9),  # 未知类别
    ]
    batch = PredictionBatch(preds, samples, "sel")
    policy = fit_abstention_policy(
        batch, "region", COSTS, 0.0, strategy="per_group"
    )
    evaluation = evaluate_policy(policy, batch, n_replicates=50, seed=11)
    groups = {g.group: g for g in evaluation.groups}
    assert set(groups) == {"x", "y", "z"}
    # 空组（整组缺预测）仍呈现
    assert groups["z"].n_samples == 1
    assert groups["z"].forced_abstain_weight == pytest.approx(5.0)
    assert groups["z"].coverage == 0.0
    # 分组账目加总等于总体账目
    for field in (
        "n_samples",
        "total_weight",
        "accepted_weight",
        "abstained_weight",
        "forced_abstain_weight",
        "total_cost",
    ):
        assert sum(getattr(g, field) for g in evaluation.groups) == pytest.approx(
            getattr(evaluation.overall, field)
        )
    # 覆盖率与代价口径一致：错误接受代价 2 > 拒答代价 1，y 组全拒答更优，
    # 仅 s1（权重 2，正确）被接受，其余 10 权重全部拒答
    assert evaluation.overall.accepted_weight == pytest.approx(2.0)
    assert evaluation.overall.coverage == pytest.approx(2.0 / 12.0)
    assert evaluation.overall.total_cost == pytest.approx(2 * 0.0 + 10 * 1.0)
    assert evaluation.overall.mean_cost == pytest.approx(10.0 / 12.0)
    # 区间与计数同源：点估计等于报告的覆盖率/代价
    assert evaluation.overall.coverage_ci.point == pytest.approx(
        evaluation.overall.coverage
    )
    assert evaluation.overall.cost_ci.point == pytest.approx(
        evaluation.overall.mean_cost
    )
    assert evaluation.overall.coverage_ci.lower <= evaluation.overall.coverage_ci.upper


def test_compare_uniform_vs_per_group():
    batch = make_selection()
    per_group = fit_abstention_policy(batch, "region", COSTS, 0.5, strategy="per_group")
    uniform = fit_abstention_policy(batch, "region", COSTS, 0.5, strategy="uniform")
    comparison = compare_policies([per_group, uniform], batch, n_replicates=20, seed=7)
    assert [e.strategy for e in comparison.evaluations] == ["per_group", "uniform"]
    for evaluation in comparison.evaluations:
        assert evaluation.overall.coverage == pytest.approx(0.8)
    with pytest.raises(ValueError, match="互不相同"):
        compare_policies([per_group, per_group], batch)


def test_costs_must_be_explicit_and_valid():
    with pytest.raises(ValueError, match="CostModel"):
        fit_abstention_policy(make_selection(), "region", None, 0.5)
    with pytest.raises(ValueError):
        CostModel(-1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        fit_abstention_policy(make_selection(), "region", COSTS, 1.5)
