"""重叠保护组：一个样本同时属于多个组时的定版、核对与保存重开。"""

import json

import pytest

from ml_eval import (
    AbstentionPolicy,
    CostModel,
    Prediction,
    PredictionBatch,
    Sample,
    SampleSet,
    evaluate_policy,
    fit_abstention_policy,
)

COSTS = CostModel(false_accept=2.0, correct_accept=0.0, reject=1.0)
ATTRS = ["region", "risk"]


def make_overlap_batch(name="selection"):
    samples = SampleSet(
        [
            Sample("a1", "A", 1.0, {"region": "x", "risk": "low"}),
            Sample("a2", "A", 0.5, {"region": "x", "risk": "low"}),
            Sample("b1", "B", 2.5, {"region": "y", "risk": "low"}),
            # ov 同时属于 region=x 与 risk=high 两个保护组
            Sample("ov", "A", 4.0, {"region": "x", "risk": "high"}),
            Sample("h1", "B", 1.0, {"region": "y", "risk": "high"}),
        ]
    )
    preds = [
        Prediction("a1", "A", 0.9),   # 接受则正确
        Prediction("a2", "B", 0.8),   # 接受则错误
        Prediction("b1", "B", 0.7),   # 接受则正确
        Prediction("ov", "A", 0.6),   # 接受则正确
        Prediction("h1", "B", 0.5),   # 接受则正确
    ]
    return PredictionBatch(preds, samples, name)


def test_overlap_sample_counts_once_overall_but_in_each_group():
    policy = fit_abstention_policy(
        make_overlap_batch(), ATTRS, COSTS,
        {"region=x": 0.5, "risk=high": 0.5}, strategy="per_group",
    )
    evaluation = evaluate_policy(policy, make_overlap_batch("holdout"),
                                 bootstrap=False)
    # 总体账目每个样本只计一次（含非整数权重）
    assert evaluation.overall.n_samples == 5
    assert evaluation.overall.total_weight == pytest.approx(9.0)
    # 组级账目分别核对：ov 同时计入 region=x 与 risk=high
    groups = {g.group: g for g in evaluation.groups}
    assert set(groups) == {"region=x", "region=y", "risk=high", "risk=low"}
    assert groups["region=x"].n_samples == 3
    assert groups["region=x"].total_weight == pytest.approx(5.5)
    assert groups["risk=high"].n_samples == 2
    assert groups["risk=high"].total_weight == pytest.approx(5.0)
    # 重叠组不拆成互斥组：组账目加总可以超过总体，总体仍只计一次
    assert sum(g.n_samples for g in evaluation.groups) == 10
    assert sum(g.total_weight for g in evaluation.groups) == pytest.approx(18.0)
    # 逐样本决策里 ov 带着全部归属组，且只出现一次
    ov = [d for d in evaluation.decisions if d.sample_id == "ov"]
    assert len(ov) == 1
    assert ov[0].groups == ("region=x", "risk=high")
    # 各组覆盖约束确实满足（ok 组）
    for group in ("region=x", "risk=high"):
        assert groups[group].coverage + 1e-12 >= groups[group].min_coverage


def test_strictest_threshold_applies_to_overlapping_sample():
    # region=x 要求高覆盖（阈值会压低），risk=high 的阈值更高时，
    # ov 必须通过两者中最严的阈值才会被接受
    policy = fit_abstention_policy(
        make_overlap_batch(), ATTRS, COSTS,
        {"region=x": 0.9, "risk=high": 0.0}, strategy="per_group",
    )
    thresholds = dict(policy.thresholds)
    evaluation = evaluate_policy(policy, make_overlap_batch("h"), bootstrap=False)
    ov = next(d for d in evaluation.decisions if d.sample_id == "ov")
    expected = max(thresholds["region=x"], thresholds["risk=high"])
    assert ov.threshold == pytest.approx(expected)
    assert ov.accepted == (0.6 >= expected)


def test_high_weight_overlap_sample_flips_feasibility():
    def build(big_weight, name):
        samples = SampleSet(
            [
                Sample("m1", "A", 1.0, {"region": "x", "risk": "high"}),
                Sample("m2", "A", 1.0, {"region": "x", "risk": "low"}),
                Sample("m3", "B", 1.0, {"region": "y", "risk": "high"}),
                # 高权重重叠样本缺预测：同时压低两个组的可达覆盖率
                Sample("big", "A", big_weight,
                       {"region": "x", "risk": "high"}),
            ]
        )
        preds = [
            Prediction("m1", "A", 0.9),
            Prediction("m2", "A", 0.8),
            Prediction("m3", "B", 0.7),
        ]
        return PredictionBatch(preds, samples, name)

    reqs = {"region=x": 0.6, "risk=high": 0.6}
    # big 权重 1：region=x 与 risk=high 可达覆盖率均为 2/3，约束可行
    feasible = fit_abstention_policy(build(1.0, "sel"), ATTRS, COSTS, reqs)
    status = {s.group: s for s in feasible.group_statuses}
    assert status["region=x"].status == "ok"
    assert status["risk=high"].status == "ok"
    # big 权重 8：两组可达覆盖率同时降为 2/10，约束组合不可行并被明确标记
    infeasible = fit_abstention_policy(build(8.0, "sel"), ATTRS, COSTS, reqs)
    status = {s.group: s for s in infeasible.group_statuses}
    assert status["region=x"].status == "infeasible"
    assert status["risk=high"].status == "infeasible"
    assert status["region=x"].max_coverage == pytest.approx(0.2)
    assert "0.2" in status["region=x"].reason
    assert "0.6" in status["region=x"].reason
    # 当前策略实际达到的各组结果随策略保留：不可行组整组拒答
    achieved = {r.group: r for r in infeasible.selection_results}
    assert achieved["region=x"].coverage == 0.0
    assert achieved["risk=high"].coverage == 0.0
    # 总体分母保留缺预测的高权重样本（总权重 11），它不消失
    assert achieved[None].total_weight == pytest.approx(11.0)
    assert achieved[None].n_samples == 4


def test_conflicting_uniform_requirements_reported_with_achieved_results():
    # region=x 含一个缺预测样本，可达覆盖率 5.5/6.5 低于 0.9 的要求：
    # 不存在满足要求的统一阈值，明确报告不可行并给出各组实际结果
    samples = SampleSet(
        [
            Sample("a1", "A", 1.0, {"region": "x", "risk": "low"}),
            Sample("a2", "A", 0.5, {"region": "x", "risk": "low"}),
            Sample("ov", "A", 4.0, {"region": "x", "risk": "high"}),
            Sample("m9", "B", 1.0, {"region": "x", "risk": "low"}),  # 缺预测
            Sample("h1", "B", 1.0, {"region": "y", "risk": "high"}),
        ]
    )
    batch = PredictionBatch(
        [
            Prediction("a1", "A", 0.9),
            Prediction("a2", "B", 0.8),
            Prediction("ov", "A", 0.6),
            Prediction("h1", "B", 0.5),
        ],
        samples,
        "selection",
    )
    policy = fit_abstention_policy(
        batch, ATTRS, COSTS, {"region=x": 0.9, "*": 0.0}, strategy="uniform",
    )
    status = {s.group: s for s in policy.group_statuses}
    assert status["*"].status == "infeasible"
    assert "统一阈值" in status["*"].reason
    assert status["region=x"].status == "infeasible"
    assert status["region=x"].max_coverage == pytest.approx(5.5 / 6.5)
    achieved = {r.group: r for r in policy.selection_results}
    # 不可行时各组按拒答处理，实际达到的结果仍然可查
    assert achieved["region=x"].coverage == 0.0
    assert achieved[None].status == "partial"
    # 缺预测样本保留在总体分母中
    assert achieved[None].total_weight == pytest.approx(7.5)


def test_save_reload_reproduces_per_sample_decisions():
    policy = fit_abstention_policy(
        make_overlap_batch(), ATTRS, COSTS,
        {"region=x": 0.5, "risk=high": 0.5}, strategy="per_group",
    )
    payload = json.dumps(policy.to_dict(), ensure_ascii=False)
    reopened = AbstentionPolicy.from_dict(json.loads(payload))
    assert reopened.policy_version == policy.policy_version
    assert reopened.to_dict() == policy.to_dict()

    holdout_samples = SampleSet(
        [
            Sample("n1", "A", 1.5, {"region": "x", "risk": "high"}),
            Sample("n2", "B", 0.5, {"region": "y", "risk": "low"}),
            Sample("n3", "A", 2.0, {"region": "z", "risk": "low"}),  # 新组
            Sample("n4", "B", 1.0, {"region": "x", "risk": "low"}),  # 缺预测
        ]
    )
    holdout = PredictionBatch(
        [
            Prediction("n1", "A", 0.65),
            Prediction("n2", "B", 0.55),
            Prediction("n3", "A", 0.99),
        ],
        holdout_samples,
        "holdout",
    )
    original = evaluate_policy(policy, holdout, n_replicates=20, seed=3)
    replayed = evaluate_policy(reopened, holdout, n_replicates=20, seed=3)
    # 保存重开后逐样本决定完全一致
    assert original.decisions == replayed.decisions
    assert original.to_dict() == replayed.to_dict()
    # 冻结策略不在核对资料上重优化：阈值与选择资料版本保持一致
    assert replayed.thresholds_source == "selection"
    assert replayed.selection_sample_set_version == (
        policy.selection_sample_set_version
    )
    # 缺预测的 n4 与策略未覆盖的新组 region=z 都按拒答留在账上
    decisions = {d.sample_id: d for d in replayed.decisions}
    assert decisions["n4"].status == "missing_prediction"
    assert not decisions["n4"].accepted
    assert decisions["n3"].status == "abstain_no_threshold"
    assert replayed.overall.total_weight == pytest.approx(5.0)
    groups = {g.group: g for g in replayed.groups}
    assert groups["region=z"].status == "undetermined"
    # 篡改内容后拒绝载入
    tampered = json.loads(payload)
    tampered["thresholds"]["region=x"] = 0.01
    with pytest.raises(ValueError, match="policy_version"):
        AbstentionPolicy.from_dict(tampered)


def test_group_membership_correction_keeps_denominator_and_history():
    samples_v1 = SampleSet(
        [
            Sample("s1", "A", 2.5, {"region": "x", "risk": "low"}),
            Sample("s2", "B", 1.0, {"region": "x", "risk": "high"}),
        ]
    )
    batch_v1 = PredictionBatch(
        [Prediction("s1", "A", 0.9), Prediction("s2", "B", 0.8)],
        samples_v1,
        "selection",
    )
    policy_v1 = fit_abstention_policy(batch_v1, ATTRS, COSTS, 0.0)
    # 归属组后来修正：s1 从 region=x 改到 region=z，样本集版本随之改变
    samples_v2 = SampleSet(
        [
            Sample("s1", "A", 2.5, {"region": "z", "risk": "low"}),
            Sample("s2", "B", 1.0, {"region": "x", "risk": "high"}),
        ]
    )
    assert samples_v2.version != samples_v1.version
    batch_v2 = PredictionBatch(
        [Prediction("s1", "A", 0.9), Prediction("s2", "B", 0.8)],
        samples_v2,
        "selection",
    )
    policy_v2 = fit_abstention_policy(batch_v2, ATTRS, COSTS, 0.0)
    # 历史依据保留：新旧策略版本不同，旧策略仍记录旧选择资料版本
    assert policy_v2.policy_version != policy_v1.policy_version
    assert policy_v1.selection_sample_set_version == samples_v1.version
    assert policy_v2.selection_sample_set_version == samples_v2.version
    # 旧策略应用到修正后的数据：总体分母不变，s1 按新归属计入 region=z
    evaluation = evaluate_policy(policy_v1, batch_v2, bootstrap=False)
    assert evaluation.overall.total_weight == pytest.approx(3.5)
    assert evaluation.overall.n_samples == 2
    groups = {g.group: g for g in evaluation.groups}
    assert groups["region=z"].n_samples == 1
    assert groups["region=z"].status == "undetermined"
    assert groups["region=x"].n_samples == 1
