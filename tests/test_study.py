"""多重保护组约束下的策略选择研究：可行性、差异、未知归属与重开复算。"""

import json

import pytest

from ml_eval import (
    CostModel,
    PolicyStudy,
    Prediction,
    PredictionBatch,
    Sample,
    SampleSet,
    evaluate_policy,
    fit_abstention_policy,
    run_policy_study,
)

COSTS = CostModel(false_accept=2.0, correct_accept=0.0, reject=1.0)
ATTRS = ["region", "risk"]


def make_batch(name):
    samples = SampleSet(
        [
            Sample("a1", "A", 1.0, {"region": "x", "risk": "low"}),
            Sample("a2", "A", 0.5, {"region": "x", "risk": "low"}),
            Sample("b1", "B", 2.5, {"region": "y", "risk": "low"}),
            # ov 同时属于 region=x 与 risk=high 两个保护组
            Sample("ov", "A", 4.0, {"region": "x", "risk": "high"}),
            # h1 是 risk=high 的极小组样本（非整数权重）
            Sample("h1", "B", 0.75, {"region": "y", "risk": "high"}),
            # u1 缺失 region 标签：保留未知归属，不归入任何已知侧
            Sample("u1", "B", 0.25, {"risk": "low"}),
        ]
    )
    preds = [
        Prediction("a1", "A", 0.9),   # 接受则正确
        Prediction("a2", "B", 0.8),   # 接受则错误
        Prediction("b1", "B", 0.7),   # 接受则正确
        Prediction("ov", "A", 0.6),   # 接受则正确
        Prediction("h1", "B", 0.5),   # 接受则正确
        Prediction("u1", "B", 0.4),   # 接受则正确
    ]
    return PredictionBatch(preds, samples, name)


def make_policies():
    selection = make_batch("selection")
    loose = fit_abstention_policy(selection, ATTRS, COSTS, 0.0, strategy="per_group")
    strict = fit_abstention_policy(
        selection, ATTRS, COSTS, {"*": 0.9}, strategy="per_group"
    )
    return loose, strict


def make_batch_with_missing(name):
    """在 region=x / risk=low 中加入缺预测样本：强制拒答拉低这两组覆盖。"""
    base = make_batch(name)
    samples = SampleSet(
        list(base.sample_set.samples)
        + [Sample("m1", "A", 0.5, {"region": "x", "risk": "low"})]
    )
    return PredictionBatch(
        [
            Prediction("a1", "A", 0.9),
            Prediction("a2", "B", 0.8),
            Prediction("b1", "B", 0.7),
            Prediction("ov", "A", 0.6),
            Prediction("h1", "B", 0.5),
            Prediction("u1", "B", 0.4),
        ],
        samples,
        name,
    )


def test_feasible_selection_and_two_objectives():
    loose, strict = make_policies()
    holdout = make_batch("holdout")
    study = run_policy_study(
        [loose, strict], holdout,
        cost_cap=100.0, min_coverage=0.3, max_disparity=1.0,
    )
    assert all(a.feasible for a in study.assessments)
    # min_cost 选可行策略中总体代价最小者；worst_group 选最差组覆盖最高者
    by_cost = min(study.assessments, key=lambda a: (a.total_cost, a.policy_version))
    assert study.selection_for("min_cost") == by_cost.policy_version
    by_worst = min(
        study.assessments,
        key=lambda a: (-a.worst_group_coverage, a.total_cost, a.policy_version),
    )
    assert study.selection_for("worst_group") == by_worst.policy_version
    comparison = {c["objective"]: c for c in study.objective_comparison}
    assert comparison["min_cost"]["total_cost"] == by_cost.total_cost
    assert (
        comparison["worst_group"]["worst_group_coverage"]
        == by_worst.worst_group_coverage
    )


def test_overall_counts_once_overlaps_counted_per_group():
    loose, _ = make_policies()
    study = run_policy_study(
        [loose], make_batch("holdout"),
        cost_cap=100.0, min_coverage=0.0, max_disparity=1.0,
    )
    assessment = study.assessments[0]
    evaluation = evaluate_policy(loose, make_batch("holdout"), bootstrap=False)
    # 研究只重放冻结策略：总体账目与独立核对完全一致，每个样本只计一次
    assert assessment.total_cost == pytest.approx(evaluation.overall.total_cost)
    assert evaluation.overall.n_samples == 6
    assert evaluation.overall.total_weight == pytest.approx(9.0)
    coverages = dict(assessment.group_coverages)
    # 重叠样本 ov 分别计入 region=x 与 risk=high；缺失标签的 u1 自成未知组
    assert set(coverages) == {
        "region=x", "region=y", "region=__unassigned__",
        "risk=high", "risk=low",
    }
    group_weight_sum = sum(g.total_weight for g in evaluation.groups)
    assert group_weight_sum > evaluation.overall.total_weight


def test_conflicting_constraints_report_groups_and_limits():
    loose, strict = make_policies()
    study = run_policy_study(
        [loose, strict], make_batch_with_missing("holdout"),
        cost_cap=0.01, min_coverage=0.99, max_disparity=1.0,
    )
    # 无法同时满足：两种目标都不做选择，绝不调整留出集或阈值制造可行
    assert study.selection_for("min_cost") is None
    assert study.selection_for("worst_group") is None
    assert all(not a.feasible for a in study.assessments)
    kinds = {v.constraint for a in study.assessments for v in a.violations}
    assert "min_coverage" in kinds
    assert "cost_cap" in kinds
    for assessment in study.assessments:
        for violation in assessment.violations:
            assert violation.shortfall > 0
            if violation.constraint == "min_coverage":
                assert violation.group is not None
                assert violation.actual < violation.limit
            if violation.constraint == "cost_cap":
                assert violation.limit == pytest.approx(0.01)
    notes = {c["objective"]: c["note"] for c in study.objective_comparison}
    assert all(n for n in notes.values())


def test_disparity_constraint_flags_tiny_group_gap():
    loose, _ = make_policies()
    evaluation = evaluate_policy(
        loose, make_batch_with_missing("holdout"), bootstrap=False
    )
    coverages = {g.group: g.coverage for g in evaluation.groups}
    spread = max(coverages.values()) - min(coverages.values())
    # 把差异上限压到实际差异之下：极小组（risk=high 两个样本）拉开差距
    study = run_policy_study(
        [loose], make_batch_with_missing("holdout"),
        cost_cap=100.0, min_coverage=0.0, max_disparity=spread - 0.01,
    )
    assessment = study.assessments[0]
    assert not assessment.feasible
    disparity_violations = [
        v for v in assessment.violations if v.constraint == "max_disparity"
    ]
    assert len(disparity_violations) == 1
    violation = disparity_violations[0]
    assert violation.actual == pytest.approx(spread)
    assert violation.limit == pytest.approx(spread - 0.01)
    assert assessment.disparity == pytest.approx(spread)


def test_unknown_group_label_kept_and_checked():
    loose, _ = make_policies()
    # 未知归属组被要求 0.99 覆盖：u1 只有 0.25 权重，约束按未知组单独核对
    study = run_policy_study(
        [loose], make_batch("holdout"),
        cost_cap=100.0,
        min_coverage={"region=__unassigned__": 0.99, "*": 0.0},
        max_disparity=1.0,
    )
    assessment = study.assessments[0]
    coverages = dict(assessment.group_coverages)
    assert "region=__unassigned__" in coverages
    if coverages["region=__unassigned__"] < 0.99:
        assert not assessment.feasible
        groups = {
            v.group for v in assessment.violations
            if v.constraint == "min_coverage"
        }
        assert "region=__unassigned__" in groups
    else:
        assert assessment.feasible


def test_roundtrip_reopen_recomputes_identical_version():
    loose, strict = make_policies()
    study = run_policy_study(
        [loose, strict], make_batch("holdout"),
        cost_cap=100.0, min_coverage=0.3, max_disparity=0.9,
    )
    payload = json.dumps(study.to_dict(), ensure_ascii=False)
    reopened = PolicyStudy.from_dict(json.loads(payload))
    assert reopened.to_dict() == study.to_dict()
    assert reopened.study_version == study.study_version
    assert reopened.selections == study.selections
    # 篡改内容后拒绝载入
    tampered = study.to_dict()
    tampered["cost_cap"] = tampered["cost_cap"] + 1.0
    with pytest.raises(ValueError):
        PolicyStudy.from_dict(tampered)


def test_new_policy_forms_new_study_old_frozen():
    loose, strict = make_policies()
    holdout = make_batch("holdout")
    first = run_policy_study(
        [loose], holdout, cost_cap=100.0, min_coverage=0.3, max_disparity=1.0,
    )
    second = run_policy_study(
        [loose, strict], holdout,
        cost_cap=100.0, min_coverage=0.3, max_disparity=1.0,
    )
    # 新策略到达只形成新研究：版本不同，旧研究的选择与核对保持冻结
    assert first.study_version != second.study_version
    assert len(first.assessments) == 1
    assert len(second.assessments) == 2
    assert (
        first.assessment_for(loose.policy_version).to_dict()
        == second.assessment_for(loose.policy_version).to_dict()
    )
    assert first.selection_for("min_cost") == loose.policy_version
    # 重开旧研究复算结果一致
    reopened = PolicyStudy.from_dict(first.to_dict())
    assert reopened.study_version == first.study_version


def test_invalid_constraints_rejected():
    loose, _ = make_policies()
    with pytest.raises(ValueError):
        run_policy_study([loose], make_batch("holdout"), cost_cap=-1.0)
    with pytest.raises(ValueError):
        run_policy_study(
            [loose], make_batch("holdout"), cost_cap=1.0, max_disparity=1.5,
        )
    with pytest.raises(ValueError):
        run_policy_study([], make_batch("holdout"), cost_cap=1.0)
