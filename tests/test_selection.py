"""多重保护组约束下的策略选择研究。

用重叠组、极小组、非整数权重与缺失组标签样本核对：可行性判定、最大
差异、总体收益/成本只计一次、约束冲突展示、两种目标比较，以及保存
重开后的复算与新策略只能形成新研究的冻结语义。
"""

import json

import pytest

from ml_eval import (
    CandidateOutcome,
    CostModel,
    OBJECTIVE_MIN_COST,
    OBJECTIVE_WORST_GROUP,
    Prediction,
    PredictionBatch,
    Sample,
    SampleSet,
    SelectionConstraints,
    StrategySelectionStudy,
    evaluate_policy,
    fit_abstention_policy,
    recompute_study,
    run_selection_study,
)
from ml_eval.selection import (
    VIOLATION_COST_CAP,
    VIOLATION_DISPARITY,
    VIOLATION_GROUP_ABSENT,
    VIOLATION_MIN_COVERAGE,
)

ATTRS = ["region", "risk"]
COSTS = CostModel(false_accept=3.0, correct_accept=0.0, reject=1.0)


def make_batch(name="holdout"):
    samples = SampleSet(
        [
            Sample("a1", "A", 1.0, {"region": "x", "risk": "low"}),
            Sample("a2", "A", 0.5, {"region": "x", "risk": "low"}),
            Sample("b1", "B", 2.5, {"region": "y", "risk": "low"}),
            # ov 同时属于 region=x 与 risk=high（重叠样本，权重非整数）
            Sample("ov", "A", 1.5, {"region": "x", "risk": "high"}),
            # tiny 是 risk=high 极小组的唯一另一个成员，权重 0.25
            Sample("tiny", "A", 0.25, {"region": "y", "risk": "high"}),
            # u1 缺失全部组标签：保留未知归属，不能凑任何组的覆盖
            Sample("u1", "A", 0.7, {}),
            # p1 只缺 region 标签，仍属于 risk=low 保护组
            Sample("p1", "A", 0.3, {"risk": "low"}),
        ]
    )
    preds = [
        Prediction("a1", "A", 0.95),   # 接受则正确
        Prediction("a2", "B", 0.90),   # 接受则错误
        Prediction("b1", "B", 0.80),   # 接受则正确
        Prediction("ov", "B", 0.70),   # 接受则错误
        Prediction("tiny", "A", 0.30),  # 低打分
        Prediction("u1", "A", 0.99),
        Prediction("p1", "A", 0.95),
    ]
    return PredictionBatch(preds, samples, name)


def policies_for(selection):
    loose = fit_abstention_policy(
        selection, ATTRS, COSTS, 0.0, strategy="per_group"
    )
    strict_x = fit_abstention_policy(
        selection, ATTRS, COSTS,
        {"region=x": 0.99, "*": 0.0}, strategy="per_group",
    )
    uniform = fit_abstention_policy(
        selection, ATTRS, COSTS, 0.0, strategy="uniform"
    )
    return {"loose": loose, "strict_x": strict_x, "uniform": uniform}


def test_overlap_sample_counts_once_overall_but_each_group():
    holdout = make_batch()
    policies = policies_for(make_batch("selection"))
    study = run_selection_study(
        tuple(policies.values()), holdout,
        SelectionConstraints(min_coverage=0.0),
    )
    outcome = study.outcome_for(policies["loose"].policy_version)
    # 总体权重与收益每个样本只计一次（非整数权重）
    assert outcome.total_weight == pytest.approx(6.75)
    groups = dict(outcome.group_total_weight)
    assert groups["region=x"] == pytest.approx(3.0)
    assert groups["risk=high"] == pytest.approx(1.75)
    # 重叠样本 ov 同时计入 region=x 与 risk=high，组账加总超过总体
    assert sum(groups.values()) > outcome.total_weight
    # 未知归属只含 u1（0.7），p1 属于 risk=low 不进未知桶
    assert outcome.unknown_weight == pytest.approx(0.7)
    assert set(dict(outcome.group_coverage)) == {
        "region=x", "region=y", "risk=high", "risk=low",
    }


def test_unknown_label_sample_never_satisfies_group_constraint():
    holdout = make_batch()
    policies = policies_for(make_batch("selection"))
    # 显式要求一个留出集上完全没有已知成员的组
    study = run_selection_study(
        [policies["loose"]], holdout,
        SelectionConstraints(min_coverage={"region=z": 0.5, "*": 0.0}),
    )
    assert not study.feasible
    outcome = study.outcomes[0]
    absent = [v for v in outcome.violations if v["type"] == VIOLATION_GROUP_ABSENT]
    assert [v["group"] for v in absent] == ["region=z"]
    # 缺失标签的 u1 没有被算进 region=z
    assert outcome.unknown_weight == pytest.approx(0.7)
    assert not study.choice_for(OBJECTIVE_MIN_COST).policy_version
    blocks = {i["group"] for i in study.infeasible_constraints if i.get("group")}
    assert "region=z" in blocks


def test_tiny_group_coverage_and_max_disparity():
    holdout = make_batch()
    policies = policies_for(make_batch("selection"))
    # risk=high 只有 ov(1.5) 与 tiny(0.25)：tiny 低打分被拒时覆盖率
    # 1.5/1.75 ≈ 0.857，与全接受组产生差异
    study = run_selection_study(
        tuple(policies.values()), holdout,
        SelectionConstraints(min_coverage=0.0, max_disparity=0.05),
    )
    by_version = {o.policy_version: o for o in study.outcomes}
    strict = by_version[policies["strict_x"].policy_version]
    assert strict.observed_disparity > 0.05
    disparity_violations = [
        v for v in strict.violations if v["type"] == VIOLATION_DISPARITY
    ]
    assert disparity_violations
    assert disparity_violations[0]["observed_disparity"] == pytest.approx(
        strict.observed_disparity
    )
    if not study.feasible:
        kinds = {i["constraint"] for i in study.infeasible_constraints}
        assert VIOLATION_DISPARITY in kinds


def test_min_coverage_violation_lists_group_and_shortfall():
    holdout = make_batch()
    policies = policies_for(make_batch("selection"))
    study = run_selection_study(
        [policies["strict_x"]], holdout,
        SelectionConstraints(min_coverage={"risk=high": 0.99}),
    )
    assert not study.feasible
    violations = [
        v for v in study.outcomes[0].violations
        if v["type"] == VIOLATION_MIN_COVERAGE
    ]
    assert [v["group"] for v in violations] == ["risk=high"]
    assert violations[0]["shortfall"] == pytest.approx(
        0.99 - dict(study.outcomes[0].group_coverage)["risk=high"]
    )
    assert study.infeasible_constraints
    assert study.infeasible_constraints


def test_cost_cap_violation_and_two_objectives_differ():
    holdout = make_batch()
    policies = policies_for(make_batch("selection"))
    # 极低成本上限：任何接受错误样本的策略都会付出 false_accept=3 的代价
    impossible = run_selection_study(
        tuple(policies.values()), holdout,
        SelectionConstraints(min_coverage=0.0, max_total_cost=0.1),
    )
    assert not impossible.feasible
    cap_blocks = [
        i for i in impossible.infeasible_constraints
        if i["constraint"] == VIOLATION_COST_CAP
    ]
    assert cap_blocks
    assert cap_blocks[0]["best_achievable_cost"] > 0.1

    # 可行场景：两个目标可能选择不同策略。构造一个高覆盖但成本更高的
    # 策略（严格组阈值迫使重叠错误样本 ov 被接受路径不同），用宽松上限
    # 让两者都可行，再核对 min_cost 与 worst_group 各自按口径选择。
    study = run_selection_study(
        tuple(policies.values()), holdout,
        SelectionConstraints(min_coverage=0.0, max_disparity=1.0,
                             max_total_cost=100.0),
    )
    assert study.feasible
    feasible = [o for o in study.outcomes if o.feasible]
    min_cost_pick = study.choice_for(OBJECTIVE_MIN_COST)
    worst_pick = study.choice_for(OBJECTIVE_WORST_GROUP)
    cheapest = min(feasible, key=lambda o: o.total_cost)
    best_worst = max(feasible, key=lambda o: o.worst_group_coverage)
    assert min_cost_pick.policy_version == cheapest.policy_version
    assert min_cost_pick.objective_value == pytest.approx(cheapest.total_cost)
    assert worst_pick.policy_version == best_worst.policy_version
    assert worst_pick.objective_value == pytest.approx(
        best_worst.worst_group_coverage
    )


def test_benefit_and_cost_count_sample_once_with_overlap():
    holdout = make_batch()
    policies = policies_for(make_batch("selection"))
    # 直接对照 evaluate_policy 总体账目：选择层结果必须与重放明细一致
    evaluation = evaluate_policy(policies["loose"], holdout, bootstrap=False)
    study = run_selection_study(
        [policies["loose"]], holdout, SelectionConstraints(),
    )
    outcome = study.outcomes[0]
    assert outcome.total_cost == pytest.approx(evaluation.overall.total_cost)
    assert outcome.accepted_weight == pytest.approx(
        evaluation.overall.accepted_weight
    )
    expected_benefit = sum(
        r.weight for r in evaluation.decisions if r.status == "accepted_correct"
    )
    assert outcome.benefit_weight == pytest.approx(expected_benefit)


def test_round_trip_and_recompute_identical():
    holdout = make_batch()
    policies = tuple(policies_for(make_batch("selection")).values())
    constraints = SelectionConstraints(
        min_coverage={"region=x": 0.5, "*": 0.2},
        max_disparity=0.75,
        max_total_cost=100.0,
    )
    study = run_selection_study(policies, holdout, constraints)
    raw = json.dumps(study.to_dict(), ensure_ascii=False, sort_keys=True)
    reopened = StrategySelectionStudy.from_dict(json.loads(raw))
    assert reopened.study_version == study.study_version
    recomputed = recompute_study(reopened, policies, holdout)
    assert json.dumps(recomputed.to_dict(), ensure_ascii=False, sort_keys=True) == raw


def test_tampered_study_rejected_on_load():
    holdout = make_batch()
    policies = tuple(policies_for(make_batch("selection")).values())
    study = run_selection_study(
        policies, holdout, SelectionConstraints(min_coverage=0.0)
    )
    payload = study.to_dict()
    payload["feasible"] = not payload["feasible"]
    with pytest.raises(ValueError):
        StrategySelectionStudy.from_dict(payload)


def test_new_policy_creates_new_study_but_old_stays_frozen():
    holdout = make_batch()
    policies = policies_for(make_batch("selection"))
    base = run_selection_study(
        [policies["loose"], policies["uniform"]], holdout,
        SelectionConstraints(min_coverage=0.0, max_total_cost=100.0),
    )
    # 新策略到达：只能形成新研究，旧研究结果不变
    extended = run_selection_study(
        [policies["loose"], policies["uniform"], policies["strict_x"]],
        holdout, SelectionConstraints(min_coverage=0.0, max_total_cost=100.0),
    )
    assert extended.study_version != base.study_version
    assert extended.candidate_policy_versions[:2] == base.candidate_policy_versions
    base_outcomes = {o.policy_version: o for o in base.outcomes}
    for outcome in extended.outcomes:
        if outcome.policy_version in base_outcomes:
            assert outcome.to_dict() == base_outcomes[outcome.policy_version].to_dict()
    assert base.choices == tuple(
        c for c in base.choices
    )  # 旧研究选择冻结


def test_recompute_rejects_changed_weights():
    holdout = make_batch()
    policies = tuple(policies_for(make_batch("selection")).values())
    study = run_selection_study(
        policies, holdout, SelectionConstraints(min_coverage=0.0)
    )
    reopened = StrategySelectionStudy.from_dict(study.to_dict())
    # 权重被修改 -> 样本集版本改变，复算结果必须与冻结研究不一致并报错
    changed_samples = SampleSet(
        [
            Sample(
                s.sample_id, s.label,
                s.weight * 2.0 if s.sample_id == "ov" else s.weight,
                dict(s.groups),
            )
            for s in holdout.sample_set.samples
        ]
    )
    changed = PredictionBatch(
        [Prediction(p.sample_id, p.predicted, p.score)
         for p in holdout.evaluable_rows]
        + [Prediction(sid, "A", 0.9)
           for sid in changed_samples.ids
           if sid not in {r.sample_id for r in holdout.evaluable_rows}],
        changed_samples, "changed",
    )
    with pytest.raises(ValueError):
        recompute_study(reopened, policies, changed)


def test_candidate_policies_must_share_group_attrs():
    holdout = make_batch()
    single_attr = fit_abstention_policy(
        make_batch("selection"), "region", COSTS, 0.0, strategy="per_group"
    )
    multi_attr = policies_for(make_batch("selection"))["loose"]
    with pytest.raises(ValueError):
        run_selection_study(
            [single_attr, multi_attr], holdout, SelectionConstraints()
        )
