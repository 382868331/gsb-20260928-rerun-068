import pytest

from ml_eval import (
    Costs,
    GroupConstraint,
    PolicySpec,
    Prediction,
    PredictionBatch,
    Sample,
    SampleSet,
    apply_policy,
    select_policy,
)
from ml_eval.policy import (
    GROUP_FEASIBLE,
    GROUP_INFEASIBLE,
    GROUP_UNDETERMINED,
    POLICY_GROUPED,
    POLICY_UNIFORM,
)


def make_set():
    return SampleSet(
        [
            Sample("a1", "A", 1.0, {"g": "x"}),
            Sample("a2", "A", 2.0, {"g": "x"}),
            Sample("b1", "B", 3.0, {"g": "y"}),
            Sample("b2", "B", 4.0, {"g": "y"}),
        ]
    )


def make_selection():
    return PredictionBatch(
        [
            Prediction("a1", "A", 0.9),
            Prediction("a2", "B", 0.8),
            Prediction("b1", "B", 0.7),
            Prediction("b2", "A", 0.6),
        ],
        make_set(),
        "selection",
    )


def spec(mode=POLICY_GROUPED, **constraints):
    return PolicySpec(
        group_attr="g",
        mode=mode,
        costs=Costs(false_accept=2.0, correct_accept=0.0, abstain=0.5),
        group_constraints=constraints,
    )


def test_grouped_selection_minimizes_cost_under_group_coverage():
    selected = select_policy(
        make_selection(),
        spec(x=GroupConstraint(0.5), y=GroupConstraint(0.5)),
        n_replicates=50,
        seed=7,
    )
    groups = {group.group: group for group in selected.groups}
    assert selected.status == GROUP_FEASIBLE
    assert groups["x"].threshold == pytest.approx(0.8)
    assert groups["y"].threshold == pytest.approx(0.6)
    assert groups["x"].coverage == pytest.approx(1.0)
    assert groups["y"].accepted_weight == pytest.approx(7.0)
    assert selected.overall.total_cost == pytest.approx(4.0 + 8.0)
    assert selected.overall.total_weight == pytest.approx(10.0)
    assert selected.overall.abstained_weight == 0.0
    assert selected.overall.correct_accepted_weight == pytest.approx(4.0)
    assert selected.overall.false_accepted_weight == pytest.approx(6.0)
    assert selected.confidence is not None
    assert (
        selected.confidence.coverage_lower
        <= selected.overall.coverage
        <= selected.confidence.coverage_upper
    )


def selection_with_missing_score():
    preds = make_selection()
    return PredictionBatch(
        [
            pred if pred.sample_id != "b2" else Prediction("b2", "A", None)
            for pred in [
                Prediction("a1", "A", 0.9),
                Prediction("a2", "B", 0.8),
                Prediction("b1", "B", 0.7),
                Prediction("b2", "A", 0.6),
            ]
        ],
        make_set(),
        "selection",
    )


def test_impossible_constraint_is_retained_not_dropped():
    selected = select_policy(
        selection_with_missing_score(),
        spec(x=GroupConstraint(1.0), y=GroupConstraint(0.5)),
        n_replicates=20,
    )
    assert selected.status == GROUP_INFEASIBLE
    groups = {group.group: group for group in selected.groups}
    assert groups["x"].status == GROUP_FEASIBLE
    assert groups["y"].status == GROUP_INFEASIBLE
    assert "no_threshold_reaches_min_coverage" in groups["y"].reason
    assert groups["y"].max_feasible_coverage == pytest.approx(3.0/7.0)
    assert groups["y"].total_weight == pytest.approx(7.0)
    assert groups["y"].accepted_weight == 0.0
    assert groups["y"].abstained_weight == pytest.approx(7.0)
    assert selected.overall.total_weight == pytest.approx(10.0)


def test_undetermined_group_is_retained_for_insufficient_selection_data():
    samples = SampleSet(
        [
            Sample("x1", "A", 1.0, {"g": "x"}),
            Sample("z1", "A", 1.0, {"g": "z"}),
            Sample("z2", "B", 1.0, {"g": "z"}),
        ]
    )
    batch = PredictionBatch(
        [Prediction("x1", "A", 0.9), Prediction("z1", "A", 0.9)],
        samples,
    )
    policy = PolicySpec(
        "g",
        POLICY_GROUPED,
        Costs(1, 0, 0.5),
        {"x": GroupConstraint(min_samples=2), "z": GroupConstraint(min_samples=2)},
    )
    selected = select_policy(batch, policy, n_replicates=20)
    assert selected.status == GROUP_UNDETERMINED
    group = {item.group: item for item in selected.groups}["x"]
    assert group.status == GROUP_UNDETERMINED
    assert "below_min_samples" in group.reason
    assert group.total_weight == 1.0
    assert group.abstained_weight == 1.0


def test_uniform_policy_uses_one_threshold_and_records_conflicting_groups():
    selected = select_policy(
        selection_with_missing_score(),
        spec(POLICY_UNIFORM, x=GroupConstraint(1.0), y=GroupConstraint(0.5)),
        n_replicates=20,
    )
    assert selected.status == GROUP_INFEASIBLE
    assert selected.to_dict()["uniform_threshold"] is None
    groups = {group.group: group for group in selected.groups}
    assert groups["x"].status == GROUP_INFEASIBLE
    assert "uniform_threshold_violates" in groups["x"].reason
    assert groups["y"].status == GROUP_INFEASIBLE


def test_audit_freezes_threshold_and_keeps_missing_conflicting_and_weights():
    selected = select_policy(
        make_selection(),
        spec(x=GroupConstraint(0.5), y=GroupConstraint(0.5)),
        n_replicates=30,
    )
    audit = PredictionBatch(
        [
            Prediction("a1", "B", 0.95),
            Prediction("a2", "A", 0.80),
            Prediction("a2", "B", 0.80),
            Prediction("b2", "Z", 0.61),
        ],
        make_set(),
        "audit",
    )
    reported = apply_policy(audit, selected, n_replicates=40, seed=99)
    assert reported.policy_version == selected.policy_version
    groups = {group.group: group for group in reported.groups}
    assert groups["x"].threshold == pytest.approx(0.8)
    assert groups["y"].threshold == pytest.approx(0.6)
    assert groups["x"].accepted_weight == pytest.approx(1.0)
    assert groups["x"].false_accepted_weight == pytest.approx(1.0)
    assert groups["y"].accepted_weight == 0.0
    assert reported.overall.total_weight == pytest.approx(10.0)
    assert reported.overall.abstained_weight == pytest.approx(9.0)
    assert reported.overall.total_cost == pytest.approx(2.0 + 0.5 * 9.0)
    assert [group.threshold for group in reported.groups] == [
        group.threshold for group in selected.groups
    ]


def test_empty_audit_group_is_undetermined_not_silently_removed():
    samples = SampleSet(
        [
            Sample("x1", "A", 1.0, {"g": "x"}),
            Sample("z1", "A", 1.0, {"g": "z"}),
        ]
    )
    selection = PredictionBatch([Prediction("x1", "A", 0.9)], samples)
    audit = PredictionBatch([Prediction("z1", "A", 0.9)], samples)
    policy = PolicySpec("g", POLICY_GROUPED, Costs(1, 0, 0.5), {})
    selected = select_policy(selection, policy, n_replicates=10)
    reported = apply_policy(audit, selected, n_replicates=10)
    by_group = {group.group: group for group in reported.groups}
    assert set(by_group) == {"x", "z"}
    assert by_group["z"].status == GROUP_UNDETERMINED
    assert "no_scored_predictions" in by_group["z"].reason
    assert by_group["z"].total_weight == 1.0
    assert by_group["x"].total_weight == 1.0


def test_changing_costs_creates_a_new_traceable_policy_version():
    batch = make_selection()
    first = select_policy(batch, spec(), n_replicates=10)
    changed = PolicySpec("g", POLICY_GROUPED, Costs(3, 0, 0.5), {})
    second = select_policy(batch, changed, n_replicates=10)
    assert first.policy_version != second.policy_version
    audit = apply_policy(batch, first, n_replicates=10)
    assert audit.policy_version == first.policy_version
    assert audit.policy_version != second.policy_version
