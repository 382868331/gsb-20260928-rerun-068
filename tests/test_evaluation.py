import math

import pytest

from ml_eval import (
    Prediction,
    PredictionBatch,
    Sample,
    SampleSet,
    bootstrap_ci,
    evaluate_batch,
)


def make_samples():
    return SampleSet(
        [
            Sample("s1", "A", 1.0, {"region": "x"}),
            Sample("s2", "A", 2.0, {"region": "x"}),
            Sample("s3", "B", 1.0, {"region": "y"}),
            Sample("s4", "B", 1.0, {"region": "y"}),
            Sample("s5", "A", 3.0, {"region": "x"}),
            Sample("s6", "B", 1.0, {"region": "y"}),
            Sample("s7", "A", 1.0, {"region": "x"}),
        ]
    )


def make_predictions():
    return [
        Prediction("s1", "A"),
        Prediction("s2", "B"),
        Prediction("s3", "B"),
        Prediction("s4", "A"),
        Prediction("s5", "A"),
        Prediction("s5", "A"),
        Prediction("s999", "A"),
        Prediction("s7", "Z"),
    ]


def shuffled_predictions():
    return list(reversed(make_predictions()))


def conflicting_predictions():
    return [
        Prediction("s1", "A"),
        Prediction("s2", "B"),
        Prediction("s3", "B"),
        Prediction("s4", "A"),
        Prediction("s5", "A"),
        Prediction("s5", "B"),
        Prediction("s999", "A"),
        Prediction("s7", "Z"),
    ]


def test_data_quality_counts_are_explicit_and_reconcilable():
    batch = PredictionBatch(make_predictions(), make_samples())
    s = batch.summary

    assert s.n_samples == 7
    assert s.n_predictions_received == 8
    assert s.n_duplicate_predictions == 1
    assert s.n_unknown_sample_ids == 1
    assert s.n_missing_predictions == 1
    assert s.n_unknown_predicted_labels == 1
    assert s.n_participating_samples == 7
    assert s.n_valid_predictions == 5
    assert s.total_weight_participating == pytest.approx(10.0)
    assert s.total_weight_valid == pytest.approx(8.0)
    assert s.total_weight_missing == pytest.approx(1.0)
    assert s.total_weight_unknown_predicted == pytest.approx(1.0)
    assert tuple(s.duplicate_sample_ids) == ("s5",)
    assert tuple(s.missing_sample_ids) == ("s6",)
    assert tuple(s.unknown_sample_ids) == ("s999",)
    assert tuple(s.unknown_label_rows) == (("s7", "Z"),)
    assert tuple(s.participating_sample_ids) == tuple(f"s{i}" for i in range(1, 8))


def test_conflicting_duplicates_fail_regardless_of_order():
    with pytest.raises(ValueError, match="s5"):
        PredictionBatch(conflicting_predictions(), make_samples())
    with pytest.raises(ValueError, match="s5"):
        PredictionBatch(list(reversed(conflicting_predictions())), make_samples())


def test_identical_duplicates_are_order_insensitive():
    r1 = evaluate_batch(PredictionBatch(make_predictions(), make_samples(), "b1"))
    r2 = evaluate_batch(PredictionBatch(shuffled_predictions(), make_samples(), "b1"))

    assert r1.overall.metrics.accuracy == r2.overall.metrics.accuracy
    assert r1.prediction_batch_version == r2.prediction_batch_version
    assert r1.participating_version == r2.participating_version
    assert r1.sample_ids == r2.sample_ids


def test_overall_metrics_include_missing_and_unknown_as_errors():
    report = evaluate_batch(
        PredictionBatch(make_predictions(), make_samples()), group_attr="region"
    )
    m = report.overall.metrics

    assert m.total_weight == pytest.approx(10.0)
    assert m.accuracy == pytest.approx(5.0 / 10.0)

    a = m.per_class["A"]
    assert a.support == pytest.approx(7.0)
    assert a.predicted_support == pytest.approx(5.0)
    assert a.precision == pytest.approx(0.8)
    assert a.recall == pytest.approx(4.0 / 7.0)
    assert a.f1 == pytest.approx(2.0 / 3.0)

    b = m.per_class["B"]
    assert b.support == pytest.approx(3.0)
    assert b.predicted_support == pytest.approx(3.0)
    assert b.precision == pytest.approx(1.0 / 3.0)
    assert b.recall == pytest.approx(1.0 / 3.0)
    assert b.f1 == pytest.approx(1.0 / 3.0)

    assert m.macro_f1 == pytest.approx((2.0 / 3.0 + 1.0 / 3.0) / 2.0)
    assert m.weighted_f1 == pytest.approx(((2.0 / 3.0) * 7 + (1.0 / 3.0) * 3) / 10)


def test_groups_use_same_participating_sample_set_as_overall():
    report = evaluate_batch(
        PredictionBatch(make_predictions(), make_samples()), "region"
    )
    groups = {g.group: g for g in report.groups}

    assert groups["x"].n_samples == 4
    assert groups["x"].metrics.total_weight == pytest.approx(7.0)
    assert groups["x"].metrics.accuracy == pytest.approx(4.0 / 7.0)
    assert groups["y"].n_samples == 3
    assert groups["y"].metrics.total_weight == pytest.approx(3.0)
    assert groups["y"].metrics.accuracy == pytest.approx(1.0 / 3.0)
    assert sum(g.n_samples for g in report.groups) == report.overall.n_samples
    assert not math.isclose(
        report.overall.metrics.accuracy, (4.0 / 7.0 + 1.0 / 3.0) / 2.0
    )


def test_bootstrap_uses_all_participating_samples_and_is_order_stable():
    rows1 = PredictionBatch(make_predictions(), make_samples()).evaluable_rows
    rows2 = PredictionBatch(shuffled_predictions(), make_samples()).evaluable_rows
    r1 = bootstrap_ci(rows1, "accuracy", n_replicates=200, seed=42)
    r2 = bootstrap_ci(list(reversed(rows2)), "accuracy", n_replicates=200, seed=42)
    r3 = bootstrap_ci(rows1, "accuracy", n_replicates=200, seed=43)

    assert len(rows1) == 7
    assert (r1.point, r1.lower, r1.upper) == (r2.point, r2.lower, r2.upper)
    assert r1.replicate_version == r2.replicate_version
    assert r1.replicate_version != r3.replicate_version


def test_versions_change_for_weights_but_not_input_permutation():
    samples = make_samples()
    report1 = evaluate_batch(PredictionBatch(make_predictions(), samples, "b1"))
    report2 = evaluate_batch(
        PredictionBatch(shuffled_predictions(), make_samples(), "b1")
    )

    changed = SampleSet(
        [
            Sample(
                s.sample_id,
                s.label,
                s.weight + (1.0 if s.sample_id == "s1" else 0.0),
                s.groups,
            )
            for s in samples.samples
        ]
    )
    report3 = evaluate_batch(PredictionBatch(make_predictions(), changed, "b1"))

    assert report1.prediction_batch_version == report2.prediction_batch_version
    assert report1.participating_version == report2.participating_version
    assert report3.sample_set_version != report1.sample_set_version
    assert report3.participating_version != report1.participating_version


def test_unknown_prediction_still_participates_as_error():
    samples = SampleSet([Sample("s1", "A")])
    report = evaluate_batch(PredictionBatch([Prediction("s1", "Z")], samples))

    assert report.overall.n_samples == 1
    assert report.data_quality.n_unknown_predicted_labels == 1
    assert report.overall.metrics.total_weight == pytest.approx(1.0)
    assert report.overall.metrics.accuracy == 0.0
