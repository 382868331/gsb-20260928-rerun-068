import math

import pytest

from ml_eval import (
    PredictionBatch,
    Sample,
    SampleSet,
    Prediction,
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
            Sample("s6", "B", 1.0, {"region": "y"}),  # 缺预测
            Sample("s7", "A", 1.0, {"region": "x"}),  # 预测成未知类别
        ]
    )


def make_predictions():
    return [
        Prediction("s1", "A"),
        Prediction("s2", "B"),  # 错（权重 2）
        Prediction("s3", "B"),
        Prediction("s4", "A"),  # 错
        Prediction("s5", "A"),
        Prediction("s5", "B"),  # 重复预测
        Prediction("s999", "A"),  # 未知样本身份
        Prediction("s7", "Z"),  # 未知类别
    ]


def test_data_quality_counts_are_kept_separate():
    batch = PredictionBatch(make_predictions(), make_samples())
    s = batch.summary
    assert s.n_samples == 7
    assert s.n_predictions_received == 8
    assert s.n_duplicate_predictions == 1
    assert s.n_unknown_sample_ids == 1
    assert s.n_missing_predictions == 1
    assert s.n_unknown_predicted_labels == 1
    assert s.n_evaluable == 5
    # 难例必须保留在统计里，不能被悄悄删掉：各计数之和自洽
    assert tuple(s.duplicate_sample_ids) == ("s5",)
    assert tuple(s.missing_sample_ids) == ("s6",)
    assert tuple(s.unknown_sample_ids) == ("s999",)
    assert tuple(s.unknown_label_rows) == (("s7", "Z"),)


def test_overall_metrics_hand_computed():
    batch = PredictionBatch(make_predictions(), make_samples())
    report = evaluate_batch(batch, group_attr="region")
    m = report.overall.metrics

    # 参与计算权重 1+2+1+1+3=8；正确权重 1+1+3=5
    assert m.total_weight == pytest.approx(8.0)
    assert m.accuracy == pytest.approx(5.0 / 8.0)

    a = m.per_class["A"]
    assert a.precision == pytest.approx(0.8)       # 4 / (4+1)
    assert a.recall == pytest.approx(4.0 / 6.0)    # 4 / (4+2)
    assert a.f1 == pytest.approx(8.0 / 11.0)
    b = m.per_class["B"]
    assert b.precision == pytest.approx(1.0 / 3.0)
    assert b.recall == pytest.approx(0.5)
    assert b.f1 == pytest.approx(0.4)

    assert m.macro_f1 == pytest.approx((8.0 / 11.0 + 0.4) / 2.0)
    # 加权平均按真实标签支持度，而不是组平均：(0.5*6 + 0.4*2)/8
    assert m.weighted_f1 == pytest.approx(((8.0 / 11.0) * 6 + 0.4 * 2) / 8)


def test_group_metrics_are_computed_independently():
    report = evaluate_batch(PredictionBatch(make_predictions(), make_samples()), "region")
    groups = {g.group: g for g in report.groups}
    # x: s1 对(1), s2 错(2), s5 对(3) -> 4/6
    assert groups["x"].n_samples == 3
    assert groups["x"].metrics.accuracy == pytest.approx(4.0 / 6.0)
    assert groups["x"].metrics.per_class["A"].f1 == pytest.approx(0.8)
    assert groups["x"].metrics.weighted_f1 == pytest.approx(0.8)
    # y: s3 对, s4 错 -> 1/2
    assert groups["y"].n_samples == 2
    assert groups["y"].metrics.accuracy == pytest.approx(0.5)
    assert groups["y"].metrics.per_class["B"].f1 == pytest.approx(2.0 / 3.0)
    # 总体不等于组指标的简单平均：5/8 != (4/6 + 1/2)/2
    assert not math.isclose(report.overall.metrics.accuracy, (4.0 / 6.0 + 0.5) / 2)


def test_bootstrap_is_seed_deterministic():
    rows = PredictionBatch(make_predictions(), make_samples()).evaluable_rows
    r1 = bootstrap_ci(rows, "accuracy", n_replicates=200, seed=42)
    r2 = bootstrap_ci(list(rows), "accuracy", n_replicates=200, seed=42)
    r3 = bootstrap_ci(rows, "accuracy", n_replicates=200, seed=43)
    assert (r1.lower, r1.upper) == (r2.lower, r2.upper)
    assert r1.replicate_version == r2.replicate_version
    assert r1.replicate_version != r3.replicate_version
    assert r1.lower <= r1.upper


def test_versions_trace_back_to_data():
    samples = make_samples()
    batch = PredictionBatch(make_predictions(), samples, "b1")
    report1 = evaluate_batch(batch)
    # 用同样内容重建，版本一致（与记录读入顺序无关）
    rebuilt_batch = PredictionBatch(list(reversed(make_predictions())), make_samples(), "b1")
    report2 = evaluate_batch(rebuilt_batch)
    assert report1.sample_set_version == report2.sample_set_version
    assert report1.prediction_batch_version == report2.prediction_batch_version
    assert report1.participating_version == report2.participating_version

    changed = SampleSet(
        [Sample(s.sample_id, s.label, s.weight + (1.0 if s.sample_id == "s1" else 0.0), s.groups)
         for s in samples.samples]
    )
    report3 = evaluate_batch(PredictionBatch(make_predictions(), changed, "b1"))
    assert report3.sample_set_version != report1.sample_set_version
    assert report3.participating_version != report1.participating_version


def test_no_evaluable_rows_still_reports_quality():
    samples = SampleSet([Sample("s1", "A")])
    batch = PredictionBatch([Prediction("s1", "Z")], samples)
    report = evaluate_batch(batch)
    assert report.overall.n_samples == 0
    assert report.data_quality.n_unknown_predicted_labels == 1
    assert report.overall.metrics.accuracy == 0.0
