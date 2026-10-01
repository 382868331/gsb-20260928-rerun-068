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
from ml_eval.compare import compare_reports
from ml_eval.resample import bootstrap_scored_accuracy


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
        Prediction("s5", "A"),  # 一致重复预测：折叠，不改变结果
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
    assert s.n_conflicting_samples == 0
    assert s.n_evaluable == 5
    # 难例必须保留在统计里，不能被悄悄删掉：各计数之和自洽
    assert tuple(s.duplicate_sample_ids) == ("s5",)
    assert tuple(s.missing_sample_ids) == ("s6",)
    assert tuple(s.unknown_sample_ids) == ("s999",)
    assert tuple(s.unknown_label_rows) == (("s7", "Z"),)


def test_identical_duplicates_fold_order_independently():
    # 把一致重复放到最前、并整体乱序：指标与版本身份必须完全一致
    shuffled = list(reversed(make_predictions()))
    b1 = PredictionBatch(make_predictions(), make_samples(), "b1")
    b2 = PredictionBatch(shuffled, make_samples(), "b1")
    assert b1.version == b2.version
    assert b1.evaluable_rows == b2.evaluable_rows
    r1 = evaluate_batch(b1, "region", "accuracy", n_replicates=50, seed=7)
    r2 = evaluate_batch(b2, "region", "accuracy", n_replicates=50, seed=7)
    assert r2.prediction_batch_version == r1.prediction_batch_version
    assert r2.participating_version == r1.participating_version
    assert r2.overall.accuracy == r1.overall.accuracy
    assert [g.to_dict() for g in r2.groups] == [g.to_dict() for g in r1.groups]
    assert (
        r2.overall.confidence.replicate_version
        == r1.overall.confidence.replicate_version
    )


def test_conflicting_duplicates_are_explicitly_excluded():
    samples = make_samples()
    # s5 冲突：A 与 B 谁先出现都不能决定成绩
    preds = [
        Prediction("s1", "A"),
        Prediction("s2", "B"),
        Prediction("s3", "B"),
        Prediction("s4", "A"),
        Prediction("s5", "A"),
        Prediction("s5", "B"),
        Prediction("s7", "Z"),
    ]
    b1 = PredictionBatch(preds, samples, "b1")
    b2 = PredictionBatch(list(reversed(preds)), samples, "b1")
    assert b1.version == b2.version
    for batch in (b1, b2):
        s = batch.summary
        assert s.n_duplicate_predictions == 1
        assert s.n_conflicting_samples == 1
        assert s.conflicting_rows == (("s5", ("A", "B")),)
        assert s.total_weight_conflicting == 3.0
        assert s.duplicate_sample_ids == ("s5",)
        assert "s5" not in s.evaluable_sample_ids
        assert len(batch.evaluable_rows) == 4
    r1 = evaluate_batch(b1, "region", "accuracy", n_replicates=50, seed=7)
    r2 = evaluate_batch(b2, "region", "accuracy", n_replicates=50, seed=7)
    assert r1.participating_version == r2.participating_version
    assert r1.overall.accuracy == r2.overall.accuracy
    # 冲突样本 s5（权重 3）按错误计入全样本口径：正确权重 2 / 总权重 10
    assert r1.accounting.accuracy_all_samples == pytest.approx(0.2)
    assert r1.overall.accuracy == pytest.approx(0.2)
    assert r1.accounting.conflicting_weight == 3.0
    assert r1.accounting.accuracy_evaluable == pytest.approx(2.0 / 5.0)
    assert r1.overall.diagnostic.accuracy == pytest.approx(2.0 / 5.0)


def test_accounting_counts_all_excluded_weights():
    report = evaluate_batch(PredictionBatch(make_predictions(), make_samples()))
    a = report.accounting
    # 总权重 1+2+1+1+3+1+1 = 10；可评估 8；缺预测 s6=1；未知类别 s7=1
    assert a.total_weight == pytest.approx(10.0)
    assert a.evaluable_weight == pytest.approx(8.0)
    assert a.excluded_weight == pytest.approx(2.0)
    assert a.missing_weight == pytest.approx(1.0)
    assert a.unknown_label_weight == pytest.approx(1.0)
    assert a.conflicting_weight == 0.0
    assert a.correct_weight == pytest.approx(5.0)
    assert a.accuracy_all_samples == pytest.approx(0.5)
    assert a.accuracy_evaluable == pytest.approx(5.0 / 8.0)
    # 可评估权重 + 被排除权重 = 全部样本权重，账目闭合
    assert a.evaluable_weight + a.excluded_weight == pytest.approx(a.total_weight)


def test_replicate_version_tracks_labels_and_weights():
    rows = PredictionBatch(make_predictions(), make_samples()).evaluable_rows
    r1 = bootstrap_ci(rows, "accuracy", n_replicates=50, seed=7)
    # 同身份同预测但权重不同：重采样版本必须不同
    reweighted = [
        type(r)(r.sample_id, r.label, r.predicted, r.weight * 2) for r in rows
    ]
    r2 = bootstrap_ci(reweighted, "accuracy", n_replicates=50, seed=7)
    assert r2.replicate_version != r1.replicate_version
    # 同身份但翻转某个预测：版本也必须不同
    flipped = [
        type(r)(r.sample_id, r.label, "B" if r.label == "A" else "A", r.weight)
        for r in rows
    ]
    r3 = bootstrap_ci(flipped, "accuracy", n_replicates=50, seed=7)
    assert r3.replicate_version != r1.replicate_version


def test_overall_metrics_hand_computed():
    batch = PredictionBatch(make_predictions(), make_samples())
    report = evaluate_batch(batch, group_attr="region")
    # 主口径为全样本：正确权重 5 / 总权重 10
    assert report.overall.scope == "all_samples"
    assert report.overall.n_samples == 7
    assert report.overall.accuracy == pytest.approx(0.5)
    assert report.overall.coverage == pytest.approx(0.8)

    # 参与计算权重 1+2+1+1+3=8；正确权重 1+1+3=5
    m = report.overall.diagnostic
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
    # x: s1 对(1), s2 错(2), s5 对(3), s7 未知类别按错(1) -> 4/7
    assert groups["x"].n_samples == 4
    assert groups["x"].accuracy == pytest.approx(4.0 / 7.0)
    assert groups["x"].diagnostic.accuracy == pytest.approx(4.0 / 6.0)
    assert groups["x"].diagnostic.per_class["A"].f1 == pytest.approx(0.8)
    assert groups["x"].diagnostic.weighted_f1 == pytest.approx(0.8)
    # y: s3 对, s4 错, s6 缺预测按错 -> 1/3
    assert groups["y"].n_samples == 3
    assert groups["y"].accuracy == pytest.approx(1.0 / 3.0)
    assert groups["y"].diagnostic.accuracy == pytest.approx(0.5)
    assert groups["y"].diagnostic.per_class["B"].f1 == pytest.approx(2.0 / 3.0)
    # 总体不等于组指标的简单平均：5/10 != (4/7 + 1/3)/2
    assert not math.isclose(report.overall.accuracy, (4.0 / 7.0 + 1.0 / 3.0) / 2)
    # 分组正确权重之和必须回到总体正确权重（口径一致可核对）
    assert sum(g.correct_weight for g in report.groups) == pytest.approx(
        report.overall.correct_weight
    )


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
    assert report.overall.n_samples == 1
    assert report.overall.n_evaluable == 0
    assert report.data_quality.n_unknown_predicted_labels == 1
    assert report.overall.accuracy == 0.0
    assert report.overall.diagnostic is None


def make_mixed_batch():
    """重复 + 冲突 + 未知类别 + 缺预测同时出现的组合场景。"""
    samples = SampleSet(
        [
            Sample("a1", "A", 1.0, {"region": "x"}),
            Sample("a2", "B", 2.0, {"region": "x"}),
            Sample("a3", "A", 3.0, {"region": "y"}),  # 冲突
            Sample("a4", "B", 4.0, {"region": "y"}),  # 未知类别
            Sample("a5", "A", 5.0, {"region": "y"}),  # 缺预测
        ]
    )
    preds = [
        Prediction("a1", "A"),
        Prediction("a1", "A"),  # 一致重复，折叠
        Prediction("a2", "A"),  # 错
        Prediction("a3", "A"),
        Prediction("a3", "B"),  # 冲突
        Prediction("a4", "Z"),  # 未知类别
        Prediction("a9", "A"),  # 未知身份
    ]
    return samples, PredictionBatch(preds, samples)


def test_mixed_anomalies_reconcile_counts_weights_and_intervals():
    samples, batch = make_mixed_batch()
    s = batch.summary
    # 计数闭合：可评估 + 缺预测 + 冲突 + 未知类别 = 样本总数
    assert (
        s.n_evaluable
        + s.n_missing_predictions
        + s.n_conflicting_samples
        + s.n_unknown_predicted_labels
    ) == s.n_samples
    # 权重闭合：各去向之和 = 总权重
    assert (
        s.total_weight_evaluable
        + s.total_weight_missing
        + s.total_weight_conflicting
        + s.total_weight_unknown_label
    ) == pytest.approx(s.total_weight_samples)
    # scored_rows 每个样本恰好一行，状态计数与 summary 一致
    assert len(batch.scored_rows) == s.n_samples
    by_status = {}
    for row in batch.scored_rows:
        by_status[row.status] = by_status.get(row.status, 0) + 1
    assert by_status == {
        "evaluable": s.n_evaluable,
        "missing": s.n_missing_predictions,
        "conflicting": s.n_conflicting_samples,
        "unknown_label": s.n_unknown_predicted_labels,
    }

    report = evaluate_batch(batch, "region", "accuracy", n_replicates=100, seed=3)
    a = report.accounting
    # 总权重 15；可评估 3（a1 对 1，a2 错 2）；缺 5；冲突 3；未知类别 4
    assert a.total_weight == pytest.approx(15.0)
    assert a.evaluable_weight == pytest.approx(3.0)
    assert a.missing_weight == pytest.approx(5.0)
    assert a.conflicting_weight == pytest.approx(3.0)
    assert a.unknown_label_weight == pytest.approx(4.0)
    assert a.correct_weight == pytest.approx(1.0)
    # 主口径、对账、区间说明三者互相核对
    assert report.overall.accuracy == pytest.approx(1.0 / 15.0)
    assert a.accuracy_all_samples == pytest.approx(report.overall.accuracy)
    ci = report.overall.confidence
    assert ci.scope == "all_samples"
    assert ci.n_rows == s.n_samples
    assert ci.point == pytest.approx(report.overall.accuracy)
    assert ci.lower <= ci.point <= ci.upper
    # 诊断口径单独命名，不与主口径混淆
    assert report.overall.diagnostic.accuracy == pytest.approx(1.0 / 3.0)
    assert a.accuracy_evaluable == pytest.approx(1.0 / 3.0)
    # 分组主口径也覆盖全组样本，且各组正确权重之和回到总体
    groups = {g.group: g for g in report.groups}
    assert groups["y"].n_samples == 3
    assert groups["y"].accuracy == pytest.approx(0.0)
    assert groups["y"].confidence.n_rows == 3
    assert sum(g.correct_weight for g in report.groups) == pytest.approx(
        report.overall.correct_weight
    )


def test_scored_bootstrap_version_tracks_status():
    samples, batch = make_mixed_batch()
    r1 = bootstrap_scored_accuracy(batch.scored_rows, n_replicates=50, seed=7)
    assert r1.scope == "all_samples"
    assert r1.n_rows == len(samples)
    # 同一身份从可评估变为缺预测：版本必须变化
    changed = PredictionBatch(
        [p for p in [
            Prediction("a2", "A"),
            Prediction("a3", "A"),
            Prediction("a3", "B"),
            Prediction("a4", "Z"),
        ]],
        samples,
    )
    r2 = bootstrap_scored_accuracy(changed.scored_rows, n_replicates=50, seed=7)
    assert r2.replicate_version != r1.replicate_version
    assert r2.n_rows == r1.n_rows


def test_compare_models_with_different_coverage_and_missing_groups():
    samples = SampleSet(
        [
            Sample("s1", "A", 1.0, {"region": "x"}),
            Sample("s2", "A", 1.0, {"region": "x"}),
            Sample("s3", "B", 1.0, {"region": "y"}),
            Sample("s4", "B", 1.0, {"region": "z"}),
        ]
    )
    full = PredictionBatch(
        [Prediction("s1", "A"), Prediction("s2", "A"),
         Prediction("s3", "B"), Prediction("s4", "A")],
        samples,
        "full",
    )
    # partial 只覆盖 x 组且全对：evaluable 口径满分，全样本口径 1/2
    partial = PredictionBatch(
        [Prediction("s1", "A"), Prediction("s2", "A")],
        samples,
        "partial",
    )
    reports = {
        "full": evaluate_batch(full, "region"),
        "partial": evaluate_batch(partial, "region"),
    }
    comparison = compare_reports(reports, group_attr="region")
    models = {m.model: m for m in comparison.models}
    assert models["full"].coverage == pytest.approx(1.0)
    assert models["partial"].coverage == pytest.approx(0.5)
    # 主口径不被满分掩盖：partial 全样本准确率 2/4
    assert models["partial"].accuracy_all_samples == pytest.approx(0.5)
    assert models["partial"].accuracy_evaluable == pytest.approx(1.0)
    assert any("覆盖率不一致" in w for w in comparison.warnings)
    # 分组并集：partial 缺失的 y/z 组显式出现为 None
    table = {group: cells for group, cells in comparison.group_table}
    assert set(table) == {"x", "y", "z"}
    # partial 在 y/z 组完全未覆盖：组仍呈现，覆盖率 0、按错误计
    assert table["y"]["partial"].coverage == pytest.approx(0.0)
    assert table["y"]["partial"].accuracy == pytest.approx(0.0)
    assert table["z"]["partial"].coverage == pytest.approx(0.0)
    assert table["y"]["full"].accuracy == pytest.approx(1.0)
    assert table["z"]["full"].accuracy == pytest.approx(0.0)
    assert table["x"]["partial"].accuracy == pytest.approx(1.0)


def test_compare_groups_union_when_group_attr_differs():
    # 同一模型集合、不同分组键缺失时，并集逻辑直接呈现缺失单元格
    samples = SampleSet(
        [
            Sample("s1", "A", 1.0, {"region": "x"}),
            Sample("s2", "B", 1.0),  # 无 region 属性
        ]
    )
    batch = PredictionBatch(
        [Prediction("s1", "A"), Prediction("s2", "B")], samples, "m",
    )
    report = evaluate_batch(batch, "region")
    groups = {g.group for g in report.groups}
    assert groups == {"x", "__unassigned__"}
