import json

import pytest

from ml_eval import (
    AbstentionPolicy,
    CostModel,
    Prediction,
    PredictionBatch,
    ProtectionGroupSpec,
    Sample,
    SampleSet,
    evaluate_policy,
    fit_abstention_policy,
)


COSTS = CostModel(false_accept=2.0, correct_accept=0.0, reject=1.0)


def make_dataset():
    # s2 同时属于 small_region 与 high_risk，权重 5 且缺预测；
    # s3 也同时属于两组（权重 1）；s4 权重 1.5（非整数），只属 high_risk。
    samples = SampleSet([
        Sample('s1', 'A', 1.0, {'region': 'small', 'risk': 'low'}),
        Sample('s2', 'A', 5.0, {'region': 'small', 'risk': 'high'}),
        Sample('s3', 'A', 1.0, {'region': 'small', 'risk': 'high'}),
        Sample('s4', 'B', 1.5, {'region': 'large', 'risk': 'high'}),
    ])
    preds = [
        Prediction('s1', 'A', 0.9),   # 接受则正确
        Prediction('s3', 'A', 0.8),   # 真值 B：接受则错误
        Prediction('s4', 'B', 0.95),  # 接受则正确
    ]
    return PredictionBatch(preds, samples, 'selection')


SPECS = [
    ProtectionGroupSpec('small_region', {'region': 'small'}),
    ProtectionGroupSpec('high_risk', {'risk': 'high'}),
]


def test_overlapping_group_accounting_counts_sample_once_overall():
    batch = make_dataset()
    policy = fit_abstention_policy(
        batch, costs=COSTS,
        min_coverage={'small_region': 0.2, 'high_risk': 0.3},
        strategy='per_group', protection_groups=SPECS,
    )
    evaluation = evaluate_policy(policy, batch, bootstrap=False)
    groups = {g.group: g for g in evaluation.groups}

    # 总体分母 8.5，每个样本只计一次
    assert evaluation.overall.total_weight == pytest.approx(8.5)
    assert evaluation.overall.n_samples == 4
    # 两个组的分母允许重叠：small = 7, high = 7.5
    assert groups['small_region'].total_weight == pytest.approx(7.0)
    assert groups['high_risk'].total_weight == pytest.approx(7.5)
    assert evaluation.sum_group_weight == pytest.approx(14.5)
    # 重叠样本 s2(5)+s3(1) 的权重在 multi_member_weight 里显式保留
    assert groups['small_region'].multi_member_weight == pytest.approx(6.0)
    assert groups['high_risk'].multi_member_weight == pytest.approx(6.0)
    assert evaluation.overlapping_sample_ids == ('s2', 's3')
    assert evaluation.overlapping_weight == pytest.approx(6.0)
    # 缺预测的高权重重叠样本对两组都按拒答贡献
    assert groups['small_region'].forced_abstain_weight == pytest.approx(5.0)
    assert groups['high_risk'].forced_abstain_weight == pytest.approx(5.0)


def test_high_weight_overlapping_sample_changes_feasibility():
    batch = make_dataset()
    # high_risk 组：s2 缺预测（权重 5），可达覆盖率只有 2.5/7.5 = 1/3。
    # 高权重 s2 同时属于两组，它的缺失把两组的可达覆盖率一起压低，
    # 使要求 high_risk >= 0.5 的约束组合不可行。
    feasible = fit_abstention_policy(
        batch, costs=COSTS,
        min_coverage={'small_region': 0.2, 'high_risk': 0.3},
        strategy='per_group', protection_groups=SPECS,
    )
    overall_ok = next(
        r for r in feasible.selection_results if r.group is None
    )
    assert overall_ok.status == 'ok'

    infeasible = fit_abstention_policy(
        batch, costs=COSTS,
        min_coverage={'small_region': 0.2, 'high_risk': 0.5},
        strategy='per_group', protection_groups=SPECS,
    )
    status = {s.group: s for s in infeasible.group_statuses}
    assert status['high_risk'].status == 'infeasible'
    assert status['high_risk'].optimistic_coverage == pytest.approx(1 / 3)
    assert '0.500000' in status['high_risk'].reason
    overall = next(
        r for r in infeasible.selection_results if r.group is None
    )
    assert overall.status == 'infeasible'

    evaluation = evaluate_policy(infeasible, batch, bootstrap=False)
    groups = {g.group: g for g in evaluation.groups}
    # 不可行约束不被拆改：small_region 实际达标，但整体组合仍标记不可行；
    # 各组实际结果如实报告。
    assert groups['small_region'].satisfies_min_coverage is True
    assert groups['high_risk'].satisfies_min_coverage is False
    assert groups['high_risk'].coverage == pytest.approx(1 / 3)
    assert evaluation.overall.total_weight == pytest.approx(8.5)


def test_holdout_uses_frozen_thresholds_and_records_membership_fix():
    batch = make_dataset()
    policy = fit_abstention_policy(
        batch, costs=COSTS,
        min_coverage={'small_region': 0.2, 'high_risk': 0.3},
        strategy='per_group', protection_groups=SPECS,
    )
    frozen_thresholds = dict(policy.thresholds)
    holdout_samples = SampleSet([
        Sample('s1', 'B', 1.0, {'region': 'small', 'risk': 'low'}),
        # 归属组后来被修正：s2 从 high_risk 移出
        Sample('s2', 'A', 5.0, {'region': 'small', 'risk': 'low'}),
        Sample('s3', 'A', 1.0, {'region': 'small', 'risk': 'high'}),
        Sample('s4', 'B', 1.5, {'region': 'large', 'risk': 'high'}),
        Sample('h9', 'A', 2.0, {'region': 'large', 'risk': 'high'}),
    ])
    holdout = PredictionBatch([
        Prediction('s1', 'B', 0.99),  # 高分但错误：也不重调阈值
        Prediction('s2', 'A', 0.9),
        Prediction('s3', 'A', 0.8),
        Prediction('s4', 'B', 0.95),
        Prediction('h9', 'A', 0.9),
    ], holdout_samples, 'holdout')

    evaluation = evaluate_policy(policy, holdout, bootstrap=False)
    assert evaluation.thresholds_source == 'selection'
    assert dict(policy.thresholds) == frozen_thresholds
    drift = evaluation.membership_drift
    assert drift is not None and drift.has_drift
    changed = {sid: (old, new) for sid, old, new in drift.changed}
    assert changed['s2'] == (
        ('high_risk', 'small_region'), ('small_region',)
    )
    assert drift.new_sample_ids == ('h9',)
    # 冻结口径：s2 仍按定版快照计入 high_risk，分母与历史依据不变
    groups = {g.group: g for g in evaluation.groups}
    assert groups['high_risk'].n_samples == 3  # s2,s3,s4（不含新样本 h9）
    assert groups['high_risk'].total_weight == pytest.approx(7.5)
    assert evaluation.overall.total_weight == pytest.approx(10.5)

    # 当前口径：归属修正生效，但阈值依旧冻结
    current_eval = evaluate_policy(
        policy, holdout, bootstrap=False, membership_mode='current'
    )
    cgroups = {g.group: g for g in current_eval.groups}
    assert cgroups['high_risk'].n_samples == 3  # s3,s4,h9
    assert dict(policy.thresholds) == frozen_thresholds


def test_save_reload_reproduces_per_sample_decisions():
    batch = make_dataset()
    policy = fit_abstention_policy(
        batch, costs=COSTS,
        min_coverage={'small_region': 0.2, 'high_risk': 0.3},
        strategy='per_group', protection_groups=SPECS,
    )
    restored = AbstentionPolicy.from_dict(
        json.loads(json.dumps(policy.to_dict()))
    )
    assert restored.policy_version == policy.policy_version
    before = evaluate_policy(policy, batch, bootstrap=False)
    after = evaluate_policy(restored, batch, bootstrap=False)
    key = lambda d: (
        d.sample_id, d.status, d.accepted, d.threshold, tuple(d.groups),
        tuple(d.applied_thresholds), d.cost,
    )
    assert [key(d) for d in before.decisions] == [
        key(d) for d in after.decisions
    ]


def test_uniform_overlapping_and_non_integer_weights():
    batch = make_dataset()
    policy = fit_abstention_policy(
        batch, costs=COSTS, min_coverage=0.1,
        strategy='uniform', protection_groups=SPECS,
    )
    assert dict(policy.thresholds).keys() == {'*'}
    evaluation = evaluate_policy(policy, batch, bootstrap=False)
    # 非整数权重 s4 原样保留：接受权重 = s1(1)+s3(1)+s4(1.5) = 3.5
    assert evaluation.overall.accepted_weight == pytest.approx(3.5)
    assert evaluation.overall.total_weight == pytest.approx(8.5)
    groups = {g.group: g for g in evaluation.groups}
    assert groups['high_risk'].accepted_weight == pytest.approx(2.5)
    # 不属于任何保护组的样本在 uniform 策略下仍套用统一阈值，
    # 不会被错误地一律拒答（本例所有样本都有归属，单独构造一例）。
    samples = SampleSet([
        Sample('u1', 'A', 1.0, {'region': 'other'}),
        Sample('g1s', 'A', 1.0, {'region': 'small'}),
    ])
    batch_u = PredictionBatch([
        Prediction('u1', 'A', 0.95),
        Prediction('g1s', 'A', 0.95),
    ], samples)
    policy_u = fit_abstention_policy(
        batch_u, costs=COSTS, min_coverage=0.0,
        strategy='uniform',
        protection_groups=[ProtectionGroupSpec('small_region',
                                               {'region': 'small'})],
    )
    evaluation_u = evaluate_policy(policy_u, batch_u, bootstrap=False)
    assert evaluation_u.decision_for('u1').accepted is True
    assert evaluation_u.unassigned_weight == pytest.approx(1.0)
