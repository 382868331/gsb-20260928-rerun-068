"""拒答策略：覆盖约束定版、独立核对与重叠保护组。

流程约束：

* 阈值与分组选择只使用 fit_abstention_policy 收到的选择资料；
  evaluate_policy 只把已定版策略应用到核对资料上，绝不根据核对结果
  重新调整阈值或分组；
* 代价（错误接受/正确接受/拒答）与各组最低覆盖要求是策略内容的一部分：
  修改任一者都会得到新的 policy_version，旧策略与旧结果仍可追溯；
* 一个样本可以同时属于多个保护组（例如同一条记录既属于小样本地区、
  又属于特定风险等级）。覆盖核对逐组独立进行，同一条记录的权重在每个
  所属组的分子/分母里都出现一次，但总体分母里每个样本只计一次；重叠组
  不会被拆成互斥组，因此原始约束保持不变；
* 资料不足（选择资料中无可用打分预测）的组保留 undetermined 状态；
  各组要求互相冲突、不存在任何阈值组合能同时满足时，策略标记 infeasible
  并保留各组实际可达覆盖率与尽力而为阈值，核对时如实报告各组结果；
* 缺预测、冲突预测、未知预测类别的样本一律按拒答计入人数、覆盖率、
  代价与区间；非整数权重原样保留，总体分母始终是全样本权重之和。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .predictions import PredictionBatch
from .resample import BootstrapResult, bootstrap_weighted_stat
from .versioning import content_version

STRATEGY_UNIFORM = "uniform"
STRATEGY_PER_GROUP = "per_group"
STRATEGIES = (STRATEGY_UNIFORM, STRATEGY_PER_GROUP)

STATUS_OK = "ok"
STATUS_UNDETERMINED = "undetermined"
STATUS_INFEASIBLE = "infeasible"
STATUS_PARTIAL = "partial"

DECISION_ACCEPTED_CORRECT = "accepted_correct"
DECISION_ACCEPTED_WRONG = "accepted_wrong"
DECISION_ABSTAIN_LOW_SCORE = "abstain_low_score"
DECISION_ABSTAIN_MISSING_SCORE = "abstain_missing_score"
DECISION_ABSTAIN_NO_THRESHOLD = "abstain_no_threshold"
DECISION_MISSING = "missing_prediction"
DECISION_CONFLICTING = "conflicting_prediction"
DECISION_UNKNOWN_LABEL = "unknown_predicted_label"

FORCED_ABSTAIN_STATUSES = (
    DECISION_MISSING,
    DECISION_CONFLICTING,
    DECISION_UNKNOWN_LABEL,
)

ALL_GROUPS = "*"
UNASSIGNED = "__unassigned__"
EPS = 1e-12


@dataclass(frozen=True)
class ProtectionGroupSpec:
    """一个重叠保护组：按样本属性条件判定归属，可与其他组重叠。

    `name` 是组名；`match` 是 {分组属性: 允许取值（一个或多个）} 形式的
    合取条件，例如 ``{"region": "small", "risk": ("high", "med")}``
    表示小样本地区且风险等级为 high/med 的样本。条件在样本自带的分组
    属性上判定，选择时与应用时使用同一份规则；定版时还会把实际归属快照
    随策略保存，作为历史依据。
    """

    name: str
    match: Mapping[str, str | Sequence[str]]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("保护组名称不能为空")
        if not isinstance(self.match, Mapping) or not self.match:
            raise ValueError(f"保护组 {self.name} 必须给出非空 match 条件")
        normalized: dict[str, tuple[str, ...]] = {}
        for key, value in self.match.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"保护组 {self.name} 的属性名非法")
            if isinstance(value, str):
                values = (value,)
            elif isinstance(value, Sequence) and not isinstance(value, (bytes,)):
                values = tuple(str(v) for v in value)
            else:
                raise ValueError(
                    f"保护组 {self.name} 属性 {key} 的取值必须是字符串或序列"
                )
            if not values or any(v == "" for v in values):
                raise ValueError(f"保护组 {self.name} 属性 {key} 的取值不能为空")
            normalized[key] = tuple(sorted(values))
        object.__setattr__(self, "match", dict(sorted(normalized.items())))

    def contains(self, sample) -> bool:
        for key, values in self.match.items():
            value = sample.groups.get(key)
            if value is None or str(value) not in values:
                return False
        return True

    def match_to_dict(self) -> dict[str, list[str]]:
        return {key: list(values) for key, values in sorted(self.match.items())}

    @property
    def match_key(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple((key, values) for key, values in sorted(self.match.items()))


@dataclass(frozen=True)
class MembershipPlan:
    """保护组归属结果：组名 -> 属于该组的样本身份（保留权重与多重归属）。"""

    members: tuple[tuple[str, tuple[str, ...]], ...]
    groups_of_sample: tuple[tuple[str, tuple[str, ...]], ...]
    unassigned: tuple[str, ...]
    overlapping_sample_ids: tuple[str, ...]
    overlapping_weight: float
    membership_version: str

    def group_ids(self, group: str) -> tuple[str, ...]:
        return dict(self.members).get(group, ())

    def groups_for(self, sample_id: str) -> tuple[str, ...]:
        return dict(self.groups_of_sample).get(sample_id, ())


def build_membership(
    sample_set,
    specs: Sequence[ProtectionGroupSpec],
) -> MembershipPlan:
    '''按规格计算每个样本的全部保护组归属；不把重叠拆成互斥组。'''
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError('保护组名称不能重复')
    members: dict[str, list[str]] = {name: [] for name in names}
    groups_of: dict[str, list[str]] = {}
    unassigned: list[str] = []
    overlapping: list[str] = []
    overlapping_weight = 0.0
    for sample in sample_set.samples:
        owning = [spec.name for spec in specs if spec.contains(sample)]
        if owning:
            groups_of[sample.sample_id] = owning
            for name in owning:
                members[name].append(sample.sample_id)
            if len(owning) > 1:
                overlapping.append(sample.sample_id)
                overlapping_weight += sample.weight
        else:
            unassigned.append(sample.sample_id)
    membership_version = content_version(
        [
            (
                'membership',
                [
                    [name, sorted(ids)]
                    for name, ids in sorted(members.items())
                ],
            )
        ]
    )
    return MembershipPlan(
        members=tuple((name, tuple(sorted(members[name]))) for name in names),
        groups_of_sample=tuple(
            (sid, tuple(groups_of[sid]))
            for sid in sorted(groups_of)
        ),
        unassigned=tuple(sorted(unassigned)),
        overlapping_sample_ids=tuple(sorted(overlapping)),
        overlapping_weight=overlapping_weight,
        membership_version=membership_version,
    )


@dataclass(frozen=True)
class CostModel:
    '''单次决策代价：错误接受、正确接受与拒答。'''

    false_accept: float
    correct_accept: float
    reject: float

    def __post_init__(self) -> None:
        for name in ('false_accept', 'correct_accept', 'reject'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'代价 {name} 必须是有限数值')
            if value < 0:
                raise ValueError(f'代价 {name} 不能为负')

    def to_dict(self) -> dict[str, float]:
        return {
            'false_accept': self.false_accept,
            'correct_accept': self.correct_accept,
            'reject': self.reject,
        }


@dataclass(frozen=True)
class GroupStatus:
    '''单个组（或统一阈值整体）在选择资料上的定版状态。'''

    group: str
    status: str
    reason: str | None
    threshold: float | None
    min_coverage: float
    max_coverage: float | None
    n_scored: int
    n_members: int = 0
    optimistic_coverage: float | None = None
    joint_coverage: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'group': self.group,
            'status': self.status,
            'reason': self.reason,
            'threshold': _threshold_to_json(self.threshold),
            'min_coverage': self.min_coverage,
            'max_coverage': self.max_coverage,
            'n_scored': self.n_scored,
            'n_members': self.n_members,
            'optimistic_coverage': self.optimistic_coverage,
            'joint_coverage': self.joint_coverage,
        }


def _threshold_to_json(threshold: float | None):
    if threshold is None:
        return None
    if math.isinf(threshold):
        return 'Infinity'
    return float(threshold)


def _threshold_from_json(value) -> float | None:
    if value is None:
        return None
    if value == 'Infinity':
        return math.inf
    return float(value)


@dataclass(frozen=True)
class DecisionRow:
    '''单个样本在策略下的决策结果，携带权重、状态、阈值与所属组。'''

    sample_id: str
    weight: float
    status: str
    cost: float
    accepted: bool
    groups: tuple[str, ...] = ()
    threshold: float | None = None
    applied_thresholds: tuple[tuple[str, float | None], ...] = ()

    @property
    def group(self) -> str:
        # 兼容旧口径：单属性模式下每个样本恰好属于一个组。
        return self.groups[0] if self.groups else UNASSIGNED


@dataclass(frozen=True)
class GroupPolicyResult:
    '''单组（或总体）的策略结果：人数、覆盖率、代价与区间。

    重叠保护组下，组间分母允许重叠；总体结果里每个样本只计一次，因此
    组权重之和可以大于总体权重，multi_member_weight 显式说明重叠量。
    '''

    group: str | None
    status: str
    reason: str | None
    threshold: float | None
    min_coverage: float
    n_samples: int
    total_weight: float
    accepted_weight: float
    abstained_weight: float
    forced_abstain_weight: float
    coverage: float | None
    mean_cost: float | None
    total_cost: float
    status_counts: tuple[tuple[str, int], ...]
    coverage_ci: BootstrapResult | None = None
    cost_ci: BootstrapResult | None = None
    multi_member_weight: float = 0.0
    satisfies_min_coverage: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        def ci_dict(ci: BootstrapResult | None) -> dict[str, Any] | None:
            if ci is None:
                return None
            return {
                'metric': ci.metric,
                'point': ci.point,
                'lower': ci.lower,
                'upper': ci.upper,
                'n_replicates': ci.n_replicates,
                'seed': ci.seed,
                'confidence_level': ci.confidence_level,
                'replicate_version': ci.replicate_version,
            }

        return {
            'group': self.group,
            'status': self.status,
            'reason': self.reason,
            'threshold': _threshold_to_json(self.threshold),
            'min_coverage': self.min_coverage,
            'n_samples': self.n_samples,
            'total_weight': self.total_weight,
            'accepted_weight': self.accepted_weight,
            'abstained_weight': self.abstained_weight,
            'forced_abstain_weight': self.forced_abstain_weight,
            'coverage': self.coverage,
            'mean_cost': self.mean_cost,
            'total_cost': self.total_cost,
            'status_counts': dict(self.status_counts),
            'coverage_ci': ci_dict(self.coverage_ci),
            'cost_ci': ci_dict(self.cost_ci),
            'multi_member_weight': self.multi_member_weight,
            'satisfies_min_coverage': self.satisfies_min_coverage,
        }


@dataclass(frozen=True)
class AbstentionPolicy:
    '''定版后的拒答策略：阈值、代价、覆盖要求、归属快照与选择资料版本。'''

    strategy: str
    costs: CostModel
    min_coverage_default: float
    min_coverage_by_group: tuple[tuple[str, float], ...]
    thresholds: tuple[tuple[str, float | None], ...]
    group_statuses: tuple[GroupStatus, ...]
    selection_sample_set_version: str
    selection_prediction_batch_version: str
    selection_results: tuple[GroupPolicyResult, ...]
    policy_version: str
    group_attr: str | None = None
    mode: str = 'single_attr'
    protection_specs: tuple[ProtectionGroupSpec, ...] = ()
    frozen_membership: tuple[tuple[str, tuple[str, ...]], ...] = ()
    frozen_membership_version: str | None = None
    frozen_overlapping: tuple[str, ...] = ()
    frozen_overlapping_weight: float = 0.0

    def threshold_for(self, group: str) -> float | None:
        table = dict(self.thresholds)
        if self.strategy == STRATEGY_UNIFORM:
            return table.get(ALL_GROUPS)
        return table.get(group)

    def min_coverage_for(self, group: str) -> float:
        return dict(self.min_coverage_by_group).get(
            group, self.min_coverage_default
        )

    def status_for(self, group: str) -> GroupStatus | None:
        for status in self.group_statuses:
            if status.group == group:
                return status
        return None

    def frozen_groups_for(self, sample_id: str) -> tuple[str, ...]:
        result = [
            name
            for name, ids in self.frozen_membership
            if sample_id in ids
        ]
        return tuple(sorted(result))

    def to_dict(self) -> dict[str, Any]:
        return {
            'strategy': self.strategy,
            'mode': self.mode,
            'group_attr': self.group_attr,
            'costs': self.costs.to_dict(),
            'min_coverage_default': self.min_coverage_default,
            'min_coverage_by_group': dict(self.min_coverage_by_group),
            'thresholds': {
                g: _threshold_to_json(t) for g, t in self.thresholds
            },
            'group_statuses': [s.to_dict() for s in self.group_statuses],
            'selection_sample_set_version': self.selection_sample_set_version,
            'selection_prediction_batch_version': (
                self.selection_prediction_batch_version
            ),
            'selection_results': [r.to_dict() for r in self.selection_results],
            'policy_version': self.policy_version,
            'protection_specs': [
                {'name': spec.name, 'match': spec.match_to_dict()}
                for spec in self.protection_specs
            ],
            'frozen_membership': [
                [name, list(ids)] for name, ids in self.frozen_membership
            ],
            'frozen_membership_version': self.frozen_membership_version,
            'frozen_overlapping': list(self.frozen_overlapping),
            'frozen_overlapping_weight': self.frozen_overlapping_weight,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'AbstentionPolicy':
        costs = CostModel(**data['costs'])
        statuses = tuple(
            GroupStatus(
                group=item['group'],
                status=item['status'],
                reason=item.get('reason'),
                threshold=_threshold_from_json(item.get('threshold')),
                min_coverage=item['min_coverage'],
                max_coverage=item.get('max_coverage'),
                n_scored=item['n_scored'],
                n_members=item.get('n_members', 0),
                optimistic_coverage=item.get('optimistic_coverage'),
                joint_coverage=item.get('joint_coverage'),
            )
            for item in data['group_statuses']
        )
        specs = tuple(
            ProtectionGroupSpec(item['name'], item['match'])
            for item in data.get('protection_specs', [])
        )
        # selection_results 仅用于展示，反序列化时不恢复 bootstrap 区间。
        results = tuple(
            GroupPolicyResult(
                group=item['group'],
                status=item['status'],
                reason=item.get('reason'),
                threshold=_threshold_from_json(item.get('threshold')),
                min_coverage=item['min_coverage'],
                n_samples=item['n_samples'],
                total_weight=item['total_weight'],
                accepted_weight=item['accepted_weight'],
                abstained_weight=item['abstained_weight'],
                forced_abstain_weight=item['forced_abstain_weight'],
                coverage=item.get('coverage'),
                mean_cost=item.get('mean_cost'),
                total_cost=item['total_cost'],
                status_counts=tuple(
                    (k, v) for k, v in item['status_counts'].items()
                ),
                multi_member_weight=item.get('multi_member_weight', 0.0),
                satisfies_min_coverage=item.get('satisfies_min_coverage'),
            )
            for item in data.get('selection_results', [])
        )
        return cls(
            strategy=data['strategy'],
            group_attr=data.get('group_attr'),
            mode=data.get('mode', 'single_attr'),
            costs=costs,
            min_coverage_default=data['min_coverage_default'],
            min_coverage_by_group=tuple(
                (k, v) for k, v in data['min_coverage_by_group'].items()
            ),
            thresholds=tuple(
                (g, _threshold_from_json(t))
                for g, t in data['thresholds'].items()
            ),
            group_statuses=statuses,
            selection_sample_set_version=data['selection_sample_set_version'],
            selection_prediction_batch_version=data[
                'selection_prediction_batch_version'
            ],
            selection_results=results,
            policy_version=data['policy_version'],
            protection_specs=specs,
            frozen_membership=tuple(
                (name, tuple(ids))
                for name, ids in data.get('frozen_membership', [])
            ),
            frozen_membership_version=data.get('frozen_membership_version'),
            frozen_overlapping=tuple(data.get('frozen_overlapping', [])),
            frozen_overlapping_weight=data.get(
                'frozen_overlapping_weight', 0.0
            ),
        )


@dataclass(frozen=True)
class MembershipDrift:
    '''应用资料上的归属与定版时归属快照的差异（归属组后来被修正）。'''

    added: tuple[tuple[str, tuple[str, ...]], ...]
    removed: tuple[tuple[str, tuple[str, ...]], ...]
    changed: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]
    new_sample_ids: tuple[str, ...]
    selection_membership_version: str | None
    current_membership_version: str

    @property
    def has_drift(self) -> bool:
        return bool(
            self.added or self.removed or self.changed or self.new_sample_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'added': [[sid, list(g)] for sid, g in self.added],
            'removed': [[sid, list(g)] for sid, g in self.removed],
            'changed': [
                [sid, list(old), list(new)]
                for sid, old, new in self.changed
            ],
            'new_sample_ids': list(self.new_sample_ids),
            'selection_membership_version': self.selection_membership_version,
            'current_membership_version': self.current_membership_version,
            'has_drift': self.has_drift,
        }


@dataclass(frozen=True)
class PolicyEvaluation:
    '''定版策略在独立核对资料上的结果。'''

    policy_version: str
    evaluation_version: str
    strategy: str
    sample_set_version: str
    prediction_batch_version: str
    selection_sample_set_version: str
    selection_prediction_batch_version: str
    thresholds_source: str
    overall: GroupPolicyResult
    groups: tuple[GroupPolicyResult, ...]
    decisions: tuple[DecisionRow, ...] = ()
    membership_drift: MembershipDrift | None = None
    overlapping_sample_ids: tuple[str, ...] = ()
    overlapping_weight: float = 0.0
    sum_group_weight: float = 0.0
    unassigned_weight: float = 0.0

    def decision_for(self, sample_id: str) -> DecisionRow | None:
        for row in self.decisions:
            if row.sample_id == sample_id:
                return row
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            'policy_version': self.policy_version,
            'evaluation_version': self.evaluation_version,
            'strategy': self.strategy,
            'mode': (
                'overlapping' if self.membership_drift is not None
                else 'single_attr'
            ),
            'sample_set_version': self.sample_set_version,
            'prediction_batch_version': self.prediction_batch_version,
            'selection_sample_set_version': self.selection_sample_set_version,
            'selection_prediction_batch_version': (
                self.selection_prediction_batch_version
            ),
            'thresholds_source': self.thresholds_source,
            'overall': self.overall.to_dict(),
            'groups': [g.to_dict() for g in self.groups],
            'decisions': [
                {
                    'sample_id': row.sample_id,
                    'weight': row.weight,
                    'groups': list(row.groups),
                    'status': row.status,
                    'accepted': row.accepted,
                    'cost': row.cost,
                    'threshold': _threshold_to_json(row.threshold),
                    'applied_thresholds': [
                        [g, _threshold_to_json(t)]
                        for g, t in row.applied_thresholds
                    ],
                }
                for row in self.decisions
            ],
            'membership_drift': (
                None if self.membership_drift is None
                else self.membership_drift.to_dict()
            ),
            'overlapping_sample_ids': list(self.overlapping_sample_ids),
            'overlapping_weight': self.overlapping_weight,
            'sum_group_weight': self.sum_group_weight,
            'unassigned_weight': self.unassigned_weight,
        }


@dataclass(frozen=True)
class PolicyComparison:
    '''多个定版策略（如统一阈值 vs 分组阈值）在同一核对资料上的并排比较。'''

    comparison_version: str
    sample_set_version: str
    evaluations: tuple[PolicyEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            'comparison_version': self.comparison_version,
            'sample_set_version': self.sample_set_version,
            'evaluations': [e.to_dict() for e in self.evaluations],
        }


def _normalize_min_coverage(
    min_coverage: float | Mapping[str, float],
) -> tuple[float, tuple[tuple[str, float], ...]]:
    '''接受单一数值或 {组: 要求}（可用 * 作为缺省），返回 (缺省, 分组表)。'''

    def check(value: float) -> float:
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(f'最低覆盖要求必须在 [0, 1] 内: {value!r}')
        return float(value)

    if isinstance(min_coverage, Mapping):
        table = {str(k): check(v) for k, v in min_coverage.items()}
        default = table.pop(ALL_GROUPS, 0.0)
        return default, tuple(sorted(table.items()))
    return check(min_coverage), ()


def _cost_for(status: str, costs: CostModel) -> float:
    if status == DECISION_ACCEPTED_CORRECT:
        return costs.correct_accept
    if status == DECISION_ACCEPTED_WRONG:
        return costs.false_accept
    return costs.reject


def _summarize(
    group: str | None,
    rows: list[DecisionRow],
    status: str,
    reason: str | None,
    threshold: float | None,
    min_coverage: float,
    with_ci: bool,
    n_replicates: int,
    seed: int,
    confidence_level: float,
    multi_member_ids: set[str] | None = None,
) -> GroupPolicyResult:
    total_weight = sum(r.weight for r in rows)
    accepted_weight = sum(r.weight for r in rows if r.accepted)
    forced_weight = sum(
        r.weight for r in rows if r.status in FORCED_ABSTAIN_STATUSES
    )
    total_cost = sum(r.cost * r.weight for r in rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    coverage = accepted_weight / total_weight if total_weight > 0 else None
    mean_cost = total_cost / total_weight if total_weight > 0 else None
    multi_weight = (
        sum(
            r.weight for r in rows
            if multi_member_ids is not None and r.sample_id in multi_member_ids
        )
        if group is not None
        else 0.0
    )
    satisfies = (
        None if coverage is None
        else coverage + EPS >= min_coverage
    )
    coverage_ci = None
    cost_ci = None
    if with_ci and rows:
        coverage_ci = bootstrap_weighted_stat(
            rows,
            lambda r: 1.0 if r.accepted else 0.0,
            'coverage',
            n_replicates=n_replicates,
            seed=seed,
            confidence_level=confidence_level,
        )
        cost_ci = bootstrap_weighted_stat(
            rows,
            lambda r: r.cost,
            'mean_cost',
            n_replicates=n_replicates,
            seed=seed,
            confidence_level=confidence_level,
        )
    return GroupPolicyResult(
        group=group,
        status=status,
        reason=reason,
        threshold=threshold,
        min_coverage=min_coverage,
        n_samples=len(rows),
        total_weight=total_weight,
        accepted_weight=accepted_weight,
        abstained_weight=total_weight - accepted_weight,
        forced_abstain_weight=forced_weight,
        coverage=coverage,
        mean_cost=mean_cost,
        total_cost=total_cost,
        status_counts=tuple(sorted(counts.items())),
        coverage_ci=coverage_ci,
        cost_ci=cost_ci,
        multi_member_weight=multi_weight,
        satisfies_min_coverage=satisfies,
    )


def _decision_context(batch: PredictionBatch):
    quality = batch.summary
    eval_index = {r.sample_id: r for r in batch.evaluable_rows}
    missing_ids = set(quality.missing_sample_ids)
    conflicting_ids = {sid for sid, _ in quality.conflicting_rows}
    unknown_label_ids = {sid for sid, _ in quality.unknown_label_rows}
    return eval_index, missing_ids, conflicting_ids, unknown_label_ids


def _make_decision(
    sample, groups, thresholds_for_groups, costs, ctx,
    unassigned_threshold=None,
):
    '''一个样本一条决策；同时属多组时取相关组阈值的最大值。

    取最大值保证：只要被激活的某个保护组阈值要求接受该样本，它才被接受，
    从而所有相关组的覆盖约束都按同一套接受集合核对，不需要拆成互斥组。
    任一相关组未定版/不可行（阈值 None）时，该样本按无阈值拒答。
    '''
    eval_index, missing_ids, conflicting_ids, unknown_label_ids = ctx
    sid = sample.sample_id
    if sid in missing_ids:
        status = DECISION_MISSING
        ev = None
    elif sid in conflicting_ids:
        status = DECISION_CONFLICTING
        ev = None
    elif sid in unknown_label_ids:
        status = DECISION_UNKNOWN_LABEL
        ev = None
    elif eval_index.get(sid) is None:
        status = DECISION_ABSTAIN_NO_THRESHOLD
        ev = None
    else:
        ev = eval_index[sid]
        if ev.score is None:
            status = DECISION_ABSTAIN_MISSING_SCORE
        else:
            status = None

    applied = tuple((g, thresholds_for_groups(g)) for g in groups)
    active_thresholds = [t for _, t in applied]
    if not active_thresholds and unassigned_threshold is not None:
        active_thresholds = [unassigned_threshold]
    has_none = any(t is None for t in active_thresholds)
    effective = (
        max(t for t in active_thresholds if t is not None)
        if active_thresholds and not has_none else None
    )
    accepted = False
    if status in FORCED_ABSTAIN_STATUSES:
        pass
    elif has_none:
        status = DECISION_ABSTAIN_NO_THRESHOLD
    elif effective is None:
        status = DECISION_ABSTAIN_NO_THRESHOLD
    elif ev.score is None:
        status = DECISION_ABSTAIN_MISSING_SCORE
    elif ev.score >= effective:
        accepted = True
        status = (
            DECISION_ACCEPTED_CORRECT
            if ev.predicted == ev.label
            else DECISION_ACCEPTED_WRONG
        )
    else:
        status = DECISION_ABSTAIN_LOW_SCORE
    return DecisionRow(
        sample_id=sid,
        weight=sample.weight,
        status=status,
        cost=_cost_for(status, costs),
        accepted=accepted,
        groups=tuple(groups),
        threshold=effective,
                applied_thresholds=applied,
            )


def _evaluate_vector(
    batch: PredictionBatch,
    groups_of_samples: Sequence[tuple[Any, tuple[str, ...]]],
    thresholds: Mapping[str, float | None],
    costs: CostModel,
    ctx,
    unassigned_threshold=None,
) -> tuple[list[DecisionRow], dict[str, dict[str, float]]]:
    '''在给定阈值向量下逐样本决策，并汇总每个组与总体的覆盖/代价。

    每个样本只产生一条决策（总体只计一次），其权重计入它所属的每个组。
    '''
    decisions: list[DecisionRow] = []
    stats: dict[str, dict[str, float]] = {}

    def bucket() -> dict[str, float]:
        return {'total': 0.0, 'accepted': 0.0, 'cost': 0.0, 'n': 0.0}

    for sample, owned in groups_of_samples:
        row = _make_decision(
            sample,
            owned,
            lambda g, _t=thresholds: _t.get(g),
            costs,
            ctx,
            unassigned_threshold=unassigned_threshold,
        )
        decisions.append(row)
        overall = stats.setdefault(None, bucket())
        overall['total'] += row.weight
        overall['accepted'] += row.weight if row.accepted else 0.0
        overall['cost'] += row.cost * row.weight
        overall['n'] += 1
        for group in owned:
            item = stats.setdefault(group, bucket())
            item['total'] += row.weight
            item['accepted'] += row.weight if row.accepted else 0.0
            item['cost'] += row.cost * row.weight
            item['n'] += 1
    return decisions, stats


def _coverage(item: dict[str, float]) -> float:
    return item['accepted'] / item['total'] if item['total'] > 0 else 0.0


def _shortfall(stats, req_for, groups) -> float:
    total = 0.0
    for group in groups:
        item = stats.get(group)
        if item is None or item['total'] <= 0:
            continue
        gap = req_for(group) - item['accepted'] / item['total']
        if gap > 0:
            total += gap * item['total']
    return total


def _search_vectors(
    batch: PredictionBatch,
    groups_of_samples,
    group_names: list[str],
    candidates: dict[str, list[float | None]],
    costs: CostModel,
    ctx,
    req_for,
    unassigned_threshold_fn=None,
):
    '''枚举阈值向量（DFS + 剪枝），返回 (最优可行, 最优尽力而为)。

    所有组同时取各自最低候选时覆盖率在每组都达到全局上界；任一已锁定组
    在其余组全部取最低的乐观情形下仍不达标，即可剪枝。
    '''
    order = group_names
    minimums = {
        g: next((c for c in candidates[g] if c is not None), None)
        for g in order
    }

    feasible_best = None
    effort_best = None

    # 每组在全部阈值都取最低（最宽松）向量下的覆盖率：若每组都达标，
    # 该向量本身就是可行解；只有某组在此上界下仍不达标时才可能不可行，
    # 此时关闭剪枝，完整枚举以便选出尽力而为（缺口最小）的阈值向量。
    upper_vec = {g: minimums[g] for g in order}
    _, upper_stats = _evaluate_vector(
        batch, groups_of_samples, upper_vec, costs, ctx,
        unassigned_threshold=(
            None if unassigned_threshold_fn is None
            else unassigned_threshold_fn(upper_vec)
        ),
    )
    all_reachable = all(
        upper_stats.get(g, {'total': 0.0})['total'] <= 0
        or _coverage(upper_stats[g]) + EPS >= req_for(g)
        for g in order
    )

    def vector_key(vec):
        # inf 不能参与元组排序（与任意 float 比较结果不可靠地用于平局），
        # 用 None 表示全拒答候选并放在更靠后的位置。
        return tuple(None if t is None or math.isinf(t) else float(t)
                     for t in vec)

    def better(cost_a, vec_a, cost_b, vec_b):
        if cost_b is None or cost_a < cost_b - EPS:
            return True
        if abs(cost_a - cost_b) <= EPS and vector_key(vec_a) < vector_key(vec_b):
            return True
        return False

    def dfs(index: int, locked: dict[str, float | None]):
        nonlocal feasible_best, effort_best
        if all_reachable:
            # 乐观剪枝：已锁定组在其余组取最低时仍不达标即可剪。
            if 0 < index < len(order):
                opt_vec = dict(locked)
                for g in order:
                    opt_vec.setdefault(g, minimums[g])
                _, opt_stats = _evaluate_vector(
                    batch, groups_of_samples, opt_vec, costs, ctx,
                    unassigned_threshold=(
                        None if unassigned_threshold_fn is None
                        else unassigned_threshold_fn(opt_vec)
                    ),
                )
                for g in order[:index]:
                    item = opt_stats.get(g)
                    if item is not None and _coverage(item) + EPS < req_for(g):
                        return
        if index == len(order):
            decisions, stats = _evaluate_vector(
                batch, groups_of_samples, locked, costs, ctx,
                unassigned_threshold=(
                    None if unassigned_threshold_fn is None
                    else unassigned_threshold_fn(locked)
                ),
            )
            total_item = stats[None]
            total_w = total_item['total']
            mean_cost = total_item['cost'] / total_w if total_w > 0 else 0.0
            vector_key = tuple(locked[g] for g in order)
            feasible = all(
                stats.get(g, {'total': 0.0})['total'] <= 0
                or _coverage(stats[g]) + EPS >= req_for(g)
                for g in order
            )
            if feasible:
                vec = [locked[g] for g in order]
                if feasible_best is None or better(
                    mean_cost, vec,
                    feasible_best[0], feasible_best[1],
                ):
                    feasible_best = (mean_cost, vec, decisions, stats)
            gap = _shortfall(stats, req_for, order)
            vec = [locked[g] for g in order]
            if effort_best is None:
                take = True
            elif gap < effort_best[0] - EPS:
                take = True
            elif abs(gap - effort_best[0]) <= EPS and better(
                mean_cost, vec, effort_best[1], effort_best[2]
            ):
                take = True
            else:
                take = False
            if take:
                effort_best = (gap, mean_cost, vec, decisions, stats)
            return
        group = order[index]
        for candidate in candidates[group]:
            locked[group] = candidate
            dfs(index + 1, locked)
        locked.pop(group, None)

    dfs(0, {})
    return feasible_best, effort_best


def fit_abstention_policy(
    selection_batch: PredictionBatch,
    group_attr: str | None = None,
    costs: CostModel | None = None,
    min_coverage: float | Mapping[str, float] = 0.0,
    strategy: str = STRATEGY_PER_GROUP,
    protection_groups=None,
) -> AbstentionPolicy:
    '''只在选择资料上定版拒答策略。

    两种口径：

    * 单属性模式（protection_groups 为 None）：按 group_attr 把样本划入
      互不重叠的组，沿用原始约束，不拆分、不合并；
    * 重叠保护组模式（给出 protection_groups）：一个样本可同时属于多个
      组，组约束逐组核对，总体每个样本只计一次。per_group 在联合接受
      语义（取所属组阈值的最大值）下搜索总体代价最小且各组覆盖都达标的
      阈值向量；uniform 搜索同一个统一阈值。

    不存在能同时满足各组最低覆盖的阈值组合时，策略整体标记 infeasible，
    保留每组乐观可达覆盖率与尽力而为阈值，核对资料上报告各组实际结果。
    '''
    if costs is None:
        raise ValueError(
            '必须显式给出 CostModel（错误接受/正确接受/拒答代价）'
        )
    if strategy not in STRATEGIES:
        raise ValueError(f'未知策略 {strategy!r}，可选: {STRATEGIES}')
    default_req, group_reqs = _normalize_min_coverage(min_coverage)
    req_table = dict(group_reqs)

    def req_for(group: str) -> float:
        return req_table.get(group, default_req)

    if protection_groups is None:
        return _fit_single_attr(
            selection_batch, group_attr, costs, default_req, group_reqs,
            req_for, strategy,
        )
    specs = _normalize_specs(protection_groups)
    return _fit_overlapping(
        selection_batch, specs, costs, default_req, group_reqs,
        req_for, strategy,
    )


def _normalize_specs(protection_groups) -> tuple[ProtectionGroupSpec, ...]:
    if isinstance(protection_groups, Mapping):
        specs = [
            ProtectionGroupSpec(str(name), match)
            for name, match in protection_groups.items()
        ]
    else:
        specs = list(protection_groups)
    for spec in specs:
        if not isinstance(spec, ProtectionGroupSpec):
            raise ValueError(
                'protection_groups 必须是 ProtectionGroupSpec 或 '
                '组名到条件的映射'
            )
    return tuple(sorted(specs, key=lambda s: s.name))


def _legacy_group_key(sample, group_attr: str | None) -> str:
    if group_attr is None:
        return ALL_GROUPS
    value = sample.groups.get(group_attr)
    return UNASSIGNED if value is None else str(value)


def _legacy_decisions(batch, group_attr, costs, threshold_for):
    ctx = _decision_context(batch)
    rows = []
    for sample in batch.sample_set.samples:
        group = _legacy_group_key(sample, group_attr)
        rows.append(
            _make_decision(
                sample, (group,),
                lambda g: threshold_for(g), costs, ctx,
            )
        )
    return rows


def _fit_single_attr(
    selection_batch, group_attr, costs, default_req, group_reqs,
    req_for, strategy,
):
    sample_set = selection_batch.sample_set
    groups = sorted(
        {_legacy_group_key(sample, group_attr)
         for sample in sample_set.samples}
    )
    scored_by_group: dict[str, list[float]] = {g: [] for g in groups}
    for row in selection_batch.evaluable_rows:
        if row.score is not None:
            group = _legacy_group_key(
                sample_set.get(row.sample_id), group_attr
            )
            scored_by_group.setdefault(group, []).append(row.score)

    def group_stats(group: str, threshold):
        decisions = _legacy_decisions(
            selection_batch, group_attr, costs, lambda g: threshold
        )
        sub = [r for r in decisions if r.group == group]
        total = sum(r.weight for r in sub)
        accepted = sum(r.weight for r in sub if r.accepted)
        cost = sum(r.cost * r.weight for r in sub)
        return (
            accepted / total if total > 0 else 0.0,
            cost / total if total > 0 else 0.0,
            total,
        )

    statuses: list[GroupStatus] = []
    thresholds: dict[str, float | None] = {}

    if strategy == STRATEGY_PER_GROUP:
        for group in groups:
            req = req_for(group)
            scores = sorted(set(scored_by_group.get(group, [])))
            n_scored = len(scored_by_group.get(group, []))
            n_members = sum(
                1 for s in sample_set.samples
                if _legacy_group_key(s, group_attr) == group
            )
            if not scores:
                statuses.append(GroupStatus(
                    group=group, status=STATUS_UNDETERMINED,
                    reason='选择资料中该组没有可用打分预测，无法定阈值',
                    threshold=None, min_coverage=req, max_coverage=None,
                    n_scored=0, n_members=n_members,
                ))
                thresholds[group] = None
                continue
            max_cov, _, _ = group_stats(group, min(scores))
            best = None
            for threshold in scores + [math.inf]:
                cov, cost, _ = group_stats(group, threshold)
                if cov + EPS >= req and (
                    best is None or cost < best[1] - EPS
                ):
                    best = (threshold, cost, cov)
            if best is None:
                statuses.append(GroupStatus(
                    group=group, status=STATUS_INFEASIBLE,
                    reason=(
                        f'可达覆盖率 {max_cov:.6f} 低于最低要求 {req:.6f}'
                    ),
                    threshold=None, min_coverage=req, max_coverage=max_cov,
                    n_scored=n_scored, n_members=n_members,
                ))
                thresholds[group] = None
            else:
                statuses.append(GroupStatus(
                    group=group, status=STATUS_OK, reason=None,
                    threshold=best[0], min_coverage=req, max_coverage=max_cov,
                    n_scored=n_scored, n_members=n_members,
                ))
                thresholds[group] = best[0]
    else:
        _fit_single_attr_uniform(
            selection_batch, group_attr, groups, scored_by_group,
            group_stats, req_for, default_req, statuses, thresholds,
        )

    threshold_table = (
        thresholds
        if strategy == STRATEGY_PER_GROUP
        else {ALL_GROUPS: thresholds[ALL_GROUPS]}
    )
    selection_decisions = _legacy_decisions(
        selection_batch, group_attr, costs,
        lambda g: (
            threshold_table[ALL_GROUPS]
            if strategy == STRATEGY_UNIFORM
            else threshold_table.get(g)
        ),
    )
    selection_results: list[GroupPolicyResult] = []
    for group in groups:
        sub = [r for r in selection_decisions if r.group == group]
        st = next(s for s in statuses if s.group == group)
        selection_results.append(
            _summarize(
                group, sub, st.status, st.reason, st.threshold,
                req_for(group), False, 0, 0, 0.95,
            )
        )
    non_ok = [s for s in statuses if s.group != ALL_GROUPS and s.status != STATUS_OK]
    overall_status = STATUS_OK if not non_ok else STATUS_PARTIAL
    overall_reason = (
        None
        if not non_ok
        else '存在未确定或不可行的组: ' + ', '.join(s.group for s in non_ok)
    )
    selection_results.append(
        _summarize(
            None, selection_decisions, overall_status, overall_reason,
            threshold_table.get(ALL_GROUPS), default_req, False, 0, 0, 0.95,
        )
    )
    policy_version = content_version([
        ('mode', 'single_attr'),
        ('strategy', strategy),
        ('group_attr', group_attr),
        ('costs', costs.to_dict()),
        ('min_coverage_default', default_req),
        ('min_coverage_by_group', list(group_reqs)),
        ('thresholds', sorted(
            (g, None if t is None else float(t))
            for g, t in threshold_table.items()
        )),
        ('group_statuses', [
            [s.group, s.status, s.reason,
             _threshold_to_json(s.threshold), s.min_coverage]
            for s in statuses
        ]),
        ('selection_sample_set_version', sample_set.version),
        ('selection_prediction_batch_version', selection_batch.version),
    ])
    return AbstentionPolicy(
        strategy=strategy,
        costs=costs,
        min_coverage_default=default_req,
        min_coverage_by_group=group_reqs,
        thresholds=tuple(sorted(threshold_table.items())),
        group_statuses=tuple(statuses),
        selection_sample_set_version=sample_set.version,
        selection_prediction_batch_version=selection_batch.version,
        selection_results=tuple(selection_results),
        policy_version=policy_version,
        group_attr=group_attr,
        mode='single_attr',
    )


def _fit_single_attr_uniform(
    batch, group_attr, groups, scored_by_group, group_stats,
    req_for, default_req, statuses, thresholds,
):
    all_scores = sorted(
        {s for scores in scored_by_group.values() for s in scores}
    )
    best_uniform = None
    for threshold in all_scores + [math.inf]:
        feasible = True
        total_cost_w = 0.0
        total_w = 0.0
        for group in groups:
            cov, cost, weight = group_stats(group, threshold)
            if cov + EPS < req_for(group):
                feasible = False
            total_cost_w += cost * weight
            total_w += weight
        mean_cost = total_cost_w / total_w if total_w > 0 else 0.0
        if feasible and (
            best_uniform is None or mean_cost < best_uniform[1] - EPS
        ):
            best_uniform = (threshold, mean_cost)
    if best_uniform is None:
        for group in groups:
            req = req_for(group)
            scores = scored_by_group.get(group, [])
            if not scores:
                statuses.append(GroupStatus(
                    group=group, status=STATUS_UNDETERMINED,
                    reason='选择资料中该组没有可用打分预测，无法定阈值',
                    threshold=None, min_coverage=req, max_coverage=None,
                    n_scored=0,
                ))
                continue
            max_cov, _, _ = group_stats(group, min(scores))
            statuses.append(GroupStatus(
                group=group,
                status=(
                    STATUS_INFEASIBLE if max_cov + EPS < req else STATUS_OK
                ),
                reason=(
                    None if max_cov + EPS >= req
                    else f'可达覆盖率 {max_cov:.6f} 低于最低要求 {req:.6f}'
                ),
                threshold=None, min_coverage=req, max_coverage=max_cov,
                n_scored=len(scores),
            ))
        statuses.append(GroupStatus(
            group=ALL_GROUPS, status=STATUS_INFEASIBLE,
            reason='不存在同时满足各组最低覆盖要求的统一阈值',
            threshold=None, min_coverage=default_req, max_coverage=None,
            n_scored=sum(len(s) for s in scored_by_group.values()),
        ))
        thresholds[ALL_GROUPS] = None
    else:
        for group in groups:
            req = req_for(group)
            scores = scored_by_group.get(group, [])
            max_cov = (
                group_stats(group, min(scores))[0] if scores else None
            )
            statuses.append(GroupStatus(
                group=group, status=STATUS_OK, reason=None,
                threshold=best_uniform[0], min_coverage=req,
                max_coverage=max_cov, n_scored=len(scores),
            ))
        thresholds[ALL_GROUPS] = best_uniform[0]


def _fit_overlapping(
    selection_batch, specs, costs, default_req, group_reqs,
    req_for, strategy,
):
    batch = selection_batch
    sample_set = batch.sample_set
    membership = build_membership(sample_set, specs)
    names = [spec.name for spec in specs]
    groups_of = dict(membership.groups_of_sample)
    ctx = _decision_context(batch)

    # 每个样本携带它所属的全部组；不属于任何保护组的样本保留在总体账上，
    # 按统一阈值（uniform）或缺省阈值处理。
    groups_of_samples = [
        (sample, tuple(groups_of.get(sample.sample_id, ())))
        for sample in sample_set.samples
    ]

    scored_values: dict[str, set[float]] = {name: set() for name in names}
    for row in batch.evaluable_rows:
        if row.score is None:
            continue
        for name in groups_of.get(row.sample_id, ()):
            scored_values[name].add(row.score)

    undetermined: set[str] = set()
    for name in names:
        ids = membership.group_ids(name)
        has_scored = bool(scored_values[name])
        if ids and not has_scored:
            undetermined.add(name)

    if strategy == STRATEGY_UNIFORM:
        all_scores = sorted({
            score for values in scored_values.values() for score in values
        })
        candidates = {
            name: (
                [None] if name in undetermined
                else all_scores + [math.inf]
            )
            for name in names
        }
    else:
        candidates = {}
        for name in names:
            if name in undetermined:
                candidates[name] = [None]
            else:
                candidates[name] = sorted(scored_values[name]) + [math.inf]

    feasible_best, effort_best = _search_vectors(
        batch, groups_of_samples, names, candidates, costs, ctx, req_for,
        unassigned_threshold_fn=(
            (lambda vec: vec[names[0]])
            if strategy == STRATEGY_UNIFORM and names else None
        ),
    )

    chosen = feasible_best if feasible_best is not None else effort_best
    if feasible_best is not None:
        _, vec_values, fit_decisions, fit_stats = feasible_best
    else:
        _, _, vec_values, fit_decisions, fit_stats = effort_best
    vector = dict(zip(names, vec_values))

    # 乐观（各组独立取最低阈值、忽略联合冲突）覆盖率，用于说明每组上限。
    optimistic_vec = {}
    for name in names:
        finite = [c for c in candidates[name] if c is not None]
        optimistic_vec[name] = min(finite) if finite else None
    _, opt_stats = _evaluate_vector(
        batch, groups_of_samples, optimistic_vec, costs, ctx,
        unassigned_threshold=(
            optimistic_vec[names[0]]
            if strategy == STRATEGY_UNIFORM and names else None
        ),
    )

    statuses: list[GroupStatus] = []
    for name in names:
        ids = membership.group_ids(name)
        finite = [c for c in candidates[name] if c is not None]
        item = fit_stats.get(name)
        joint_cov = (
            item['accepted'] / item['total']
            if item is not None and item['total'] > 0 else 0.0
        )
        opt_item = opt_stats.get(name)
        opt_cov = (
            opt_item['accepted'] / opt_item['total']
            if opt_item is not None and opt_item['total'] > 0 else 0.0
        )
        req = req_for(name)
        if name in undetermined:
            statuses.append(GroupStatus(
                group=name, status=STATUS_UNDETERMINED,
                reason='选择资料中该组没有可用打分预测，无法定阈值',
                threshold=None, min_coverage=req, max_coverage=None,
                n_scored=0, n_members=len(ids),
                optimistic_coverage=None, joint_coverage=joint_cov,
            ))
        elif feasible_best is None:
            statuses.append(GroupStatus(
                group=name, status=STATUS_INFEASIBLE,
                reason=(
                    f'各组最低覆盖要求互相冲突：该组乐观可达 '
                    f'{opt_cov:.6f}、联合定版后实际 {joint_cov:.6f}，'
                    f'最低要求 {req:.6f}'
                ),
                threshold=vector[name], min_coverage=req,
                max_coverage=opt_cov, n_scored=len(scored_values[name]),
                n_members=len(ids), optimistic_coverage=opt_cov,
                joint_coverage=joint_cov,
            ))
        else:
            statuses.append(GroupStatus(
                group=name, status=STATUS_OK, reason=None,
                threshold=vector[name], min_coverage=req,
                max_coverage=opt_cov, n_scored=len(scored_values[name]),
                n_members=len(ids), optimistic_coverage=opt_cov,
                joint_coverage=joint_cov,
            ))

    multi_ids = set(membership.overlapping_sample_ids)

    def group_result(name, decisions, with_ci):
        member_ids = set(membership.group_ids(name))
        sub = [r for r in decisions if r.sample_id in member_ids]
        st = next(s for s in statuses if s.group == name)
        return _summarize(
            name, sub, st.status, st.reason, st.threshold,
            req_for(name), with_ci, 0, 0, 0.95,
            multi_member_ids=multi_ids,
        )

    selection_results = [
        group_result(name, fit_decisions, False) for name in names
    ]
    if feasible_best is None:
        failing = [
            s.group for s in statuses
            if s.status == STATUS_INFEASIBLE
            or (
                s.status == STATUS_UNDETERMINED
                and req_for(s.group) > EPS
            )
        ]
        overall_status = STATUS_INFEASIBLE
        reason = (
            '不存在同时满足各组最低覆盖要求的阈值组合，未达标组: '
            + ', '.join(failing)
            if failing
            else '不存在同时满足各组最低覆盖要求的阈值组合'
        )
    elif any(s.status == STATUS_UNDETERMINED for s in statuses):
        overall_status = STATUS_PARTIAL
        reason = (
            '存在没有可用打分预测的组: '
            + ', '.join(
                s.group for s in statuses
                if s.status == STATUS_UNDETERMINED
            )
        )
    else:
        overall_status = STATUS_OK
        reason = None

    overall_threshold = (
        vector[names[0]]
        if strategy == STRATEGY_UNIFORM and names else None
    )
    if strategy == STRATEGY_UNIFORM and len({
        vector[n] for n in names
    }) == 1 and names:
        overall_threshold = vector[names[0]]

    overall = _summarize(
        None, fit_decisions, overall_status, reason,
        overall_threshold, default_req, False, 0, 0, 0.95,
        multi_member_ids=multi_ids,
    )
    selection_results.append(overall)

    threshold_table: dict[str, float | None]
    if strategy == STRATEGY_UNIFORM:
        uniform_value = vector[names[0]] if names else None
        threshold_table = {ALL_GROUPS: uniform_value}
    else:
        threshold_table = {name: vector[name] for name in names}

    policy_version = content_version([
        ('mode', 'overlapping'),
        ('strategy', strategy),
        ('costs', costs.to_dict()),
        ('min_coverage_default', default_req),
        ('min_coverage_by_group', list(group_reqs)),
        ('protection_specs', [
            [spec.name, list(spec.match_key)] for spec in specs
        ]),
        ('frozen_membership_version', membership.membership_version),
        ('thresholds', sorted(
            (g, _threshold_to_json(t)) for g, t in threshold_table.items()
        )),
        ('group_statuses', [
            [s.group, s.status, s.reason,
             _threshold_to_json(s.threshold), s.min_coverage,
             s.optimistic_coverage, s.joint_coverage]
            for s in statuses
        ]),
        ('selection_sample_set_version', sample_set.version),
        ('selection_prediction_batch_version', batch.version),
    ])
    return AbstentionPolicy(
        strategy=strategy,
        costs=costs,
        min_coverage_default=default_req,
        min_coverage_by_group=group_reqs,
        thresholds=tuple(sorted(threshold_table.items())),
        group_statuses=tuple(statuses),
        selection_sample_set_version=sample_set.version,
        selection_prediction_batch_version=batch.version,
        selection_results=tuple(selection_results),
        policy_version=policy_version,
        group_attr=None,
        mode='overlapping',
        protection_specs=tuple(specs),
        frozen_membership=membership.members,
        frozen_membership_version=membership.membership_version,
        frozen_overlapping=membership.overlapping_sample_ids,
        frozen_overlapping_weight=membership.overlapping_weight,
    )


def evaluate_policy(
    policy: AbstentionPolicy,
    holdout_batch: PredictionBatch,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
    bootstrap: bool = True,
    membership_mode: str = 'frozen',
) -> PolicyEvaluation:
    '''把已定版策略应用到独立核对资料上。

    阈值完全来自 policy（即选择资料），这里只重放逐样本决策，不因核对
    结果重新调阈值。重叠模式默认按定版时保存的归属快照（frozen）判定
    组贡献，归属组后来被修正只在 membership_drift 中记录，不改变总体
    分母与历史依据；显式传 membership_mode='current' 可按当前属性核对，
    但阈值仍然冻结，漂移信息照样报告。
    '''
    if membership_mode not in ('frozen', 'current'):
        raise ValueError("membership_mode 必须是 'frozen' 或 'current'")

    sample_set = holdout_batch.sample_set
    ctx = _decision_context(holdout_batch)
    drift: MembershipDrift | None = None
    overlapping_ids: tuple[str, ...] = ()
    overlapping_weight = 0.0

    if policy.mode == 'overlapping':
        current = build_membership(sample_set, policy.protection_specs)
        current_map = dict(current.groups_of_sample)
        frozen_map: dict[str, tuple[str, ...]] = {
            sid: policy.frozen_groups_for(sid)
            for sid in sample_set.ids
        }
        added: list[tuple[str, tuple[str, ...]]] = []
        removed: list[tuple[str, tuple[str, ...]]] = []
        changed: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        for sid in sorted(set(frozen_map) | set(current_map)):
            old = tuple(sorted(frozen_map.get(sid, ())))
            new = tuple(sorted(current_map.get(sid, ())))
            if old == new:
                continue
            if old and not new:
                removed.append((sid, old))
            elif new and not old:
                added.append((sid, new))
            else:
                changed.append((sid, old, new))
        selection_id_set = set()
        for _, ids in policy.frozen_membership:
            selection_id_set.update(ids)
        new_samples = tuple(
            sorted(set(sample_set.ids) - selection_id_set)
        )
        drift = MembershipDrift(
            added=tuple(added),
            removed=tuple(removed),
            changed=tuple(changed),
            new_sample_ids=new_samples,
            selection_membership_version=policy.frozen_membership_version,
            current_membership_version=current.membership_version,
        )
        if membership_mode == 'frozen':
            effective_map = frozen_map
        else:
            effective_map = current_map
        # 多重归属与重叠权重按实际生效的归属计算。
        ov_ids = [
            sid for sid, owned in effective_map.items() if len(owned) > 1
        ]
        overlapping_ids = tuple(sorted(ov_ids))
        overlapping_weight = sum(
            sample_set.get(sid).weight for sid in ov_ids
        )
        groups_of_samples = [
            (sample, tuple(sorted(effective_map.get(
                sample.sample_id, ()
            ))))
            for sample in sample_set.samples
        ]
        known_groups = {spec.name for spec in policy.protection_specs}
        threshold_lookup = policy.threshold_for
        unassigned_threshold = (
            policy.threshold_for(ALL_GROUPS)
            if policy.strategy == STRATEGY_UNIFORM else None
        )
        decisions = [
            _make_decision(
                sample, owned, threshold_lookup, policy.costs, ctx,
                unassigned_threshold,
            )
            for sample, owned in groups_of_samples
        ]
    else:
        decisions = _legacy_decisions(
            holdout_batch, policy.group_attr, policy.costs,
            policy.threshold_for,
        )
        known_groups = {
            s.group for s in policy.group_statuses if s.group != ALL_GROUPS
        }

    groups_present = sorted({
        g for row in decisions for g in row.groups
    } | known_groups)
    multi_id_set = set(overlapping_ids)
    group_results: list[GroupPolicyResult] = []
    for group in groups_present:
        sub = [r for r in decisions if group in r.groups]
        status = policy.status_for(group)
        if status is None:
            g_status = STATUS_UNDETERMINED
            reason = '策略未覆盖该组（选择资料中不存在），整组按拒答处理'
            threshold = None
        else:
            g_status = status.status
            reason = status.reason
            threshold = status.threshold
        group_results.append(
            _summarize(
                group, sub, g_status, reason, threshold,
                policy.min_coverage_for(group),
                bootstrap, n_replicates, seed, confidence_level,
                multi_member_ids=multi_id_set,
            )
        )

    non_ok = [g for g in group_results if g.status != STATUS_OK]
    if (
        policy.mode == 'overlapping'
        and any(s.status == STATUS_INFEASIBLE for s in policy.group_statuses)
    ):
        overall_status = STATUS_INFEASIBLE
    else:
        overall_status = STATUS_OK if not non_ok else STATUS_PARTIAL
    if overall_status == STATUS_INFEASIBLE:
        overall_reason = next(
            (s.reason for s in policy.group_statuses
             if s.status == STATUS_INFEASIBLE and s.reason),
            '各组最低覆盖要求互相冲突，无阈值组合可同时满足',
        )
    else:
        overall_reason = (
            None if not non_ok
            else '存在未确定或不可行的组: '
            + ', '.join(g.group for g in non_ok)
        )
    overall_threshold = policy.threshold_for(ALL_GROUPS)
    overall = _summarize(
        None, decisions, overall_status, overall_reason,
        overall_threshold, policy.min_coverage_default,
        bootstrap, n_replicates, seed, confidence_level,
        multi_member_ids=multi_id_set,
    )
    sum_group_weight = sum(g.total_weight for g in group_results)
    unassigned_weight = sum(
        row.weight for row in decisions if not row.groups
    )
    evaluation_version = content_version([
        ('policy_version', policy.policy_version),
        ('holdout_sample_set_version', holdout_batch.sample_set.version),
        ('holdout_prediction_batch_version', holdout_batch.version),
        ('membership_mode', membership_mode),
        ('bootstrap', [n_replicates, seed, confidence_level, bootstrap]),
    ])
    return PolicyEvaluation(
        policy_version=policy.policy_version,
        evaluation_version=evaluation_version,
        strategy=policy.strategy,
        sample_set_version=holdout_batch.sample_set.version,
        prediction_batch_version=holdout_batch.version,
        selection_sample_set_version=policy.selection_sample_set_version,
        selection_prediction_batch_version=(
            policy.selection_prediction_batch_version
        ),
        thresholds_source='selection',
        overall=overall,
        groups=tuple(group_results),
        decisions=tuple(decisions),
        membership_drift=drift,
        overlapping_sample_ids=overlapping_ids,
        overlapping_weight=overlapping_weight,
        sum_group_weight=sum_group_weight,
        unassigned_weight=unassigned_weight,
    )


def compare_policies(
    policies,
    holdout_batch: PredictionBatch,
    n_replicates: int = 1000,
    seed: int = 20260929,
    confidence_level: float = 0.95,
) -> PolicyComparison:
    '''在同一核对资料上并排比较多个定版策略（如统一阈值 vs 分组阈值）。'''
    policies = tuple(policies)
    if not policies:
        raise ValueError('比较至少需要一个策略')
    versions = [p.policy_version for p in policies]
    if len(set(versions)) != len(versions):
        raise ValueError('参与比较的策略版本必须互不相同')
    evaluations = tuple(
        evaluate_policy(
            policy, holdout_batch,
            n_replicates=n_replicates, seed=seed,
            confidence_level=confidence_level,
        )
        for policy in policies
    )
    comparison_version = content_version([
        ('holdout_sample_set_version', holdout_batch.sample_set.version),
        ('evaluations', [e.evaluation_version for e in evaluations]),
    ])
    return PolicyComparison(
        comparison_version=comparison_version,
        sample_set_version=holdout_batch.sample_set.version,
        evaluations=evaluations,
    )
