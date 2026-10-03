"""参与范围（scope）：把全样本与可评估样本两种口径显式化。

* `all_samples`（默认）：样本集里的每一个样本都进入主成绩。缺预测、
  冲突预测、未知预测类别一律落到拒答伪类别并按错误计；
* `evaluable`：诊断视角，只包含通过全部校验的行。

主报告读取到的默认成绩（`default_score`）按显式选择的范围计算并标注；
诊断指标、分组指标与 bootstrap 区间始终基于可评估集合并单独标注，
不允许诊断集合悄悄代替主成绩。
"""

from __future__ import annotations

from dataclasses import dataclass

SCOPE_ALL = "all_samples"
SCOPE_EVALUABLE = "evaluable"
SCOPES = (SCOPE_ALL, SCOPE_EVALUABLE)
ABSTAIN_LABEL = "__abstain__"

# 行状态：用于版本指纹与对账，区分不同原因导致的拒答。
STATUS_EVALUABLE = "evaluable"
STATUS_MISSING = "missing_prediction"
STATUS_CONFLICTING = "conflicting_prediction"
STATUS_UNKNOWN_LABEL = "unknown_predicted_label"


@dataclass(frozen=True)
class ScopeRow:
    sample_id: str
    label: str
    predicted: str
    weight: float
    status: str
    score: float | None = None

    @property
    def is_abstention(self) -> bool:
        return self.status != STATUS_EVALUABLE


def validate_scope(scope: str) -> str:
    if scope not in SCOPES:
        raise ValueError(f"未知参与范围 {scope!r}，可选: {SCOPES}")
    return scope


def build_scope_rows(batch, scope: str, abstain_label: str = ABSTAIN_LABEL):
    """按给定范围构造参与行，返回 ScopeRow 列表。

    可评估行的 ```predicted``` 为真实类别，其余样本的 ```predicted```
    统一为拒答伪类别，但 ```status``` 保留缺预测/冲突/未知类别的区分，
    便于人数、权重与代价对账。
    """
    validate_scope(scope)
    quality = batch.summary
    sample_set = batch.sample_set

    missing_ids = set(quality.missing_sample_ids)
    conflicting_ids = {sid for sid, _ in quality.conflicting_rows}
    unknown_label_ids = {sid for sid, _ in quality.unknown_label_rows}

    rows: list[ScopeRow] = []
    if scope == SCOPE_EVALUABLE:
        for r in batch.evaluable_rows:
            rows.append(
                ScopeRow(
                    r.sample_id, r.label, r.predicted, r.weight,
                    STATUS_EVALUABLE, r.score,
                )
            )
        return rows

    evaluable_index = {r.sample_id: r for r in batch.evaluable_rows}
    for sample in sample_set.samples:
        sid = sample.sample_id
        if sid in missing_ids:
            status = STATUS_MISSING
        elif sid in conflicting_ids:
            status = STATUS_CONFLICTING
        elif sid in unknown_label_ids:
            status = STATUS_UNKNOWN_LABEL
        else:
            status = STATUS_EVALUABLE
        if status == STATUS_EVALUABLE:
            row = evaluable_index[sid]
            predicted = row.predicted
            score = row.score
        else:
            predicted = abstain_label
            score = None
        rows.append(
            ScopeRow(sid, sample.label, predicted, sample.weight, status, score)
        )
    return rows
