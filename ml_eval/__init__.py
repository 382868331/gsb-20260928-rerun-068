"""机器学习评估平台：样本集、预测批次与指标计算基础。

当前范围：对本地提供的分类预测做总体与分组评估。
训练与在线模型不在范围内。
"""

from .samples import Sample, SampleSet
from .predictions import Prediction, PredictionBatch, AlignmentSummary, EvaluableRow
from .metrics import binary_metrics, multiclass_metrics, confusion_counts
from .resample import bootstrap_ci
from .evaluation import Accounting, GroupReport, evaluate_batch
from .comparison import ComparisonReport, ModelEntry, compare_batches
from .policy import (
    AbstentionPolicy,
    CostModel,
    DecisionRow,
    GroupPolicyResult,
    GroupStatus,
    PolicyComparison,
    PolicyEvaluation,
    compare_policies,
    evaluate_policy,
    fit_abstention_policy,
)

__all__ = [
    "Sample",
    "SampleSet",
    "Prediction",
    "PredictionBatch",
    "AlignmentSummary",
    "EvaluableRow",
    "binary_metrics",
    "multiclass_metrics",
    "confusion_counts",
    "bootstrap_ci",
    "GroupReport",
    "Accounting",
    "evaluate_batch",
    "ComparisonReport",
    "ModelEntry",
    "compare_batches",
    "AbstentionPolicy",
    "CostModel",
    "DecisionRow",
    "GroupPolicyResult",
    "GroupStatus",
    "PolicyComparison",
    "PolicyEvaluation",
    "fit_abstention_policy",
    "evaluate_policy",
    "compare_policies",
]

__version__ = "0.1.0"
