"""机器学习评估平台：样本集、预测批次与指标计算基础。

当前范围：对本地提供的分类预测做总体与分组评估。
训练与在线模型不在范围内。
"""

from .samples import Sample, SampleSet
from .predictions import Prediction, PredictionBatch, AlignmentSummary, EvaluableRow
from .metrics import binary_metrics, multiclass_metrics, confusion_counts
from .resample import bootstrap_ci, bootstrap_scored_accuracy
from .evaluation import Accounting, ScopeReport, evaluate_batch
from .compare import ComparisonReport, compare_reports
from .predictions import ScoredRow

__all__ = [
    "Sample",
    "SampleSet",
    "PredictionBatch",
    "AlignmentSummary",
    "EvaluableRow",
    "ScoredRow",
    "binary_metrics",
    "multiclass_metrics",
    "confusion_counts",
    "bootstrap_ci",
    "bootstrap_scored_accuracy",
    "ScopeReport",
    "Accounting",
    "evaluate_batch",
    "ComparisonReport",
    "compare_reports",
]

__version__ = "0.1.0"
