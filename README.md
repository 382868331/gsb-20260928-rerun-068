# 机器学习评估平台（首轮）

首轮范围只包含本地分类预测批次评估：样本集、预测批次、分组指标、加权指标、数据质量统计，以及固定种子的自助法置信区间。训练模型、在线推理和模型管理不在当前范围内。

## 数据模型

- `SampleSet`：以 `sample_id` 作为唯一身份，保存真实类别、正权重和多个分组字段。
- `PredictionBatch`：保存预测行的原始批次和批次版本，不预先删除重复行。
- 对齐结果：记录参与计算的 `scored_sample_ids`、样本集版本、预测批次版本和对齐版本。

## 异常处理

- 缺预测：样本保留在数据质量报告中，但不进入指标分母。
- 重复预测：该样本的所有预测不参与指标计算，样本 ID 和样本权重单独报告。
- 未知类别：未知预测标签仍进入指标分母并计为错误，避免删除难例后抬高结果。
- 未知样本：预测中存在但样本集中没有的 ID 单独统计。

## 指标口径

- 总指标使用样本权重计算加权准确率和错误率。
- 分组指标在每个组内独立加权计算，不把组结果与总体加权结果混用。
- `group_macro_accuracy` 是组间简单平均；总体准确率不是组平均。
- 分类报告包含每类 precision、recall、F1、support，以及 macro F1 和 weighted F1。

## 可复现实验证

样本集、预测批次、对齐结果和评估结果都带 SHA-256 版本号。自助法区间使用 Python 标准库 `random.Random(seed)` 并按对齐后的固定样本顺序抽样；相同样本、指标、迭代数、种子和置信度会得到相同结果版本。

## 命令行

```bash
.venv/bin/python -m mleval.cli examples/samples.csv examples/predictions.csv
.venv/bin/python -m mleval.cli examples/samples.csv examples/predictions.csv --bootstrap-iterations 1000 --bootstrap-seed 7
```

## 手算核对

示例中 `s1,s2,s3` 参与计算，`s4` 因重复排除，`s5` 因缺预测排除。参与权重为 `1+1+2=4`，正确权重为 `1+2=3`，总准确率为 `3/4=0.75`。north 组准确率为 `1/2=0.5`，south 可计算组只有 `s3`，准确率为 `1.0`，组简单平均为 `(0.5+1.0)/2=0.75`。
