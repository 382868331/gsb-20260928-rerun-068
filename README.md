# 机器学习评估平台（首轮：分类预测分组评估）

本首轮只建立评估基础设施并对**本地提供的分类预测**做评估；训练与在线模型
不在范围内。后续新指标可以直接复用样本身份、权重与分组信息。

## 核心概念

- `ml_eval/samples.py`：`SampleSet`/`Sample`。样本身份 `id` 唯一，携带真实标签、
  非负权重与任意分组属性；样本集有内容版本号。
- `ml_eval/predictions.py`：`PredictionBatch`。预测对齐到样本集，并把
  完全相同的重复记录幂等接受，冲突预测会直接报错。所有已知样本都进入指标计算，
  缺预测和未知预测类别按错误计入，未知样本身份单独计数。
- `ml_eval/metrics.py`：加权准确率、每类 precision/recall/F1、macro 与
  weighted 平均、加权混淆矩阵。平均方式显式命名，分组指标在组内独立计算，
  总体指标不是组指标的平均。
- `ml_eval/resample.py`：固定种子（默认 `20260929`，numpy PCG64）的加权
  bootstrap 置信区间；重采样版本与批次版本共同定位参与计算的数据版本。
- `ml_eval/evaluation.py`：组装总体/分组报告，输出 `sample_set_version`、
  `prediction_batch_version`、`participating_version` 三个版本号。

## 本地使用

样本 CSV：`id,label,weight,g_<分组名>...`；预测 CSV：`id,predicted`。也支持 JSON。

```bash
python -m ml_eval \
  --samples examples/samples.csv \
  --predictions examples/predictions.csv \
  --group region \
  --bootstrap-metric accuracy --replicates 1000 --seed 20260929 \
  -o report.json
```

## 可手算核对（examples）

参与计算：s1–s7（s6 缺预测，s7 预测为未知类别 Z，二者都按错误计入；s5 的
重复记录与原预测相同；s999 为未知身份，不改变已知样本分母）。总权重 10，
正确权重 5，加权准确率 5/10；A 类 P/R/F1 = 4/5 / 4/7 / 2/3，B 类
P/R/F1 = 1/3 / 1/3 / 1/3；macro F1 = 1/2，weighted F1 = 17/30。
组 x 准确率 4/7，组 y 准确率 1/3，均独立计算，但使用同一个参与样本集合。

数据质量区列出参与样本、有效预测、缺预测和未知类别预测的数量与权重，可直接
核对：有效权重 8 + 缺预测权重 1 + 未知类别权重 1 = 参与总权重 10。

## 测试

```bash
python -m pytest -q
```
