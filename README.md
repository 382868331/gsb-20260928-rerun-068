# 机器学习评估平台（首轮：分类预测分组评估）

本首轮只建立评估基础设施并对**本地提供的分类预测**做评估；训练与在线模型
不在范围内。后续新指标可以直接复用样本身份、权重与分组信息。

## 核心概念

- `ml_eval/samples.py`：`SampleSet`/`Sample`。样本身份 `id` 唯一，携带真实标签、
  非负权重与任意分组属性；样本集有内容版本号。
- `ml_eval/predictions.py`：`PredictionBatch`。预测对齐到样本集，并把
  **缺预测、重复预测、未知样本身份、未知预测类别**分别计数保留，只有合法行
  进入指标计算，难例不会被先删掉。重复处理与输入顺序无关：同一身份的一致
  重复折叠为一条；同一身份给出**冲突预测**（类别不同）时不采用先到先得，
  该身份被明确标记为冲突样本、排除出可计算集合，并在全样本口径中按错误计。
- `ml_eval/metrics.py`：加权准确率、每类 precision/recall/F1、macro 与
  weighted 平均、加权混淆矩阵。平均方式显式命名，分组指标在组内独立计算，
  总体指标不是组指标的平均。
- `ml_eval/resample.py`：固定种子（默认 `20260929`，numpy PCG64）的加权
  bootstrap 置信区间；重采样版本包含参与行的身份、真值、预测与权重，与批次
  版本共同定位参与计算的数据版本。分组统计与重采样使用同一个可评估集合。
- `ml_eval/evaluation.py`：组装总体/分组报告，输出 `sample_set_version`、
  `prediction_batch_version`、`participating_version` 三个版本号，并给出
  `accounting` 全样本对账：缺预测/冲突/未知类别的权重分母与全样本准确率
  （异常样本按错误计），避免剔除异常后总体指标虚高。
- `ml_eval/comparison.py`：多模型并排比较。每个模型先独立走同一套
  `evaluate_batch` 口径，比较层只汇总不重算：`coverage`（可评估权重 /
  全样本权重）把覆盖率不一致摆到明面，可评估口径（诊断视角）与全样本口径
  并列展示；分组比较表取各模型分组的并集，某模型没有可评估行的组也显式
  出现（指标为 null，权重去向在该模型的组级 `accounting` 中列明）。
  参与比较的批次必须对齐同一样本集版本。

## 两种准确率口径

- `accuracy_evaluable`：仅在可评估样本上计算的诊断视角；
- `accuracy_all_samples`：全样本口径，缺预测/冲突/未知类别一律按错误计。

总体与每个分组都带 `accounting`，列出 total/evaluable/correct/excluded 及
missing/conflicting/unknown_label 的权重去向，分组账目加总等于总体账目；
没有任何可评估行的分组仍然呈现（`metrics` 与置信区间为 null）。

## 本地使用

样本 CSV：`id,label,weight,g_<分组名>...`；预测 CSV：`id,predicted`。也支持 JSON。

```bash
python -m ml_eval \
  --samples examples/samples.csv \
  --predictions examples/predictions.csv \
  --group region \
  --bootstrap-metric accuracy --replicates 1000 --seed 20260929 \
  -o report.json

# 多模型比较（覆盖率不一致时并排核对）
python -m ml_eval \
  --samples examples/samples.csv \
  --model model_a=examples/predictions.csv \
  --model model_b=other_predictions.csv \
  --group region -o comparison.json
```

## 可手算核对（examples）

参与计算：s1、s2、s3、s4、s5（s6 缺预测，s7 预测为未知类别 Z，s5 出现一次
内容一致的重复，折叠为一条，s999 为未知身份）。总权重 8，正确权重 5，
可评估口径加权准确率 5/8；全样本口径（总权重 10，s6、s7 按错误计）准确率
为 5/10，两类口径的权重账目在 `accounting` 中闭合；
A 类 P/R/F1 = 0.8 / (4/6) / (8/11)，B 类 P/R/F1 = 1/3 / 1/2 / 0.4；
macro F1 = (8/11 + 0.4)/2，weighted F1 = ((8/11)*6 + 0.4*2)/8。
组 x 准确率 4/6，组 y 准确率 1/2，均独立计算。

若把 s5 的重复改成冲突（如一次 A、一次 B），无论两条记录的先后顺序如何，
s5 都会被标记为冲突样本排除（`conflicting_rows` 记录全部冲突取值），指标与
三个版本号保持一致；其权重 3 计入 `total_weight_conflicting` 并在全样本
准确率中按错误处理。

## 测试

```bash
python -m pytest -q
```
