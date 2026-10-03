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

主成绩默认使用明确的 `--scope all_samples`（全体参与范围）：缺预测、冲突预测、
未知类别都以拒答伪类别留在主指标、分组指标和区间中。`--scope evaluable` 只作
诊断视角，报告中始终带有 `scope: evaluable`，不能冒充默认成绩。

## 分组拒答策略定版

研究者可按组设置最低覆盖要求，并设置错误接受、正确接受、拒答三种代价：

```bash
python -m ml_eval \
  --samples examples/samples.csv \
  --selection-predictions selection.csv \
  --audit-predictions audit.csv \
  --policy grouped_threshold --group region \
  --cost-false-accept 2 --cost-correct-accept 0 --cost-abstain 0.5 \
  --min-coverage x=0.8 --min-coverage y=0.6 \
  --min-samples z=20 \
  -o policy.json
```

阈值扫描只读取 `selection`：候选阈值来自选择资料中实际出现的分数，
`grouped_threshold` 为每组独立选择，`uniform_threshold` 强制全体共用一个阈值。
缺分数、缺预测、冲突预测、未知类别都进入拒答与代价分母。若某组资料人数不足，
该组保留 `undetermined` 与具体原因；若没有阈值能达到覆盖要求，则保留
`infeasible`、可达到的最高覆盖率和全拒答保守后备，不会删除该组粉饰总体结果。

`audit` 只冻结应用选择阶段的阈值并重新计算人数、权重、覆盖率、加权代价及
bootstrap 区间；核对覆盖率不满足约束只会标记为 `infeasible`，不会重新换阈值。
`policy_version` 对代价、覆盖要求、选择样本集和预测批次做内容寻址：更换任一
参数都会形成新策略，旧策略及其核对结果仍可通过原版本追溯。

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
