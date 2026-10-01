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
  `scored_rows` 把每个样本展开成恰好一行计分行（状态 + 是否计对），是
  主口径与对账互相核对的共同基础。
- `ml_eval/metrics.py`：加权准确率、每类 precision/recall/F1、macro 与
  weighted 平均、加权混淆矩阵。平均方式显式命名，分组指标在组内独立计算，
  总体指标不是组指标的平均。这些是**诊断视角**，只在可评估样本上计算。
- `ml_eval/resample.py`：固定种子（默认 `20260929`，numpy PCG64）的加权
  bootstrap 置信区间。主口径区间 `bootstrap_scored_accuracy` 在全样本
  计分行上重采样（`scope=all_samples`）；诊断区间 `bootstrap_ci` 只在
  可评估行上计算（`scope=evaluable_only`）。区间结果携带 `scope` 与
  `n_rows`，与对账计数可直接核对。
- `ml_eval/evaluation.py`：组装总体/分组报告。**主口径（headline）一律使用
  `all_samples` 参与范围**：总体、分组与重采样都覆盖样本集中的每个样本，
  缺预测/冲突/未知类别按错误计，最显眼的成绩不会掩盖未覆盖样本；仅在
  可评估样本上计算的指标单独命名为 `diagnostic`（`evaluable_only`）。
  输出 `sample_set_version`、`prediction_batch_version`、
  `participating_version` 三个版本号，并给出 `accounting` 全样本对账：
  `weight_disposition` 列出每一份权重的去向（evaluable/missing/
  conflicting/unknown_label），四者之和等于总权重。
- `ml_eval/compare.py`：`compare_reports` 多模型比较。主口径为
  `all_samples`，覆盖率（可评估权重占比）与 `evaluable_only` 诊断准确率
  一并列出；覆盖率不一致时给出警告，提示不要用诊断口径直接排名。
  分组表取各模型分组的并集，某模型没有样本的分组显式出现（单元格为
  `null`），不会被省略。

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

多模型比较（`--predictions` 可重复，用 `name=path` 命名）：

```bash
python -m ml_eval \
  --samples examples/samples.csv \
  --predictions modelA=runs/a.csv \
  --predictions modelB=runs/b.csv \
  --group region -o comparison.json
```

## 可手算核对（examples）

参与计算：s1、s2、s3、s4、s5（s6 缺预测，s7 预测为未知类别 Z，s5 出现一次
内容一致的重复，折叠为一条，s999 为未知身份）。**主口径（all_samples）**：
总权重 10，正确权重 5，准确率 5/10，s6、s7 按错误计；总体与分组区间都在
这 7 个样本上重采样。**诊断口径（evaluable_only）**：可评估权重 8，
准确率 5/8；两类口径的权重账目在 `accounting.weight_disposition` 中闭合；
A 类 P/R/F1 = 0.8 / (4/6) / (8/11)，B 类 P/R/F1 = 1/3 / 1/2 / 0.4；
macro F1 = (8/11 + 0.4)/2，weighted F1 = ((8/11)*6 + 0.4*2)/8。
组 x 主口径准确率 4/7（含 s7 按错），诊断口径 4/6；组 y 主口径 1/3
（含 s6 按错），诊断口径 1/2，均独立计算。

若把 s5 的重复改成冲突（如一次 A、一次 B），无论两条记录的先后顺序如何，
s5 都会被标记为冲突样本排除（`conflicting_rows` 记录全部冲突取值），指标与
三个版本号保持一致；其权重 3 计入 `total_weight_conflicting` 并在全样本
准确率中按错误处理。

当重复、冲突预测与未知类别同时出现时，下列账目必须互相闭合（有测试锁定）：
计数上 `n_evaluable + n_missing + n_conflicting + n_unknown_label == n_samples`；
权重上四类去向之和等于总权重；`scored_rows` 每个样本恰好一行且状态计数与
summary 一致；主口径区间 `confidence.scope == "all_samples"` 且
`n_rows == n_samples`，其 `point` 等于 `overall.accuracy` 与
`accounting.accuracy_all_samples`。

## 测试

```bash
python -m pytest -q
```
