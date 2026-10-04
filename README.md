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
- `ml_eval/scope.py`：参与范围显式化。主成绩 `default_score` 按显式选择的
  范围计算：`all_samples`（默认，异常样本按错误计）或 `evaluable`
  （诊断口径）。诊断指标、分组指标与 bootstrap 区间始终基于可评估集合并
  通过 `diagnostic_accuracy`/`accounting.accuracy_evaluable` 单独标注，
  诊断集合不会悄悄代替主成绩。
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

## 拒答策略（分组定版与独立核对）

`ml_eval/policy.py` 支持按分组约束定版拒答策略：

- 研究者显式给出 `CostModel`（错误接受/正确接受/拒答代价）与各组最低
  覆盖要求（单一数值或 `{组: 要求}`，`"*"` 为缺省）；`group_attr`
  可传入属性名列表（如 `["region", "risk"]`），此时一个样本可同时属于
  多个保护组（组键为 `属性=取值`）：总体账目中每个样本只计一次，各组
  分别核对该样本对本组覆盖约束的贡献，重叠组不拆成互斥组，组账目加总
  因此可能超过总体；同时属于多个组的样本必须通过其所有组的阈值
  （取最严者）才会被接受；
- `fit_abstention_policy` 只在**选择资料**上选阈值：`strategy="uniform"`
  选一个同时满足各组覆盖要求的统一阈值，`strategy="per_group"` 每组独立
  选阈值；候选阈值取选择资料中的打分值，满足覆盖要求者中取代价最小。
  组间存在重叠样本时约束互相耦合：per_group 先以“全部接受”确定各组
  可达覆盖率上限，再以坐标下降逐组降低总体代价，最后做联合约束终检；
- 资料不足（无可用打分预测）的组标记 `undetermined`，可达覆盖率低于要求
  的组（含多组约束互相冲突的组合）标记 `infeasible`，原因与可达覆盖率
  随策略保留，当前策略实际达到的各组结果保留在 `selection_results`；
  核对时这些组整组按拒答处理并留在总体账目里，不会被悄悄排除；
- 代价、覆盖要求、阈值、分组状态与选择资料版本共同决定 `policy_version`：
  更换代价或覆盖要求即形成新策略，旧结果仍可追溯；
- `evaluate_policy` 把定版策略应用到**独立核对资料**上，只重放决策，
  不因核对结果不理想而重调阈值；`compare_policies` 在同一核对资料上
  并排比较统一阈值与分组阈值策略；逐样本决策（含每个样本的全部归属组
  与所用阈值）随 `decisions` 输出；
- `AbstentionPolicy.to_dict/from_dict` 支持保存后重开：重开时按内容重算
  `policy_version` 并校验一致，重开的策略与原策略产生完全相同的逐样本
  决策；归属组后来修正会改变样本集版本，重新定版得到新策略，旧策略与其
  选择资料版本仍可追溯，总体分母不因修正而丢失样本；
- 缺预测、冲突预测、未知类别按拒答计入人数、覆盖率、代价与 bootstrap
  区间；组间无重叠时分组账目加总等于总体账目。

```python
from ml_eval import (CostModel, PredictionBatch, fit_abstention_policy,
                     evaluate_policy, compare_policies)

costs = CostModel(false_accept=2.0, correct_accept=0.0, reject=1.0)
per_group = fit_abstention_policy(selection_batch, "region", costs,
                                  {"x": 0.5, "*": 0.3}, strategy="per_group")
uniform = fit_abstention_policy(selection_batch, "region", costs, 0.5,
                                strategy="uniform")
evaluation = evaluate_policy(per_group, holdout_batch)      # 独立核对
comparison = compare_policies([per_group, uniform], holdout_batch)
```

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
