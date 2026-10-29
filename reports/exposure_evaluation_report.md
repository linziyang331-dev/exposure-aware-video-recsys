# 曝光偏差评估模块：实际运行报告

**模块实现、实际实验和重算核验已完成。真实KuaiRand结果属于观察性来源诊断及概率校准；IPS/SNIPS/DR的可识别性与真值误差在合成环境验证。**

正式推荐流程仍为原双塔V2＋DIN V2，候选及Top20排序未改变。本模块不训练或部署新的真实推荐策略，不报告真实日志的无偏策略价值或CQL收益。

## 实验与识别边界

- 运行类型：完整预定义预算。
- 真实日志没有已确认的逐事件动作概率；未从is_rand、随机曝光占比或1/视频数量制造propensity。
- OPE估计对象为一次动作的期望奖励，既不是整张Top20列表的价值，也不是长期RL回报。
- 完整行为分布可用时拒绝无支持目标；仅给已选动作pscore时，支持性标为无法完全验证。
- 真实用户簇bootstrap保留事件加权分母；SNIPS在每次bootstrap中重新计算分子和分母。
- 真实测试曾被旧阶段查看，本轮为迭代诊断。校准方法预先固定，不按测试选择。

## 真实数据：冻结DIN的来源差异

| 分区/子集 | 曝光数 | 正事件查询 | 正反馈率 | AUC | LogLoss | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard/test/all | 166618 | 75430 | 0.452712 | 0.719374 | 0.609120 | 0.134482 | 0.054840 |
| random/test/all | 897427 | 160735 | 0.179106 | 0.660279 | 0.575784 | 0.005164 | 0.001856 |
| standard/test/tab_1 | 130767 | 64789 | 0.495454 | 0.669117 | 0.650444 | 0.128124 | 0.051663 |
| random/test/tab_1 | 892610 | 160685 | 0.180017 | 0.658260 | 0.578465 | 0.005165 | 0.001856 |
| standard/test/common_warm_user_item_tab1 | 125082 | 61705 | 0.493316 | 0.670050 | 0.650111 | 0.126343 | 0.050820 |
| random/test/common_warm_user_item_tab1 | 649085 | 118647 | 0.182791 | 0.668870 | 0.588993 | 0.005967 | 0.002139 |

共同暖用户数=22869，共同已知视频数=5965。共同范围不等于相同请求或相同视频曝光频率；不能将来源差值直接解释为曝光机制的因果效应。

## 共同用户-tab层标准化

共21234个共享用户-tab层，每层在两个来源中均获得相同总权重。只标准化曝光指标，不改Top20分母。

| 来源 | 纳入曝光 | 加权正反馈率 | 加权AUC | 加权LogLoss | 加权Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard/test | 125647 | 0.504112 | 0.670687 | 0.648872 | 0.228502 |
| random/test | 620085 | 0.172380 | 0.676658 | 0.571205 | 0.194693 |

## 真实反馈概率的校准对照

standard-validation校准只使用标准验证标签；random-validation校准使用更早随机反馈，属于有随机反馈可用时的校准，不是零样本跨来源泛化。两者都严格早于测试时间。尺度约束为正，反馈分数次序保持不变；正式排名仍直接引用冻结的原排名。

| 测试来源 | 概率方法 | AUC | LogLoss | Brier | ECE（15等宽桶或配置桶数） |
| --- | --- | ---: | ---: | ---: | ---: |
| standard/test | uncalibrated | 0.719374 | 0.609120 | 0.211674 | 0.021285 |
| standard/test | platt_standard_validation | 0.719374 | 0.607901 | 0.211328 | 0.013968 |
| standard/test | platt_random_validation | 0.719374 | 0.755570 | 0.271366 | 0.232683 |
| random/test | uncalibrated | 0.660279 | 0.575784 | 0.196781 | 0.220399 |
| random/test | platt_standard_validation | 0.660279 | 0.583047 | 0.199517 | 0.232045 |
| random/test | platt_random_validation | 0.660279 | 0.446079 | 0.140103 | 0.007534 |

校准的LogLoss/Brier差值95%用户簇区间（校准后减原预测，负值表示损失降低）：

| 测试来源 | 拟合来源 | LogLoss差 | 95%区间 |
| --- | --- | ---: | --- |
| standard/test | platt_standard_validation | -0.001220 | [-0.001547, -0.000925] |
| standard/test | platt_random_validation | 0.146450 | [0.142585, 0.150733] |
| random/test | platt_standard_validation | 0.007263 | [0.006878, 0.007592] |
| random/test | platt_random_validation | -0.129705 | [-0.132731, -0.126733] |

区间条件于已经拟合的模型与校准系数，不包含系数拟合、早期选模或训练的不确定性。ECE依赖桶数；各子组为描述性分析，不作多重比较校正后的显著性宣称。

## 合成环境：已知真实策略价值

有限总体含4096个上下文、10个动作、5维特征。独立日志重复40次，logging强度=[0.0, 1.5, 3.0]，样本量=[1000, 5000, 20000]，权重上限=[None, 5.0, 20.0]。

精确有限总体真值：uniform=0.369921；reward_softmax=0.550840。后者是模拟器中预定义的策略，不是真实推荐模型的收益。

reward model另用独立日志拟合。其训练只读取记录动作的反馈，不读取模拟器的完整奖励表；oracle q仅用于检验数学性质。五种误设条件均保留，不用某一胜出条件代表所有情况。

### 主表：logging强度3.0，样本量20000，目标uniform，无截断

| 条件 | 估计器 | 偏差 | RMSE | 重复间SD |
| --- | --- | ---: | ---: | ---: |
| correct_pscore_oracle_q | ips | -0.000751 | 0.013657 | 0.013810 |
| correct_pscore_oracle_q | snips | -0.000370 | 0.010317 | 0.010441 |
| correct_pscore_oracle_q | dm | 0.000070 | 0.000789 | 0.000796 |
| correct_pscore_oracle_q | dr | -0.001042 | 0.007806 | 0.007834 |
| correct_pscore_wrong_q | ips | -0.000751 | 0.013657 | 0.013810 |
| correct_pscore_wrong_q | snips | -0.000370 | 0.010317 | 0.010441 |
| correct_pscore_wrong_q | dm | -0.269921 | 0.269921 | 0.000000 |
| correct_pscore_wrong_q | dr | -0.000650 | 0.012193 | 0.012331 |
| wrong_pscore_oracle_q | ips | -0.155390 | 0.155415 | 0.002801 |
| wrong_pscore_oracle_q | snips | 0.055462 | 0.055653 | 0.004669 |
| wrong_pscore_oracle_q | dm | 0.000070 | 0.000789 | 0.000796 |
| wrong_pscore_oracle_q | dr | 0.000046 | 0.002111 | 0.002138 |
| wrong_pscore_wrong_q | ips | -0.155390 | 0.155415 | 0.002801 |
| wrong_pscore_wrong_q | snips | 0.055462 | 0.055653 | 0.004669 |
| wrong_pscore_wrong_q | dm | -0.269921 | 0.269921 | 0.000000 |
| wrong_pscore_wrong_q | dr | -0.105823 | 0.105854 | 0.002609 |
| correct_pscore_learned_q | ips | -0.000751 | 0.013657 | 0.013810 |
| correct_pscore_learned_q | snips | -0.000370 | 0.010317 | 0.010441 |
| correct_pscore_learned_q | dm | 0.151157 | 0.151161 | 0.001025 |
| correct_pscore_learned_q | dr | -0.000139 | 0.012701 | 0.012863 |

同一条件中，直接使用日志奖励均值的偏差=0.129843、RMSE=0.129899。它估计行为策略的日志均值，不应当作目标策略收益。

## 权重截断与有效样本量

下表ESS和权重来自一个预先固定种子的代表日志；RMSE来自全部独立重复。两者统计单位不同。

| 上限 | 原始ESS | 使用后ESS | 被截断比例 | 使用后最大权重 | IPS的RMSE | DR的RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 2032.1 | 2032.1 | 0.000000 | 19.999984 | 0.013657 | 0.012701 |
| 5.0 | 2032.1 | 4485.2 | 0.047800 | 5.000000 | 0.123248 | 0.095707 |
| 20.0 | 2032.1 | 2032.1 | 0.000000 | 19.999984 | 0.013657 | 0.012701 |

截断可能降低方差，也可能引入偏差；截断DR不再自动保留通常的双重稳健保证。ESS表示权重集中程度，不是独立用户数或真实数据量。无支持目标已被程序拒绝。

## Bootstrap区间的经验覆盖率

logging强度3.0、5000条记录，每个条件40次独立日志、每次200次重采样。95%是区间名义水平，不是保证。经验覆盖率同时报告Wilson区间，以体现重复次数有限。

| 条件 | 估计器 | 覆盖次数/重复数 | 经验覆盖率 | 覆盖率Wilson 95%区间 |
| --- | --- | --- | ---: | --- |
| correct_pscore_oracle_q | ips | 37/40 | 0.925 | [0.801, 0.974] |
| correct_pscore_oracle_q | dm | 39/40 | 0.975 | [0.871, 0.996] |
| correct_pscore_oracle_q | dr | 37/40 | 0.925 | [0.801, 0.974] |
| correct_pscore_oracle_q | snips | 36/40 | 0.900 | [0.769, 0.960] |
| correct_pscore_wrong_q | ips | 37/40 | 0.925 | [0.801, 0.974] |
| correct_pscore_wrong_q | dm | 0/40 | 0.000 | [0.000, 0.088] |
| correct_pscore_wrong_q | dr | 37/40 | 0.925 | [0.801, 0.974] |
| correct_pscore_wrong_q | snips | 36/40 | 0.900 | [0.769, 0.960] |
| wrong_pscore_oracle_q | ips | 0/40 | 0.000 | [0.000, 0.088] |
| wrong_pscore_oracle_q | dm | 39/40 | 0.975 | [0.871, 0.996] |
| wrong_pscore_oracle_q | dr | 38/40 | 0.950 | [0.835, 0.986] |
| wrong_pscore_oracle_q | snips | 0/40 | 0.000 | [0.000, 0.088] |
| wrong_pscore_wrong_q | ips | 0/40 | 0.000 | [0.000, 0.088] |
| wrong_pscore_wrong_q | dm | 0/40 | 0.000 | [0.000, 0.088] |
| wrong_pscore_wrong_q | dr | 0/40 | 0.000 | [0.000, 0.088] |
| wrong_pscore_wrong_q | snips | 0/40 | 0.000 | [0.000, 0.088] |

## 已知倾向概率下的加权反馈学习

实际训练了同一简化logistic反馈模型，比较普通BCE、IPS-BCE和上限5的IPS-BCE。奖励模型有意省略模拟器非线性交互，因此不能预设普通监督与加权学习必然等效。全动作真实风险在已知总体上计算，没有用该风险选超参数。

| 训练损失 | 种子数 | 真实均匀动作LogLoss均值±SD | 贪心策略真实奖励均值 | 最大目标梯度 |
| --- | ---: | --- | ---: | ---: |
| unweighted | 5 | 0.736841 ± 0.008100 | 0.349404 | 0.000001 |
| ips | 5 | 0.632044 ± 0.001990 | 0.465432 | 0.000003 |
| ips_clip5 | 5 | 0.640998 ± 0.003396 | 0.477515 | 0.000002 |

这是模拟条件成立时的去偏加权损失验证，不是对KuaiRand主模型的去偏训练结果。所有失败或退化情况均保留。

## 工程核验与复现

- 数学手算、DR精确总体期望、零权重/支持、截断、SNIPS比率、用户簇分母、加权梯度、拟合隔离、单调校准和输出篡改均有自动测试。
- synthetic/real各自保存配置、源码、输入、环境版本指纹以及全部输出哈希；完成目录拒绝不一致复用，不覆盖原实验。
- 验证器重算代表日志估计器与bootstrap、全部Monte Carlo汇总、覆盖率汇总及一个完整覆盖日志、加权学习真实风险；真实分组、校准和用户簇区间全部重新运行核对。
- 数值重算容差relative=1e-8、absolute=1e-10；这是数值一致性验证，不声明所有浮点值跨环境逐位相同。

```powershell
. .\scripts\enter_env.ps1
python -B -m src.exposure.run --phase all --output-directory experiments/exposure_reproduction
python -B -m src.exposure.run --phase verify --output-directory experiments/exposure_reproduction
python -B -m src.exposure.run --phase report --output-directory experiments/exposure_reproduction
```

独立复现依赖本项目已准备的数据、冻结DIN预测/排名及其证据文件；本轮未验收全新机器从原始下载开始的完整重建。

## 文件与边界

本次目录：experiments/exposure_evaluation。synthetic_metrics.csv、synthetic_coverage.csv、synthetic_replicates.csv位于synthetic/；真实子集与校准桶CSV位于real/。逐条模拟数组在被忽略的artifacts/内。

真实日志仍不可据此报告IPS/SNIPS/DR因果价值；CQL和完整真实状态转移仍未完成。概率校准不等于消除了曝光偏差，Top20收益也不等于一次动作期望奖励。

## 参考

- [Open Bandit Pipeline公式与接口](https://github.com/st-tech/zr-obp/blob/master/obp/ope/estimators.py)。本项目独立实现数学内核，未安装或复制该包。
- [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601)。
- [The Self-Normalized Estimator for Counterfactual Learning](https://papers.nips.cc/paper_files/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html)。
