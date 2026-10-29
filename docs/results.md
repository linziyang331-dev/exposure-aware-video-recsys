# v0.1.0 结果与结论边界

结果来自本机已完成实验的聚合快照，主流程是双塔 V2 seed42 → 固定 Top100 → DIN V2 seed42 → Top20。机器可读摘要见 [summary.json](../results/v0.1.0/summary.json)。

## 推荐效果

| 标准测试 | Recall@20 | NDCG@20 | Recall@100 |
| --- | ---: | ---: | ---: |
| 热门度，全目录 | 6.1355% | 0.025171 | 19.7309% |
| ItemCF，全目录 | 9.6871% | 0.037764 | 30.2400% |
| 双塔 V2，全目录 | 10.3579% | 0.038619 | 32.5109% |
| 双塔 V2 → DIN V2 | **13.4482%** | **0.054840** | **32.5109%** |

DIN 在同一 Top100 内重排，标准 NDCG@20 相对双塔原顺序提高 42.01%。它不是全目录 DIN 与全目录双塔的独立模型对比。排序版标准曝光 AUC 为 0.719374，低于旧纯 DIN 的 0.745358；不能同时宣称 CTR 和 Top20 全面改善。

标准测试分母为 75,430 个正反馈事件，每个事件只有一个相关目标；允许重复推荐，冷视频目标保留为未命中。全部训练已知视频参与召回，禁止向候选池注入目标。测试在此前阶段已查看，本轮是迭代复测，不是全新盲测或线上 A/B 测试。

随机来源测试的正式链路 Recall@20 为 0.5164%、NDCG@20 为 0.001856，Recall@100 为 2.2298%。跨来源表现仍弱。

## 主要瓶颈

标准测试有 50,907 个目标未进入 Top100，占 67.49%。标准长尾正目标 18,035 条，仅 965 条进入 Top100（5.35%），287 条进入最终 Top20（1.59%）。头部按训练期正反馈次数最高的 20% 视频定义，其余已知视频为长尾，冷视频另列。

下一轮优先优化长尾召回，再匹配新教师与候选重新适配 DIN；以最终 Top20 验收。详情见 [优化路线](../reports/system_optimization_roadmap.md)。V3 的 Recall@20=13.4549% 属于未采用实验；相对正式 DIN 的增益区间跨零。

![候选损失漏斗](../experiments/system_evaluation_main/figures/candidate_funnel.png)

## 其他模块

- SASRec 标准全目录 Recall@20 为 9.1489%、NDCG@20 为 0.035887，未替代主链路；长尾与收敛仍需改进。
- 随机验证集拟合的单调校准将随机测试 LogLoss 从 0.575784 降至 0.446079，但应用于标准来源会退化；校准不改变排名，不是 Top20 增益。
- 真实日志没有已确认的逐事件 propensity，因此真实因果 IPS/DR 价值未被识别。IPS/SNIPS/DM/DR 和加权训练的收益限定已知倾向概率的合成实验。
- 合成狭窄覆盖、2,000 会话条件下，三次运行平均精确折扣回报：离线 DQN=3.680326，主 CQL=5.410894；它不是 KuaiRand 真实 RL 收益。
- RTX4060 Laptop GPU 上预构造上下文到 Top20 的暖推理 p50≈4.27ms、p95≈7.18ms；不含历史查询、网络、并发或冷启动，不能作为线上 SLO。

详细报告：[召回](../reports/retrieval_optimization_report.md)、[精排](../reports/ranking_v2_report.md)、[SASRec](../reports/sequential_report.md)、[曝光](../reports/exposure_evaluation_report.md)、[CQL](../reports/offline_rl_report.md)、[统一评估](../reports/system_evaluation_report.md)。这些是历史实验报告，原有测试计数表示对应阶段；首版新增交付检查另见 [发布说明](release.md)。
