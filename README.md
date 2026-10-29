# 面向曝光偏差的多阶段短视频推荐与离线策略优化

基于260万条KuaiRand-Pure日志，实现双塔召回 → Top100 → DIN精排 → Top20，并提供曝光来源诊断、合成OPE和独立CQL/DQN实验。**v0.2.0新增长尾根因诊断、邻域共享表示召回及新教师DIN适配。**

## 长尾优化结果

标准测试75,430个正反馈事件，目录7,538个训练已知视频。每事件一个相关目标，冷目标计为未命中；采用严格过去的标准正反馈历史。

| 主种子标准测试 | 原双塔V2＋DIN V2 | 新长尾召回＋适配DIN |
| --- | ---: | ---: |
| 最终Recall@20 | 13.4482% | **14.1840%** |
| 最终NDCG@20 | 0.054840 | **0.057311** |
| 长尾Recall@100 | 5.3507% | **8.5168%** |
| 最终长尾Recall@20 | 1.5914% | **2.3565%** |

完成20组配置验证和多种子复测。在新模型自己的同一Top100池上，DIN将Recall@20从10.7557%提高到14.1840%。三个精排种子共享各自冻结召回seed42，不能解释为三条独立完整链路。

随机来源总体Recall@20三种子均值从0.5222%降至0.5023%；标准长尾仍有超过91%的目标未进入Top100。新模型保留为实验候选，原`FrozenPipeline`默认基准未替换。测试集已用于迭代复测，未证明线上收益；新模型未重新拟合概率校准。V3的13.4549%属于历史未采用实验。

## 首次运行

验收范围为Windows、D盘项目、已有Python 3.12和匹配依赖。环境入口把可控缓存和临时文件放在D盘，不安装Python或PyTorch。请先阅读[环境与复现前提](docs/reproduction.md)。

```powershell
. .\scripts\enter_env.ps1
python -B -m unittest discover -s tests -v
python -B scripts\demo_cpu.py --output-directory .runtime/demos/first-run
```

CPU演示使用自己生成的数据，不需要真实数据或旧权重。正式训练与冻结资产复核需要本地历史数据、教师及选型记录；公开仓库不包含这些资产，也没有验收新机器一键重建全部历史实验。

## 阅读入口

| 内容 | 入口 |
| --- | --- |
| 方法、根因、效果和限制 | [长尾优化](docs/longtail.md) / [v0.2.0聚合摘要](results/v0.2.0/summary.json) |
| 结构与模型职责 | [项目结构](docs/architecture.md) |
| 环境、CPU演示、训练与资产前提 | [运行与复现](docs/reproduction.md) |
| 旧正式基准与历史结果 | [结果说明](docs/results.md) / [v0.1.0摘要](results/v0.1.0/summary.json) |
| 时间划分、严格历史和许可 | [数据协议](docs/data.md) |
| 发布范围与验证 | [发布说明](docs/release.md) / [验证记录](docs/release_validation.md) |
| 版本与引用 | [变更记录](CHANGELOG.md) / [第三方声明](THIRD_PARTY_NOTICES.md) |

发布包含源码、配置、测试、运行脚本和聚合证据。数据、模型、词表、逐事件预测、依赖和缓存保留在本地。发布范围由[白名单](configs/release_v0.2.0.json)约束；原创代码采用[MIT](LICENSE)，数据及上游组件保留各自许可。
