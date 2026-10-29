# 面向曝光偏差的多阶段短视频推荐与离线策略优化

在 KuaiRand-Pure 上实现双塔召回、DIN 精排及统一离线评估，并分别完成真实曝光来源诊断、合成 OPE 和独立 CQL 实验。当前首版为 **v0.1.0：离线实验源码与聚合结果**，正式链路为 **双塔 V2 → Top100 → DIN V2 → Top20**。

## 核心结果

标准测试共有 75,430 个正反馈事件，召回目录包含 7,538 个训练已知视频；每事件一个相关目标，冷目标计为未命中。

| 模型/链路 | Recall@20 | NDCG@20 | Recall@100 |
| --- | ---: | ---: | ---: |
| ItemCF，全目录 | 9.6871% | 0.037764 | 30.2400% |
| 双塔 V2，全目录 | 10.3579% | 0.038619 | 32.5109% |
| 双塔 V2 → DIN V2 | **13.4482%** | **0.054840** | **32.5109%** |

DIN 在同一 Top100 上使标准 NDCG@20 相对提高 **42.01%**。真实推荐、观察性曝光分析和模拟策略价值分别报告；不把合成 CQL 收益或概率校准当作 Top20 提升。

已知不足：标准长尾 Recall@100 仅 5.35%、最终 Recall@20 为 1.59%；随机来源最终 Recall@20 为 0.5164%。下一阶段优先优化长尾召回。旧测试已被查看，这些是迭代复测成绩，未证明线上收益。V3 的 13.4549% 属于未采用实验。

## 首次运行

当前支持范围为 Windows / D 盘项目 / 已有 Python 3.12 和匹配依赖。环境入口将可控缓存、临时文件及项目新增包放到 D 盘；它不会安装 Python 或 PyTorch。新机器安装尚未验收，请先阅读 [环境前提与复现层级](docs/reproduction.md)。

```powershell
. .\scripts\enter_env.ps1
# 已有基础环境，项目新增包尚未安装时才执行：
# .\scripts\install_dependencies.ps1
python -B -m unittest discover -s tests -v
python -B scripts\demo_cpu.py --output-directory .runtime/demos/first-run
```

CPU 演示生成自己的日志、训练小规模 CQL/DQN 并重算结果，不需要真实数据和旧模型。它用于检查流程，不能复现正式预算的模型质量。源码仓库不包含真实数据、权重或逐事件产物；正式训练、精排和冻结资产验收的前提见运行说明。

## 阅读入口

| 内容 | 入口 |
| --- | --- |
| 目录结构、正式链路与实验模型 | [项目结构](docs/architecture.md) |
| 安装边界、CPU 演示、数据/训练与资产验收命令 | [运行与复现](docs/reproduction.md) |
| 最终 Top20、长尾瓶颈及结论限制 | [结果说明](docs/results.md) / [机器摘要](results/v0.1.0/summary.json) |
| 日历划分、严格历史、数据获取与许可 | [数据协议](docs/data.md) |
| 提交范围、导出/核验与首次提交 | [发布说明](docs/release.md) |
| 阶段成果和后续缺口 | [版本记录](CHANGELOG.md) |
| 上游组件、兼容补丁与引用 | [第三方声明](THIRD_PARTY_NOTICES.md) |

## 首版交付范围

包含模块源码、配置、测试、运行脚本、兼容补丁和选定的聚合证据。原始/处理后数据、模型、词表、逐事件排名/预测、依赖、缓存、机器快照及重复代码副本均不提交。发布文件由 [白名单](configs/release_v0.1.0.json) 和 `.gitignore` 共同约束。

```powershell
python -B scripts\prepare_release.py --check
python -B scripts\prepare_release.py --output-directory .runtime/releases/v0.1.0
python -B scripts\prepare_release.py --verify-directory .runtime/releases/v0.1.0
```

导出产生独立源码目录、ZIP 和内容哈希清单，不执行远程上传。现有目录拒绝覆盖。历史报告中的 PASS 表示当时本机资产验收，公开聚合快照不等于已恢复模型。新环境完整恢复、线上部署和 GitHub 远程发布仍待完成。

项目原创代码首版采用 [MIT](LICENSE)；RecBole 与数据保留各自许可，详见第三方声明。
