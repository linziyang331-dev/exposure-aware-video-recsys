# 上游来源与许可范围

项目原创代码的首版采用 [MIT](LICENSE)。以下软件、数据、论文与模型方法保留各自权利；项目许可不替代上游条款。

## RecBole 1.2.1

使用 [RUCAIBox/RecBole v1.2.1](https://github.com/RUCAIBox/RecBole/tree/v1.2.1) 的 DIN、LR、SASRec、嵌入/注意力/Dice/MLP、训练与配置组件。通过 import 复用，没有将整个库复制进仓库。RecBole 许可为 MIT，保留 [上游许可全文](licenses/RecBole-MIT.txt)，版权为 2020 RUCAIBox。

项目兼容边界为 `src/compat/recbole.py` 与 `patches/recbole-1.2.1-pandas3.patch`。补丁只用于项目本地安装，修复 pandas 缺失值赋值与转换后共享数组的问题，未覆盖全局包。补丁包含上游少量上下文，适用上游许可。运行记录中的源码哈希只标识实际安装来源，不冒充远程逐字节核验。

## 数据

KuaiRand 数据及条款见 [数据说明](docs/data.md) 和 [官方主页](https://kuairand.com/)。仓库不重新分发真实逐条数据或用项目 MIT 重新授权数据。RecBole 示例使用其提供的 MovieLens-100K，只用于框架 smoke 验证，须遵守 [GroupLens 使用条件](https://grouplens.org/datasets/movielens/100k/)，不作为 KuaiRand 成绩。

## 方法参考

| 来源 | 本项目使用方式 |
| --- | --- |
| [DIN](https://arxiv.org/abs/1706.06978) / [SASRec](https://arxiv.org/abs/1808.09781) | 通过 RecBole 复用模型组件或模型；本项目补充数据/候选/评测适配 |
| [Open Bandit Pipeline](https://github.com/st-tech/zr-obp) | 对照 OPE 公式和接口，独立 NumPy 实现；未安装或复制该包 |
| [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601) | 合成单动作 DR 估计 |
| [CQL](https://arxiv.org/abs/2006.04779) | 独立 PyTorch 离散 CQL(H) 原型，未复制上游实现 |
| [Pearl](https://github.com/facebookresearch/Pearl) | 参考公开 API/保守损失接口；未安装、未声称通过其运行兼容测试 |
| [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) | 结构/特征连接参考；未安装或复制该库 |

其他依赖如 PyTorch、NumPy、pandas、SciPy 和 matplotlib 通过环境导入，仓库不分发其二进制。版本范围与兼容例外见 [复现说明](docs/reproduction.md)。引用上游论文不等同于复现其论文指标。
