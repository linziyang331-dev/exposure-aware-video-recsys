# 双塔 V2 固定基准

状态：`FROZEN_FOR_DOWNSTREAM`。按用户要求固定双塔 V2，后续模块以本版本为召回基准。版本清单为 [retrieval_v2_freeze.json](retrieval_v2_freeze.json)，记录89个文件的SHA-256，并已逐项核对。

## 后续统一使用的版本

| 项目 | 固定值 |
| --- | --- |
| 模型 | `two_tower_v2_seed42`，不挑选测试分数最高的种子 |
| 选定配置 | `configs/retrieval_v2_selected.yaml` |
| 权重 | `experiments/retrieval_v2/artifacts/tuning/residual_recency/best.pt` |
| 权重 SHA-256 | `742ea704ed744fdc9bb595f4d376359ea66aba369a6ed6b494b423fdc6930103` |
| 结构与目标 | 32维双塔，近期历史残差聚合，正反馈全候选softmax |
| 历史 | 最多50个严格过去标准正反馈；位置衰减0.9 |
| 评分 | 用户向量与视频向量点积 / 0.1；无视频偏置 |
| 候选 | 7,538个训练已知视频，Top100；允许重复推荐 |
| 评测 | 原时间切分、滚动历史、冷目标计为未命中及并列ID规则保持不变 |

种子43/44用于复核和不确定性记录，历史关闭模型用于消融；后续默认模型是种子42，不对三个模型进行隐式融合。

## 保留的质量结论

- 标准测试主种子Recall@20=0.103579，Recall@100=0.325109；三个种子Recall@20均值0.104989，样本标准差0.001419。
- 标准Recall@20高于当前ItemCF；NDCG差异区间跨零。覆盖率、长尾表现和随机来源Recall@20仍有不足。
- 旧测试在优化前已经查看，本轮是迭代复测，不是全新未见的独立确认。
- 完整结果及限制保留在 [双塔优化报告](retrieval_optimization_report.md)，不因后续模块加入而改写本阶段结论。

## 已固定的产物

清单覆盖数据准备清单与五个Parquet、词表、评分相关代码、选模/复核证据、选定配置、三种子及消融权重、逐查询排名、聚合结果、快照向量与候选、验证曝光候选及校验证据。原始下载文件由既有数据清单管理。

现有候选包括26,211个训练结束用户快照Top100，以及122,315条标准验证曝光Top100；不表示所有后续分区的候选已经生成。后续所需的新候选沿用本权重和上下文协议，另存到对应模块的产物目录，不覆盖冻结文件。未曝光候选不能据此获得虚构标签。

## 后续修改规则

1. 后续DIN等模块读取本版本，不在原实验目录追加召回调参或重新选种子。
2. 重训使用独立目录；如需改变模型、目标、采样、历史或数据协议，另建V3配置、产物和报告。
3. 不覆盖清单中的代码、权重、候选和阶段报告。需要改共享实现时，先保留V2可运行源码与依赖版本，再建立新版本。
4. 回头优化召回后，需要重新评测下游；若下游训练依赖候选采样或动作集合，再判断是否重建数据和重训。
5. 不将新阶段增加的测试数或指标写回冻结的阶段报告；新验证单独记录。

## 检查版本完整性

在项目根目录运行，无需重训：

```powershell
. .\scripts\enter_env.ps1
@'
import json
import yaml
from src.data.common import ROOT
from src.retrieval_v2.provenance import verify_evidence
f = json.loads((ROOT / 'reports/retrieval_v2_freeze.json').read_text(encoding='utf-8'))
cfg = yaml.safe_load((ROOT / f['config_path']).read_text(encoding='utf-8'))
verify_evidence(ROOT, f['evidence'], cfg)
print(f['version'], f['status'], len(f['evidence']['files']), 'files: PASS')
'@ | python -B -
```

本次仅固定基准，没有开展融合、DIN或其他下一阶段实验。
