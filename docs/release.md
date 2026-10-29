# v0.2.0发布说明

2026-10-30整理长尾优化源码与聚合证据。公开范围由[v0.2.0白名单](../configs/release_v0.2.0.json)和`.gitignore`约束；[v0.1.0白名单](../configs/release_v0.1.0.json)保留为历史记录。

提交源码、配置、测试、脚本、文档、声明和聚合结果。原始/处理数据、词表、权重、逐事件排名、完整实验、依赖、缓存、机器快照和安装日志不提交。历史报告中的PASS和哈希表示当时本机完整资产验收，不能据此恢复模型，也不表示当前克隆已重算报告。

## 生成与核验

```powershell
. .\scripts\enter_env.ps1
python -B scripts\prepare_release.py --policy configs/release_v0.2.0.json --check
python -B scripts\prepare_release.py --policy configs/release_v0.2.0.json --output-directory .runtime/releases/v0.2.0
python -B scripts\prepare_release.py --verify-directory .runtime/releases/v0.2.0
```

源码目录、相邻ZIP及哈希清单位于D盘。已有目标拒绝覆盖，再次导出请换目录名。工具检查路径、目录联接、类型/体积、常见凭据形态、个人目录与Markdown本地链接，并逐字节验证目录和ZIP。工具本身不执行Git上传。

从干净导出目录暂存时，核对Git文件集合严格等于发布清单加`release_manifest.json`，然后提交到原仓库分支。清单证明文件内容一致，不证明模型效果。具体范围见[验证记录](release_validation.md)。

新方法和限制见[长尾优化](longtail.md)。默认冻结基准仍为原双塔V2＋DIN V2；新机器完整恢复、线上部署和线上收益未验证。
