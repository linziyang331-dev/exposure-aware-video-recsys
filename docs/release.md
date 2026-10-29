# v0.1.0 首次提交说明

版本日期：2026-10-29。定位为“已运行的离线推荐实验与评估源码首版”，正式模型为双塔 V2＋DIN V2；长尾优化尚未开展。发布源文件由 [显式白名单](../configs/release_v0.1.0.json) 决定。

## 提交与保留范围

- 提交：模块源码、配置、测试、运行/交付脚本、兼容补丁、公开文档、上游声明、聚合结果，以及少量旧选模/哈希元数据和聚合图表。
- 本机保留但不提交：原始/处理后数据、词表、逐事件排名和预测、模型、完整实验、依赖目录、缓存、机器快照、安装日志、TensorBoard、阶段计划与旧代码副本。
- `experiments/ranking_v2/selection.json` 是历史选模元数据；`reports/` 中的 PASS 和哈希属于当时完整本地资产的历史记录。它们不是当前克隆已重算的证明，也不能恢复缺失资产。
- `results/v0.1.0/summary.json` 是选择聚合字段的首版结果快照，记录来源文件摘要；`release_manifest.json` 只校验公开发布文件内容，不证明模型效果。

## 本机生成和核验

```powershell
. .\scripts\enter_env.ps1
python -B scripts\prepare_release.py --check
python -B scripts\prepare_release.py --output-directory .runtime/releases/v0.1.0
python -B scripts\prepare_release.py --verify-directory .runtime/releases/v0.1.0
```

生成目录及相邻 ZIP 都位于 D 盘。已存在目标拒绝覆盖；需要重建时换一个新目录名。工具检查相对路径、链接/目录联接、文件类型/体积、常见密钥形态、个人用户目录及公开 Markdown 的本地链接；它不代替人工审查全部内容。

`.gitignore` 也按白名单组织，用于防止在完整工作区误暂存资产；**首次提交建议使用生成的干净目录**。生成过程不执行 Git、不创建远程、不上传。整理交付时可以在干净目录本地 `git init -b main` 和 `git add --all`，先检查 `git diff --cached --stat`，再由作者确定身份并作首次提交。

建议首条提交信息：`chore: prepare v0.1.0 offline recommendation baseline`。创建远程仓库、公开可见性、提交和 push 分别处理，不在本轮整理中执行。

## 验收范围

首版在完整工作区和无数据/权重的导出目录分别运行测试与 CPU 演示，核对 Git 暂存集合等于发布清单。导出目录仍复用本机 D 盘依赖，因此不等于全新环境安装验收。[本轮验证结果](release_validation.md) 记录范围和跳过项；实际文件数/体积见生成的 `release_manifest.json`。ZIP SHA-256 由本机交付记录另存；不把历史阶段测试数相加。

后续版本优先：长尾召回诊断与独立优化 → 新教师/候选匹配 DIN → 最终 Top20 稳定性验收。新环境恢复、依赖解析与线上部署另列任务。
