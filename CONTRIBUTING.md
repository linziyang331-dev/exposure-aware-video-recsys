# 开发约定

请先阅读 [运行说明](docs/reproduction.md) 和 [结果边界](docs/results.md)。本项目首版验证平台为 Windows / D 盘 / Python 3.12；尚未承诺 Linux 或新机器安装可用。

模型优化使用新的实验目录和版本，保留冻结双塔 V2/DIN V2。标准训练/验证用于选型，既有测试属于迭代复测，不能称为新的盲测；主验收指标为最终 Recall@20 和 NDCG@20，同时记录 Recall@100、头尾分组、覆盖率与成本。

提交前运行 `python -B -m unittest discover -s tests -v` 和 `python -B scripts/prepare_release.py --check`。新公开文件必须进入发布白名单及对应 `.gitignore` 例外；不要提交数据、权重、逐事件产物、个人目录或凭据。

报告问题时提供 Python/依赖版本、命令、预期行为与精简错误信息；不附真实用户数据或个人机器快照。提出效果改进时说明候选空间、分母、训练种子、选型协议及资源预算。
