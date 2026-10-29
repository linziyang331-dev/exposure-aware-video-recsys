# 依赖文件说明

| 文件 | 用途 | 使用限制 |
| --- | --- | --- |
| [main.txt](main.txt) | 原有 Python/PyTorch 基础环境上的项目新增包 | 由 `scripts/install_dependencies.ps1` 安装到 D 盘；使用 `--no-deps` |
| [observed-python312.txt](observed-python312.txt) | 早期环境的实测依赖记录 | 历史快照，含上游兼容例外，不是通用锁文件 |
| [public-runtime-observed.txt](public-runtime-observed.txt) | 首版整理时实际使用的核心包，包括后续绘图依赖 | 核心版本参考，不含完整传递闭包；不是新机器安装成功证明 |

运行协议见 [复现说明](../docs/reproduction.md)。本轮没有安装新依赖，没有验证 Linux/CPU PyTorch 替换方案或新机器依赖解析；现有 torch CUDA 构建和 RecBole 固定范围的例外不能通过一条普通全局 pip 命令假定解决。
