# 运行与复现说明

当前验证平台为 Windows、Python 3.12.10、PyTorch 2.6.0+cu124、RecBole 1.2.1。训练/正式冻结接口使用 RTX4060 Laptop GPU。公开源码不附带真实数据、模型和逐事件排名。

## 环境前提

将项目放在 D 盘，准备已有 Python 3.12 和基础依赖。入口将可控缓存、临时目录和项目新增依赖放在项目 `.runtime/`；它不会创建 Python，也不会安装 PyTorch。CPU 演示和单元测试不要求 CUDA，但正式训练配置默认使用 CUDA。

基础依赖至少涉及 PyTorch、NumPy、pandas、SciPy、scikit-learn、PyYAML、psutil、requests、packaging、matplotlib。文件职责和首版核心实测版本见 [依赖说明](../requirements/README.md)；历史 [observed-python312.txt](../requirements/observed-python312.txt) 记录早期环境，**不是可直接由解析器安装的通用锁文件**。`main.txt` 仅记录原有基础环境上的新增包。

```powershell
. .\scripts\enter_env.ps1
python --version
# 仅当项目新增包尚未安装时执行；目标为 D 盘项目 .runtime/python312-packages
.\scripts\install_dependencies.ps1
```

安装脚本使用显式 `--target`、`--no-deps`，因此基础依赖必须事先准备好。RecBole 的 Ray/colorama 固定范围有已记录的兼容例外；脚本应用本地 pandas 3 补丁。首版没有重新执行下载或安装，也没有验证全新机器安装。请勿把这两份 requirements 当作一条通用全局 pip 安装命令。

## 首次运行：无需真实数据或旧模型

```powershell
. .\scripts\enter_env.ps1
python -B -m unittest discover -s tests -v
python -B scripts\demo_cpu.py --output-directory .runtime/demos/first-run
```

测试覆盖数据时间协议、模型数学/梯度/训练、候选与标签隔离、OPE 支持、CQL、交付哈希与路径保护。部分 CPU 小训练依赖 RecBole 及其本地兼容补丁；完整测试仍要求上述已有依赖。真实符号链接测试在 Windows 未获创建权限时会显示跳过。

演示生成自己的短会话，完成少量 CPU CQL/DQN 训练和保存结果重算，并运行手算可核对的单动作 OPE。输出包含实际检查点，但仅在忽略目录中。它是流程演示，预算小于正式实验，不能复现或证明正式报告中的效果数值。完成目录不可覆盖，第二次运行请换目录名。

`scripts/check_environment.py` 是原工作机包清单不变性审计，依赖未公开的安装前机器快照；它不是新克隆的安装检查入口。

## 获取数据并训练选定双塔

先阅读 [数据说明](data.md)。以下下载需要网络并写入 D 盘项目，首版整理未重新执行下载。

```powershell
. .\scripts\enter_env.ps1
python -B -m src.data.download
python -B -m src.data.inspect_dataset
python -B -m src.data.prepare
python -B -m src.data.validate
python -B -m src.data.report
python -B scripts\train_retrieval_v2.py --config configs/retrieval_v2_selected.yaml --output-directory experiments/retrieval_v2_new/seed42 --seed 42
```

数据准备和验证全量访问真实数据；需要足够磁盘、内存和时间。处理或训练目录发生签名不一致时不会自动覆盖。该训练命令使用冻结的选定配置和训练/验证数据，输出新的 `best.pt` 与曲线；它不会自动重建全部历史调参台账、测试预测、精排教师和候选缓存。

## 需要完整本地资产的入口

下列入口是**已有冻结资产的复核/复现入口**，不是无资产源码克隆的一键全链路命令：

```powershell
python -B -m src.retrieval_v2.verify
python -B scripts\train_ranking_v2.py --output-directory experiments/ranking_v2_new/seed42 --seed 42
python -B scripts\train_sequential.py --output-directory experiments/sequential_new/seed42 --seed 42
python -B -m src.exposure.run --phase all
python -B -m src.offline_rl.run --phase all
python -B -m src.system_evaluation.run --phase all
```

它们依赖本地历史 manifest、selection、数据、教师、候选或报告签名。公开的聚合快照不能替代这些资产；缺少文件或哈希不符时应失败。合成 CQL 的正式 runner 也绑定旧阶段证据，纯源码试运行请使用上面的 `demo_cpu.py`。

首版已验证本机算法和资产的同机复查；公开源码目录的测试/演示验收仍复用这台机器已有依赖。**从全新环境恢复所有历史冻结资产的一键脚本、新机器全链路重训和线上部署均未完成。**
