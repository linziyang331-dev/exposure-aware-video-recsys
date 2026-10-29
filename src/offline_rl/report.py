"""Report synthetic policy learning, including failures and support limitations."""
import json
from pathlib import Path
import numpy as np

from src.data.common import ROOT
from src.data.download import digest_file


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _f(value,digits=5):return '未定义' if value is None else f'{value:.{digits}f}'


def require_verified_experiment(output):
    output=Path(output);verification=_read(output/'verification.json')
    fingerprint=_read(output/'experiment/component.json')['fingerprint']
    if (verification['status']!='PASS' or verification.get('experiment_fingerprint')!=fingerprint
            or verification.get('experiment_manifest_sha256')!=digest_file(output/'experiment/component.json')):
        raise ValueError('Verification does not belong to this exact completed RL experiment')
    return verification


def _plots(config,result,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.offline_rl.experiment import method_name
    folder.mkdir(parents=True,exist_ok=True)
    scenarios=config['scenarios'];sizes=config['training_episodes']
    colors=[plt.get_cmap('tab10')(i%10) for i in range(len(config['alphas']))]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    for kind,label,name in [('discounted_return','Exact discounted return','reward_curves'),
                            ('train_bellman','Bellman MSE (fixed diagnostic rows)','loss_curves')]:
        fig,axes=plt.subplots(len(scenarios),len(sizes),figsize=(6*len(sizes),3.9*len(scenarios)),squeeze=False,layout='constrained')
        for i,scenario in enumerate(scenarios):
            for j,n in enumerate(sizes):
                ax=axes[i,j]
                for color,alpha in zip(colors,config['alphas']):
                    method=method_name(alpha)
                    rows=[r for r in result['curves'] if r['scenario']==scenario and r['training_episodes']==n and r['method']==method]
                    steps=sorted({r['step'] for r in rows});values=np.array([[r[kind] for r in rows if r['step']==step] for step in steps])
                    mean=values.mean(1);sd=values.std(1,ddof=1) if values.shape[1]>1 else np.zeros(len(mean))
                    title='Offline DQN' if alpha==0 else 'CQL alpha='+str(alpha)
                    ax.plot(steps,mean,label=title,color=color,linewidth=1.8)
                    if values.shape[1]>1:ax.fill_between(steps,mean-sd,mean+sd,color=color,alpha=.13,linewidth=0)
                if kind=='discounted_return':
                    for method,color,label_text in [('logging_policy','#777777','Logging policy'),('optimal_oracle','#222222','Privileged optimal reference')]:
                        baseline=next(r['discounted_return_mean'] for r in result['aggregate']
                                      if r['scenario']==scenario and r['training_episodes']==n and r['method']==method)
                        ax.axhline(baseline,color=color,linestyle='--',linewidth=1.,label=label_text)
                ax.set(title=f'{scenario}; {n} training episodes',xlabel='Gradient updates',ylabel=label)
                ax.grid(alpha=.18);ax.legend(fontsize=8,loc='best')
        fig.suptitle('Synthetic finite-horizon MDP\nShading: run SD (not confidence intervals)',fontsize=11)
        fig.savefig(folder/(name+'.png'),dpi=180);fig.savefig(folder/(name+'.svg'));plt.close(fig)


def create_report(config,output):
    from src.offline_rl.run import validate_component,signature
    output=Path(output);validate_component(output/'experiment',signature(config))
    verification=require_verified_experiment(output)
    result=_read(output/'experiment/results.json');_plots(config,result,output/'figures')
    lines=['# CQL离线策略优化：实际实验报告','',
        '**本模块评价已知有限模拟环境中的累计奖励，不代表KuaiRand真实RL、真实用户收益或Top20改善。**','',
        f'运行预算：{"SMOKE快速检查" if config["smoke"] else "完整预定义实验"}。',
        f'环境为{config["world"]["topics"]}类兴趣×{config["world"]["fatigue_levels"]}级疲劳，共{config["world"]["topics"]*config["world"]["fatigue_levels"]}个完全可观测状态、{config["world"]["topics"]*2}个动作、{config["world"]["horizon"]}步会话，折扣gamma={config["model"]["gamma"]}。兴趣/疲劳为人为定义的模拟状态，没有从真实用户日志拟合。','',
        '## 实现与真实数据适用性','',
        '- KuaiRand-Pure缺失池外交互，当前没有完整状态转移证据；不把相邻日志当作已验证真实MDP。',
        f'- 本机Pearl可导入路径是否存在：{result["framework_assessment"]["pearl_installed"]}。已核对其官方DeepQLearning、保守损失和依赖声明；没有进行Pearl运行兼容性验收，也未安装Gym/MuJoCo等新依赖。',
        '- 本项目使用现有PyTorch独立实现离散CQL(H)，不是Pearl运行结果或论文基准复现。',
        '- 损失为MSE Bellman项＋alpha×mean(logsumexp(Q(s,:))−Q(s,a_logged))，温度1；目标网络无梯度，有限时域最后一步不bootstrap。',
        '- t与state共同输入网络。训练只读离线记录，完整奖励/转移表只用于生成和评估。','',
        '## 预定义训练协议','',
        f'- 日志机制={config["scenarios"]}；每次训练会话数={config["training_episodes"]}；独立验证会话数={config["validation_episodes"]}；种子={config["seeds"]}。',
        '- exploratory对所有动作有正概率；narrow每个状态只支持两个动作，其余是真实零支持。二者都从固定模拟器生成完整会话。',
        f'- 所有模型共享同条件日志、初始化、batch序列与训练预算；alpha={config["alphas"]}，主CQL alpha预先固定为{config["main_alpha"]}。',
        f'- MLP隐藏层={config["model"]["hidden"]}；学习率={config["model"]["learning_rate"]}；batch={config["model"]["batch_size"]}；固定{config["model"]["steps"]}次更新，目标网络每{config["model"]["target_update"]}步硬同步。',
        '- 固定最后检查点，未按验证或评估奖励选择轮次/alpha；训练完成后才从保存Q快照计算奖励曲线。训练损失曲线使用前2048条固定诊断记录，验证Bellman指标用完整验证日志。','',
        '## 策略价值与覆盖结果','',
        '回报是每个策略在完整已知模拟器中的精确有限时域期望，由反向DP和前向占用交叉核对。随机/日志行为/经验行为克隆是可用基线；myopic_oracle和optimal_oracle具有真实奖励/转移特权，仅作参考。数值越高表示当前模拟目标下累计正反馈越多。','',
        '| 日志 | 训练会话 | 策略 | 精确折扣回报均值 | 跨运行SD | 相对DQN配对差均值 | 零支持动作占用 | 训练未见动作占用 |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in result['aggregate']:
        lines.append(f'| {r["scenario"]} | {r["training_episodes"]} | {r["method"]} | {_f(r["discounted_return_mean"])} | {_f(r["discounted_return_sample_std"])} | {_f(r["paired_exact_difference_vs_dqn_mean"])} | {_f(r["zero_support_fraction_mean"])} | {_f(r["unobserved_action_fraction_mean"])} |')
    lines+=['','跨运行变化同时包括离线日志与网络初始化变化；CQL/DQN差值在相同种子的同一日志上配对。运行次数有限，不将样本SD当作高精度显著性证明。','',
        '## 预定义主模型的实际对照','']
    for scenario in config['scenarios']:
        for n in config['training_episodes']:
            lookup={r['method']:r for r in result['aggregate'] if r['scenario']==scenario and r['training_episodes']==n}
            main=lookup[result['main_method']];dqn=lookup['offline_dqn'];logging=lookup['logging_policy']
            lines.append(f'- {scenario}、{n}条训练会话：主CQL平均回报{_f(main["discounted_return_mean"])}，DQN为{_f(dqn["discounted_return_mean"])}，日志策略为{_f(logging["discounted_return_mean"])}；主CQL减DQN={_f(main["paired_exact_difference_vs_dqn_mean"])}。')
    lines+=['','以上对照按预定义主模型汇报，未从alpha敏感性表中事后挑选胜出配置。','',
        '## Q值诊断','',
        'Q偏差为模型预测的初始策略价值减该策略的精确真实价值。负值表示在此定义下低估，不自动证明理论下界；也不保证回报更高。','',
        '| 日志 | 训练会话 | 学习模型 | 平均初始Q偏差 |','| --- | ---: | --- | ---: |']
    for r in result['aggregate']:
        if r['q_start_bias_mean'] is not None:
            lines.append(f'| {r["scenario"]} | {r["training_episodes"]} | {r["method"]} | {_f(r["q_start_bias_mean"])} |')
    lines+=['','## 独立会话模拟与不确定性','',
        f'每策略另用{config["rollout_episodes"]}条独立会话模拟回报，使用{config["bootstrap_repeats"]}次整会话bootstrap。相同条件策略共享随机数构成配对；区间条件于固定训练后策略，不包含重新训练的不确定性。全部均值/区间、相对日志/BC/DQN配对差保存在完整JSON。',
        '精确DP价值不依赖Monte Carlo置信区间；程序用6个标准误加0.03的诊断阈值核对采样与精确值，并不要求所有名义95%区间都包含真值。','',
        '## 曲线','',
        '奖励曲线为保存Q快照对应策略的精确值；阴影为跨运行样本SD，不是置信区间。验证与训练损失不直接等于策略回报。','',
        '![Reward curves](figures/reward_curves.png)','',
        '![Bellman diagnostic curves](figures/loss_curves.png)','',
        '## 核验和复现','',
        f'- 重算PASS：{verification["conditions_replayed"]}个日志条件，{verification["models_reloaded"]}个检查点重载，{verification["policy_metrics_recomputed"]}组策略指标，{verification["curve_rows_recomputed"]}条曲线。',
        '- 所有训练/验证轨迹按固定种子重放；完整会话边界、终止、实际行为概率与支持核对；全部策略模拟回报重放。',
        '- 全部模型最终Q与检查点逐值一致；初始参数与batch序列重新核验；保存在线/目标Q快照重算诊断损失及全部奖励曲线；CSV与JSON逐值核对。',
        '- 累计梯度范数是训练时记录的元数据，仅核对有限性与单调性，未逐步重放梯度；诊断样本数另作核对。',
        '- 输出绑定配置、源码、环境、旧证据指纹和全部产物哈希，完成目录拒绝不一致复用。','',
        '```powershell',r'. .\scripts\enter_env.ps1',
        'python -B -m src.offline_rl.run --phase all --output-directory experiments/offline_rl_reproduction',
        'python -B -m src.offline_rl.run --phase verify --output-directory experiments/offline_rl_reproduction',
        'python -B -m src.offline_rl.run --phase report --output-directory experiments/offline_rl_reproduction','```','',
        '## 结果解释的限制','',
        '- 神经CQL并不保证优于DQN、行为克隆或日志策略；所有预定义条件和退化结果均保留。',
        '- 对日志零支持动作的真实价值来自已知模拟器；不声称仅凭离线日志可以识别其收益。',
        '- 这不是由单动作IPS/DR推算的整会话收益，也没有将Top20列表当作一个具有已知概率的动作。',
        '- 真实推荐主流程、真实Top20与曝光校准模块未被替换；新机器环境复现、完整真实转移与线上验收不在本轮完成范围。','',
        '## 参考','',
        '- [CQL原论文](https://arxiv.org/abs/2006.04779)。',
        '- [Pearl DeepTDLearning](https://github.com/facebookresearch/Pearl/blob/main/pearl/policy_learners/sequential_decision_making/deep_td_learning.py)、[保守损失接口](https://github.com/facebookresearch/Pearl/blob/main/pearl/utils/functional_utils/learning/loss_fn_utils.py)、[依赖声明](https://github.com/facebookresearch/Pearl/blob/main/pyproject.toml)。']
    # The canonical report uses figures local to the experiments report, not broken relative report links.
    markdown='\n'.join(lines)+'\n';(output/'report.md').write_text(markdown,encoding='utf-8')
    summary={'status':'COMPLETE','smoke':config['smoke'],'real_rl_value_claimed':False,'real_top20_changed':False,
        'main_method':result['main_method'],'aggregate':result['aggregate'],'verification':verification,
        'output_directory':output.relative_to(ROOT).as_posix(),'limits':result['limits']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    if not config['smoke']:
        prefix='../'+output.relative_to(ROOT).as_posix()+'/figures/'
        canonical=markdown.replace('(figures/','('+prefix)
        (ROOT/'reports/offline_rl_report.md').write_text(canonical,encoding='utf-8')
        (ROOT/'reports/offline_rl_results.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print('REPORT '+str(output/'report.md'),flush=True);return summary
