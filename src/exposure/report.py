"""Source-grounded reporting of identifiable simulations and observed real logs."""
import json
from pathlib import Path

import numpy as np

from src.data.common import ROOT


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _f(value, digits=6):return '未定义' if value is None else f'{value:.{digits}f}'


def require_verified_components(output):
    output=Path(output);verification=_read(output/'verification.json')
    expected={phase:_read(output/phase/'component.json')['fingerprint'] for phase in ['synthetic','real']}
    if verification['status']!='PASS' or verification.get('component_fingerprints')!=expected:
        raise ValueError('Verification does not belong to the exact completed exposure components')
    return verification


def create_report(config,output):
    verification=require_verified_components(output)
    output=Path(output);simulation=_read(output/'synthetic/synthetic_results.json');real=_read(output/'real/real_results.json')
    cfg=config['synthetic'];strength=max(cfg['strengths']);n=max(cfg['sample_sizes'])
    def metric(case,estimator,cap=None,rows=n):
        matches=[r for r in simulation['metrics'] if r['strength']==strength and r['rows']==rows
                 and r['target']=='uniform' and r['case']==case and r['cap']==cap and r['estimator']==estimator]
        if len(matches)!=1:raise ValueError('Report cell missing or duplicated')
        return matches[0]
    lines=['# 曝光偏差评估模块：实际运行报告','',
        '**模块实现、实际实验和重算核验已完成。真实KuaiRand结果属于观察性来源诊断及概率校准；IPS/SNIPS/DR的可识别性与真值误差在合成环境验证。**','',
        '正式推荐流程仍为原双塔V2＋DIN V2，候选及Top20排序未改变。本模块不训练或部署新的真实推荐策略，不报告真实日志的无偏策略价值或CQL收益。',
        '', '## 实验与识别边界','',
        f'- 运行类型：{"SMOKE快速验证，不能替代正式预算" if config["smoke"] else "完整预定义预算"}。',
        '- 真实日志没有已确认的逐事件动作概率；未从is_rand、随机曝光占比或1/视频数量制造propensity。',
        '- OPE估计对象为一次动作的期望奖励，既不是整张Top20列表的价值，也不是长期RL回报。',
        '- 完整行为分布可用时拒绝无支持目标；仅给已选动作pscore时，支持性标为无法完全验证。',
        '- 真实用户簇bootstrap保留事件加权分母；SNIPS在每次bootstrap中重新计算分子和分母。',
        '- 真实测试曾被旧阶段查看，本轮为迭代诊断。校准方法预先固定，不按测试选择。',
        '', '## 真实数据：冻结DIN的来源差异','',
        '| 分区/子集 | 曝光数 | 正事件查询 | 正反馈率 | AUC | LogLoss | Recall@20 | NDCG@20 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for cohort in ['all','tab_1','common_warm_user_item_tab1']:
        for part in ['standard/test','random/test']:
            g=real['groups'][part][cohort];e=g['exposure'];r=g['ranking'].get('din',{})
            lines.append(f"| {part}/{cohort} | {e['rows']} | {r.get('queries',0)} | {_f(e.get('positive_rate'))} | {_f(e.get('auc'))} | {_f(e.get('logloss'))} | {_f(r.get('recall@20'))} | {_f(r.get('ndcg@20'))} |")
    lines+=['',f"共同暖用户数={real['common_cohort']['warm_users']}，共同已知视频数={real['common_cohort']['known_items']}。共同范围不等于相同请求或相同视频曝光频率；不能将来源差值直接解释为曝光机制的因果效应。",'',
        '## 共同用户-tab层标准化','',
        f"共{real['standardization']['definition']['shared_strata']}个共享用户-tab层，每层在两个来源中均获得相同总权重。只标准化曝光指标，不改Top20分母。",'',
        '| 来源 | 纳入曝光 | 加权正反馈率 | 加权AUC | 加权LogLoss | 加权Brier |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for part,m in real['standardization']['metrics'].items():
        lines.append(f"| {part} | {m['positive_weight_rows']} | {_f(m['positive_rate'])} | {_f(m['auc'])} | {_f(m['logloss'])} | {_f(m['brier'])} |")
    lines+=['', '## 真实反馈概率的校准对照','',
        'standard-validation校准只使用标准验证标签；random-validation校准使用更早随机反馈，属于有随机反馈可用时的校准，不是零样本跨来源泛化。两者都严格早于测试时间。尺度约束为正，反馈分数次序保持不变；正式排名仍直接引用冻结的原排名。','',
        '| 测试来源 | 概率方法 | AUC | LogLoss | Brier | ECE（15等宽桶或配置桶数） |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    calibration_improvements={}
    for part,methods in real['calibration_results'].items():
        calibration_improvements[part]={}
        baseline=methods['uncalibrated']['all']
        for name,groups in methods.items():
            m=groups['all'];lines.append(f"| {part} | {name} | {_f(m['auc'])} | {_f(m['logloss'])} | {_f(m['brier'])} | {_f(m['ece'])} |")
            if name!='uncalibrated':calibration_improvements[part][name]=m['logloss']-baseline['logloss']
    lines+=['','校准的LogLoss/Brier差值95%用户簇区间（校准后减原预测，负值表示损失降低）：','',
            '| 测试来源 | 拟合来源 | LogLoss差 | 95%区间 |','| --- | --- | ---: | --- |']
    for part,methods in real['uncertainty']['calibration_delta'].items():
        for name,ci in methods.items():
            lines.append(f"| {part} | {name} | {_f(ci['difference'][0])} | [{_f(ci['low'][0])}, {_f(ci['high'][0])}] |")
    lines+=['','区间条件于已经拟合的模型与校准系数，不包含系数拟合、早期选模或训练的不确定性。ECE依赖桶数；各子组为描述性分析，不作多重比较校正后的显著性宣称。','',
            '## 合成环境：已知真实策略价值','',
            f"有限总体含{cfg['n_contexts']}个上下文、{cfg['n_actions']}个动作、{cfg['context_dim']}维特征。独立日志重复{cfg['repeats']}次，logging强度={cfg['strengths']}，样本量={cfg['sample_sizes']}，权重上限={cfg['clips']}。",'',
            f"精确有限总体真值：uniform={_f(simulation['known_values']['uniform'])}；reward_softmax={_f(simulation['known_values']['reward_softmax'])}。后者是模拟器中预定义的策略，不是真实推荐模型的收益。",'',
            'reward model另用独立日志拟合。其训练只读取记录动作的反馈，不读取模拟器的完整奖励表；oracle q仅用于检验数学性质。五种误设条件均保留，不用某一胜出条件代表所有情况。','',
            f'### 主表：logging强度{strength}，样本量{n}，目标uniform，无截断','',
            '| 条件 | 估计器 | 偏差 | RMSE | 重复间SD |','| --- | --- | ---: | ---: | ---: |']
    for case in ['correct_pscore_oracle_q','correct_pscore_wrong_q','wrong_pscore_oracle_q','wrong_pscore_wrong_q','correct_pscore_learned_q']:
        for estimator in ['ips','snips','dm','dr']:
            m=metric(case,estimator)
            lines.append(f"| {case} | {estimator} | {_f(m['bias'])} | {_f(m['rmse'])} | {_f(m['sample_std'])} |")
    naive=metric('correct_pscore_learned_q','naive')
    lines+=['',f"同一条件中，直接使用日志奖励均值的偏差={_f(naive['bias'])}、RMSE={_f(naive['rmse'])}。它估计行为策略的日志均值，不应当作目标策略收益。",'',
            '## 权重截断与有效样本量','',
            '下表ESS和权重来自一个预先固定种子的代表日志；RMSE来自全部独立重复。两者统计单位不同。','',
            '| 上限 | 原始ESS | 使用后ESS | 被截断比例 | 使用后最大权重 | IPS的RMSE | DR的RMSE |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for cap in cfg['clips']:
        d=simulation['representative'][str(cap)]['weight_diagnostics']
        lines.append(f"| {cap} | {_f(d['raw']['ess'],1)} | {_f(d['used']['ess'],1)} | {_f(d['clipped_fraction'])} | {_f(d['used']['quantiles']['1'])} | {_f(metric('correct_pscore_learned_q','ips',cap)['rmse'])} | {_f(metric('correct_pscore_learned_q','dr',cap)['rmse'])} |")
    lines+=['','截断可能降低方差，也可能引入偏差；截断DR不再自动保留通常的双重稳健保证。ESS表示权重集中程度，不是独立用户数或真实数据量。无支持目标已被程序拒绝。','',
            '## Bootstrap区间的经验覆盖率','',
            f"logging强度{strength}、{cfg['coverage_rows']}条记录，每个条件{cfg['coverage_repeats']}次独立日志、每次{cfg['coverage_bootstrap_repeats']}次重采样。95%是区间名义水平，不是保证。经验覆盖率同时报告Wilson区间，以体现重复次数有限。",'',
            '| 条件 | 估计器 | 覆盖次数/重复数 | 经验覆盖率 | 覆盖率Wilson 95%区间 |','| --- | --- | --- | ---: | --- |']
    for row in simulation['coverage']:
        if row['estimator'] not in ['ips','snips','dm','dr']:continue
        lo,hi=row['coverage_wilson95']
        lines.append(f"| {row['case']} | {row['estimator']} | {row['covered']}/{row['repeats']} | {_f(row['coverage_over_all_repeats'],3)} | [{_f(lo,3)}, {_f(hi,3)}] |")
    lines+=['','## 已知倾向概率下的加权反馈学习','',
            '实际训练了同一简化logistic反馈模型，比较普通BCE、IPS-BCE和上限5的IPS-BCE。奖励模型有意省略模拟器非线性交互，因此不能预设普通监督与加权学习必然等效。全动作真实风险在已知总体上计算，没有用该风险选超参数。','',
            '| 训练损失 | 种子数 | 真实均匀动作LogLoss均值±SD | 贪心策略真实奖励均值 | 最大目标梯度 |',
            '| --- | ---: | --- | ---: | ---: |']
    learning_summary={}
    for name in ['unweighted','ips','ips_clip5']:
        rows=[r['models'][name] for r in simulation['weighted_learning']]
        losses=np.array([r['true_uniform_logloss'] for r in rows]);values=np.array([r['true_greedy_policy_value'] for r in rows])
        sd=float(losses.std(ddof=1)) if len(rows)>1 else None
        learning_summary[name]={'seeds':len(rows),'true_uniform_logloss_mean':float(losses.mean()),'sample_std':sd,
                                'true_greedy_policy_value_mean':float(values.mean())}
        lines.append(f"| {name} | {len(rows)} | {_f(float(losses.mean()))} ± {_f(sd)} | {_f(float(values.mean()))} | {_f(max(r['objective_gradient_max'] for r in rows))} |")
    lines+=['','这是模拟条件成立时的去偏加权损失验证，不是对KuaiRand主模型的去偏训练结果。所有失败或退化情况均保留。','',
            '## 工程核验与复现','',
            '- 数学手算、DR精确总体期望、零权重/支持、截断、SNIPS比率、用户簇分母、加权梯度、拟合隔离、单调校准和输出篡改均有自动测试。',
            '- synthetic/real各自保存配置、源码、输入、环境版本指纹以及全部输出哈希；完成目录拒绝不一致复用，不覆盖原实验。',
            '- 验证器重算代表日志估计器与bootstrap、全部Monte Carlo汇总、覆盖率汇总及一个完整覆盖日志、加权学习真实风险；真实分组、校准和用户簇区间全部重新运行核对。',
            '- 数值重算容差relative=1e-8、absolute=1e-10；这是数值一致性验证，不声明所有浮点值跨环境逐位相同。','',
            '```powershell',r'. .\scripts\enter_env.ps1',
            'python -B -m src.exposure.run --phase all --output-directory experiments/exposure_reproduction',
            'python -B -m src.exposure.run --phase verify --output-directory experiments/exposure_reproduction',
            'python -B -m src.exposure.run --phase report --output-directory experiments/exposure_reproduction','```','',
            '独立复现依赖本项目已准备的数据、冻结DIN预测/排名及其证据文件；本轮未验收全新机器从原始下载开始的完整重建。','',
            '## 文件与边界','',
            f"本次目录：{output.relative_to(ROOT).as_posix()}。synthetic_metrics.csv、synthetic_coverage.csv、synthetic_replicates.csv位于synthetic/；真实子集与校准桶CSV位于real/。逐条模拟数组在被忽略的artifacts/内。",'',
            '真实日志仍不可据此报告IPS/SNIPS/DR因果价值；CQL和完整真实状态转移仍未完成。概率校准不等于消除了曝光偏差，Top20收益也不等于一次动作期望奖励。','',
            '## 参考','',
            '- [Open Bandit Pipeline公式与接口](https://github.com/st-tech/zr-obp/blob/master/obp/ope/estimators.py)。本项目独立实现数学内核，未安装或复制该包。',
            '- [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601)。',
            '- [The Self-Normalized Estimator for Counterfactual Learning](https://papers.nips.cc/paper_files/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html)。']
    summary={'status':'COMPLETE','smoke':config['smoke'],'real_ope_identified':False,
             'real_ope_status':real['real_ope_status'],'real_propensity_fabricated':False,
             'original_top20':real['frozen_top20'],'calibration_logloss_deltas':calibration_improvements,
             'synthetic_known_values':simulation['known_values'],'synthetic_config':cfg,
             'synthetic_weighted_learning':learning_summary,'verification':verification,
             'output_directory':output.relative_to(ROOT).as_posix(),
             'limits':['Single-action synthetic OPE, not Top20 slate or RL value',
                       'Real source differences and calibration are observational',
                       'Existing test sets have been viewed; no independent new-test claim']}
    text='\n'.join(lines)+'\n'
    (output/'report.md').write_text(text,encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if not config['smoke']:
        (ROOT/'reports/exposure_evaluation_report.md').write_text(text,encoding='utf-8')
        (ROOT/'reports/exposure_evaluation_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('REPORT '+str(output/'report.md'),flush=True)
    return summary
