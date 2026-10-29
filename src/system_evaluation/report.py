"""Separate-task scorecards, chain bottlenecks, and bounded offline timing reports."""
import json
from pathlib import Path
import numpy as np
from src.data.common import ROOT
from src.data.download import digest_file


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def require_verified(output):
    output=Path(output);v=read(output/'verification.json');path=output/'evaluation/component.json'
    component=read(path)
    if (v['status']!='PASS' or v.get('component_fingerprint')!=component['fingerprint']
            or v.get('manifest_sha256')!=digest_file(path)):raise ValueError('Verification does not bind current inputs and outputs')
    from src.system_evaluation.run import verification_record,compare_analysis
    expected=verification_record(component['signature']['config'],component,path.parent,
        read(path.parent/'analysis.json'),read(path.parent/'pipeline.json'))
    compare_analysis(expected,v)
    return v


def budget_description(config):
    return ('SMOKE；' if config['smoke'] else '正式全量排名/预测重算；')+(
        '曝光/RL深度重放' if config['deep_audit'] else '旧曝光/RL仅检查冻结资产，未深度重放')


def delivery_paths(config,output):
    output=Path(output)
    files=[output/name for name in ['report.md','summary.json','figures/ranking_comparison.png',
        'figures/ranking_comparison.svg','figures/candidate_funnel.png','figures/candidate_funnel.svg']]
    if not config['smoke'] and config['output_directory']=='experiments/system_evaluation_main':
        files += [ROOT/'reports/system_evaluation_report.md',ROOT/'reports/system_evaluation_results.json']
    return files


def write_report_delivery(config,output):
    output=Path(output)
    data={'status':'COMPLETE','evaluation_manifest_sha256':digest_file(output/'evaluation/component.json'),
        'verification_sha256':digest_file(output/'verification.json'),
        'output_hashes':{p.relative_to(ROOT).as_posix():digest_file(p) for p in delivery_paths(config,output)}}
    (output/'report_delivery.json').write_text(json.dumps(data,indent=2),encoding='utf-8')


def validate_report_delivery(config,output):
    output=Path(output);data=read(output/'report_delivery.json')
    expected={'status':'COMPLETE','evaluation_manifest_sha256':digest_file(output/'evaluation/component.json'),
        'verification_sha256':digest_file(output/'verification.json'),
        'output_hashes':{p.relative_to(ROOT).as_posix():digest_file(p) for p in delivery_paths(config,output)}}
    from src.exposure.verify import assert_result_equal
    assert_result_equal(expected,data)
    if {p.name for p in (output/'figures').iterdir() if p.is_file()}!={p.name for p in delivery_paths(config,output) if p.parent==output/'figures'}:
        raise ValueError('Report figure file set differs')
    return data


def _f(x):return '未定义' if x is None else f'{x:.6f}'


def _plots(result,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    names={'popularity':'Popularity','itemcf':'ItemCF','two_tower_seed42':'TwoTower V1',
        'two_tower_v2_seed42':'TwoTower V2','sasrec_seed42':'SASRec','v2_original_order':'V2 order',
        'din_v2_seed42':'DIN V2','din_v3_experimental_seed42':'DIN V3*','old_lr':'Old LR','old_din_seed42':'Old DIN'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,space in enumerate(['catalog','fixed_v2_top100']):
        rows=[r for r in result['rank_results'] if r['partition']=='standard/test' and r['space']==space and r['primary']]
        for j,metric in enumerate(['recall@20','ndcg@20']):
            ax=axes[i,j];x=np.arange(len(rows))
            for offset,part,color in [(-.18,'standard/test','#326a9f'),(.18,'random/test','#cf7d36')]:
                values=[next(v['metrics'][metric] for v in result['rank_results']
                    if v['space']==space and v['partition']==part and v['model']==r['model']) for r in rows]
                bars=ax.bar(x+offset,values,.35,label=part,color=color)
                ax.bar_label(bars,fmt='%.4f',fontsize=6,padding=2,rotation=90)
            ax.set_xticks(x,[names[r['model']] for r in rows],rotation=25,ha='right')
            ax.set_title(('Full catalog' if space=='catalog' else 'Same V2 Top100')+' / '+metric)
            ax.set_ylim(top=ax.get_ylim()[1]*1.25);ax.legend(fontsize=7);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Observed positive-event ranking; separate candidate spaces\n*V3 remains experimental; no causal or online-gain interpretation',fontsize=11)
    fig.savefig(directory/'ranking_comparison.png',dpi=180);fig.savefig(directory/'ranking_comparison.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained');bottom=np.zeros(2)
    for field,title,color in [('top20_hits','Final Top20 hit','#438c73'),('pool_not_top20','In Top100, outside Top20','#cf7d36'),
                              ('outside_pool','Outside Top100','#8c99a5')]:
        value=np.array([result['candidate_funnel'][p]['all'][field]/result['candidate_funnel'][p]['all']['queries'] for p in ['standard/test','random/test']])
        ax.bar(['Standard test','Random-source test'],value,bottom=bottom,label=title,color=color)
        for i,n in enumerate(value):
            if n>.03:ax.text(i,bottom[i]+n/2,f'{n:.1%}',ha='center',va='center',fontsize=11)
        bottom+=value
    ax.set(ylim=(0,1),ylabel='Fraction of all observed positive-event queries',title='Frozen V2 -> Top100 -> DIN: disjoint loss accounting')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.1),ncol=1,frameon=False)
    fig.savefig(directory/'candidate_funnel.png',dpi=180);fig.savefig(directory/'candidate_funnel.svg');plt.close(fig)


def create_report(config,output):
    from src.system_evaluation.run import signature,validate_component
    output=Path(output);validate_component(output/'evaluation',signature(config));v=require_verified(output)
    result=read(output/'evaluation/analysis.json');pipe=read(output/'evaluation/pipeline.json')
    _plots(result,output/'figures')
    lines=['# 统一评估与整体效果诊断','',
        '**本阶段完成统一核验、真实推荐接口抽样贯通与瓶颈诊断，没有训练新模型或改变正式Top20。**','',
        '预算：'+budget_description(config)+'。',
        '当前主流程：固定双塔V2 seed42 → 同一Top100 → DIN V2 seed42 → Top20。全目录、固定池、真实反馈概率、合成OPE与合成RL分别评价，没有混合总分。',
        '真实指标为日志观测正事件的单正目标代理；冷记录保留，测试此前已查看，属于迭代复测。','',
        '## 真实排序：同口径对照','']
    for space,title in [('catalog','全目录检索'),('fixed_v2_top100','固定同一V2 Top100排序')]:
        lines+=['### '+title,'','| 来源 | 模型 | 正事件数 | Recall@20 | NDCG@20 | Recall@100 | MRR | MRR范围 |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
        for row in result['rank_results']:
            if row['space']!=space or not row['primary'] or not row['partition'].endswith('/test'):continue
            m=row['metrics'];name=row['model']+('（实验，未采用）' if row['experimental'] else '')
            lines.append(f'| {row["partition"]} | {name} | {m["queries"]} | {_f(m["recall@20"])} | {_f(m["ndcg@20"])} | {_f(m["recall@100"])} | {_f(m["mrr"])} | {m["mrr_cutoff"]} |')
    std={r['model']:r['metrics'] for r in result['rank_results'] if r['partition']=='standard/test' and r['space']=='fixed_v2_top100'}
    lines+=['','早期产物缺少宏平均@100时，从原始排名重算；未补造保存期望值。全部原本保存的核心指标均核对，新增宏平均值单独保留在完整JSON。','',
        f'标准测试V3 Recall@20={std["din_v3_experimental_seed42"]["recall@20"]:.4%}属于实验版本；当前正式DIN V2={std["din_v2_seed42"]["recall@20"]:.4%}。本阶段延续冻结阶段的未采用状态，不因微小点估计变化切换模型。SASRec是独立全目录/固定池对照。','',
        '![Comparable ranking](figures/ranking_comparison.png)','',
        '## Top20损失漏斗','',
        '| 来源 | 正事件分母 | 最终Top20命中 | 进入100但未入20 | 未进100 | Recall@100 | 池内Top20命中率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for part in ['standard/test','random/test']:
        f=result['candidate_funnel'][part]['all']
        lines.append(f'| {part} | {f["queries"]} | {f["top20_hits"]} | {f["pool_not_top20"]} | {f["outside_pool"]} | {_f(f["recall@100"])} | {_f(f["conditional_top20_given_pool"])} |')
    lines+=['','三类计数互斥且加总为全部正事件；R20=R100×池内Top20命中率。已进入候选后才计算的指标不能替代端到端指标。假定知道标签、把每个候选内目标排第一所得R100上限仅是不可部署的oracle诊断，不是可获得收益承诺。','',
        '![Candidate loss funnel](figures/candidate_funnel.png)','',
        '## 正式模型的分组瓶颈','',
        '| 来源 | 分组 | 正事件数 | 未入Top100比例 | 已入池但未入20比例 | 最终Recall@20 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for part in ['standard/test','random/test']:
        for name in ['head_target','tail_target','cold_user','cold_target','empty_history','history_target_seen']:
            f=result['candidate_funnel'][part][name];n=f['queries']
            lines.append(f'| {part} | {name} | {n} | {_f(f.get("outside_pool",0)/n if n else None)} | {_f(f.get("pool_not_top20",0)/n if n else None)} | {_f(f.get("recall@20"))} |')
    lines+=['','热门/长尾只用标准训练期计数定义；历史只取严格过去。冷热用户、历史分组可以重叠；不把这些行加总。样本极少的冷目标组只作描述，不据其零值宣称普遍失败。','',
        '## 精排新增与损失命中','',
        '| 来源 | V2命中被保留 | DIN新增命中 | DIN丢失原命中 | 净Recall变化 | NDCG变化95%用户簇区间 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for part in ['standard/test','random/test']:
        t=result['top20_transitions'][part];ci=result['paired_user_intervals'][part]
        lines.append(f'| {part} | {t["retained_top20_hits"]} | {t["new_top20_hits"]} | {t["lost_top20_hits"]} | {_f(t["net_recall_change"])} | [{_f(ci["low"][1])}, {_f(ci["high"][1])}] |')
    lines+=['','区间条件于固定模型，只反映用户簇抽样；不包含训练、历史选模或分布变化。三种子均值/样本SD另存JSON；V3三个继续训练种子共享原seed42骨干，不能称为三个独立初始化。','',
        '## 真实反馈概率：与排序分开','',
        '| 来源 | 预测模型 | 曝光数 | AUC | LogLoss | Brier | ECE |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in result['pointwise_results']:
        if not row['partition'].endswith('/test'):continue
        m=row['metrics'];lines.append(f'| {row["partition"]} | {row["model"]} | {m["rows"]} | {_f(m["auc"])} | {_f(m["logloss"])} | {_f(m["brier"])} | {_f(m["ece"])} |')
    lines+=['','排序DIN的AUC取舍保留；BCE对照概率更好不等于Top20更好。曝光模块的早期验证校准、跨来源退化及分组结果保留在独立任务JSON/报告中，校准不改变排序。','',
        '## 合成实验的独立证据','',
        '- 曝光OPE：已知有限总体、真实行为概率和单动作目标策略，估计偏差/RMSE/覆盖率；不能解释成整张Top20价值。',
        '- CQL：已知有限MDP中的折扣累计奖励；不等于真实KuaiRand用户收益。真实Pure的完整转移仍未建立。',
        '- 完整合成结果原样保留在analysis.json的other_task_results，与真实排序分开；本阶段未把它们拼成综合提升率。','',
        '## 冻结推荐接口与离线性能观测','',
        f'加载原检查点，以encoded user、tab和严格过去history50产生Top100→Top20。四分区共{v["pipeline_queries_replayed"]}条抽样上下文重新生成候选/打分/Top20并核对，接口不接收当前目标或反馈。该抽样不是所有请求的新一轮全模型重算。',
        f'设备={pipe["device"]}，硬件={pipe["hardware"]}，PyTorch={pipe["torch_version"]}，线程={pipe["torch_threads"]}。','',
        '| batch | 实测批次数 | p50 ms/批 | p95 ms/批 | p99 ms/批 | 请求吞吐/秒 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in pipe['latency'].values():lines.append(f'| {m["batch_size"]} | {m["measured_batches"]} | {m["p50_ms"]:.3f} | {m["p95_ms"]:.3f} | {m["p99_ms"]:.3f} | {m["requests_per_second"]:.1f} |')
    lines+=['','计时包括预构造上下文到Top20及CPU/GPU传输，CUDA前后同步；不含历史查询、网络、调度、并发和冷启动。p99仅为本批有限观测的描述性分位数，不是线上SLO。响应一致性核验采用检索batch512、精排batch256；其他批形只作计时，不宣称跨批形浮点/Top20完全相同。','',
        '## 下一轮优化的优先级','']
    for part in ['standard/test','random/test']:
        f=result['candidate_funnel'][part]['all']
        lines.append(f'- {part}：{f["outside_pool"]/f["queries"]:.2%}的正目标未进入Top100，{f["pool_not_top20"]/f["queries"]:.2%}已进入但未排进20。先区分召回与池内排序损失，再分配优化预算。')
    lines+=['- 优先评估双塔对长尾及跨来源的候选覆盖，保持既有精排对照；固定池结构修改不能找回池外目标。',
        '- 池内排序仍有空间，下一训练实验可检查选择损失、困难替代项与历史重复特征的取舍；需要验证集选型及独立对照，不把未曝光项补为真实负反馈。',
        '- 实验V3微小变动不足以替换正式版本；SASRec先检验训练是否充分收敛及长尾表现，继续作为独立对照。',
        '- 后续只用训练/验证决定配置；当前测试是迭代诊断，新的独立确认需未使用数据或新数据。当前没有宣称已实现新模型改善。','',
        '## 复现与验收','',
        f'本次重算{v["rank_artifacts_recomputed"]}份排名、{v["exposure_predictions_recomputed"]}份曝光预测和{v["subgroup_rows_recomputed"]}行分组，deep_audit={v["deep_module_recomputation"]}。',
        '配置/源码/原始资产/环境及所有输出哈希绑定，verification额外绑定实际manifest SHA256。报告自身重新检查组件和验证绑定。数据准备与原模型完整重训仍沿用各模块命令，新机器/线上验收未完成。','',
        '```powershell',r'. .\scripts\enter_env.ps1','python -B -m src.system_evaluation.run --phase all',
        'python -B -m src.system_evaluation.run --phase verify','python -B -m src.system_evaluation.run --phase report','```']
    markdown='\n'.join(lines)+'\n';(output/'report.md').write_text(markdown,encoding='utf-8')
    summary={'status':'COMPLETE','smoke':config['smoke'],'main_pipeline':result['main_pipeline'],
        'candidate_funnel':result['candidate_funnel'],'top20_transitions':result['top20_transitions'],
        'verification':v,'latency':pipe['latency'],'hardware':pipe['hardware'],
        'new_model_trained':False,'new_quality_gain_claimed':False,'output_directory':output.relative_to(ROOT).as_posix(),
        'limits':result['limits']}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if not config['smoke'] and config['output_directory']=='experiments/system_evaluation_main':
        prefix='../'+output.relative_to(ROOT).as_posix()+'/figures/'
        (ROOT/'reports/system_evaluation_report.md').write_text(markdown.replace('(figures/','('+prefix),encoding='utf-8')
        (ROOT/'reports/system_evaluation_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_report_delivery(config,output);validate_report_delivery(config,output)
    print('REPORT '+str(output/'report.md'),flush=True);return summary
