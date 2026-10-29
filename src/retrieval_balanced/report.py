"""Complete positive/negative experiments, selection tradeoffs and final Top20."""
import numpy as np
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval_longtail.run import load_selection
from src.retrieval_longtail.train import read_json,write_json
from src.retrieval_balanced.verify import verify,validate_delivery


def p(x):return f'{100*x:.4f}%'


def summary(m):
    return {k:{'mean':float(np.mean(v)),'sample_sd':float(np.std(v,ddof=1)) if len(v)>1 else 0.}
        for k,v in m.items()}


def report(cfg,out):
    v=verify(cfg,out);f=load_selection(out);rep=read_json(out/'replication.json');ev=read_json(out/'evaluation.json')
    lines=['# 根因驱动的长尾召回与精排适配：实际实验','',
        '仅标准训练/验证选型；同一7538项目录、严格过去历史50、冷目标保留、重复允许。测试已查看，全部为迭代复测。真实Top20、观察性诊断与合成OPE/RL分开，本轮不涉及后两者。',
        '', '## 修改与方法边界','',
        '视频侧新增标准训练二值共点击余弦邻域共享，历史使用同一视频表示；仍为单模型、可独立导出视频向量的点积检索，不融合ItemCF预测分数。它借鉴[LightGCN的邻域传播思想](https://arxiv.org/abs/2002.02126)，不是完整LightGCN复现。',
        '损失权重改为自然事件/有点击视频均匀风险的rho混合，正事件均值1且有效组份额随rho线性变化；零正例项没有凭空生成正例。另以[训练内logit adjustment](https://arxiv.org/abs/2007.07314)做小范围对照，推理仍用点积；不称采样logQ修复、IPS或因果去偏。',
        '视频基本表上传日未晚于首次训练曝光，但快照字段历史可用性未完全证明，本轮未接入。新增模型依赖仅现有PyTorch/SciPy；无下载/安装，全部新产物D盘。',
        '', '## 训练前固定的验证协议','',
        '20组配置，每组从头训练seed42、固定8轮，保存逐轮总体/头尾曲线。合格轮要求总体R20/N20及头部R20保留98%，长尾R20绝对至少+0.002、R100至少+0.01；按两项长尾相对比值的等权平均选最佳合格轮。无合格轮时保存总体N20最佳作为明确失败，不冒充准入。',
        '最多2个验证候选（三种子复核前冻结）：长尾效用最高，以及各运行所选检查点中总体N20最高。三种子均值按本轮同预算新控制重算相同门槛，且每个种子两项长尾指标均不得倒退。测试不替补配置。98%保留及绝对增益是本轮预定实验取舍，不是用户给出的业务SLO。',
        '', '## 全部验证配置（seed42）','',
        '| 配置 | 轮次（从0） | 总体R20 | N20 | 长尾R20 | 长尾R100 | 合格 |',
        '|---|---:|---:|---:|---:|---:|:---:|']
    for name,r in f['runs'].items():
        m=r['best_validation'];t=m['groups']['tail_target']
        lines.append(f'| {name} | {r["best_epoch"]} | {p(m["recall@20"])} | {m["ndcg@20"]:.6f} | {p(t["recall@20"])} | {p(t["recall@100"])} | {r["qualified"]} |')
    lines+=['',f'冻结复核候选：{f["candidate_names"]}。通过三种子门槛：{rep["eligible_candidates"]}；进入适配：`{rep["selected_for_adaptation"]}`。控制seed42骨干参数精确复现原V2。',
        '', '## 独立召回复测（三种子均值 ± 样本标准差）','',
        '| 分区 | 模型 | R20 | N20 | R100 | 长尾R20 | 长尾R100 | 目录覆盖20 |','|---|---|---:|---:|---:|---:|---:|---:|']
    tables=[]
    for part in ['standard/validation','standard/test','random/validation','random/test']:
        for name,group in ev['results'].items():
            m=[g[part] for g in group.values()]
            s=summary({'r20':[a['recall@20'] for a in m],'n20':[a['ndcg@20'] for a in m],
                'r100':[a['recall@100'] for a in m],'tail20':[a['groups']['tail_target']['recall@20'] for a in m],
                'tail100':[a['groups']['tail_target']['recall@100'] for a in m],'coverage20':[a['coverage_at20'] for a in m]})
            tables.append({'partition':part,'model':name,'metrics':s})
            cells=[f'{x["mean"]:.6f} ± {x["sample_sd"]:.6f}' if k=='n20' else f'{p(x["mean"])} ± {p(x["sample_sd"])}' for k,x in s.items()]
            lines.append('| '+part+' | '+name+' | '+' | '.join(cells)+' |')
    lines+=['','主种子配对用户簇95%区间、每种子完整头尾/频次/用户宏平均见summary.json和逐事件rank。区间固定500次重采样，不包含训练/调参不确定性；不由单个显著区间推断线上收益。','',
        '## 完整链路：新教师、新候选及DIN适配','']
    adapter=v['adaptation'];adapt_tables=[]
    if adapter is not None and adapter['status']=='PASS':
        lines+=['旧DIN不直接更换教师。两条链路均用各自冻结召回器seed42及自己的训练/验证Top100，保持原RichDIN结构/选择损失/真实曝光辅助BCE、8轮上限/patience2、64000查询/轮等预算。排序种子42/43/44共享各自冻结seed42召回器，因此这不是三条独立召回+精排全链路训练。',
            '候选未注入目标；候选内不明反馈仍是选择替代项，不作为CTR负标签。不同召回池的完整链路比较不称精排架构消融；新链路也须与其裸召回比较。','',
            '| 分区 | 链路教师 | 最终R20均值 | 最终N20均值 | 候选R100均值 | 最终长尾R20均值 |','|---|---|---:|---:|---:|---:|']
        for part in ['standard/validation','standard/test','random/validation','random/test']:
            for name,group in adapter['metrics'].items():
                m=[x[part] for x in group.values()]
                s=summary({'r20':[x['recall@20'] for x in m],'n20':[x['ndcg@20'] for x in m],
                    'r100':[x['recall@100'] for x in m],'tail20':[x['groups']['tail_target']['recall@20'] for x in m]})
                adapt_tables.append({'partition':part,'model':name,'metrics':s})
                lines.append(f'| {part} | {name} | {p(s["r20"]["mean"])} | {s["n20"]["mean"]:.6f} | {p(s["r100"]["mean"])} | {p(s["tail20"]["mean"])} |')
        lines+=['','原正式主种子V2→DIN的standard/test R20=13.4482%、N20=0.054840，仅作为冻结主种子参考；上表为适配排序三种子均值，不将两者混成同一统计口径。']
    else:lines.append('没有候选通过三种子门槛，未强行适配DIN。原正式链路保持不变。')
    lines+=['','## 资源、来源与核验','',
        f'实际核验{v["training_components_checked"]}个召回训练组件、{v["rank_files_recomputed"]}个完整排名文件、{v["actual_retriever_context_replays"]}条实际召回上下文；主种子控制/采用候选另建目录从头重训，参数、损失曲线与所选验证排名一致。',
        '训练流程时间包括逐轮验证、保存、重载；峰值Torch已分配显存不是整卡占用，推理计时不是线上SLO。每运行实际秒数/显存/参数量及混合权重ESS见training.json。',
        f'最终正式目录为`{out.relative_to(ROOT).as_posix()}`。retrieval_balanced_main保存20组训练，main_v2保存审查后的图与三种子训练；最终目录按实际图数组/训练源码/数据/配置签名只读复用这些组件，新进程冻结选择。GPU适配probe设备修复前的未完成目录保留，不充当完成证据；没有按测试添加配置或改变模型选型。引用的目录均需保留。',
        '正式旧源码、数据、V2/DIN及已发布首版未覆盖。本轮是独立候选及效果证据，不自动改动旧FrozenPipeline或GitHub发布。',
        '', '## 运行','', '```powershell',r'. .\scripts\enter_env.ps1',
        'python -B -m src.retrieval_balanced.run --phase tune',
        'python -B -m src.retrieval_balanced.run --phase replicate',
        'python -B -m src.retrieval_balanced.run --phase evaluate',
        'python -B -m src.retrieval_balanced.run --phase adapt',
        'python -B -m src.retrieval_balanced.run --phase reproduce',
        'python -B -m src.retrieval_balanced.run --phase verify',
        'python -B -m src.retrieval_balanced.run --phase report','```',
        '冻结目录拒绝重复调参/复测；独立主种子从头重训用reproduce。图和训练组件已经完成时只做指纹核验复用。清空或重命名产物不构成新的独立盲测。']
    result={'retrieval_tables':tables,'adapted_pipeline_tables':adapt_tables,'verification':v,
        'validation_candidates':f['candidate_names'],'selected_for_adaptation':rep['selected_for_adaptation'],
        'original_formal_pipeline_replaced':False,'test_is_iterative_reassessment':True}
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');write_json(out/'summary.json',result)
    mirrors={}
    if out==(ROOT/cfg['output_directory']).resolve():
        for name,source in [('reports/balanced_retrieval_report.md',out/'report.md'),('reports/balanced_retrieval_results.json',out/'summary.json')]:
            (ROOT/name).write_bytes(source.read_bytes());mirrors[name]=digest_file(ROOT/name)
    paths=[out/'report.md',out/'summary.json',out/'verification.json',out/'evaluation_manifest.json',out/'reproduction.json']
    if (out/'adaptation_manifest.json').exists():paths.append(out/'adaptation_manifest.json')
    write_json(out/'report_manifest.json',{'output_hashes':{str(x.relative_to(out)):digest_file(x) for x in paths},'global_mirrors':mirrors})
    validate_delivery(out);return result
