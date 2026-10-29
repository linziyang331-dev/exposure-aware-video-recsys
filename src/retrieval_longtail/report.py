"""Chinese results with validation/test and retrieval/end-to-end boundaries."""
from pathlib import Path
import numpy as np
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval_longtail.train import read_json,write_json
from src.retrieval_longtail.run import load_selection
from src.retrieval_longtail.verify import verify,validate_outputs


def pct(x):return f'{100*x:.4f}%'


def report(cfg,output):
    verification=verify(cfg,output);selection=load_selection(output)
    evaluation=read_json(output/'evaluation.json');replication=read_json(output/'replication.json')
    diagnosis=read_json(output/'diagnosis.json');rows=[]
    text=['# 第一阶段长尾召回优化：实际结果','',
        '本轮只优化单模型召回，旧双塔V2与DIN冻结不变。标准训练/验证选型，随机来源不训练、不选型；测试是已看过数据的迭代复测。所有百分数的Recall均为单正反馈事件命中率。',
        '', '## 诊断与做了什么','',
        f'训练目录{diagnosis["catalog_items"]}项、标准训练正事件{diagnosis["positive_training_events"]:,}条；其中{diagnosis["catalog_support"]["freq_zero"]["catalog_items"]}项无训练点击正例，视频正例数中位数{diagnosis["training_positive_count_quantiles"][2]:g}。完整频次桶、独立正用户数、验证排名和固定分层抽样分数差见diagnosis.json。',
        '原模型只有视频ID嵌入与固定位置衰减历史。本轮对照：较慢学习率、全局归一且原始比例上限3的频次加权选择损失、候选无关的上下文注意力和逐查询历史门控。没有新增侧信息或多路融合，也没有把频次当作真实曝光概率。',
        '固定8配置、相同12轮上限/patience3、按标准验证NDCG20保存最佳轮次；门槛为总体R20/N20至少保留99%、头部R20至少保留98%、长尾R20与R100均提高。先设门槛再运行；这些保留比例是本轮实验取舍，用户未指定业务容忍度。',
        '', '## 标准验证选型（seed42）','',
        '| 配置 | 最佳轮（从0） | R20 | N20 | 头部R20 | 长尾R20 | 长尾R100 | 通过门槛 |',
        '|---|---:|---:|---:|---:|---:|---:|:---:|']
    from src.retrieval_longtail.contracts import selection_passes
    baseline=selection['metrics']['control']
    for name,run in selection['runs'].items():
        m=run['best_validation'];g=m['groups'];passed=selection_passes(m,baseline,cfg['gate'])
        text.append(f'| {name} | {run["best_epoch"]} | {pct(m["recall@20"])} | {m["ndcg@20"]:.6f} | {pct(g["head_target"]["recall@20"])} | {pct(g["tail_target"]["recall@20"])} | {pct(g["tail_target"]["recall@100"])} | {"是" if passed else "否"} |')
    chosen=selection['selected_trial']
    text+=['',f'冻结候选：`{chosen}`。原控制从头训练参数摘要与冻结V2完全一致。候选与同预算控制均独立从头运行42/43/44；并非共享骨干继续训练。三种子验证均值门槛：{replication["three_seed_validation_gate_passed"]}。','',
        '## 冻结后复测（三种子均值 ± 样本标准差）','',
        '| 来源/分区 | 模型 | R20 | N20 | R100 | 长尾R20 | 长尾R100 | Top20目录覆盖率 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for part in ['standard/validation','standard/test','random/validation','random/test']:
        for group,seeds in evaluation['results'].items():
            values=[r[part] for r in seeds.values()]
            row={'partition':part,'group':group,'seeds':list(seeds)}
            cells=[]
            for label,getter in [('recall20',lambda m:m['recall@20']),('ndcg20',lambda m:m['ndcg@20']),
                ('recall100',lambda m:m['recall@100']),('tail_recall20',lambda m:m['groups']['tail_target']['recall@20']),
                ('tail_recall100',lambda m:m['groups']['tail_target']['recall@100']),('coverage20',lambda m:m['coverage_at20'])]:
                v=[getter(m) for m in values];mean=float(np.mean(v));sd=float(np.std(v,ddof=1)) if len(v)>1 else 0.
                row[label]={'mean':mean,'sample_sd':sd}
                cells.append(f'{mean:.6f} ± {sd:.6f}' if label=='ndcg20' else f'{pct(mean)} ± {pct(sd)}')
            rows.append(row);text.append('| '+part+' | '+group+' | '+' | '.join(cells)+' |')
    text+=['','## 主种子配对区间','',
        '按相同正事件与用户对齐，对完整用户簇重采样500次。差值为候选减控制；区间不包含训练种子和调参选择不确定性。这里是单模型召回，不是DIN最终Top20。','',
        '| 分区 | 分组 | R20差 | 95%区间 | R100差 | 95%区间 |','|---|---|---:|---|---:|---|']
    for part,groups in verification['paired_seed42_user_cluster_intervals'].items():
        for group,cis in groups.items():
            a=cis['recall@20'];b=cis['recall@100']
            text.append(f'| {part} | {group} | {a["difference"]:.6f} | [{a["ci95_low"]:.6f}, {a["ci95_high"]:.6f}] | {b["difference"]:.6f} | [{b["ci95_low"]:.6f}, {b["ci95_high"]:.6f}] |')
    text+=['','## 资源与核验','',
        '| 模型 | 参数量 | 训练流程秒 | 峰值Torch已分配显存MiB |','|---|---:|---:|---:|']
    for group,runs in replication['groups'].items():
        for seed,r in runs.items():text.append(f'| {group} seed{seed} | {r["parameter_count"]:,} | {r["elapsed_seconds"]:.2f} | {r["peak_torch_allocated_MiB"]} |')
    text+=['',f'核验{verification["rank_files_recomputed"]}个全量逐事件排名文件，实际检查点重放{verification["checkpoint_context_rank_replays"]:,}条上下文；{verification["training_runs_validated"]}次训练组件哈希/最佳验证排名重算通过，旧V2冻结文件{verification["old_v2_frozen_files_unchanged"]}个不变。主种子独立目录从头重训参数、每轮损失及验证排名一致。',
        '训练流程耗时包含每轮验证、保存和重载；显存为Torch已分配峰值，进程内存为累计峰值。评测耗时不是在线SLO，实际推理重放固定batch512，不保证其他批大小逐位一致。',
        '', '## 结论与下一步','']
    if chosen and replication['three_seed_validation_gate_passed']:
        text.append('候选通过预定三种子验证均值门槛，可进入新教师/新候选上的DIN适配对照；它仍是实验召回器，尚未替换正式链路。测试复测的具体取舍须按上表和逐组结果解释，不根据测试另选配置。')
    else:text.append('本轮没有候选通过完整验证门槛，继续保留旧双塔V2作为正式召回；失败配置及取舍全部保留，不依据测试扩大本轮调参预算。')
    text+=['长尾是否彻底解决不能由相对涨幅或目录覆盖率判断；仍需看绝对命中率、头部损失和完整链路。原DIN的教师特征绑定旧V2，新召回必须重新生成候选并适配DIN，之后才可判断最终Top20是否改善。',
        '零训练正例的目录项和真正目录外冷目标分开报告；本轮未引入经过时点审计的内容特征，因此没有解决普遍的新视频冷启动。随机来源诊断不是无偏线上价值。',
        '', '## 命令','', '```powershell','. .\scripts\enter_env.ps1',
        'python -B -m src.retrieval_longtail.run --phase all',
        'python -B -m src.retrieval_longtail.run --phase verify',
        'python -B -m src.retrieval_longtail.run --phase report','```',
        '已完成目录不允许重复调参/复测；完整重跑使用新的 experiments/retrieval_longtail_* 目录，--output-directory指定。主种子同机复现可使用--phase reproduce；原始产物不覆盖。']
    summary={'selected_trial':chosen,'three_seed_validation_gate_passed':replication['three_seed_validation_gate_passed'],
        'summary_rows':rows,'verification':verification,'end_to_end_din_adapted':False,
        'original_formal_pipeline_unchanged':True}
    path=output/'report.md';path.write_text('\n'.join(text)+'\n',encoding='utf-8')
    write_json(output/'summary.json',summary)
    paths=[path,output/'summary.json',output/'verification.json',output/'evaluation_manifest.json',output/'reproduction.json']
    # Only canonical output writes global mirrors. Independent runs cannot overwrite formal reports.
    if output==(ROOT/cfg['output_directory']).resolve():
        (ROOT/'reports/retrieval_longtail_report.md').write_bytes(path.read_bytes())
        write_json(ROOT/'reports/retrieval_longtail_results.json',summary)
    write_json(output/'report_manifest.json',{'output_hashes':{str(p.relative_to(output)):digest_file(p) for p in paths}})
    validate_outputs(output,output/'report_manifest.json')
    return summary
