"""Aggregate only completed frozen retrieval experiments, without positive spin."""
import csv
import json
from pathlib import Path
import numpy as np
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.provenance import verify_frozen_inputs
from src.retrieval.statistics import paired_user_bootstrap,metric_values


def main():
    from src.retrieval.run import load_setup,write_json
    config,paths,vocabulary,counts,_=load_setup([])
    selection=json.loads((paths['output']/'selection.json').read_text());verify_frozen_inputs(selection,paths['prepared'])
    if config!=selection['config']:raise ValueError('Report configuration differs from frozen selection')
    evaluated=json.loads((paths['output']/'evaluation.json').read_text())
    if evaluated['smoke_only'] or evaluated['selection_sha256']!=digest_file(paths['output']/'selection.json'):
        raise ValueError('Only current full frozen evaluation can be reported')
    results=evaluated['results'];primary=f'two_tower_seed{config["primary_seed"]}'
    ablation=f'two_tower_no_history_seed{config["ablation_seed"]}'
    summaries={};intervals={};statistics_inputs={}
    for partition in ['standard/test','random/test']:
        summaries[partition]={}
        for name in ['recall@10','recall@20','ndcg@10','ndcg@20','mrr','coverage_at20','tail_coverage_at20']:
            values=np.array([results[f'two_tower_seed{seed}'][partition][name] for seed in config['seeds']])
            summaries[partition][name]={'mean':float(values.mean()),'sample_std':float(values.std(ddof=1))}
        path=partition.replace('/','_')+'.npz'
        with np.load(paths['artifacts']/'ranks'/primary/path) as model,np.load(paths['artifacts']/'ranks/itemcf'/path) as baseline:
            if not np.array_equal(model['event_ids'],baseline['event_ids']) or not np.array_equal(model['users'],baseline['users']):
                raise ValueError('Models did not evaluate the same paired queries')
            for data,name in [(model,primary),(baseline,'itemcf')]:
                expected=[results[name][partition][metric] for metric in ['recall@20','ndcg@20','mrr']]
                np.testing.assert_allclose(metric_values(data['ranks']).mean(0),expected,rtol=1e-12,atol=1e-12)
            intervals[partition]=paired_user_bootstrap(model['ranks'],baseline['ranks'],model['users'],config['bootstrap_repeats'],config['bootstrap_seed'])
        statistics_inputs[partition]={name:digest_file(paths['artifacts']/'ranks'/name/path) for name in [primary,'itemcf']}
    export=json.loads((paths['output']/'exports.json').read_text())
    candidates=json.loads((paths['output']/'candidates_standard_validation.json').read_text())
    artifact_checks=json.loads((ROOT/'reports/retrieval_artifact_validation.json').read_text())
    if artifact_checks['status']!='PASS':raise ValueError('Actual artifact validation required')
    for filename,expected in export['files'].items():
        if digest_file(paths['artifacts']/'snapshot'/filename)!=expected:raise ValueError('Export file changed')
    if digest_file(candidates['path'])!=candidates['sha256']:raise ValueError('Event candidate file changed')
    table=[]
    for model,partitions in results.items():
        for key,value in partitions.items():
            table.append({'model':model,'partition':key,**{name:value[name] for name in ['queries','recall@10','recall@20','ndcg@10','ndcg@20','mrr','coverage_at20','tail_coverage_at20','cold_item_queries','cold_user_queries','evaluation_seconds']}})
    with (ROOT/'reports/retrieval_metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    tuning=json.loads((ROOT/'reports/retrieval_tuning.json').read_text())
    payload={'status':'VERIFIED','selected_output':str(paths['output'].relative_to(ROOT)),
             'selection_sha256':digest_file(paths['output']/'selection.json'),
             'evaluation_sha256':digest_file(paths['output']/'evaluation.json'),
             'seed_summary':summaries,'paired_primary_minus_itemcf_user_cluster_ci95':intervals,
             'bootstrap_repeats':config['bootstrap_repeats'],'bootstrap_seed':config['bootstrap_seed'],
             'results':results,'exports':export,'validation_candidates':candidates,
             'tuning':tuning,'report_code_sha256':digest_file(__file__),
             'statistics_code_sha256':digest_file(Path(__file__).parent/'statistics.py')}
    payload['statistics_input_sha256']=statistics_inputs
    payload['artifact_validation']=artifact_checks
    write_json(ROOT/'reports/retrieval_results.json',payload)
    lines=['# KuaiRand-Pure 召回实验报告','',
           '本报告只使用真实训练、冻结选模和完整评测产物。当前轻量双塔是否优于基线，以各来源完整结果为准；没有线上收益、因果去偏或离线策略价值结论。','',
           '## 数据、模型与评测协议','',
           '- 训练：标准训练期 1,125,689 条曝光，主标签为发布的点击/有效播放值，用户 26,211、视频 7,538；词表与流行度仅使用训练期。',
           '- 热门度：训练正反馈计数排序。ItemCF：训练用户—视频二值正反馈矩阵，复用 RecBole 1.2.1 ComputeSimilarity 的余弦核心，滚动历史加权召回；相似度为零的候选以热门度回填，允许重复推荐。',
           '- 双塔：用户 ID 嵌入 + 最近最多 50 个标准正反馈的均值嵌入 + 场景嵌入，经 MLP 得到归一化向量；视频 ID 经独立 MLP 得到归一化视频向量；评分为点积除以温度。RecBole DSSM 的静态特征路径不能直接表达该动态上下文，故按任务书补充轻量 PyTorch 实现。',
           '- BCE 只针对真实曝光标签；附加对比项仅使用去重后的批内正视频池，屏蔽历史中已知正反馈（目标除外）。它们是未观测匹配的代理负项，不是已知负反馈，仍有未知正样本与流行度采样偏差；没有称其为去偏损失。',
           f'- 选定对比项权重 {config["contrastive_weight"]}；用户/视频基础嵌入 {config["embedding_dim"]}，输出 {config["output_dim"]}，隐藏层 {config["hidden_dim"]}，温度 {config["temperature"]}；训练种子 {config["seeds"]}。',
           '- 同一候选集为全部训练期已知 7,538 个视频，排除填充索引 0，允许重复推荐，不强行插入目标，也不做负采样候选评测。逐事件读取已审计的严格过去历史，随机反馈不更新历史。',
           '- 评测查询仅包含该来源实际正反馈事件，每条只有一个已知相关目标，因此这里 Recall@K 等于命中率；NDCG 为单目标折损，MRR 遍历完整候选集，不是 MRR@20。总体按事件加权，同时提供按用户等权平均。未知目标仍留在分母，计为未命中。',
           '- 分数按 float32 排名，相同分数按编码视频 ID 升序；编码与原始视频 ID 顺序一致。TopK 使用整数排序键保留精确 float32 次序，不用浮点扰动打破并列。',
           '- 验证集用于选邻居数和训练轮次；正式测试在所有模型选定后运行。小样例只验证程序、标记 smoke_only，不纳入实验表和调参依据。','',
           '## 验证集选模','', '| 对比项权重（主种子42） | 最佳验证 NDCG@20 |','| --- | ---: |']
    for row in tuning['trials']:lines.append(f'| {row["contrastive_weight"]} | {row["best_validation_ndcg20"]:.6f} |')
    lines += ['', '初始配置的验证表现弱于 ItemCF，随后只在标准验证集比较三档权重，没有扩大测试集调参。选中的配置在独立目录重训 42/43/44，主结果预先指定种子42，不挑选最好种子。历史关闭对照也使用相同权重和种子42，具有相同训练预算与验证早停规则；选中的轮次可能不同。','',
              '| 模型 | 最佳验证轮次（从0起） | 最佳验证 NDCG@20 | 训练流程秒数 | 峰值已分配显存 MiB |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name,run in selection['training_runs'].items():
        lines.append(f'| {name} | {run["best_epoch"]} | {run["best_validation_ndcg20"]:.6f} | {run["elapsed_seconds"]:.2f} | {run["peak_torch_allocated_MiB"]:.2f} |')
    lines += ['', f'ItemCF 从 {config["itemcf_neighbors"]} 中按标准验证 NDCG@20 选中 k={selection["selected_itemcf_neighbors"]}；余弦拟合约 {selection["itemcf_fit_seconds"]:.2f} 秒。每个训练流程耗时含验证、保存及重载检查；学习曲线记录逐轮 BCE、对比项和验证指标。','']
    for partition in ['standard/test','random/test']:
        lines += [f'## {partition} 完整结果','',
                  '| 模型 | 查询数 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MRR | Top20覆盖率 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for model,partitions in results.items():
            row=partitions[partition]
            lines.append(f'| {model} | {row["queries"]:,} | {row["recall@10"]:.6f} | {row["recall@20"]:.6f} | {row["ndcg@10"]:.6f} | {row["ndcg@20"]:.6f} | {row["mrr"]:.6f} | {row["coverage_at20"]:.4%} |')
        mean=summaries[partition]['recall@20'];comparison=intervals[partition]['recall@20']
        lines += ['',f'双塔三种子 Recall@20：{mean["mean"]:.6f} ± {mean["sample_std"]:.6f}（样本标准差）。主种子相对 ItemCF 的绝对差值为 {comparison["difference"]:.6f}，配对用户簇 bootstrap 95% 区间 [{comparison["ci95_low"]:.6f}, {comparison["ci95_high"]:.6f}]。负值表示低于 ItemCF，不能写成提升。',
                  f'区间采用 {config["bootstrap_repeats"]} 次按完整用户簇重采样，保持每事件计数的分母；这是固定已训练模型与已记录查询上的抽样不确定性，不包含模型选择、种子及线上策略价值的不确定性。','',
                  '| 模型 | 用户等权 Recall@20 | 尾部目标 Recall@20 | 尾部 Top20覆盖率 | 冷目标查询数 | 冷用户查询数 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for model in ['popularity','itemcf',primary,ablation]:
            row=results[model][partition];tail=row['groups']['tail_target']
            lines.append(f'| {model} | {row["user_macro"]["recall@20"]:.6f} | {tail.get("recall@20",0):.6f} | {row["tail_coverage_at20"]:.4%} | {row["cold_item_queries"]:,} | {row["cold_user_queries"]:,} |')
        main=results[primary][partition];control=results[ablation][partition]
        lines += ['',f'历史消融：主种子含历史 Recall@20={main["recall@20"]:.6f}，关闭历史={control["recall@20"]:.6f}。这是一个种子的受控对照，不声称跨种子统计显著。','']
    lines += ['## 覆盖、导出与可复现边界','',
              '- 候选覆盖率是所有正反馈查询 Top20 的视频去重数/7,538。头部按训练正反馈频次最高约20%的视频定义，其余为尾部；冷目标单独报告，不用测试期流行度分组。全字段与分组结果见 retrieval_results.json。',
              '- 评测耗时与吞吐包含打分、完整目标排名和 Top20 计算，且热门度使用预计算全局排名，不代表公平的线上单请求延迟或 P99 SLA。CPU峰值是进程累计峰值，GPU峰值是 PyTorch 已分配显存，不等同于整张显卡占用。',
              f'- 已导出 {export["users"]:,} 个训练结束上下文用户向量、{export["items"]:,} 个视频向量及每用户 Top{export["topk"]}；用户向量不是跨所有历史/场景不变的静态身份向量。点积与模型评分抽查 {export["dot_score_probe_count"]} 条，最大绝对误差 {export["dot_score_max_abs_error"]:.8g}。',
              f'- 另对 standard/validation 全部 {candidates["rows"]:,} 条曝光（含正负标签）生成 Top{candidates["topk"]}；候选文件只有分数与 ID，不给未曝光候选制造反馈标签，未强行加入目标。',
              '- 权重、用户/视频向量、逐查询排名和候选均在被 .gitignore 排除的 artifacts 下；配置、聚合指标与报告可供后续发布。输入文件、词表、评分代码、检查点与导出均保留指纹或哈希。',
              f'- 独立产物检查逐行验证了全部 {artifact_checks["candidate_rows_checked"]["snapshot"]:,} 条用户快照和 {artifact_checks["candidate_rows_checked"]["validation_events"]:,} 条事件候选的 ID、唯一性、分数次序与并列次序；对全部快照候选重算向量点积，最大误差 {artifact_checks["snapshot_all_topk_dot_max_error"]:.8g}。另重算 {artifact_checks["event_queries_independently_rescored"]:,} 条事件的全候选评分与 Top100，并将日志目标/标签改为零，确认结果不变，防止目标泄漏。',
              '- Pure 是候选池过滤后的不完整序列；当前结果只描述标准/随机来源日志中观测正反馈的全候选检索，不能当作真实 IPS/DR、无偏因果收益或 CQL 策略价值。侧信息快照和整月统计仍未作为输入。',
              '- 当前轻量双塔未被预设为胜过 ItemCF；若表中较差，应保留这一结果。可进一步研究更合适的采样、稀疏用户表示与可靠的侧信息，不能用持续查看测试集来修饰结果。','',
              '## 实际运行入口','', '```powershell', '. .\\scripts\\enter_env.ps1',
              'python -B -m src.retrieval.run --phase fit','python -B -m src.retrieval.run --phase evaluate',
              'python -B -m src.retrieval.candidates','python -B -m src.retrieval.candidates --partition standard/validation',
              'python -B scripts\\verify_retrieval_outputs.py',
              'python -B -m src.retrieval.report','python -B -m unittest discover -s tests -v','```','',
              '单独训练：`python -B -m src.retrieval.train --seed 42`；历史关闭：增加 `--no-history`；最小程序检查：`python -B -m src.retrieval.run --smoke`。改变协议/配置后选择新输出目录，禁止静默使用旧的冻结结果。','',
              f'最终实验目录：`{paths["output"].relative_to(ROOT)}`。初始配置和验证试验在 `experiments/retrieval/` 与 `reports/retrieval_tuning.json`；没有把废弃配置的种子结果混进最终三种子统计。','',
              '本阶段未进行 DIN、SASRec、IPS/SNIPS/DR 或 CQL 实验；原有 MovieLens 运行验证指标也没有混入本表。']
    (ROOT/'reports/retrieval_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Rendered reports/retrieval_report.md and retrieval_results.json',flush=True)


if __name__=='__main__':main()
