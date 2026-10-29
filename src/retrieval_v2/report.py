"""Source-grounded optimization report; preserves failures and iterative-test scope."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.metrics import summarize_ranks
from src.retrieval.statistics import paired_user_bootstrap
from src.retrieval_v2.provenance import verify_evidence
from src.retrieval_v2.run import write_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    output=ROOT/'experiments/retrieval_v2';selection=read(output/'selection.json');cfg=selection['config']
    replication=read(output/'replication.json');evaluation=read(output/'evaluation.json');tuning=read(output/'tuning.json')
    exports=read(output/'exports.json');validation=read(ROOT/'reports/retrieval_optimization_artifact_validation.json')
    verify_evidence(ROOT,selection['evidence'],cfg);verify_evidence(ROOT,replication['evidence'],cfg)
    if evaluation['selection_sha256']!=digest_file(output/'selection.json') or evaluation['replication_sha256']!=digest_file(output/'replication.json'):
        raise ValueError('Evaluation selection evidence mismatch')
    if validation['status']!='PASS':raise ValueError('Actual artifact verification required')
    test_log=(ROOT/'reports/environment_logs/all_tests_retrieval_v2_stage.txt').read_text(encoding='utf-8')
    tests=re.findall(r'Ran (\d+) tests',test_log)
    if not tests or not test_log.strip().endswith('OK'):raise ValueError('Full passing test log required')
    environment=read(ROOT/'reports/environment_python312_check.json')
    if environment['status']!='PASS':raise ValueError('Environment check must pass')
    results=evaluation['results'];seeds={};bootstrap={};rows=[];rank_hashes={}
    for partition in ['standard/validation','standard/test','random/validation','random/test']:
        ranks={};reference=None
        for name,parts in results.items():
            metrics=parts[partition];path=Path(metrics['rank_artifact']);rank_hashes[str(path.relative_to(ROOT))]=digest_file(path)
            with np.load(path) as z:
                ranks[name]={'ranks':z['ranks'].copy(),'users':z['users'].copy(),'events':z['event_ids'].copy()}
            data=ranks[name]
            if reference is None:reference=data
            else:
                np.testing.assert_array_equal(data['events'],reference['events']);np.testing.assert_array_equal(data['users'],reference['users'])
            rebuilt=summarize_ranks(data['ranks'],data['users'],ks=(10,20,100))
            for metric in ['recall@10','recall@20','ndcg@10','ndcg@20','recall@100','ndcg@100','mrr']:
                if not np.isclose(rebuilt[metric],metrics[metric],rtol=0,atol=1e-12):raise ValueError('Rank/aggregate mismatch')
            rows.append({'partition':partition,'model':name,'queries':metrics['queries'],
                         **{k:metrics[k] for k in ['recall@10','recall@20','ndcg@10','ndcg@20','recall@100','ndcg@100','mrr','coverage_at20']}})
        seeds[partition]={}
        for metric in ['recall@20','ndcg@20','recall@100','ndcg@100','mrr']:
            values=[results[f'two_tower_v2_seed{s}'][partition][metric] for s in [42,43,44]]
            seeds[partition][metric]={'mean':float(np.mean(values)),'sample_standard_deviation':float(np.std(values,ddof=1)),'values':values}
        if partition.endswith('/test'):
            main_rank=ranks['two_tower_v2_seed42'];bootstrap[partition]={}
            for baseline in ['popularity','itemcf','two_tower_seed42']:
                bootstrap[partition][baseline]=paired_user_bootstrap(main_rank['ranks'],ranks[baseline]['ranks'],main_rank['users'],
                                                                  repeats=cfg['bootstrap_repeats'],seed=cfg['bootstrap_seed'])
    payload={'status':'VERIFIED','test_is_iterative_reassessment':True,'selected_trial':selection['trial'],
             'config':cfg,'validation_tuning':tuning,'results':results,'three_seed_summary':seeds,
             'paired_user_bootstrap':bootstrap,'exports':exports,'artifact_validation':validation,
             'tests_passed':int(tests[-1]),'rank_artifact_hashes':rank_hashes,
             'selection_sha256':digest_file(output/'selection.json'),'replication_sha256':digest_file(output/'replication.json'),
             'evaluation_sha256':digest_file(output/'evaluation.json'),
             'diagnostics':read(ROOT/'reports/retrieval_optimization_diagnostics.json')}
    write_json(ROOT/'reports/retrieval_optimization_results.json',payload)
    with (ROOT/'reports/retrieval_optimization_metrics.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# 双塔核心召回优化报告','',
          '本轮只优化双塔本身，没有融合 ItemCF/热门度分数，没有进入 DIN/SASRec 或去偏/RL。旧模型、代码与实验结果保留。',
          '', '## 实验与结论边界','',
          f'仅使用标准训练/验证集比较 {len(tuning["trials"])} 组配置，按验证 NDCG@20 选择 `{selection["trial"]}`，然后固定配置复核42/43/44。',
          '旧测试集在上一阶段已查看，本轮是迭代优化后的复测，不是全新未见的独立确认。没有用本轮测试结果继续调参。',
          '候选、查询、冷目标分母、时间协议与旧实验一致：全训练词表7538项、重复允许、未知目标计为未命中。Recall@20是单正反馈事件命中率，不是CTR。',
          '', '## 诊断与受控验证','',
          '原混合目标复现参数摘要与旧主种子完全相同。64条固定训练样本的可记忆性检查达到Top1=100%，仅说明训练链路能学习，不是泛化成绩。',
          '梯度抽查方向并非始终相反，不能仅因对比损失数值大就断言它是唯一问题。分别比较曝光BCE、批内对比、混合与全候选选择目标，保留所有失败配置。',
          '全候选softmax仅训练观测正反馈的相对选择，不给其余候选制造真实负反馈标签，不是无偏或因果估计。目标对比与结构对比分阶段进行；正例采样方式也随检索目标改变，因此不能把全部收益归因于某一个公式。',
          '', '| 验证配置（种子42） | Recall@20 | NDCG@20 |', '| --- | ---: | ---: |']
    for name,trial in tuning['trials'].items():
        m=trial['best_validation'];text.append(f'| {name} | {m["recall@20"]:.6f} | {m["ndcg@20"]:.6f} |')
    text += ['', '## 选定模型','',
             f'模型类型 `{cfg["model"]}`，目标 `{cfg["objective"]}`，历史 `{cfg["history_mode"]}`，长度{cfg["history_length"]}，基础维度{cfg["embedding_dim"]}，视频偏置{cfg["item_bias"]}。',
             '用户向量只依赖用户、历史与场景，视频向量独立编码；评分为点积/温度。若启用偏置，以用户常数维和视频偏置维增广，导出评分保持等价。',
             '残差模型将历史视频嵌入归一化后聚合，与用户和场景嵌入相加，再归一化；recency使用位置衰减0.9，不声称精确时间衰减，也不是DIN候选注意力。',
             '', '| 运行 | 最佳轮次（从0起） | 训练流程秒数 | 峰值PyTorch已分配显存MiB |', '| --- | ---: | ---: | ---: |']
    for name,run in replication['runs'].items():
        text.append(f'| {name} | {run["best_epoch"]} | {run["elapsed_seconds"]:.2f} | {run["peak_torch_allocated_MiB"]:.2f} |')
    text += ['', '流程耗时含验证、保存和重载，显存不是整卡占用；CPU峰值为进程累计峰值，不是独立模型内存。推理吞吐含完整候选打分、排名和Top20计算，不代表线上P99延迟。']
    for partition in ['standard/test','random/test']:
        text += ['', f'## {partition} 完整复测','', '| 模型 | Recall@20 | NDCG@20 | Recall@100 | Top20覆盖率 |', '| --- | ---: | ---: | ---: | ---: |']
        for name,parts in results.items():
            m=parts[partition];text.append(f'| {name} | {m["recall@20"]:.6f} | {m["ndcg@20"]:.6f} | {m["recall@100"]:.6f} | {m["coverage_at20"]:.2%} |')
        mean=seeds[partition]['recall@20']
        text += ['', f'双塔三种子Recall@20均值 {mean["mean"]:.6f}，样本标准差 {mean["sample_standard_deviation"]:.6f}。',
                 '', '| 主种子减去基线 | Recall@20绝对差 | 配对用户簇95%区间 |','| --- | ---: | --- |']
        for name,value in bootstrap[partition].items():
            r=value['recall@20'];text.append(f'| {name} | {r["difference"]:.6f} | [{r["ci95_low"]:.6f}, {r["ci95_high"]:.6f}] |')
        text += ['', '500次完整用户簇重采样，分母仍按事件计数；该区间不包含调参选择、训练种子或线上策略价值的不确定性。跨零不解释成稳定提升。']
    text += ['', '## 产物与验证','',
             f'已导出{exports["users"]}个训练结束上下文用户向量、{exports["items"]}个视频向量，实际导出维度{exports["dimension"]}；用户Top100与{exports["events"]["rows"]}条验证曝光Top100。',
             f'全量检查候选ID、唯一性、排序、原始ID映射和快照点积；抽样重算{validation["event_queries_independently_rescored"]}条事件候选。目标/标签置零的{exports["target_label_independence_queries"]}条评分完全不变。',
             f'全项目{int(tests[-1])}项测试通过，环境检查PASS。完整机器可读结果见 retrieval_optimization_results.json，表格见 retrieval_optimization_metrics.csv。',
             'Pure序列仍不完整，无真实逐事件倾向概率；本轮没有曝光去偏、线上收益或RL价值结论。历史消融只有主种子，不能扩展为跨种子显著性结论。',
             '', '## 实际命令','', '```powershell', '. .\\scripts\\enter_env.ps1',
             'python -B -m src.retrieval_v2.diagnose',
             '# 每组训练配置与完整命令见计划执行记录及 reports/environment_logs/retrieval_v2_*.txt',
             'python -B -m src.retrieval_v2.finalize --phase replicate',
             'python -B -m src.retrieval_v2.finalize --phase evaluate',
             'python -B -m src.retrieval_v2.finalize --phase export',
             'python -B -m src.retrieval_v2.verify',
             'python -B -m unittest discover -s tests -v',
             'python -B scripts/check_environment.py',
             'python -B -m src.retrieval_v2.report', '```','',
             '完成选模后禁止继续调参或静默重复测试。全新复现需使用独立输出目录；原始阶段的运行入口保持不变。']
    (ROOT/'reports/retrieval_optimization_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('VERIFIED retrieval optimization report; tests='+str(tests[-1]),flush=True)


if __name__=='__main__':main()
