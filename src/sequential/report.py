"""Sequence results with catalog retrieval and fixed-pool ordering distinguished."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from src.data.common import ROOT
from src.data.download import digest_file
from src.ranking.data import write_json
from src.retrieval.statistics import paired_user_bootstrap
from src.retrieval_v2.provenance import verify_evidence
from src.sequential.provenance import read,verify_previous


def paired(a,b,cfg,bounded=False):
    with np.load(a) as x,np.load(b) as y:
        np.testing.assert_array_equal(x['event_ids'],y['event_ids']);np.testing.assert_array_equal(x['users'],y['users'])
        ci=paired_user_bootstrap(x['ranks'],y['ranks'],x['users'],cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
    if bounded:ci['mrr@100']=ci.pop('mrr')
    return ci


def main():
    verify_previous();out=ROOT/'experiments/sequential';sel=read(out/'selection.json');cfg=sel['config'];rep=read(out/'replication.json')
    ev=read(out/'evaluation.json');ledger=read(out/'tuning.json');validation=read(ROOT/'reports/sequential_artifact_validation.json')
    verify_evidence(ROOT,sel['evidence'],cfg);verify_evidence(ROOT,rep['evidence'],cfg)
    if validation['status']!='PASS':raise ValueError('Actual sequence artifacts must pass')
    for p,h in validation['artifact_hashes'].items():
        if digest_file(ROOT/p)!=h:raise ValueError('Verified sequence artifact changed')
    tests=(ROOT/'reports/environment_logs/all_tests_sequential.txt').read_text(encoding='utf-8');n=re.findall(r'Ran (\d+) tests',tests)
    if not n or not tests.strip().endswith('OK'):raise ValueError('Full passing tests required')
    if read(ROOT/'reports/environment_python312_check.json')['status']!='PASS':raise ValueError('Environment must pass')
    baselines=read(ROOT/'experiments/retrieval_v2/evaluation.json')['results']
    din=read(ROOT/'experiments/ranking_v2/evaluation.json');old_din=read(ROOT/'experiments/ranking/evaluation.json')
    seeds={};ci={};rows=[]
    for part in ['standard/validation','standard/test','random/validation','random/test']:
        seeds[part]={}
        for mode in ['catalog','fixed_v2_pool']:
            seeds[part][mode]={}
            for key in ['recall@20','ndcg@20','recall@100']:
                values=[ev['results'][f'sasrec_seed{s}'][part][mode][key] for s in [42,43,44]]
                seeds[part][mode][key]={'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
        for name in ['popularity','itemcf','two_tower_v2_seed42']:
            m=baselines[name][part];rows.append({'partition':part,'pool':'catalog','model':name,
                **{k:m[k] for k in ['recall@20','ndcg@20','recall@100']},'mrr':m['mrr'],'mrr_cutoff':'full_catalog'})
        for name,parts in ev['results'].items():
            for mode in ['catalog','fixed_v2_pool']:
                m=parts[part][mode];rows.append({'partition':part,'pool':mode,'model':name,
                    **{k:m[k] for k in ['recall@20','ndcg@20','recall@100']},
                    'mrr':m['mrr'] if mode=='catalog' else m['mrr@100'],'mrr_cutoff':'full_catalog' if mode=='catalog' else '100'})
        for name,m in [('v2_order',din['baseline'][part]),('din_revision',din['results']['din_seed42'][part]['ranking']),
                       ('old_pure_din',old_din['results']['din_seed42'][part]['reranking']['results']['pure'])]:
            rows.append({'partition':part,'pool':'fixed_v2_pool','model':name,
                         **{k:m[k] for k in ['recall@20','ndcg@20','recall@100']},'mrr':m['mrr@100'],'mrr_cutoff':'100'})
        if part.endswith('/test'):
            primary=ev['results']['sasrec_seed42'][part]
            ci[part]={'catalog_vs_v2':paired(primary['catalog_rank_artifact'],baselines['two_tower_v2_seed42'][part]['rank_artifact'],cfg),
                      'catalog_vs_itemcf':paired(primary['catalog_rank_artifact'],baselines['itemcf'][part]['rank_artifact'],cfg),
                      'catalog_vs_no_position':paired(primary['catalog_rank_artifact'],ev['results']['no_position'][part]['catalog_rank_artifact'],cfg),
                      'pool_vs_v2':paired(primary['pool_rank_artifact'],din['baseline'][part]['artifact'],cfg,True),
                      'pool_vs_din':paired(primary['pool_rank_artifact'],din['results']['din_seed42'][part]['rank_artifact'],cfg,True)}
    std=ev['results']['sasrec_seed42']['standard/test'];v2=baselines['two_tower_v2_seed42']['standard/test']
    assessment={'module_implemented_run_verified':True,'standard_catalog_ndcg20':std['catalog']['ndcg@20'],
                'standard_catalog_v2_ndcg20':v2['ndcg@20'],'standard_catalog_ndcg20_vs_v2_relative':std['catalog']['ndcg@20']/v2['ndcg@20']-1,
                'standard_catalog_vs_v2_ci_positive':ci['standard/test']['catalog_vs_v2']['ndcg@20']['ci95_low']>0,
                'standard_pool_vs_din_ci_positive':ci['standard/test']['pool_vs_din']['ndcg@20']['ci95_low']>0,
                'current_main_pipeline_replaced':False,'no_ctr_or_causal_claim':True}
    payload={'status':'VERIFIED','selected_trial':sel['trial'],'config':cfg,'quality_assessment':assessment,
             'test_is_iterative_reassessment':True,'validation_trials':ledger['trials'],'results':ev,
             'three_seed_summary':seeds,'paired_user_bootstrap':ci,'baselines':baselines,
             'artifact_validation':validation,'tests_passed':int(n[-1]),
             'selection_sha256':digest_file(out/'selection.json'),'replication_sha256':digest_file(out/'replication.json'),
             'evaluation_sha256':digest_file(out/'evaluation.json'),'test_log_sha256':digest_file(ROOT/'reports/environment_logs/all_tests_sequential.txt')}
    write_json(ROOT/'reports/sequential_results.json',payload)
    with (ROOT/'reports/sequential_metrics.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# SASRec序列推荐实际实验报告','',
          f'已实现、已训练、已验证。标准测试全候选NDCG@20={std["catalog"]["ndcg@20"]:.6f}，V2={v2["ndcg@20"]:.6f}，相对变化={assessment["standard_catalog_ndcg20_vs_v2_relative"]:.2%}。是否领先以同口径对照和区间为准。',
          'SASRec是独立序列推荐模型；现有主流程仍为固定双塔V2加DIN残差精排，本阶段没有强制串联或替换它。','',
          '## 上游复用与模型适配','',
          '- [SASRec论文](https://arxiv.org/abs/1808.09781)',
          '- [实际采用RecBole1.2.1 SASRec](https://github.com/RUCAIBox/RecBole/blob/v1.2.1/recbole/model/sequential_recommender/sasrec.py)','',
          '直接实例化原版SASRec，复用TransformerEncoder、位置嵌入、原生因果mask、forward和calculate_loss流程；未重写注意力、未改上游包、未安装依赖。只用薄数据/损失适配层：排除padding类别0，初始化padding向量为0，空历史回退训练期热门度。无用户ID嵌入或DIN交互头。',
          'RecBole此版用全连接前馈层，属于其SASRec实现；不冒称逐行复现原作者TensorFlow代码。CE是下一个观测正项的隐式分类损失，其他类别不是已知曝光负反馈，也不是CTR或propensity。','',
          '## 数据与时间协议','',
          '沿用既有训练/验证/测试时间边界、训练词表、标准正历史和数据指纹。历史严格早于当前事件，同毫秒事件不互相进入历史；随机来源反馈不更新历史。验证/测试可使用更早的标准已观测反馈，模型参数始终冻结，这是滚动评测。',
          '526000条训练正事件中499735条历史非空，进入序列CE；26265条空历史只在序列训练中跳过。评测仍保留全部正事件及冷记录，空历史明确回退训练期log1p热门度，不把当前目标拼回历史。',
          '截取最近H项、保留从早到晚顺序、右侧padding，长度不含0。Pure已删除候选池外交互，所以预测的是保留日志中的后续正项，不能代表完整真实下一次行为。','',
          '## 有限验证实验与选模','',
          '两组主配置H20/H50，隐藏32、2层2头、全候选CE、Adam lr.001；最多8轮、patience2，完整标准验证NDCG20选checkpoint。再作匹配的固定零位置嵌入训练消融；此控制仍保留因果mask，不称彻底删除时序信息。种子42预定主结果，43/44复核，控制模型不混入主模型选取。',
          f'选定`{sel["trial"]}`，历史{cfg["history_length"]}，batch{cfg["batch_size"]}。初始epoch-1仅保存基线，正式best必须来自已训练轮次。','',
          '| 验证配置 | 最佳轮次（从0起） | Recall@20 | NDCG@20 | Recall@100 |', '| --- | ---: | ---: | ---: | ---: |']
    for name,t in ledger['trials'].items():
        m=t['best_validation'];text.append(f'| {name} | {t["best_epoch"]} | {m["recall@20"]:.6f} | {m["ndcg@20"]:.6f} | {m["recall@100"]:.6f} |')
    text+=['','首个1024批大小预运行未完成一轮即中断，源代码/初始检查点保留在sequential_preflight_20261028；修复评测可选保存和输出目录保护后，从新目录正式运行。真实批次吞吐测量选择8192，测量不包含数据IO、不作为质量结果；证据sequential_throughput.json。','',
           '## 指标可比性','',
           '全候选：搜索全部7538个训练视频，允许重复，目标未知记0，完整正事件分母不变；与热门度/ItemCF/V2直接比较，MRR计到全部已知候选。',
           '固定池：只在同一V2Top100中用SASRec打分，未召回目标仍记0，Recall100保持V2值；与DIN比较该口径，MRR记为MRR@100。两表候选空间不同，不能将差异全部归于序列架构。',
           '没有单独训练真实曝光概率预测器，因此不对SASRec报告CTR AUC/LogLoss。测试此前已被旧阶段查看，本轮为迭代复测；随机来源不参与调参，未根据测试改配置。','']
    for part in ['standard/test','random/test']:
        text+=['## '+part,'']
        for mode,title in [('catalog','全候选检索'),('fixed_v2_pool','固定V2Top100重排')]:
            text += ['### '+title,'','| 模型 | Recall20 | NDCG20 | Recall100 | '+('MRR全候选' if mode=='catalog' else 'MRR@100')+' |',
                     '| --- | ---: | ---: | ---: | ---: |']
            for row in rows:
                if row['partition']==part and row['pool']==mode:
                    text.append(f'| {row["model"]} | {row["recall@20"]:.6f} | {row["ndcg@20"]:.6f} | {row["recall@100"]:.6f} | {row["mrr"]:.6f} |')
            s=seeds[part][mode]['ndcg@20'];text+=['',f'三种子NDCG20均值{s["mean"]:.6f}，样本标准差{s["sample_sd"]:.6f}。','']
        text+=['| SASRec主种子减对照：用户簇配对bootstrap500次 | NDCG20差 | 95%区间 |','| --- | ---: | --- |']
        for name,value in ci[part].items():
            c=value['ndcg@20'];text.append(f'| {name} | {c["difference"]:.6f} | [{c["ci95_low"]:.6f}, {c["ci95_high"]:.6f}] |')
        text+=['','区间跨0时不宣称优于该对照；区间仅反映用户抽样不确定性，不包含训练/调参方差。','']
        groups=ev['results']['sasrec_seed42'][part]['catalog']['groups']
        text+=['| 全候选分组 | 正事件数 | NDCG20 |','| --- | ---: | ---: |']
        for name in ['head_target','tail_target','cold_user','cold_target','empty_history','nonempty_history']:
            g=groups[name];text.append(f'| {name} | {g["queries"]} | '+(f'{g["ndcg@20"]:.6f}' if g['queries'] else '—')+' |')
        text+=['']
    text+=['## 实际训练与复核证据','', '| 运行 | 最佳轮次 | 总耗时秒 | CUDA已分配峰值MiB | 进程历史工作集峰值MiB |',
           '| --- | ---: | ---: | ---: | ---: |']
    for name,run in rep['runs'].items():text.append(f'| {name} | {run["best_epoch"]} | {run["elapsed_seconds"]:.1f} | {run["peak_torch_allocated_MiB"]:.1f} | {run["peak_process_working_set_MiB"]:.1f} |')
    text+=['','总耗时含完整验证与保存重载；显存为PyTorch已分配峰值，进程峰值口径为同进程累计，不是整机占用或线上P99。逐分区推理时间和覆盖率详见机器结果。',
           f'{n[-1]}项测试及环境检查通过。主种子42四分区完整全候选/固定池预测与Top100导出逐条重算一致；所有模型排名/事件/用户及由排名派生的分组、宏平均和指标聚合核对通过，覆盖率另核验主种子42，旧V2/DIN冻结证据未改变。',
           '种子43/44完成实际评测及完整验证重算，不声称对其测试预测全部重复打分；新环境完整重训尚未验收。',
           '机器结果sequential_results.json，平表sequential_metrics.csv，产物核验sequential_artifact_validation.json。模型、Top100、视频向量在experiments/sequential/artifacts。','',
           '```powershell','. .\\scripts\\enter_env.ps1',
           'python -B scripts\\train_sequential.py --output-directory experiments/sequential_reproduction/seed42 --seed 42',
           'python -B -m src.sequential.verify','python -B -m src.sequential.report','```','',
           '训练目录已冻结，不追加trial或覆盖旧结果。SASRec模块完成不等同项目全部完成；IPS/SNIPS/DR、CQL与最终GitHub/简历交付尚待后续阶段。']
    (ROOT/'reports/sequential_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(assessment,indent=2),flush=True);return payload


if __name__=='__main__':main()
