"""Evidence-based DIN/LR report: exposure prediction is distinct from reranking."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.statistics import paired_user_bootstrap
from src.retrieval_v2.provenance import verify_evidence
from src.ranking.data import write_json
from src.ranking.provenance import verify_v2


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    frozen=verify_v2();output=ROOT/'experiments/ranking'
    selection=read(output/'selection.json');cfg=selection['config'];replication=read(output/'replication.json')
    evaluation=read(output/'evaluation.json');tuning=read(output/'tuning.json')
    validation=read(ROOT/'reports/ranking_artifact_validation.json');upstream=read(ROOT/'reports/ranking_upstream_audit.json')
    verify_evidence(ROOT,selection['evidence'],cfg);verify_evidence(ROOT,replication['evidence'],cfg)
    if validation['status']!='PASS':raise ValueError('Actual artifacts must pass validation')
    for name,expected in validation['artifact_hashes'].items():
        if digest_file(ROOT/name)!=expected:raise ValueError('Verified artifact changed')
    log=(ROOT/'reports/environment_logs/all_tests_ranking_stage.txt').read_text(encoding='utf-8')
    counts=re.findall(r'Ran (\d+) tests',log)
    if not counts or not log.strip().endswith('OK'):raise ValueError('Complete passing test log required')
    if read(ROOT/'reports/environment_python312_check.json')['status']!='PASS':raise ValueError('Environment must pass')
    results=evaluation['results'];seeds={};bootstrap={};rows=[]
    for partition,baseline in evaluation['baseline'].items():
        row={'partition':partition,'model':'v2','mode':'original','auc':None,'logloss':None,
             **{key:baseline[key] for key in ['recall@20','ndcg@20','recall@100','mrr@100']}}
        rows.append(row)
        for name,parts in results.items():
            for mode,m in parts[partition]['reranking']['results'].items():
                rows.append({'partition':partition,'model':name,'mode':mode,
                             **{k:parts[partition]['pointwise'][k] for k in ['auc','logloss']},
                             **{key:m[key] for key in ['recall@20','ndcg@20','recall@100','mrr@100']}})
        seeds[partition]={}
        for metric in ['auc','logloss']:
            values=[results[f'din_seed{s}'][partition]['pointwise'][metric] for s in [42,43,44]]
            seeds[partition][metric]={'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
        for mode in ['pure',selection['mode']]:
            seeds[partition][mode]={}
            for metric in ['ndcg@20','recall@20','recall@100','mrr@100']:
                values=[results[f'din_seed{s}'][partition]['reranking']['results'][mode][metric] for s in [42,43,44]]
                seeds[partition][mode][metric]={'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
        if partition.endswith('/test'):
            with np.load(baseline['artifact']) as z:base={k:z[k] for k in z.files}
            bootstrap[partition]={}
            for mode in sorted(set(['pure',selection['mode']])):
                path=Path(results['din_seed42'][partition]['rank_directory'])/(mode+'.npz')
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['event_ids'],base['event_ids']);np.testing.assert_array_equal(z['users'],base['users'])
                    interval=paired_user_bootstrap(z['ranks'],base['ranks'],z['users'],cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
                interval['mrr@100']=interval.pop('mrr');bootstrap[partition][mode]=interval
    payload={'status':'VERIFIED','stage':'upstream DIN ranking','pure_din_and_score_combination_separate':True,
             'test_is_iterative_reassessment':True,'selected_trial':selection['trial'],'selected_mode':selection['mode'],
             'selected_alpha':selection['alpha'],'config':cfg,'upstream_reuse':upstream,
             'validation_tuning':tuning,'results':results,'v2_baseline':evaluation['baseline'],
             'three_seed_summary':seeds,'paired_user_bootstrap_vs_v2':bootstrap,
             'artifact_validation':validation,'exports':evaluation['exports'],'tests_passed':int(counts[-1]),
             'v2_checkpoint_sha256':frozen['checkpoint_sha256'],'selection_sha256':digest_file(output/'selection.json'),
             'replication_sha256':digest_file(output/'replication.json'),'evaluation_sha256':digest_file(output/'evaluation.json')}
    base=evaluation['baseline']['standard/test']['ndcg@20']
    pure=results['din_seed42']['standard/test']['reranking']['results']['pure']['ndcg@20']
    selected=results['din_seed42']['standard/test']['reranking']['results'][selection['mode']]['ndcg@20']
    payload['quality_assessment']={'standard_pure_din_ndcg20':pure,'standard_v2_ndcg20':base,
        'standard_selected_pipeline_ndcg20':selected,'pipeline_uses_din':selection['alpha']!=0,
        'ndcg20_point_estimate_improved':selected>base,
        'ndcg20_paired_ci_above_zero':bootstrap['standard/test'][selection['mode']]['ndcg@20']['ci95_low']>0}
    write_json(ROOT/'reports/ranking_results.json',payload)
    with (ROOT/'reports/ranking_metrics.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# DIN 精排实际实验报告','',
          f'标准测试：纯DIN NDCG20={pure:.6f}，固定V2={base:.6f}，选定流程={selected:.6f}。是否提升以该对照及区间为准；不因模型更复杂而预设提升。',
          '若选中alpha0，表示DIN尚未被采用到最终排序，输出保留V2原顺序；这不是DIN带来的提升。','',
          '直接调用RecBole1.2.1完整DIN和LR；核心注意力、Dice/MLP与预测头未复制或重写。模型来源：',
          '- [RecBole DIN v1.2.1](https://github.com/RUCAIBox/RecBole/blob/v1.2.1/recbole/model/sequential_recommender/din.py)',
          '- [RecBole LR v1.2.1](https://github.com/RUCAIBox/RecBole/blob/v1.2.1/recbole/model/context_aware_recommender/lr.py)',
          '', '项目代码仅适配已审计ID、历史及训练/评测，DIN另外使用用户/场景加性偏置；这属于显式扩展，不声称原版DIN已使用这些上下文特征。实际上游文件SHA在ranking_upstream_audit.json及训练签名中。没有新装依赖，也没有使用DeepCTR-Torch替代运行结果。',
          '', '## 协议与选模','',
          '只用标准训练期真实曝光label0/1优化BCE，不把未曝光候选当已知负反馈。按验证LogLoss早停；标准验证固定Top100的NDCG20用于选择DIN配置及可选系数。随机来源不参与调参。',
          '标签沿用发布的点击/有效播放语义。只用ID、tab和严格过去标准正历史；冷记录保留，不使用快照/整月统计/当前反馈特征。',
          'V2权重、词表与89文件冻结证据保持不变。候选由同一V2主种子42生成；不强行插入目标。Recall100在所有重排中必须一致。',
          '每条观测正反馈仅一个已知目标，其余候选反馈未知。NDCG/Recall是此日志查询上的单正目标检索代理，不是无偏CTR、线上收益或策略价值。MRR只到候选100，记为MRR@100。',
          '旧测试在召回阶段已经查看，本次属于迭代复测；新DIN方案在完整测试前冻结，不用测试选择系数、种子或历史长度。',
          f'选定 `{selection["trial"]}`，模式 `{selection["mode"]}`，alpha={selection["alpha"]}。pure为纯DIN分数；alpha模式为V2_score+alpha*DIN_logit，必须单独解释；alpha0表示保留V2原排序。',
          '', '## 验证实验与历史敏感性','',
          '| 配置（种子42） | 历史长度 | batch | AUC | LogLoss | 纯模型NDCG20 | 最优模式 | 最优NDCG20 |',
          '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for name,trial in tuning['trials'].items():
        pw=trial['pointwise_validation'];m=trial['candidate_validation']['results'];mode=max(m,key=lambda k:m[k]['ndcg@20'])
        text.append(f'| {name} | {trial["config"]["history_length"]} | {trial["config"]["batch_size"]} | {pw["auc"]:.6f} | {pw["logloss"]:.6f} | {m["pure"]["ndcg@20"]:.6f} | {mode} | {m[mode]["ndcg@20"]:.6f} |')
    text += ['', '长度比较仅使用统一8192批大小的10/20/50正式配置。1024批大小的未完成性能预实验已中断，中间文件保留，未混入结果；合成张量吞吐测试也不作为质量指标。',
             'AUC/LogLoss与重排目标不同，不能用前者替代NDCG结论。用户/tab加性偏置在同一次查询内不改变候选顺序。',
             '', '## 选定模型实际训练','',
             '| 运行 | 最佳轮次（从0起） | 流程秒数 | 峰值PyTorch已分配MiB |', '| --- | ---: | ---: | ---: |']
    runs={'lr':selection['lr_training']}|replication['runs']
    for name,run in runs.items():text.append(f'| {name} | {run["best_epoch"]} | {run["elapsed_seconds"]:.2f} | {run["peak_torch_allocated_MiB"]:.2f} |')
    text += ['', '流程耗时包括验证、保存与重载；显存是PyTorch已分配峰值，不是整卡占用。CPU峰值为进程累计峰值；候选评分吞吐不是线上P99。']
    for partition in ['standard/test','random/test']:
        text += ['', '## '+partition,'', '| 模型 | 曝光AUC | 曝光LogLoss |', '| --- | ---: | ---: |']
        for name,parts in results.items():
            pw=parts[partition]['pointwise'];text.append(f'| {name} | {pw["auc"]:.6f} | {pw["logloss"]:.6f} |')
        text += ['', '| 同一Top100上的排序 | Recall20 | NDCG20 | Recall100 | MRR@100 |', '| --- | ---: | ---: | ---: | ---: |']
        baseline=evaluation['baseline'][partition]
        def row(label,m):return f'| {label} | {m["recall@20"]:.6f} | {m["ndcg@20"]:.6f} | {m["recall@100"]:.6f} | {m["mrr@100"]:.6f} |'
        text.append(row('V2原顺序',baseline))
        for name,parts in results.items():
            text.append(row(name+' 纯模型',parts[partition]['reranking']['results']['pure']))
            mode=selection['lr_mode'] if name=='lr' else selection['mode']
            if mode!='pure':text.append(row(name+' '+mode,parts[partition]['reranking']['results'][mode]))
        mean=seeds[partition][selection['mode']]['ndcg@20']
        text += ['', f'选定流程三种子NDCG20均值{mean["mean"]:.6f}，样本标准差{mean["sample_sd"]:.6f}。',
                 '', '| 主种子相对V2 | NDCG20绝对差 | 配对用户簇95%区间 |','| --- | ---: | --- |']
        for mode,interval in bootstrap[partition].items():
            ci=interval['ndcg@20'];text.append(f'| {mode} | {ci["difference"]:.6f} | [{ci["ci95_low"]:.6f}, {ci["ci95_high"]:.6f}] |')
        text += ['', '500次完整用户簇重采样，保持事件分母；区间不包含调参、种子或线上策略价值的不确定性。跨零不能宣称稳定提升。']
    text += ['', '## 产物与限制','',
             f'全项目{int(counts[-1])}项测试通过；实际产物校验PASS，V2冻结证据仍通过。逐条曝光预测、正反馈查询排名和重排候选均在忽略的artifacts目录。',
             '导出的重排Top100针对评测正反馈事件，不是所有曝光；只有模型分数与ID，不给未知候选附加反馈标签。完整事件数、哈希、目标/标签置零检查见ranking_artifact_validation.json。',
             '候选命中率限制精排上限，未召回目标始终为0。Pure序列不完整且没有真实逐事件倾向概率，本阶段没有去偏、因果或RL收益结论。',
             '', '## 实际运行入口','', '```powershell', '. .\\scripts\\enter_env.ps1',
             'python -B -m src.ranking.run --name lr --kind lr',
             'python -B -m src.ranking.run --name din_h10_b8192 --history-length 10',
             'python -B -m src.ranking.run --name din_h20_b8192 --history-length 20',
             'python -B -m src.ranking.run --name din_h50_b8192 --history-length 50',
             'python -B -m src.ranking.finalize --phase replicate',
             'python -B -m src.ranking.finalize --phase evaluate',
             'python -B -m src.ranking.verify',
             'python -B -m unittest discover -s tests -v',
             'python -B scripts/check_environment.py',
             'python -B -m src.ranking.report', '```','',
             '正式选模后禁止在原目录追加调参或静默重复测试；新复现实验另用输出目录。每组配置完整参数保留在tuning.json。下一阶段SASRec尚未执行。']
    (ROOT/'reports/ranking_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('VERIFIED DIN stage; tests='+counts[-1],flush=True)


if __name__=='__main__':main()
