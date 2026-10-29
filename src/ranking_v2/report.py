"""Report actual changes and component controls, including unfavorable evidence."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from src.data.common import ROOT
from src.data.download import digest_file
from src.ranking.data import write_json
from src.ranking.provenance import verify_v2
from src.retrieval.statistics import paired_user_bootstrap
from src.retrieval_v2.provenance import verify_evidence


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    frozen=verify_v2();out=ROOT/'experiments/ranking_v2';sel=read(out/'selection.json');cfg=sel['config'];rep=read(out/'replication.json')
    ev=read(out/'evaluation.json');ledger=read(out/'tuning.json');validation=read(ROOT/'reports/ranking_v2_artifact_validation.json')
    verify_evidence(ROOT,sel['evidence'],cfg);verify_evidence(ROOT,rep['evidence'],cfg)
    if validation['status']!='PASS':raise ValueError('Artifact validation required')
    for name,h in validation['artifact_hashes'].items():
        if digest_file(ROOT/name)!=h:raise ValueError('Verified artifact changed: '+name)
    log=(ROOT/'reports/environment_logs/all_tests_ranking_v2.txt').read_text(encoding='utf-8');n=re.findall(r'Ran (\d+) tests',log)
    if not n or not log.strip().endswith('OK'):raise ValueError('Passing full test log required')
    if read(ROOT/'reports/environment_python312_check.json')['status']!='PASS':raise ValueError('Environment check required')
    old=read(ROOT/'experiments/ranking/evaluation.json');seeds={};bootstrap={};rows=[]
    for part,base in ev['baseline'].items():
        seeds[part]={}
        for key in ['ndcg@20','recall@20','mrr@100']:
            values=[ev['results'][f'din_seed{s}'][part]['ranking'][key] for s in [42,43,44]]
            seeds[part][key]={'values':values,'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1))}
        mappings={'v2':base,'history_rule_penalty4':ev['history_rule'][part],
                  'old_pure_din':old['results']['din_seed42'][part]['reranking']['results']['pure']}
        mappings.update({name:parts[part]['ranking'] for name,parts in ev['results'].items()})
        for name,m in mappings.items():
            pw=ev['results'].get(name,{}).get(part,{}).get('pointwise',{})
            if name=='old_pure_din':pw=old['results']['din_seed42'][part]['pointwise']
            rows.append({'partition':part,'model':name,**{k:m[k] for k in ['ndcg@20','recall@20','recall@100','mrr@100']},
                         'auc':pw.get('auc'),'logloss':pw.get('logloss')})
        if part.endswith('/test'):
            with np.load(ev['results']['din_seed42'][part]['rank_artifact']) as z:primary={k:z[k] for k in z.files}
            comparisons={'v2':base['artifact'],'history_rule_penalty4':ev['history_rule'][part]['artifact'],
                         'no_attention':ev['results']['no_attention'][part]['rank_artifact'],
                         'linear_only':ev['results']['linear_only'][part]['rank_artifact'],
                         'no_repeat_feature':ev['results']['no_repeat_feature'][part]['rank_artifact'],
                         'old_pure_din':str(Path(old['results']['din_seed42'][part]['rank_directory'])/'pure.npz')}
            bootstrap[part]={}
            for name,path in comparisons.items():
                with np.load(path) as z:
                    np.testing.assert_array_equal(primary['event_ids'],z['event_ids']);np.testing.assert_array_equal(primary['users'],z['users'])
                    ci=paired_user_bootstrap(primary['ranks'],z['ranks'],z['users'],cfg['bootstrap_repeats'],cfg['bootstrap_seed'])
                ci['mrr@100']=ci.pop('mrr');bootstrap[part][name]=ci
    std=ev['results']['din_seed42']['standard/test']['ranking'];base=ev['baseline']['standard/test']
    assessment={'standard_ndcg20_improved':std['ndcg@20']>base['ndcg@20'],
                'standard_ndcg20_relative_gain':std['ndcg@20']/base['ndcg@20']-1,
                'paired_ci_vs_v2_positive':bootstrap['standard/test']['v2']['ndcg@20']['ci95_low']>0,
                'paired_ci_vs_rule_positive':bootstrap['standard/test']['history_rule_penalty4']['ndcg@20']['ci95_low']>0,
                'attention_contribution_ci_positive':bootstrap['standard/test']['no_attention']['ndcg@20']['ci95_low']>0}
    payload={'status':'VERIFIED','selected_trial':sel['trial'],'config':cfg,'test_is_iterative_reassessment':True,
             'v2_checkpoint_sha256':frozen['checkpoint_sha256'],'quality_assessment':assessment,'three_seed_summary':seeds,
             'paired_user_bootstrap':bootstrap,'results':ev,'validation_trials':ledger['trials'],
             'artifact_validation':validation,'tests_passed':int(n[-1]),
             'test_log_sha256':digest_file(ROOT/'reports/environment_logs/all_tests_ranking_v2.txt')}
    write_json(ROOT/'reports/ranking_v2_results.json',payload)
    with (ROOT/'reports/ranking_v2_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# DIN 精排修订实验','',
          f'主模型：`{sel["trial"]}`，固定双塔V2 Top100后实际重排。标准测试NDCG@20={std["ndcg@20"]:.6f}，V2={base["ndcg@20"]:.6f}，相对变化={assessment["standard_ndcg20_relative_gain"]:.2%}。',
          '旧实验保留在reports/ranking_report.md；本报告只描述新版本。','',
          '## 实现与修正','',
          '复用RecBole1.2.1的DIN特征嵌入、SequenceAttLayer、MLPLayers/Dice，重新连接完整用户、候选、交互和场景输入；上游包文件未改动。旧适配中用户表无梯度，用户加性偏置不改变同一请求内的候选顺序。',
          '- [DIN论文](https://arxiv.org/abs/1706.06978)',
          '- [RecBole DIN v1.2.1](https://github.com/RUCAIBox/RecBole/blob/v1.2.1/recbole/model/sequential_recommender/din.py)',
          '- [DeepCTR-Torch DIN特征连接参考](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/din.py)',
          '', '候选排序使用条件选择softmax；每次采样训练期原本已在Top100中的正目标与15个不同未知反馈候选，目标位0。只代表相对于采样替代项的选择偏好，不把未知反馈写成CTR label0；没有采样概率校正，不声称是全候选softmax的无偏估计。辅助BCE仅使用真实曝光反馈。',
          '主残差模型分数=固定V2分数+可训练DIN修正+可训练历史重复线性项；两项修正零初始化，实际训练更新，未用手工调alpha或保持alpha0。固定V2用户/视频表示只读，学生嵌入可训练。直接choice模型不含V2分数跳连，排序与CTR共享输出头；其delta分支不参与排序，delta正则对该对照的排序输出无效，不将它描述为同等输出正则对照。',
          '重复特征只查当前时刻之前的最长50条标准正历史；不使用未来/当前目标反馈。验证目标约0.31%出现在该历史，而原Top20约25.65%是历史正项，因此必须与简单历史降权规则比较，不能把这部分收益都归于注意力。',
          '', '## 协议','',
          '六组配置只用标准训练/验证，选定注意力启用的主模型后固定参数，复核种子43/44；无注意力、仅线性和无重复显式特征是归因对照。每轮独立均匀抽取最多64k合格训练查询，轮内不放回，跨轮可重复，不保证遍历。种子42为预定主结果，不按测试选择种子。',
          f'训练固定候选中目标存在的查询为{sel["training_candidate_coverage"]["target_present_queries"]}/{sel["training_candidate_coverage"]["positive_queries"]}（{sel["training_candidate_coverage"]["fraction"]:.2%}），仅这些查询进入条件选择训练；其余曝光仍可进入辅助BCE。教师使用训练期拟合，训练候选为同训练期生成，存在候选命中选择偏差，未声称无偏。',
          '所有评测保留同一V2 Top100，未召回目标仍计0，冷记录保留，Recall100不变，不插入目标、不在评测阶段删除历史项。NDCG是日志观测正事件的单正目标代理，不是全相关性评估、线上CTR或去偏因果收益。',
          '测试集在旧阶段已经查看，本次明确属于迭代复测。随机来源仅诊断，不参与配置选择。注意力等消融同时存在训练/采样方差，单种子对照不能证明普遍因果贡献。用户特征梯度/交互测试证明数据路径有效，本轮没有用户分支的训练消融，不据此量化用户特征的排序增益。',
          '', '## 六组完整标准验证结果','',
          '| 配置 | 最佳轮次（从0起） | NDCG@20 | 曝光AUC | 曝光LogLoss |', '| --- | ---: | ---: | ---: | ---: |']
    for name,t in ledger['trials'].items():
        m=t['best_validation'];text.append(f'| {name} | {t["best_epoch"]} | {m["ranking"]["ndcg@20"]:.6f} | {m["pointwise"]["auc"]:.6f} | {m["pointwise"]["logloss"]:.6f} |')
    text+=['','rich_bce按验证LogLoss选轮，其余按验证NDCG选轮；BCE对照用于检查结构修正能否自动解决排序问题，不能把曝光指标替代NDCG。','']
    for part in ['standard/test','random/test']:
        text+=['## '+part,'','| 模型 | Recall@20 | NDCG@20 | Recall@100 | MRR@100 | 曝光AUC |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for row in rows:
            if row['partition']==part:
                auc='—' if row['auc'] is None else f'{row["auc"]:.6f}'
                text.append(f'| {row["model"]} | {row["recall@20"]:.6f} | {row["ndcg@20"]:.6f} | {row["recall@100"]:.6f} | {row["mrr@100"]:.6f} | {auc} |')
        summary=seeds[part]['ndcg@20'];text+=['',f'三种子NDCG@20均值={summary["mean"]:.6f}，样本标准差={summary["sample_sd"]:.6f}。','',
          '| 主模型减对照（用户簇配对bootstrap，500次） | NDCG@20差 | 95%区间 |','| --- | ---: | --- |']
        for name,m in bootstrap[part].items():
            c=m['ndcg@20'];text.append(f'| {name} | {c["difference"]:.6f} | [{c["ci95_low"]:.6f}, {c["ci95_high"]:.6f}] |')
        text+=['','区间跨0时，不主张主模型优于该对照。该区间描述测试用户抽样不确定性，不包含调参/训练随机性或分布漂移。','']
    text+=['## 证据与复现','',f'{n[-1]}项自动化测试通过；环境检查通过；旧V2和旧DIN冻结证据未改变。所有主模型完整预测重算一致，导出逐条与检查点分数/排名一致；修改当前目标/反馈不影响候选打分。',
          '机器结果ranking_v2_results.json，平表ranking_v2_metrics.csv，实际产物验证ranking_v2_artifact_validation.json；模型/选模/三种子在experiments/ranking_v2。',
          '新环境下完整重训尚未验证；同机检查点重载和实际完整重算已验证。','',
          '```powershell','. .\\scripts\\enter_env.ps1',
          'python -B scripts\\train_ranking_v2.py --output-directory experiments/ranking_v2_reproduction/seed42 --seed 42',
          'python -B -m src.ranking_v2.verify','python -B -m src.ranking_v2.report','```','',
          '训练入口src.ranking_v2.run --name <新trial>及其明确的对照开关；原目录已冻结，禁止追加trial。独立复现入口会读取冻结selection.json，在新输出目录调用train_run，复用只读候选并保护原实验目录。',
          '本阶段到此结束；没有开展SASRec、IPS/SNIPS/DR或CQL。']
    (ROOT/'reports/ranking_v2_report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(assessment,indent=2),flush=True)
    return payload


if __name__=='__main__':main()
