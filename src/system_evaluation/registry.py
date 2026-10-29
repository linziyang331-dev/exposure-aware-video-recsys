"""Bind existing evaluations to their frozen evidence and actual artifacts."""
import json
from pathlib import Path
from src.data.common import ROOT
from src.data.download import digest_file


class Assets:
    def __init__(self):self.hashes={}
    def check(self,path,expected=None):
        path=(ROOT/path).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Asset leaves project')
        name=path.relative_to(ROOT).as_posix()
        if name not in self.hashes:self.hashes[name]=digest_file(path)
        if expected is not None and self.hashes[name]!=expected:raise ValueError('Frozen artifact changed: '+name)
        return path
    def read(self,path):return json.loads(self.check(path).read_text(encoding='utf-8'))
    def evidence(self,stage):
        selection=self.read('experiments/'+stage+'/selection.json')
        for name in ['selection','replication']:
            path='experiments/'+stage+'/'+name+'.json'
            data=self.read(path);e=data['evidence']
            if e['config']!=selection['config']:raise ValueError('Frozen configuration differs: '+stage)
            for p,h in e['files'].items():self.check(p,h)
        return selection
    def validation(self,path):
        data=self.read(path)
        if data['status']!='PASS':raise ValueError('Original verification did not pass')
        for p,h in data['artifact_hashes'].items():self.check(p,h)
        return data


def load_registry():
    assets=Assets();ranks=[];predictions=[]
    for stage in ['ranking','ranking_v2','sequential','ranking_v3_frozen_top20']:assets.evidence(stage)
    frozen=assets.read('reports/retrieval_v2_freeze.json')
    for p,h in frozen['evidence']['files'].items():assets.check(p,h)
    for path in ['reports/ranking_artifact_validation.json','reports/ranking_v2_artifact_validation.json','reports/sequential_artifact_validation.json']:
        assets.validation(path)
    retrieval=assets.read('experiments/retrieval_v2/evaluation.json')
    report=assets.read('reports/retrieval_optimization_results.json')
    for p,h in report['rank_artifact_hashes'].items():assets.check(p,h)
    din=assets.read('experiments/ranking_v2/evaluation.json');old=assets.read('experiments/ranking/evaluation.json')
    seq=assets.read('experiments/sequential/evaluation.json');v3=assets.read('experiments/ranking_v3_frozen_top20/evaluation.json')
    if assets.read('experiments/ranking_v3_frozen_top20/verification.json')['status']!='PASS':raise ValueError('V3 must be verified')
    for stage,data in [('ranking_v3_frozen_top20',v3),('ranking_v2',din),('sequential',seq),('ranking',old),('retrieval_v2',retrieval)]:
        for key in ['selection','replication']:
            path='experiments/'+stage+'/'+key+'.json';assets.check(path,data[key+'_sha256'])
    def rank(model,part,space,path,expected,primary=True,experimental=False):
        ranks.append({'model':model,'partition':part,'space':space,'path':str(assets.check(path)),
            'expected':expected,'primary':primary,'experimental':experimental})
    def pred(model,part,row):
        predictions.append({'model':model,'partition':part,'path':str(assets.check(row['prediction_artifact'])),'expected':row['pointwise']})
    for part in din['baseline']:
        for name in ['popularity','itemcf','two_tower_seed42','two_tower_v2_seed42','two_tower_v2_seed43','two_tower_v2_seed44']:
            row=retrieval['results'][name][part];rank(name,part,'catalog',row['rank_artifact'],row,not name.endswith(('seed43','seed44')))
        rank('v2_original_order',part,'fixed_v2_top100',din['baseline'][part]['artifact'],din['baseline'][part])
        for seed in [42,43,44]:
            row=din['results'][f'din_seed{seed}'][part]
            rank(f'din_v2_seed{seed}',part,'fixed_v2_top100',row['rank_artifact'],row['ranking'],seed==42)
            if seed==42:pred('din_v2',part,row)
            row=seq['results'][f'sasrec_seed{seed}'][part]
            rank(f'sasrec_seed{seed}',part,'catalog',row['catalog_rank_artifact'],row['catalog'],seed==42)
            rank(f'sasrec_seed{seed}',part,'fixed_v2_top100',row['pool_rank_artifact'],row['fixed_v2_pool'],seed==42)
            row=v3['results'][f'seed{seed}'][part]
            assets.check(row['rank_artifact'],row['rank_sha256'])
            rank(f'din_v3_experimental_seed{seed}',part,'fixed_v2_top100',row['rank_artifact'],row['ranking'],seed==42,True)
        for model in ['lr','din_seed42']:
            row=old['results'][model][part];name='old_'+model
            rank(name,part,'fixed_v2_top100',Path(row['rank_directory'])/'pure.npz',row['reranking']['results']['pure'])
            pred(name,part,row)
        pred('rich_bce_control',part,din['results']['rich_bce'][part])
    return {'assets':assets,'ranks':ranks,'predictions':predictions,'frozen':frozen,'din':din}
