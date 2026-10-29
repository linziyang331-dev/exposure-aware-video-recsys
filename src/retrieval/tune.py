"""Small standard-validation-only loss-weight grid; no test partition is loaded."""
import json
from src.data.common import ROOT
from src.retrieval.run import load_setup,write_json
from src.retrieval.train import train_run


def main():
    config,paths,vocabulary,counts,caches=load_setup(['standard/train','standard/validation'])
    initial=json.loads((paths['output']/'selection.json').read_text())
    seed=config['primary_seed']
    trials=[{'contrastive_weight':config['contrastive_weight'],
             'best_validation_ndcg20':initial['training_runs'][f'two_tower_seed{seed}']['best_validation_ndcg20'],
             'existing_initial_run':True}]
    for weight in [1.,5.]:
        trial=dict(config);trial['contrastive_weight']=weight
        result=train_run(caches['standard/train'],caches['standard/validation'],counts,trial,
                         paths['artifacts']/'tuning'/f'contrastive_{weight}_seed{seed}',seed,True)
        trials.append({'contrastive_weight':weight,'best_validation_ndcg20':result['best_validation_ndcg20'],
                       'checkpoint_sha256':result['checkpoint_sha256'],'best_epoch':result['best_epoch']})
        print(f'Validation loss-weight trial {weight}: {result["best_validation_ndcg20"]:.6f}',flush=True)
    best=max(trials,key=lambda row:row['best_validation_ndcg20'])
    result={'status':'VALIDATION_ONLY_TUNING','seed':seed,'selection_metric':'standard validation NDCG@20',
            'test_partitions_loaded':False,'trials':trials,'selected_contrastive_weight':best['contrastive_weight'],
            'replication_policy':'Once chosen, identical config for seeds42/43/44 and history-off42; no best-seed choice'}
    write_json(ROOT/'reports/retrieval_tuning.json',result)
    print(json.dumps(result,indent=2),flush=True)
    for cache in caches.values():cache.close()


if __name__=='__main__':main()
