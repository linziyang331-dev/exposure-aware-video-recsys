"""Task/denominator contracts and exact single-positive pipeline accounting."""
import numpy as np
from src.retrieval.metrics import summarize_ranks
from src.exposure.verify import assert_result_equal


def _ranks(values):
    r=np.asarray(values)
    if r.ndim!=1 or not len(r) or r.dtype.kind not in 'iu' or (r<0).any():
        raise ValueError('Nonempty nonnegative integer ranks required')
    return r


def check_alignment(expected_events,expected_users,events,users):
    e,u=np.asarray(events),np.asarray(users)
    if e.ndim!=1 or u.shape!=e.shape or len(np.unique(e))!=len(e) or not np.array_equal(e,expected_events) or not np.array_equal(u,expected_users):
        raise ValueError('Positive event/user alignment differs or events duplicated')


def rank_metrics(ranks,users,space):
    r=_ranks(ranks)
    if space not in ['catalog','fixed_v2_top100']:raise ValueError('Unknown candidate-space contract')
    if space=='fixed_v2_top100' and (r>100).any():raise ValueError('Pool rank exceeds frozen candidates')
    result=summarize_ranks(r,users,ks=(20,100))
    result.update(candidate_space=space,mrr_cutoff='full_catalog' if space=='catalog' else '100')
    return result


def candidate_funnel(catalog,pool):
    catalog=_ranks(catalog);pool=_ranks(pool)
    if catalog.shape!=pool.shape or (pool>100).any():raise ValueError('Inconsistent candidate funnel ranks')
    present=(catalog>0)&(catalog<=100)
    if not np.array_equal(present,pool>0):raise ValueError('Reranker candidate membership changed')
    hit=(pool>0)&(pool<=20);n=len(pool);count=int(present.sum())
    result={'queries':n,'top20_hits':int(hit.sum()),'pool_not_top20':int((present&~hit).sum()),
        'outside_pool':int((~present).sum()),'recall@20':float(hit.mean()),'recall@100':float(present.mean()),
        'conditional_top20_given_pool':float(hit.sum()/count) if count else None,
        'label_oracle_recall20_ceiling':float(present.mean()),
        'oracle_is_deployable':False}
    if result['top20_hits']+result['pool_not_top20']+result['outside_pool']!=n:raise ValueError('Funnel denominator changed')
    return result


def transition_counts(catalog,pool):
    candidate_funnel(catalog,pool)
    first=(catalog>0)&(catalog<=20);second=(pool>0)&(pool<=20)
    win=int((~first&second).sum());loss=int((first&~second).sum())
    return {'queries':len(first),'new_top20_hits':win,'lost_top20_hits':loss,
        'retained_top20_hits':int((first&second).sum()),'neither_top20':int((~first&~second).sum()),
        'net_recall_change':(win-loss)/len(first)}


def cohort_masks(items,user_indices,history,counts):
    items=np.asarray(items);users=np.asarray(user_indices);history=np.asarray(history);counts=np.asarray(counts)
    if items.ndim!=1 or users.shape!=items.shape or history.ndim!=2 or len(history)!=len(items) or (items<0).any() or (items>=len(counts)).any():
        raise ValueError('Aligned query context and training catalog required')
    head=np.zeros(len(counts),dtype=bool);order=np.argsort(-counts[1:],kind='stable')+1
    head[order[:int(np.ceil(.2*len(order)))]]=True
    known=items>0;seen=known&((history==items[:,None])&(history>0)).any(1)
    return {'all':np.ones(len(items),dtype=bool),'head_target':known&head[items],
        'tail_target':known&~head[items],'cold_target':~known,'cold_user':users==0,
        'warm_user_and_item':known&(users>0),'empty_history':(history==0).all(1),
        'nonempty_history':(history>0).any(1),'history_target_seen':seen,'history_target_unseen':~seen}


def check_saved_metrics(actual,expected):
    for key in ['queries','users','recall@20','ndcg@20','recall@100','ndcg@100']:
        assert_result_equal(actual[key],expected[key])
    name='mrr' if actual['mrr_cutoff']=='full_catalog' else 'mrr@100'
    assert_result_equal(actual['mrr'],expected[name])
    for key in ['recall@20','ndcg@20','recall@100','ndcg@100']:
        if key in expected['user_macro']:
            assert_result_equal(actual['user_macro'][key],expected['user_macro'][key])
        elif key in ['recall@20','ndcg@20']:raise ValueError('Required saved macro20 metrics missing')
    assert_result_equal(actual['user_macro']['mrr'],expected['user_macro'][name])
