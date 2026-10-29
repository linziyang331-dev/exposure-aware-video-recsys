"""Material tail gains under overall/head retention; validation only."""
import numpy as np


def qualifies(m,b,gate):
    values=[m[k] for k in ['recall@20','ndcg@20']]+[m['groups'][g][k] for g,k in
        [('head_target','recall@20'),('tail_target','recall@20'),('tail_target','recall@100')]]
    return bool(np.isfinite(values).all()
        and all(m[k]>=b[k]*gate['retention'] for k in ['recall@20','ndcg@20'])
        and m['groups']['head_target']['recall@20']>=b['groups']['head_target']['recall@20']*gate['retention']
        and m['groups']['tail_target']['recall@20']>=b['groups']['tail_target']['recall@20']+gate['tail20_delta']
        and m['groups']['tail_target']['recall@100']>=b['groups']['tail_target']['recall@100']+gate['tail100_delta'])


def utility(m,b):
    t=m['groups']['tail_target'];base=b['groups']['tail_target']
    return .5*t['recall@20']/max(base['recall@20'],1e-12)+.5*t['recall@100']/max(base['recall@100'],1e-12)
