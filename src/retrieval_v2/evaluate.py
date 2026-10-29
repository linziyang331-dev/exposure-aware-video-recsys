"""Reuse exact original protocol; add full-catalog rank-derived @100 metrics."""
import tempfile
from pathlib import Path
import numpy as np
from src.retrieval.evaluate import Scorer,evaluate_partition
from src.retrieval.metrics import summarize_ranks


def evaluate(scorer,cache,counts,*,batch_size=512,artifact=None,query_limit=None):
    temporary=None
    if artifact is None:
        temporary=tempfile.TemporaryDirectory();artifact=Path(temporary.name)/'ranks.npz'
    try:
        result=evaluate_partition(scorer,cache,counts,batch_size=batch_size,artifact=artifact,query_limit=query_limit)
        with np.load(artifact) as ranks:
            extra=summarize_ranks(ranks['ranks'],ranks['users'],ks=(100,))
        result.update({k:extra[k] for k in ['recall@100','ndcg@100']})
        result['user_macro'].update(extra['user_macro'])
        return result
    finally:
        if temporary is not None:temporary.cleanup()
