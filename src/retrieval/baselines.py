"""Train-only popularity and rolling-history ItemCF using RecBole cosine core."""
import numpy as np
import scipy.sparse as sparse


def popularity_scores(positive_counts):
    return np.log1p(np.asarray(positive_counts,dtype=np.float32)[1:])


def fit_itemcf(users,items,num_users,num_items,neighbors=100):
    from recbole.model.general_recommender.itemknn import ComputeSimilarity
    matrix=sparse.csr_matrix((np.ones(len(items),dtype=np.float32),(users,items)),shape=(num_users+1,num_items+1))
    matrix.sum_duplicates();matrix.data.fill(1.)
    _,weights=ComputeSimilarity(matrix,topk=min(neighbors,num_items),method='item',normalize=True).compute_similarity()
    return weights.tocsc()


def truncate_neighbors(weights,neighbors):
    values=[];rows=[];columns=[]
    for column in range(weights.shape[1]):
        start,end=weights.indptr[column:column+2]
        ids=weights.indices[start:end];data=weights.data[start:end]
        order=np.lexsort((ids,-data))[:neighbors]
        rows.extend(ids[order]);values.extend(data[order]);columns.extend([column]*len(order))
    return sparse.csc_matrix((values,(rows,columns)),shape=weights.shape,dtype=np.float32)


def score_itemcf(histories,weights,popularity):
    histories=np.asarray(histories)
    row,column=np.nonzero(histories>0)
    profile=sparse.csr_matrix((np.ones(len(row),dtype=np.float32),(row,histories[row,column])),shape=(len(histories),weights.shape[0]))
    scores=(profile@weights)[:,1:].toarray().astype(np.float32)
    order=np.argsort(-popularity,kind='stable');ranks=np.empty(len(order),dtype=np.int32);ranks[order]=np.arange(len(order))
    fallback=-.5-ranks.astype(np.float32)/(2*max(1,len(order)))
    return np.where(scores>0,scores,fallback[None,:]).astype(np.float32)
