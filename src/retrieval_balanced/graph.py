"""Train-only binary co-click neighbors with deterministic cosine/ID ordering."""
import numpy as np
from scipy import sparse


def build_neighbors(users,items,num_users,num_items,k=32):
    users=np.asarray(users);items=np.asarray(items)
    if (users.ndim!=1 or users.shape!=items.shape or k<1 or num_items<1
            or (users<=0).any() or (users>num_users).any() or (items<=0).any() or (items>num_items).any()):
        raise ValueError('Known positive train user/item pairs required')
    x=sparse.csr_matrix((np.ones(len(items),dtype=np.float32),(users,items)),shape=(num_users+1,num_items+1))
    x.sum_duplicates();x.data.fill(1.)
    degree=np.asarray(x.sum(0)).ravel();inv=np.zeros_like(degree);np.divide(1,np.sqrt(degree),out=inv,where=degree>0)
    normalized=x@sparse.diags(inv)
    ids=np.zeros((num_items+1,k),dtype=np.int64);weights=np.zeros((num_items+1,k),dtype=np.float32)
    for start in range(1,num_items+1,256):
        stop=min(num_items+1,start+256);block=(normalized[:,start:stop].T@normalized).tocsr()
        for j in range(stop-start):
            i=start+j;begin,end=block.indptr[j:j+2];n=block.indices[begin:end];v=block.data[begin:end]
            valid=(n>0)&(n!=i)&(v>0);n=n[valid];v=v[valid]
            order=np.lexsort((n,-v))[:k];n=n[order];v=v[order]
            ids[i,:len(n)]=n
            if len(v):weights[i,:len(v)]=v/v.sum()
    return ids,weights
