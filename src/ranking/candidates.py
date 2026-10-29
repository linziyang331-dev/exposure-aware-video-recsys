"""Positive-event evaluation candidates from frozen V2, never target-injected."""
import hashlib
import json
import uuid
from pathlib import Path
import numpy as np
import torch
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval.evaluate import Scorer
from src.retrieval.metrics import stable_topk
from src.retrieval_v2.train import load_model
from src.ranking.metrics import candidate_ranks
from src.ranking.data import write_json


class CandidateCache:
    def __init__(self,path,metadata):
        self.path=path;self.metadata=metadata
        self.arrays={name:np.load(path/(name+'.npy'),mmap_mode='r') for name in metadata['array_hashes']}
    def __len__(self):return self.metadata['rows']
    def __getitem__(self,key):return self.arrays[key]
    def close(self):
        for v in self.arrays.values():v._mmap.close()


def load_candidates(partition,cache,counts,output,frozen,device='cuda'):
    signature={'partition':partition,'source':cache.metadata['signature'],'v2_checkpoint':frozen['checkpoint_sha256'],
               'v2_freeze_sha256':digest_file(ROOT/'reports/retrieval_v2_freeze.json'),
               'generator_sha256':digest_file(__file__),'topk':100,'target_injected':False}
    directory=Path(output)/'artifacts/candidates'/partition.replace('/','_')
    if directory.exists():
        metadata=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        if metadata['signature']!=signature:raise ValueError('Candidate generator/source changed')
        for name,expected in metadata['array_hashes'].items():
            if digest_file(directory/(name+'.npy'))!=expected:raise ValueError('Candidate cache hash mismatch')
        return CandidateCache(directory,metadata)
    torch.backends.cuda.matmul.allow_tf32=False
    model,_=load_model(ROOT/frozen['checkpoint'],device)
    scorer=Scorer('two_tower',np.log1p(counts[1:]).astype(np.float32),device,model=model)
    eligible=np.flatnonzero(cache['label']==1);rows=len(eligible)
    work=directory.with_name(directory.name+'.building-'+uuid.uuid4().hex);work.mkdir(parents=True)
    arrays={}
    types={'ids':('int32',(rows,100)),'scores':('float32',(rows,100)),
           'indices':('int64',(rows,)),'event_ids':('int64',(rows,)),'user_ids':('int64',(rows,)),'v2_ranks':('int32',(rows,))}
    for name,(dtype,shape) in types.items():arrays[name]=np.lib.format.open_memmap(work/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
    for start in range(0,rows,512):
        stop=min(start+512,rows);indices=eligible[start:stop]
        scores=scorer.scores(cache,indices);ids,values=stable_topk(scores,scorer.candidate_ids,100)
        targets=torch.as_tensor(np.array(cache['item_index'][indices],copy=True),device=device,dtype=torch.long)
        arrays['ids'][start:stop]=ids.cpu().numpy();arrays['scores'][start:stop]=values.cpu().numpy()
        arrays['indices'][start:stop]=indices;arrays['event_ids'][start:stop]=cache['event_id'][indices]
        arrays['user_ids'][start:stop]=cache['user_id'][indices]
        arrays['v2_ranks'][start:stop]=candidate_ranks(ids,values,targets).cpu().numpy()
    for array in arrays.values():array.flush();array._mmap.close()
    # Compare to frozen full-catalog ranks: only truncation at100 is permitted.
    reference=ROOT/'experiments/retrieval_v2/artifacts/ranks/two_tower_v2_seed42'/(partition.replace('/','_')+'.npz')
    if reference.exists():
        with np.load(reference) as z:
            np.testing.assert_array_equal(z['event_ids'],np.load(work/'event_ids.npy'))
            expected=np.where(z['ranks']<=100,z['ranks'],0)
            np.testing.assert_array_equal(expected,np.load(work/'v2_ranks.npy'))
    metadata={'rows':rows,'signature':signature,'array_hashes':{name:digest_file(work/(name+'.npy')) for name in arrays}}
    write_json(work/'metadata.json',metadata);work.rename(directory)
    print('Candidates '+partition+': '+str(rows)+' queries, no target injection',flush=True)
    return CandidateCache(directory,metadata)
