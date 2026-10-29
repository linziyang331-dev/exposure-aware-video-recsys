"""Export dynamic user snapshots, item vectors, and target-independent Top-K."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from src.data.download import digest_file
from src.retrieval.metrics import stable_topk
from src.retrieval.train import load_model
from src.retrieval.provenance import verify_frozen_inputs


def inverse_ids(mapping):
    result=np.full(len(mapping)+1,-1,dtype=np.int64)
    for raw,index in mapping.items():result[index]=int(raw)
    return result


@torch.no_grad()
def export_snapshot(model,train,vocabulary,cutoff_ms,directory,topk=100,batch_size=512):
    if int(train['time_ms'].max())>=cutoff_ms:raise ValueError('Snapshot cutoff must be after all training events')
    model.eval();device=next(model.parameters()).device;directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    user_raw=inverse_ids(vocabulary['user_raw_to_index']);item_raw=inverse_ids(vocabulary['item_raw_to_index'])
    number=len(user_raw)-1;length=train['history'].shape[1]
    histories=np.zeros((number,length),dtype=np.int32);tabs=np.zeros(number,dtype=np.int64);last_times=np.zeros(number,dtype=np.int64)
    user_indices=np.asarray(train['user_index'])
    if (np.diff(user_indices)<0).any():raise ValueError('Snapshot requires user-sorted training records')
    boundaries=np.r_[0,np.flatnonzero(np.diff(user_indices))+1,len(train)]
    seen=np.zeros(number,dtype=bool)
    for start,stop in zip(boundaries[:-1],boundaries[1:]):
        index=int(user_indices[start])-1;seen[index]=True
        positives=np.flatnonzero(train['label'][start:stop]==1)[-length:]+start
        histories[index,:len(positives)]=train['item_index'][positives]
        tabs[index]=train['tab'][stop-1];last_times[index]=train['time_ms'][stop-1]
    if not seen.all():raise ValueError('Snapshot user vocabulary incomplete')
    candidate_ids=torch.arange(1,len(item_raw),device=device)
    item_vectors=model.encode_item(candidate_ids).cpu().numpy()
    user_vectors=np.empty((number,item_vectors.shape[1]),dtype=np.float32)
    schema=pa.schema([('user_id',pa.int64()),('user_index',pa.int32()),('snapshot_time_ms',pa.int64()),
                      ('tab',pa.int8()),('candidate_item_indices',pa.list_(pa.int32())),
                      ('candidate_video_ids',pa.list_(pa.int64())),('scores',pa.list_(pa.float32()))])
    writer=pq.ParquetWriter(directory/'snapshot_candidates.parquet',schema,compression='zstd')
    try:
        for start in range(0,number,batch_size):
            stop=min(start+batch_size,number)
            uid=torch.arange(start+1,stop+1,device=device)
            user=model.encode_user(uid,torch.as_tensor(tabs[start:stop],device=device),
                                   torch.as_tensor(histories[start:stop],device=device,dtype=torch.long))
            user_vectors[start:stop]=user.cpu().numpy()
            scores=user@torch.as_tensor(item_vectors,device=device).T/model.temperature
            top,values=stable_topk(scores,candidate_ids,topk);top=top.cpu().numpy();values=values.cpu().numpy()
            rows=[{'user_id':int(user_raw[i+1]),'user_index':i+1,'snapshot_time_ms':int(cutoff_ms),
                   'tab':int(tabs[i]),'candidate_item_indices':top[j].tolist(),
                   'candidate_video_ids':item_raw[top[j]].tolist(),'scores':values[j].tolist()}
                  for j,i in enumerate(range(start,stop))]
            writer.write_table(pa.Table.from_pylist(rows,schema=schema))
    finally:writer.close()
    np.savez_compressed(directory/'item_vectors.npz',item_index=np.arange(1,len(item_raw)),video_id=item_raw[1:],vectors=item_vectors)
    np.savez_compressed(directory/'user_vectors.npz',user_index=np.arange(1,number+1),user_id=user_raw[1:],
                        vectors=user_vectors,tabs=tabs,history=histories,last_event_time_ms=last_times,snapshot_time_ms=cutoff_ms)
    count=min(16,number);probe_items=torch.arange(count,device=device)%(len(item_raw)-1)+1
    direct=model(torch.arange(1,count+1,device=device),probe_items,torch.as_tensor(tabs[:count],device=device),
                 torch.as_tensor(histories[:count],device=device,dtype=torch.long))
    dot=(torch.as_tensor(user_vectors[:count],device=device)*torch.as_tensor(item_vectors,device=device)[probe_items-1]).sum(1)/model.temperature
    torch.testing.assert_close(direct,dot,rtol=1e-5,atol=1e-5)
    return {'status':'EXPORTED','users':number,'items':len(item_raw)-1,'dimension':item_vectors.shape[1],
            'topk':min(topk,len(item_raw)-1),'snapshot_time_ms':cutoff_ms,
            'user_context':f'most recent up to {length} strictly pre-cutoff training positives, last training tab; not a universal static user vector',
            'dot_score_probe_count':count,'dot_score_max_abs_error':float((direct-dot).abs().max().item()),
            'files':{file.name:digest_file(file) for file in directory.iterdir() if file.is_file()}}


@torch.no_grad()
def export_event_candidates(scorer,cache,vocabulary,path,topk,batch_size):
    raw=inverse_ids(vocabulary['item_raw_to_index']);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    schema=pa.schema([('event_id',pa.int64()),('user_id',pa.int64()),('time_ms',pa.int64()),
                      ('candidate_item_indices',pa.list_(pa.int32())),('candidate_video_ids',pa.list_(pa.int64())),
                      ('scores',pa.list_(pa.float32()))])
    with pq.ParquetWriter(path,schema,compression='zstd') as writer:
        for start in range(0,len(cache),batch_size):
            indices=np.arange(start,min(start+batch_size,len(cache)))
            top,values=stable_topk(scorer.scores(cache,indices),scorer.candidate_ids,topk)
            top=top.cpu().numpy();values=values.cpu().numpy()
            rows=[{'event_id':int(cache['event_id'][i]),'user_id':int(cache['user_id'][i]),'time_ms':int(cache['time_ms'][i]),
                   'candidate_item_indices':top[j].tolist(),'candidate_video_ids':raw[top[j]].tolist(),'scores':values[j].tolist()}
                  for j,i in enumerate(indices)]
            writer.write_table(pa.Table.from_pylist(rows,schema=schema))
    return {'rows':len(cache),'topk':min(topk,len(raw)-1),'sha256':digest_file(path),
            'label_policy':'Scores/candidates only; unexposed candidates have no fabricated feedback labels'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--partition',choices=['standard/train','standard/validation','standard/test','random/validation','random/test'])
    args=parser.parse_args()
    from src.retrieval.run import load_setup,write_json
    from src.retrieval.evaluate import Scorer
    from src.retrieval.baselines import popularity_scores
    parts=[args.partition] if args.partition else ['standard/train']
    config,paths,vocabulary,counts,caches=load_setup(parts)
    selection=json.loads((paths['output']/'selection.json').read_text());verify_frozen_inputs(selection,paths['prepared'])
    name=f'two_tower_seed{selection["primary_seed"]}';info=selection['training_runs'][name]
    if digest_file(info['checkpoint'])!=info['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    model,checkpoint=load_model(info['checkpoint'],config['device'])
    if checkpoint['smoke_only']:raise ValueError('Cannot export a smoke model as production-stage output')
    if args.partition:
        scorer=Scorer('two_tower',popularity_scores(counts),config['device'],model=model)
        destination=paths['artifacts']/'candidates'/(args.partition.replace('/','_')+'.parquet')
        result=export_event_candidates(scorer,caches[args.partition],vocabulary,destination,config['candidate_topk'],config['eval_batch_size'])
        result.update(partition=args.partition,path=str(destination));filename='candidates_'+args.partition.replace('/','_')+'.json'
    else:
        prepared=json.loads((paths['prepared']/'manifest.json').read_text())
        result=export_snapshot(model,caches['standard/train'],vocabulary,prepared['signature']['protocol']['train_end_ms'],
                               paths['artifacts']/'snapshot',config['candidate_topk'],config['eval_batch_size'])
        filename='exports.json'
    result.update(checkpoint_sha256=info['checkpoint_sha256'],data_fingerprint=selection['data_fingerprint'],export_code_sha256=digest_file(__file__))
    write_json(paths['output']/filename,result)
    for cache in caches.values():cache.close()
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
