"""Verified Parquet -> bounded typed memory maps, without giant history lists."""
import hashlib
import json
import uuid
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from src.data.common import ROOT
from src.data.download import digest_file

FIELDS={'event_id':'int64','user_id':'int64','video_id':'int64','time_ms':'int64',
        'user_index':'int32','item_index':'int32','tab':'int8','label':'int8'}


class PartitionCache:
    def __init__(self,root,metadata):
        self.root=root;self.metadata=metadata
        self.arrays={name:np.load(root/(name+'.npy'),mmap_mode='r') for name in metadata['array_hashes']}
    def __getitem__(self,name):return self.arrays[name]
    def __len__(self):return self.metadata['rows']
    def close(self):
        for array in self.arrays.values():
            if getattr(array,'_mmap',None) is not None:array._mmap.close()
        self.arrays.clear()


def load_partition(prepared,partition,cache_root,max_history=50):
    prepared=Path(prepared).resolve();cache_root=Path(cache_root).resolve()
    if not cache_root.is_relative_to(ROOT.resolve()) or cache_root.drive.upper()!='D:':
        raise ValueError('Retrieval cache must remain in the D: project')
    if max_history<=0:raise ValueError('max_history must be positive')
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    stat=manifest['partitions'][partition];path=(prepared/stat['path']).resolve()
    if not path.is_relative_to(prepared) or digest_file(path)!=stat['sha256']:
        raise ValueError('Prepared Parquet hash mismatch')
    signature={'data_fingerprint':manifest['fingerprint'],'partition':partition,'source_sha256':stat['sha256'],
               'max_history':max_history,'cache_code_sha256':digest_file(__file__)}
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:20]
    destination=cache_root/(partition.replace('/','_')+'-'+key)
    if destination.exists():
        metadata=json.loads((destination/'metadata.json').read_text())
        if metadata['signature']!=signature:raise ValueError('Cache signature mismatch')
        for name,expected in metadata['array_hashes'].items():
            if digest_file(destination/(name+'.npy'))!=expected:raise ValueError('Cache array hash mismatch')
        return PartitionCache(destination,metadata)
    work=cache_root/(destination.name+'.building-'+uuid.uuid4().hex);work.mkdir(parents=True)
    rows=stat['rows'];arrays={}
    try:
        for name,dtype in {**FIELDS,'history_length':'int16','history':'int32'}.items():
            shape=(rows,max_history) if name=='history' else (rows,)
            arrays[name]=np.lib.format.open_memmap(work/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
        offset=0
        with pq.ParquetFile(path) as parquet:
            columns=list(FIELDS)+['history_item_indices']
            for batch in parquet.iter_batches(batch_size=8192,columns=columns):
                count=len(batch);stop=offset+count
                for name in FIELDS:
                    column=batch.column(batch.schema.get_field_index(name))
                    if column.null_count:raise ValueError('NULL retrieval input')
                    arrays[name][offset:stop]=column.to_numpy(zero_copy_only=False)
                history=batch.column(batch.schema.get_field_index('history_item_indices'))
                starts=history.offsets.to_numpy();values=history.values.to_numpy()
                lengths=np.minimum(np.diff(starts),max_history)
                padded=np.zeros((count,max_history),dtype=np.int32)
                for position in range(max_history):
                    valid=lengths>position
                    padded[valid,position]=values[starts[1:][valid]-lengths[valid]+position]
                arrays['history'][offset:stop]=padded;arrays['history_length'][offset:stop]=lengths
                offset=stop
        if offset!=rows:raise ValueError('Prepared row count mismatch')
        if not np.isin(arrays['label'],[0,1]).all():raise ValueError('Observed binary labels required')
        if partition=='standard/train' and ((arrays['item_index']<=0).any() or (arrays['user_index']<=0).any()):
            raise ValueError('Unexpected cold ID in training')
        for array in arrays.values():array.flush()
    finally:
        for array in arrays.values():array._mmap.close()
    metadata={'rows':rows,'signature':signature,'array_hashes':{name:digest_file(work/(name+'.npy')) for name in arrays}}
    (work/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');work.rename(destination)
    print(f'Cached {partition}: {rows:,} rows',flush=True)
    return PartitionCache(destination,metadata)


def training_counts(vocabulary):
    items={int(k):v for k,v in vocabulary['item_raw_to_index'].items()}
    result=np.zeros(len(items)+1,dtype=np.int64)
    for raw,index in items.items():result[index]=vocabulary['train_positive_counts'].get(str(raw),0)
    return result


def tensor_batch(cache,indices,device):
    import torch
    names={'users':'user_index','items':'item_index','tabs':'tab','history':'history','labels':'label'}
    return {name:torch.from_numpy(np.array(cache[source][indices],copy=True)).to(device=device,dtype=torch.float32 if name=='labels' else torch.long)
            for name,source in names.items()}
