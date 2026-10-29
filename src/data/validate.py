"""Independent full-output checks plus seeded SQL reconstruction of histories."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.common import ROOT, load_config
from src.data.download import digest_file
from src.data.staging import connect_database


def validate_preparation(config, protocol, *, samples_per_partition=32):
    start=time.perf_counter();output=config['output_directory']
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['signature']['protocol']==asdict(protocol), 'Calendar/history configuration changed'
    vocab_path=output/'vocabulary.json'
    assert digest_file(vocab_path)==manifest['vocabulary_sha256'], 'Vocabulary integrity failed'
    vocab=json.loads(vocab_path.read_text(encoding='utf-8'))
    users={int(k):v for k,v in vocab['user_raw_to_index'].items()}
    items={int(k):v for k,v in vocab['item_raw_to_index'].items()}
    db=connect_database(config['staging_database'])
    try:
        audit=json.loads(db.execute("SELECT value FROM metadata WHERE key='raw_audit'").fetchone()[0])
        assert manifest['signature']['raw_files']==audit['raw_files'], 'Raw input provenance changed'
        assert manifest['signature']['staging_code_sha256']==audit['staging_code_sha256']==digest_file(Path(__file__).parent/'staging.py'), 'Staging code provenance changed'
        expected_users=[r[0] for r in db.execute('SELECT DISTINCT user_id FROM events WHERE source=? AND time_ms>=? AND time_ms<? ORDER BY user_id',('standard',protocol.train_start_ms,protocol.train_end_ms))]
        expected_items=[r[0] for r in db.execute('SELECT DISTINCT video_id FROM events WHERE source=? AND time_ms>=? AND time_ms<? ORDER BY video_id',('standard',protocol.train_start_ms,protocol.train_end_ms))]
        assert users=={raw:i+1 for i,raw in enumerate(expected_users)}, 'User vocabulary leaked heldout records'
        assert items=={raw:i+1 for i,raw in enumerate(expected_items)}, 'Item vocabulary leaked heldout records'
        expected_popularity={str(r[0]):r[1] for r in db.execute('SELECT video_id,COUNT(*) FROM events WHERE source=? AND time_ms>=? AND time_ms<? AND is_click=1 GROUP BY video_id',('standard',protocol.train_start_ms,protocol.train_end_ms))}
        assert vocab['train_positive_counts']==expected_popularity, 'Popularity leaked heldout outcomes'
        db.execute('CREATE TEMP TABLE checked_rows(event_id INTEGER PRIMARY KEY,source TEXT,split TEXT,user_id INTEGER,video_id INTEGER,time_ms INTEGER,label INTEGER,long_view INTEGER,tab INTEGER,user_index INTEGER,item_index INTEGER)')
        db.execute('CREATE TEMP TABLE train_users(raw INTEGER PRIMARY KEY,idx INTEGER)')
        db.execute('CREATE TEMP TABLE train_items(raw INTEGER PRIMARY KEY,idx INTEGER)')
        db.executemany('INSERT INTO train_users VALUES(?,?)',users.items())
        db.executemany('INSERT INTO train_items VALUES(?,?)',items.items())
        bounds={'train':(protocol.train_start_ms,protocol.train_end_ms),
                'validation':(protocol.train_end_ms,protocol.validation_end_ms),
                'test':(protocol.validation_end_ms,protocol.test_end_ms)}
        expected_keys={'standard/train','standard/validation','standard/test','random/validation','random/test'}
        assert set(manifest['partitions'])==expected_keys, 'Unexpected source/split partitions'
        total,sampled=0,0;partition_results={}
        scalar_columns=['event_id','source','split','user_id','video_id','time_ms','label','long_view','tab','user_index','item_index']
        for partition_number,(key,stat) in enumerate(sorted(manifest['partitions'].items())):
            path=(output/stat['path']).resolve()
            assert path.is_relative_to(output.resolve()), 'Output file escapes preparation directory'
            assert digest_file(path)==stat['sha256'], f'File integrity failed: {key}'
            source,split=key.split('/');lower,upper=bounds[split]
            rng=np.random.default_rng(42+partition_number)
            positions=set(rng.choice(stat['rows'],size=min(samples_per_partition,stat['rows']),replace=False).tolist())
            offset=0;examples=[]
            with pq.ParquetFile(path) as parquet:
                assert parquet.metadata.num_rows==stat['rows'], 'Partition row count differs'
                for batch in parquet.iter_batches(batch_size=config['output_batch_size']):
                    assert all(batch.column(i).null_count==0 for i in range(batch.num_columns)), 'Null values in prepared fields'
                    def column(name):return batch.column(batch.schema.get_field_index(name))
                    time_ms=column('time_ms').to_numpy();lengths=column('history_length').to_numpy()
                    actual_lengths=pc.list_value_length(column('history_time_ms')).to_numpy()
                    item_lengths=pc.list_value_length(column('history_item_indices')).to_numpy()
                    assert np.array_equal(lengths,actual_lengths) and np.array_equal(lengths,item_lengths), 'History lengths disagree'
                    assert ((lengths>=0)&(lengths<=protocol.max_history)).all(), 'History exceeds truncation'
                    flat_times=pc.list_flatten(column('history_time_ms')).to_numpy()
                    flat_items=pc.list_flatten(column('history_item_indices')).to_numpy()
                    assert (flat_times<np.repeat(time_ms,lengths)).all(), 'History must be strictly earlier than target, including ties'
                    assert ((flat_items>0)&(flat_items<=len(items))).all(), 'History contains padding/OOV items'
                    groups=np.repeat(np.arange(len(batch)),lengths)
                    if len(flat_times)>1:
                        assert (np.diff(flat_times)[groups[1:]==groups[:-1]]>=0).all(), 'History is not chronological'
                    assert ((time_ms>=lower)&(time_ms<upper)).all(), 'Calendar boundary violated'
                    assert set(column('source').to_pylist())=={source}, 'Source mismatch'
                    assert set(column('split').to_pylist())=={split}, 'Split mismatch'
                    for name,index_name in [('user_seen_in_train','user_index'),('item_seen_in_train','item_index')]:
                        assert np.array_equal(column(name).to_numpy(zero_copy_only=False),column(index_name).to_numpy()!=0), 'Cold flag disagrees with index'
                    values=[column(name).to_pylist() for name in scalar_columns]
                    db.executemany('INSERT INTO checked_rows VALUES(?,?,?,?,?,?,?,?,?,?,?)',zip(*values))
                    for position in sorted(p-offset for p in positions if offset<=p<offset+len(batch)):
                        examples.extend(batch.slice(position,1).to_pylist())
                    offset+=len(batch)
            assert offset==stat['rows']
            for row in examples:
                # Reconstruct independently from raw clean events with SQL < target,
                # sorting DESC + LIMIT, rather than calling the generation function.
                history=db.execute('SELECT t.idx,r.time_ms FROM raw_events r JOIN dedup_ids d ON r.event_id=d.event_id JOIN train_items t ON r.video_id=t.raw WHERE r.user_id=? AND r.source=? AND r.is_click=1 AND r.time_ms<? ORDER BY r.time_ms DESC,r.event_id DESC LIMIT ?',
                                   (row['user_id'],'standard',row['time_ms'],protocol.max_history)).fetchall()
                history=list(reversed(history))
                assert row['history_item_indices']==[h[0] for h in history], 'Sampled history items differ from raw SQL reconstruction'
                assert row['history_time_ms']==[h[1] for h in history], 'Sampled history timestamps differ from raw SQL reconstruction'
                sampled+=1
            total+=offset
            partition_results[key]={'rows_checked':offset,'sampled_history_reconstructions':len(examples)}
            print(f'Validated {key}: {offset:,} rows',flush=True)
        assert total==audit['unique_rows']==manifest['rows_written'], 'Records missing or added'
        # PK uniqueness + equal total + every dedup ID present proves identical ID
        # sets. Joining the raw PK directly avoids materializing the full view.
        missing=db.execute('SELECT COUNT(*) FROM dedup_ids d LEFT JOIN checked_rows o ON o.event_id=d.event_id WHERE o.event_id IS NULL').fetchone()[0]
        mismatch=db.execute('SELECT COUNT(*) FROM checked_rows o LEFT JOIN raw_events e ON e.event_id=o.event_id LEFT JOIN train_users u ON e.user_id=u.raw LEFT JOIN train_items i ON e.video_id=i.raw WHERE e.event_id IS NULL OR o.source IS NOT e.source OR o.user_id IS NOT e.user_id OR o.video_id IS NOT e.video_id OR o.time_ms IS NOT e.time_ms OR o.label IS NOT e.is_click OR o.long_view IS NOT e.long_view OR o.tab IS NOT e.tab OR o.user_index IS NOT COALESCE(u.idx,0) OR o.item_index IS NOT COALESCE(i.idx,0)').fetchone()[0]
        assert missing==0 and mismatch==0, 'Prepared target fields differ from exact raw events'
        result={'status':'PASS','rows_checked':total,'missing_events':missing,'target_field_mismatches':mismatch,
                'sampled_histories_checked':sampled,'sampling_seed':42,'partitions':partition_results,
                'full_checks':['file hashes','unique complete event IDs','exact raw target fields','calendar/source splits',
                               'train-only vocab/popularity','history lengths/order/strict timestamp bound','cold flags'],
                'sample_checks':['history is exactly last K eligible standard positives reconstructed from raw SQL'],
                'elapsed_seconds':time.perf_counter()-start,
                'process_peak_working_set_MiB':psutil.Process().memory_info().peak_wset/2**20}
        return result
    finally:
        db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path)
    args=parser.parse_args();config,protocol=load_config(args.config)
    result=validate_preparation(config,protocol)
    (ROOT/'reports/data_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
