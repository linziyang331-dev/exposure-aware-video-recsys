"""Prepare source-separated, time-split exposures with strictly earlier histories."""
import argparse
import hashlib
import json
import time
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.common import ROOT, load_config
from src.data.download import digest_file
from src.data.protocol import iter_examples, learn_train_vocabulary
from src.data.staging import connect_database

SCHEMA = pa.schema([
    ('event_id',pa.int64()),('user_id',pa.int64()),('video_id',pa.int64()),
    ('user_index',pa.int32()),('item_index',pa.int32()),('time_ms',pa.int64()),
    ('label',pa.int8()),('long_view',pa.int8()),('tab',pa.int8()),
    ('source',pa.string()),('split',pa.string()),
    ('user_seen_in_train',pa.bool_()),('item_seen_in_train',pa.bool_()),
    ('history_item_indices',pa.list_(pa.int32())),('history_time_ms',pa.list_(pa.int64())),
    ('history_length',pa.int16()),
])


def prepare_dataset(config, protocol):
    expected = {'primary_label':'is_click','history_update_source':'standard',
                'history_update_label':'is_click','propensity_policy':'absent_do_not_fabricate',
                'history_policy':'rolling_past_standard_positives_in_train_vocabulary'}
    for name,value in expected.items():
        if config[name] != value:
            raise ValueError(f'Unsupported policy {name}: {config[name]}')
    if config['enabled_input_features'] != ['user_index','item_index','tab','history_item_indices']:
        raise ValueError('Model input allowlist must match the audited protocol')
    output = config['output_directory'].resolve()
    if not output.is_relative_to(ROOT.resolve()) or output.drive.upper() != 'D:':
        raise ValueError('Prepared data must remain inside the D: project')
    db = connect_database(config['staging_database'])
    try:
        audit = json.loads(db.execute("SELECT value FROM metadata WHERE key='raw_audit'").fetchone()[0])
        if audit['invalid_core_rows'] or audit['source_flag_mismatches']:
            raise ValueError('Raw audit failed core fields or source checks; no silent row deletion')
        if audit.get('staging_code_sha256')!=digest_file(Path(__file__).parent/'staging.py'):
            raise ValueError('Staging implementation changed; rerun inspection with --rebuild')
        source_hashes = {name:digest_file(Path(__file__).parent/name) for name in ['prepare.py','protocol.py']}
        signature = {'protocol':asdict(protocol),'raw_files':audit['raw_files'],
                     'staging_code_sha256':audit['staging_code_sha256'],
                     'policy':expected,'input_features':config['enabled_input_features'],'source_hashes':source_hashes}
        fingerprint = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        if output.exists():
            manifest_path = output/'manifest.json'
            if not manifest_path.exists():
                raise FileExistsError('Output exists without a completion manifest; refusing to overwrite')
            previous = json.loads(manifest_path.read_text(encoding='utf-8'))
            if previous['fingerprint'] != fingerprint:
                raise FileExistsError('Input/config/code changed: choose a new output directory, preserving existing data')
            for item in previous['partitions'].values():
                path = (output/item['path']).resolve()
                if not path.is_relative_to(output) or digest_file(path) != item['sha256']:
                    raise ValueError('Existing processed data failed integrity verification')
            if digest_file(output/'vocabulary.json') != previous['vocabulary_sha256']:
                raise ValueError('Existing vocabulary failed integrity verification')
            print('Verified existing preparation; no data rewritten',flush=True)
            return previous
        start = time.perf_counter()
        train_rows = db.execute('SELECT source,user_id,video_id,time_ms,is_click FROM events WHERE source=? AND time_ms>=? AND time_ms<?',
                               ('standard',protocol.train_start_ms,protocol.train_end_ms))
        users,items,popularity = learn_train_vocabulary(train_rows,protocol)
        if not users or not items:
            raise ValueError('No training users/items within configured boundaries')
        work = output.parent/(output.name+'.building-'+uuid.uuid4().hex)
        work.mkdir(parents=True)
        partitions = [f'standard/{split}' for split in ['train','validation','test']] + ['random/validation','random/test']
        writers,buffers,stats = {},{},{}
        for partition in partitions:
            path = work/(partition+'.parquet');path.parent.mkdir(parents=True,exist_ok=True)
            writers[partition] = pq.ParquetWriter(path,SCHEMA,compression='zstd',use_dictionary=['source','split','tab'])
            buffers[partition] = []
            stats[partition] = {'path':partition+'.parquet','rows':0,'positives':0,'cold_user_rows':0,
                                'cold_item_rows':0,'empty_history_rows':0,'sequence_eligible_rows':0,
                                'history_length_counts':Counter(),'min_time_ms':None,'max_time_ms':None}
        written = 0
        try:
            rows = db.execute('SELECT event_id,user_id,video_id,time_ms,is_click,long_view,tab,source FROM events ORDER BY user_id,time_ms,event_id')
            for row in iter_examples(rows,protocol,users,items):
                key = row['source']+'/'+row['split'];stat = stats[key]
                buffers[key].append(row);stat['rows']+=1;stat['positives']+=row['label']
                stat['cold_user_rows']+=not row['user_seen_in_train'];stat['cold_item_rows']+=not row['item_seen_in_train']
                stat['empty_history_rows']+=row['history_length']==0
                stat['sequence_eligible_rows']+=bool(row['label'] and row['history_length'] and row['item_seen_in_train'] and row['user_seen_in_train'])
                stat['history_length_counts'][row['history_length']]+=1
                t=row['time_ms'];stat['min_time_ms']=t if stat['min_time_ms'] is None else min(t,stat['min_time_ms'])
                stat['max_time_ms']=t if stat['max_time_ms'] is None else max(t,stat['max_time_ms'])
                if len(buffers[key])>=config['output_batch_size']:
                    writers[key].write_table(pa.Table.from_pylist(buffers[key],schema=SCHEMA));buffers[key].clear()
                written+=1
                if written%500000==0:print(f'Prepared {written:,} exposures',flush=True)
            for key,writer in writers.items():
                if buffers[key]:writer.write_table(pa.Table.from_pylist(buffers[key],schema=SCHEMA))
        finally:
            for writer in writers.values():writer.close()
        if written != audit['unique_rows']:
            raise AssertionError('Prepared count differs from exact-deduplicated raw events')
        vocabulary = {'user_raw_to_index':users,'item_raw_to_index':items,
                      'train_positive_counts':popularity,'unknown_padding_index':0,
                      'fit_source':'standard train exposures only'}
        vocab_path = work/'vocabulary.json'
        vocab_path.write_text(json.dumps(vocabulary,ensure_ascii=False,sort_keys=True),encoding='utf-8')
        for stat in stats.values():
            path=work/stat['path'];stat.update(bytes=path.stat().st_size,sha256=digest_file(path))
        result = {'status':'PREPARED','timestamp':datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  'fingerprint':fingerprint,'signature':signature,'rows_written':written,'partitions':stats,
                  'train_users':len(users),'train_items':len(items),
                  'items_with_zero_train_positives':sum(popularity[item]==0 for item in items),
                  'vocabulary_sha256':digest_file(vocab_path),
                  'history_policy':expected['history_policy'],'max_history':protocol.max_history,
                  'model_input_allowlist':config['enabled_input_features'],
                  'real_ope_supported':False,'full_rl_transitions_supported':False,
                  'date_mismatch_handling':'Preserved; time_ms is canonical, date/hourmin are audit-only',
                  'elapsed_seconds':time.perf_counter()-start,
                  'process_peak_working_set_MiB':psutil.Process().memory_info().peak_wset/2**20}
        (work/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        work.rename(output)
        print(f'Prepared {written:,} rows -> {output}',flush=True)
        return result
    finally:
        db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path)
    args=parser.parse_args();config,protocol=load_config(args.config)
    from src.data.inspect_dataset import inspect
    inspect(config)
    result=prepare_dataset(config,protocol)
    (ROOT/'reports/data_preparation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
