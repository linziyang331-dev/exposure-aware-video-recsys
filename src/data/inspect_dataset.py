"""Audit the actual KuaiRand release and build a bounded-memory staging database."""
import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from src.data.common import ROOT, load_config
from src.data.download import RAW, digest_file
from src.data.staging import connect_database, stage_logs, load_verified_stage


def quantiles(values):
    values = np.asarray(values)
    return {str(q): float(np.quantile(values, q)) for q in [0, .25, .5, .75, .9, .95, .99, 1]}


def verify_raw_manifest(raw_directory=None):
    expected=(RAW/'extracted/KuaiRand-Pure/data').resolve()
    if raw_directory is not None and Path(raw_directory).resolve()!=expected:
        raise ValueError('Inspection must use the verified official raw directory')
    manifest = json.loads((RAW/'manifest.json').read_text(encoding='utf-8'))
    extraction = (RAW/'extracted').resolve()
    for item in manifest['files']:
        path = (extraction/item['path']).resolve()
        if not path.is_relative_to(extraction) or digest_file(path) != item['sha256']:
            raise ValueError(f'Raw file integrity mismatch: {item["path"]}')
    return manifest


def inspect(config, *, rebuild=False):
    start = time.perf_counter()
    manifest = verify_raw_manifest(config['raw_directory'])
    raw, database = config['raw_directory'], config['staging_database']
    logs = sorted(raw.glob('log_*.csv'))
    if rebuild and database.exists():
        if not database.resolve().is_relative_to((ROOT/'.runtime').resolve()):
            raise ValueError('Rebuild can remove only the project runtime database')
        database.unlink()
    if database.exists():
        audit = load_verified_stage(logs,database)
        print('Reusing verified staging database', flush=True)
    else:
        audit = stage_logs(logs, database, chunk_size=config['chunk_size'])
    db = connect_database(database)
    try:
        all_users = {r[0] for r in db.execute('SELECT DISTINCT user_id FROM events')}
        all_items = {r[0] for r in db.execute('SELECT DISTINCT video_id FROM events')}
        audit['total_users'] = len(all_users); audit['total_videos'] = len(all_items)
        mismatch="replace(date(time_ms/1000,'unixepoch','+8 hours'),'-','') != CAST(date AS TEXT)"
        audit['date_mismatch_hours']=[dict(row) for row in db.execute(f"SELECT source,strftime('%H',time_ms/1000,'unixepoch','+8 hours') local_hour,COUNT(*) rows FROM raw_events WHERE {mismatch} GROUP BY source,local_hour")]
        next_date_count=db.execute(f"SELECT COUNT(*) FROM raw_events WHERE {mismatch} AND replace(date(time_ms/1000,'unixepoch','+8 hours','+1 day'),'-','')=CAST(date AS TEXT)").fetchone()[0]
        audit['date_mismatch_interpretation']={'stored_date_is_next_calendar_day_rows':next_date_count,
            'clock_or_bucket_mechanism_verified':False,'canonical_field':'time_ms',
            'handling':'Preserve rows; date/hourmin are audit-only, not split keys or input features'}
        audit['user_interaction_quantiles'] = quantiles([r[0] for r in db.execute('SELECT COUNT(*) FROM events GROUP BY user_id')])
        audit['video_exposure_quantiles'] = quantiles([r[0] for r in db.execute('SELECT COUNT(*) FROM events GROUP BY video_id')])
        ordered = [r[0] for r in db.execute('SELECT COUNT(*) n FROM events GROUP BY video_id ORDER BY n DESC')]
        audit['top_10_percent_video_exposure_share'] = sum(ordered[:max(1,int(np.ceil(len(ordered)*.1)))])/sum(ordered)
        audit['feature_tables'] = []
        for path in sorted(raw.glob('*features*.csv')):
            header = pd.read_csv(path,nrows=0).columns.tolist()
            frame = pd.read_csv(path,dtype={'tag':'string'} if 'tag' in header else None)
            key = 'user_id' if 'user_id' in frame else 'video_id'
            known = set(frame[key].dropna().astype('int64').tolist())
            required = all_users if key == 'user_id' else all_items
            audit['feature_tables'].append({
                'file':path.name,'rows':len(frame),'columns':frame.columns.tolist(),
                'dtypes':frame.dtypes.astype(str).to_dict(),
                'missing':{k:int(v) for k,v in frame.isna().sum().items()},
                'duplicate_key_rows':int(frame.duplicated(key).sum()),
                'log_entities_missing_metadata':len(required-known),
                'enabled_as_model_inputs':False,
                'reason':'Monthly aggregates include future periods' if 'statistic' in path.name else
                         'Snapshot availability not established; initial protocol uses IDs/tab/history only'})
        audit['propensity_fields'] = [name for name in audit['columns'] if any(token in name.lower() for token in ['propensity','pscore','probability'])]
        audit['real_ope_supported'] = False
        audit['ope_reason'] = 'No per-event behavior action probability and reconstructible action support; intervention rate is not an action propensity.'
        audit['full_rl_transitions_supported'] = False
        audit['rl_reason'] = 'Pure removes out-of-pool interactions; observed subsequences are not verified complete state transitions.'
        audit['license'] = {'zenodo_record':'cc-by-4.0',
                            'archive_license_first_line':(raw.parent/'LICENSE').read_text().splitlines()[0],
                            'metadata_conflict':True,'raw_data_redistributed_in_repository':False}
        audit['download'] = manifest
        audit['audit_timestamp'] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        audit['inspection_elapsed_seconds'] = time.perf_counter()-start
        audit['process_peak_working_set_MiB'] = psutil.Process().memory_info().peak_wset/2**20
        audit['database_bytes'] = database.stat().st_size
    finally:
        db.close()
    destination = ROOT/'reports/data_audit.json'
    destination.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['raw_rows','unique_rows','exact_duplicates_removed','invalid_core_rows','source_flag_mismatches','date_timestamp_mismatches','source_summary','inspection_elapsed_seconds','process_peak_working_set_MiB']},ensure_ascii=False,indent=2),flush=True)
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--rebuild',action='store_true',help='Explicitly rebuild only the derived runtime database')
    args=parser.parse_args()
    config,_=load_config(args.config)
    inspect(config,rebuild=args.rebuild)


if __name__=='__main__':
    main()
