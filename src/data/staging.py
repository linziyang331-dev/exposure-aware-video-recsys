"""Chunked CSV audit and exact disk-backed deduplication."""
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.download import digest_file
CORE = ['user_id','video_id','time_ms','is_click','long_view','is_rand','tab']

def connect_database(path):
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    db.execute('PRAGMA cache_size=-65536');db.execute('PRAGMA temp_store=FILE')
    return db

def load_verified_stage(paths,database):
    db=connect_database(database)
    try:
        try:
            row=db.execute("SELECT value FROM metadata WHERE key='raw_audit'").fetchone()
        except sqlite3.OperationalError as error:
            raise ValueError('Incomplete staging database; use --rebuild') from error
        if row is None:raise ValueError('Incomplete staging database; use --rebuild')
        report=json.loads(row[0])
    finally:
        db.close()
    if report.get('staging_code_sha256')!=digest_file(__file__):
        raise ValueError('staging code changed; use an explicit --rebuild')
    if report['raw_files']!={Path(p).name:digest_file(p) for p in sorted(map(Path,paths))}:
        raise ValueError('Raw input changed; use an explicit --rebuild')
    return report

def stage_logs(paths,database,*,chunk_size=100000):
    database=Path(database)
    if database.exists():raise FileExistsError(f'Refusing to replace {database}')
    if chunk_size<=0:raise ValueError('chunk_size must be positive')
    paths=sorted(map(Path,paths))
    if not paths:raise FileNotFoundError('No raw logs')
    database.parent.mkdir(parents=True,exist_ok=True)
    cols=pd.read_csv(paths[0],nrows=0).columns.tolist()
    if not set(CORE+['date']).issubset(cols):raise ValueError('Missing required log fields')
    if any(not c.replace('_','').isalnum() for c in cols):raise ValueError('Invalid column identifier')
    quoted=','.join(f'"{c}"' for c in cols)
    definitions=','.join(f'"{c}" INTEGER' for c in cols)
    db=connect_database(database)
    db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF')
    db.execute(f'CREATE TABLE raw_events(event_id INTEGER PRIMARY KEY,source TEXT,source_file TEXT,source_row INTEGER,{definitions})')
    insert=f'INSERT INTO raw_events(source,source_file,source_row,{quoted}) VALUES({",".join("?" for _ in range(len(cols)+3))})'
    report={'raw_rows':0,'columns':cols,'files':[],'invalid_core_rows':0,'source_flag_mismatches':0,'date_timestamp_mismatches':0,'raw_files':{p.name:digest_file(p) for p in paths}}
    report['staging_code_sha256']=digest_file(__file__)
    start=time.perf_counter()
    for path in paths:
        if not path.name.startswith(('log_random_','log_standard_')):raise ValueError('Unknown log filename')
        source='random' if path.name.startswith('log_random_') else 'standard'
        stats={'file':path.name,'source':source,'rows':0,'missing':Counter(),'feedback':{},'negative_duration_rows':0,'negative_play_time_rows':0,'zero_duration_rows':0,'play_longer_than_video_rows':0,'time_min_ms':None,'time_max_ms':None,'dates':Counter(),'tabs':Counter()}
        users,videos=set(),set()
        for chunk in pd.read_csv(path,chunksize=chunk_size):
            if chunk.columns.tolist()!=cols:raise ValueError('Log schemas differ')
            if any(not pd.api.types.is_numeric_dtype(chunk[c]) for c in cols):raise ValueError('Non-numeric log values')
            stats['missing'].update({c:int(n) for c,n in chunk.isna().sum().items()})
            bad=chunk[CORE].isna().any(axis=1)|~np.isfinite(chunk[CORE]).all(axis=1)
            bad|=(chunk[['user_id','video_id']]<0).any(axis=1)
            bad|=(chunk['time_ms']<=0)|~chunk['is_click'].isin([0,1])|~chunk['long_view'].isin([0,1])
            bad|=~chunk['is_rand'].isin([0,1])|~chunk['tab'].between(0,14)
            report['invalid_core_rows']+=int(bad.sum())
            report['source_flag_mismatches']+=int((chunk['is_rand']!=int(source=='random')).sum())
            dates=pd.to_datetime(chunk['time_ms'],unit='ms',utc=True,errors='coerce').dt.tz_convert('Asia/Shanghai').dt.strftime('%Y%m%d')
            report['date_timestamp_mismatches']+=int((pd.to_numeric(dates,errors='coerce')!=chunk['date']).sum())
            for c in [c for c in cols if c.startswith('is_') or c=='long_view']:
                stats['feedback'].setdefault(c,Counter()).update({str(int(k)):int(v) for k,v in chunk[c].value_counts().items()})
            for key,mask in [('negative_duration_rows',chunk['duration_ms']<0),('negative_play_time_rows',chunk['play_time_ms']<0),('zero_duration_rows',chunk['duration_ms']==0),('play_longer_than_video_rows',chunk['play_time_ms']>chunk['duration_ms'])]:stats[key]+=int(mask.sum())
            low,high=int(chunk['time_ms'].min()),int(chunk['time_ms'].max())
            stats['time_min_ms']=low if stats['time_min_ms'] is None else min(low,stats['time_min_ms'])
            stats['time_max_ms']=high if stats['time_max_ms'] is None else max(high,stats['time_max_ms'])
            for c,key in [('date','dates'),('tab','tabs')]:stats[key].update({str(int(k)):int(v) for k,v in chunk[c].value_counts().items()})
            users.update(chunk['user_id'].dropna().astype('int64').unique().tolist());videos.update(chunk['video_id'].dropna().astype('int64').unique().tolist())
            offset=stats['rows']
            db.executemany(insert,((source,path.name,offset+i+1,*row) for i,row in enumerate(chunk.itertuples(index=False,name=None))))
            db.commit();stats['rows']+=len(chunk)
        stats.update(users=len(users),videos=len(videos));report['raw_rows']+=stats['rows'];report['files'].append(stats)
        print(f'Staged {path.name}: {stats["rows"]:,} rows',flush=True)
    db.execute(f'CREATE TABLE dedup_ids AS SELECT MIN(event_id) event_id,COUNT(*) copies FROM raw_events GROUP BY source,{quoted}')
    db.execute('CREATE UNIQUE INDEX dedup_event_id ON dedup_ids(event_id)')
    db.execute('CREATE INDEX event_user_time ON raw_events(user_id,time_ms,event_id)')
    db.execute('CREATE VIEW events AS SELECT r.* FROM raw_events r JOIN dedup_ids d ON r.event_id=d.event_id')
    report['unique_rows']=db.execute('SELECT COUNT(*) FROM dedup_ids').fetchone()[0]
    report['exact_duplicates_removed']=report['raw_rows']-report['unique_rows']
    report['source_summary']=[dict(r) for r in db.execute('SELECT source,COUNT(*) rows,COUNT(DISTINCT user_id) users,COUNT(DISTINCT video_id) videos,SUM(is_click) positives,AVG(is_click) positive_rate,AVG(long_view) long_view_rate,MIN(time_ms) min_time_ms,MAX(time_ms) max_time_ms FROM events GROUP BY source')]
    report['source_tab_summary']=[dict(r) for r in db.execute('SELECT source,tab,COUNT(*) rows,AVG(is_click) positive_rate FROM events GROUP BY source,tab')]
    report['same_identity_extra_rows']=db.execute('SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM events GROUP BY source,user_id,video_id,time_ms,tab HAVING COUNT(*)>1)').fetchone()[0]
    report['elapsed_seconds']=time.perf_counter()-start
    db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)');db.execute('INSERT INTO metadata VALUES(?,?)',('raw_audit',json.dumps(report)))
    db.commit();db.close()
    return report
