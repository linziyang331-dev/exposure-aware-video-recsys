import csv
import sqlite3
import json
from contextlib import closing
import tempfile
import unittest
from pathlib import Path


COLUMNS = ['user_id','video_id','date','hourmin','time_ms','is_click','is_like','is_follow',
           'is_comment','is_forward','is_hate','long_view','play_time_ms','duration_ms',
           'profile_stay_time','comment_stay_time','is_profile_enter','is_rand','tab']


class DataStagingTests(unittest.TestCase):
    def test_exact_duplicates_removed_but_later_repeat_exposure_kept(self):
        from src.data.staging import stage_logs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/'log_standard_fixture.csv'
            first = [0,5,20220408,0,1649347200000,1,0,0,0,0,0,1,1000,1000,0,0,0,0,1]
            later = list(first);later[4] += 1000
            with path.open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(COLUMNS);writer.writerows([first,first,later])
            report=stage_logs([path],root/'staging.sqlite',chunk_size=1)
            self.assertEqual(report['raw_rows'],3)
            self.assertEqual(report['unique_rows'],2)
            self.assertEqual(report['exact_duplicates_removed'],1)
            with closing(sqlite3.connect(root/'staging.sqlite')) as db:
                self.assertEqual(db.execute('select count(*) from events').fetchone()[0],2)

    def test_file_source_disagreement_is_audited(self):
        from src.data.staging import stage_logs
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'log_random_fixture.csv'
            row=[0,5,20220408,0,1649347200000,1,0,0,0,0,0,1,1000,1000,0,0,0,0,1]
            with path.open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(COLUMNS);writer.writerow(row)
            report=stage_logs([path],root/'staging.sqlite',chunk_size=1)
            self.assertEqual(report['source_flag_mismatches'],1)

    def test_cached_staging_rejects_changed_implementation(self):
        from src.data.staging import stage_logs, load_verified_stage
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'log_standard_fixture.csv'
            row=[0,5,20220408,0,1649347200000,1,0,0,0,0,0,1,1000,1000,0,0,0,0,1]
            with path.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(COLUMNS);writer.writerow(row)
            database=root/'stage.sqlite';report=stage_logs([path],database,chunk_size=1)
            self.assertEqual(load_verified_stage([path],database)['raw_rows'],1)
            report['staging_code_sha256']='0'*64
            with closing(sqlite3.connect(database)) as db:
                db.execute('UPDATE metadata SET value=? WHERE key=?',(json.dumps(report),'raw_audit'));db.commit()
            with self.assertRaisesRegex(ValueError,'staging code changed'):
                load_verified_stage([path],database)

    def test_inspection_rejects_unbound_raw_directory(self):
        from src.data.inspect_dataset import verify_raw_manifest
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'official raw directory'):
                verify_raw_manifest(Path(directory))
