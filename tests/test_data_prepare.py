import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from tests.test_data_staging import COLUMNS


class DataPreparationIntegrationTests(unittest.TestCase):
    def test_parquet_partitions_histories_cold_entities_and_repeat_run(self):
        from src.data.prepare import prepare_dataset
        from src.data.protocol import DataProtocol
        from src.data.staging import stage_logs
        def ms(day):return int(datetime.fromisoformat(day+'T00:00:00+08:00').timestamp()*1000)
        def row(user,item,day,label,random=0,offset=0):
            return [user,item,int(day.replace('-','')),0,ms(day)+offset,label,0,0,0,0,0,label,1000,1000,0,0,0,random,1]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw=root/'raw';raw.mkdir()
            standard=[row(0,5,'2022-04-08',1),row(0,6,'2022-04-21',0),
                      row(0,5,'2022-04-22',1),row(0,6,'2022-04-29',1)]
            random=[row(0,6,'2022-04-22',1,1,1000),row(1,7,'2022-04-29',0,1,1000)]
            for name,rows in [('standard',standard),('random',random)]:
                with (raw/f'log_{name}_fixture.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(COLUMNS);writer.writerows(rows)
            database=root/'staging.sqlite';stage_logs(raw.glob('log_*.csv'),database,chunk_size=2)
            config={'staging_database':database,'output_directory':root/'processed','output_batch_size':2,
                    'primary_label':'is_click','history_update_source':'standard',
                    'history_update_label':'is_click','history_policy':'rolling_past_standard_positives_in_train_vocabulary',
                    'propensity_policy':'absent_do_not_fabricate',
                    'enabled_input_features':['user_index','item_index','tab','history_item_indices']}
            protocol=DataProtocol(ms('2022-04-08'),ms('2022-04-22'),ms('2022-04-29'),ms('2022-05-09'),2)
            result=prepare_dataset(config,protocol)
            self.assertEqual(result['rows_written'],6)
            self.assertEqual(result['partitions']['standard/train']['rows'],2)
            val=pq.read_table(root/'processed/random/validation.parquet').to_pylist()
            cold=pq.read_table(root/'processed/random/test.parquet').to_pylist()
            self.assertEqual(val[0]['history_item_indices'],[1,1])
            self.assertEqual(cold[0]['user_index'],0)
            self.assertEqual(cold[0]['item_index'],0)
            self.assertEqual(cold[0]['history_item_indices'],[])
            self.assertEqual(prepare_dataset(config,protocol)['rows_written'],6)
            from src.data.validate import validate_preparation
            checked=validate_preparation(config,protocol)
            self.assertEqual(checked['rows_checked'],6)
            self.assertEqual(checked['sampled_histories_checked'],6)
            # Even a hash-consistent file must fail semantic future-leak checks.
            path=root/'processed/random/validation.parquet'
            table=pq.read_table(path);rows=table.to_pylist()
            rows[0]['history_time_ms'][0]=rows[0]['time_ms']
            import pyarrow as pa
            pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),path)
            manifest_path=root/'processed/manifest.json'
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['partitions']['random/validation']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(AssertionError,'strictly earlier'):
                validate_preparation(config,protocol)
            rows=table.to_pylist();rows[0]['label']=None
            pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),path)
            manifest['partitions']['random/validation']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
            with self.assertRaisesRegex(AssertionError,'Null values'):
                validate_preparation(config,protocol)
