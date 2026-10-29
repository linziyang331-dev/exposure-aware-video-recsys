"""Explicit project paths and timezone-qualified calendar configuration."""
from datetime import datetime
from pathlib import Path
import yaml
from src.data.protocol import DataProtocol
ROOT=Path(__file__).resolve().parents[2]
def load_config(path=None):
    config=yaml.safe_load((Path(path) if path else ROOT/'configs/data.yaml').read_text(encoding='utf-8'))
    for key in ['raw_directory','staging_database','output_directory']:
        resolved=(ROOT/config[key]).resolve()
        if not resolved.is_relative_to(ROOT.resolve()) or resolved.drive.upper()!='D:':raise ValueError(f'{key} must stay within the D: project')
        config[key]=resolved
    if config['chunk_size']<=0 or config['output_batch_size']<=0:raise ValueError('Batch sizes must be positive')
    def ms(key):
        date=datetime.fromisoformat(config[key])
        if date.tzinfo is None:raise ValueError('Explicit timezone required')
        return int(date.timestamp()*1000)
    return config,DataProtocol(ms('train_start'),ms('train_end'),ms('validation_end'),ms('test_end'),config['max_history'])
