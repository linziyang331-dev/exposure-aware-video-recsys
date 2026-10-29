"""Freeze exact configs, source, prepared inputs and selected checkpoints."""
import json
from pathlib import Path
from src.data.download import digest_file


def freeze_evidence(root,config,paths):
    root=Path(root).resolve();files={}
    for path in paths:
        path=Path(path).resolve()
        if not path.is_relative_to(root):raise ValueError('Evidence must remain under its root')
        files[path.relative_to(root).as_posix()]=digest_file(path)
    return {'config':json.loads(json.dumps(config)),'files':files}


def verify_evidence(root,frozen,config):
    root=Path(root).resolve()
    if config!=frozen['config']:raise ValueError('Frozen configuration changed')
    for name,expected in frozen['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest_file(path)!=expected:
            raise ValueError('Frozen evidence changed: '+name)
