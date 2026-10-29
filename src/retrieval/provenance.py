"""Verify vocabulary, declared input manifests, and frozen scoring implementation."""
import json
from pathlib import Path
from src.data.download import digest_file

CORE_FILES=['data.py','metrics.py','model.py','baselines.py','evaluate.py','train.py','run.py','provenance.py']


def read_verified_vocabulary(prepared):
    prepared=Path(prepared)
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    path=prepared/'vocabulary.json'
    if digest_file(path)!=manifest['vocabulary_sha256']:raise ValueError('Prepared vocabulary hash mismatch')
    return json.loads(path.read_text(encoding='utf-8'))


def code_hashes():
    directory=Path(__file__).parent
    return {name:digest_file(directory/name) for name in CORE_FILES}


def verify_code_hashes(expected,directory=None):
    directory=Path(directory) if directory else Path(__file__).parent
    for name,digest in expected.items():
        if Path(name).name!=name or digest_file(directory/name)!=digest:
            raise ValueError(f'Frozen retrieval code changed: {name}')


def verify_frozen_inputs(selection,prepared):
    prepared=Path(prepared)
    verify_code_hashes(selection['code_hashes'])
    if digest_file(prepared/'manifest.json')!=selection['prepared_manifest_sha256']:
        raise ValueError('Prepared manifest changed after validation selection')
    if digest_file(prepared/'vocabulary.json')!=selection['vocabulary_sha256']:
        raise ValueError('Frozen vocabulary hash changed')
