"""Read the unchanged verified exposure cache; ranking never re-splits logs."""
import json
from pathlib import Path
import yaml
from src.data.common import ROOT
from src.retrieval.data import load_partition,training_counts
from src.retrieval.provenance import read_verified_vocabulary
from src.ranking.provenance import verify_v2


def setup(partitions=('standard/train','standard/validation')):
    frozen=verify_v2();cfg=yaml.safe_load((ROOT/'configs/ranking.yaml').read_text(encoding='utf-8'))
    paths={name:(ROOT/cfg[key]).resolve() for name,key in [('prepared','prepared_directory'),('cache','cache_directory'),('output','output_directory')]}
    for path in paths.values():
        if not path.is_relative_to(ROOT.resolve()) or path.drive.upper()!='D:':raise ValueError('D: project paths required')
    vocabulary=read_verified_vocabulary(paths['prepared']);counts=training_counts(vocabulary)
    caches={name:load_partition(paths['prepared'],name,paths['cache'],50) for name in partitions}
    return cfg,paths,vocabulary,counts,caches,frozen


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')
