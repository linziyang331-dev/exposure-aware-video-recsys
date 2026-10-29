"""Runtime upstream source hashes and immutable two-tower V2 evidence."""
import json
from pathlib import Path
import recbole
import yaml
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval_v2.provenance import verify_evidence


def verify_v2():
    path=ROOT/'reports/retrieval_v2_freeze.json'
    frozen=json.loads(path.read_text(encoding='utf-8'))
    config=yaml.safe_load((ROOT/frozen['config_path']).read_text(encoding='utf-8'))
    verify_evidence(ROOT,frozen['evidence'],config)
    return frozen


def implementation_hashes():
    root=Path(recbole.__file__).parent
    upstream=['model/sequential_recommender/din.py','model/context_aware_recommender/lr.py',
              'model/layers.py','model/abstract_recommender.py','data/interaction.py']
    local=Path(__file__).parent
    return {'recbole_version':recbole.__version__,'upstream':{name:digest_file(root/name) for name in upstream},
            'local':{name:digest_file(local/name) for name in ['adapter.py','model.py','metrics.py','train.py','provenance.py']}}
