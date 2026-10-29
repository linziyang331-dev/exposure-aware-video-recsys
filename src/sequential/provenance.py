"""Current native implementation and previously frozen recommendation evidence."""
import json
from pathlib import Path
import recbole
from src.data.common import ROOT
from src.data.download import digest_file
from src.ranking.provenance import verify_v2
from src.retrieval_v2.provenance import verify_evidence


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def code_hashes():
    local=Path(__file__).parent;up=Path(recbole.__file__).parent
    return {'recbole_version':recbole.__version__,
            'sequential':{n:digest_file(local/n) for n in ['model.py','train.py','evaluate.py','provenance.py']},
            'upstream':{n:digest_file(up/n) for n in ['model/sequential_recommender/sasrec.py','model/layers.py',
                         'model/abstract_recommender.py','model/loss.py','data/interaction.py']},
            'shared':{n:digest_file(ROOT/n) for n in ['src/retrieval/data.py','src/retrieval/evaluate.py',
                       'src/retrieval/metrics.py','src/retrieval_v2/evaluate.py','src/ranking/adapter.py','src/ranking/model.py']}}


def verify_previous():
    frozen=verify_v2();din=read(ROOT/'experiments/ranking_v2/selection.json')
    verify_evidence(ROOT,din['evidence'],din['config'])
    rep=read(ROOT/'experiments/ranking_v2/replication.json');verify_evidence(ROOT,rep['evidence'],din['config'])
    return frozen,din
