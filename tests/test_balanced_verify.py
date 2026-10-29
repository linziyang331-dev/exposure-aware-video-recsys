import unittest
import tempfile
from pathlib import Path
from src.data.common import ROOT
from src.data.download import digest_file
from src.retrieval_longtail.train import write_json
from src.retrieval_balanced.verify import selected_epoch


class BalancedVerifyTests(unittest.TestCase):
    def m(self,n,t,p):return {'recall@20':.1,'ndcg@20':n,'groups':{'head_target':{'recall@20':.13},'tail_target':{'recall@20':t,'recall@100':p}}}
    def test_epoch_reconstruction_uses_qualified_tail_utility(self):
        b=self.m(.04,.01,.05);gate={'retention':.98,'tail20_delta':.002,'tail100_delta':.01}
        rows=[{'epoch':0,'validation':self.m(.041,.011,.055)},
              {'epoch':1,'validation':self.m(.04,.02,.1)},
              {'epoch':2,'validation':self.m(.0393,.025,.12)}]
        self.assertEqual(selected_epoch(rows,b,gate),(2,True))
        self.assertEqual(selected_epoch(rows[:1],b,gate),(0,False))

    def test_canonical_report_cannot_omit_global_mirror_hashes(self):
        from src.retrieval_balanced.verify import validate_delivery
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/tmp') as d:
            p=Path(d);(p/'report.md').write_text('report',encoding='utf-8')
            write_json(p/'selection.json',{'config':{'output_directory':p.relative_to(ROOT).as_posix()}})
            write_json(p/'report_manifest.json',{'output_hashes':{'report.md':digest_file(p/'report.md')},'global_mirrors':{}})
            with self.assertRaises(ValueError):validate_delivery(p)
