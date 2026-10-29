"""The source release demo must run without private data or frozen weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


class PublicDemoTests(unittest.TestCase):
    def test_cpu_demo_trains_and_recomputes_without_frozen_evidence(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.demo_cpu'),'Public demo missing')
        from scripts.demo_cpu import run_demo
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'demo'
            result=run_demo(root)
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(result['verification']['models_reloaded'],4)
            self.assertEqual(result['ope']['ips'],.5)
            self.assertEqual(result['ope']['snips'],.5)
            self.assertFalse(result['real_top20_improvement_claimed'])
            self.assertTrue((root/'summary.json').is_file())
            self.assertEqual(json.loads((root/'summary.json').read_text())['status'],'PASS')
            with self.assertRaises(FileExistsError):run_demo(root)


if __name__=='__main__':unittest.main()
