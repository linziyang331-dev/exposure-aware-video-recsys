"""Public release boundaries, independent of datasets and old model artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


class ReleaseTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.prepare_release'), 'Release tool missing')
        from scripts import prepare_release
        return prepare_release

    def fixture(self, root):
        (root/'src').mkdir();(root/'src/a.py').write_text('value = 1\n')
        (root/'README.md').write_text('[Source](src/a.py)\n',encoding='utf-8')
        (root/'data').mkdir();(root/'data/private.csv').write_text('private')
        (root/'src/__pycache__').mkdir();(root/'src/__pycache__/a.pyc').write_bytes(b'cache')
        return {'version':'0.1.0','directories':['src'],'files':['README.md'],'max_file_bytes':1024,'max_total_bytes':4096}

    def test_only_allowlisted_files_are_exported_with_bound_hashes_and_no_overwrite(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';root.mkdir();cfg=self.fixture(root)
            files=api.collect_files(root,cfg)
            self.assertEqual([p.as_posix() for p in files],['README.md','src/a.py'])
            output=Path(d)/'release';manifest=api.export_release(root,cfg,output)
            self.assertEqual(manifest['file_count'],2)
            for name,digest in manifest['files'].items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(),digest['sha256'])
            with zipfile.ZipFile(Path(str(output)+'.zip')) as archive:
                self.assertEqual(set(archive.namelist()),{'README.md','src/a.py','release_manifest.json'})
            with self.assertRaises(FileExistsError):api.export_release(root,cfg,output)

    def test_dotted_version_keeps_full_name_in_zip(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';root.mkdir();cfg=self.fixture(root)
            output=Path(d)/'v0.1.0';api.export_release(root,cfg,output)
            self.assertTrue((Path(d)/'v0.1.0.zip').is_file())

    def test_missing_outside_and_repeated_paths_are_rejected(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=self.fixture(root)
            for value in ['../outside','/absolute','D:/outside','missing.py','src/../README.md']:
                with self.subTest(value=value),self.assertRaises(ValueError):
                    api.collect_files(root,{**cfg,'files':[value]})
            with self.assertRaises(ValueError):api.collect_files(root,{**cfg,'files':['README.md','README.md']})

    def test_sensitive_large_unsupported_and_broken_link_files_fail_audit(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=self.fixture(root)
            cases=['[missing](missing.md)','C:'+'\\Users\\'+'private-person\\x','ghp_'+'a'*36]
            for value in cases:
                (root/'README.md').write_text(value,encoding='utf-8')
                with self.subTest(value=value[:12]),self.assertRaises(ValueError):api.audit_files(root,cfg)
            (root/'README.md').write_text('x'*1025)
            with self.assertRaises(ValueError):api.audit_files(root,cfg)
            (root/'README.md').write_text('ok')
            (root/'src/.env').write_text('TOKEN=x')
            with self.assertRaises(ValueError):api.audit_files(root,cfg)

    def test_export_inside_published_source_and_symlinks_are_rejected(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=self.fixture(root)
            with self.assertRaises(ValueError):api.export_release(root,cfg,root/'src/release')
            try:(root/'src/link.py').symlink_to(root/'src/a.py')
            except OSError:self.skipTest('Symlink creation unavailable on this Windows account')
            with self.assertRaises(ValueError):api.collect_files(root,cfg)

    def test_export_verification_rejects_changed_extra_and_missing_files(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'source';root.mkdir();cfg=self.fixture(root)
            output=Path(d)/'release';api.export_release(root,cfg,output)
            api.verify_release(output)
            p=output/'src/a.py';original=p.read_bytes();p.write_text('modified')
            with self.assertRaises(ValueError):api.verify_release(output)
            p.write_bytes(original);(output/'extra.txt').write_text('extra')
            with self.assertRaises(ValueError):api.verify_release(output)
            (output/'extra.txt').unlink();p.unlink()
            with self.assertRaises(ValueError):api.verify_release(output)


if __name__=='__main__':unittest.main()
