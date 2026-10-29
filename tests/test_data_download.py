import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path


class DownloadIntegrityTests(unittest.TestCase):
    def test_wrong_archive_digest_is_rejected(self):
        from src.data.download import verify_archive
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.tar.gz'
            path.write_bytes(b'corrupted data')
            with self.assertRaisesRegex(ValueError, 'MD5'):
                verify_archive(path, expected_md5='0' * 32)

    def test_matching_archive_digest_is_accepted(self):
        from src.data.download import verify_archive
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.tar.gz'
            path.write_bytes(b'known fixture')
            result = verify_archive(path, expected_md5=hashlib.md5(b'known fixture').hexdigest())
            self.assertEqual(result['bytes'], 13)

    def test_archive_traversal_does_not_write_outside_destination(self):
        from src.data.download import extract_archive
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'bad.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                info = tarfile.TarInfo('../escaped.txt');info.size = 3
                archive.addfile(info, io.BytesIO(b'bad'))
            with self.assertRaises((ValueError, tarfile.FilterError)):
                extract_archive(path, root / 'output')
            self.assertFalse((root / 'escaped.txt').exists())

    def test_existing_extraction_is_not_overwritten(self):
        from src.data.download import extract_archive
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);output = root / 'output';output.mkdir()
            (output / 'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                extract_archive(root / 'unused.tar.gz', output)
            self.assertEqual((output / 'keep.txt').read_text(), 'keep')
