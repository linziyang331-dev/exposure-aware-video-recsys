"""Download and verify the official KuaiRand-Pure release, entirely on D:."""
import hashlib
import json
import shutil
import tarfile
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path, PurePosixPath

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/kuairand_pure'
URL = 'https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz?download=1'
MD5 = '0820331067a3784d9691136f772b35a7'
SIZE = 47432272


def digest_file(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path, *, expected_md5=MD5, expected_size=None):
    path = Path(path)
    md5 = digest_file(path, 'md5')
    if md5 != expected_md5:
        raise ValueError(f'MD5 mismatch for {path.name}: {md5} != {expected_md5}')
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f'Archive size mismatch: {size} != {expected_size}')
    return {'bytes': size, 'md5': md5, 'sha256': digest_file(path)}


def extract_archive(path, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Extraction destination already exists: {destination}')
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) > 100 or sum(member.size for member in members) > 2 * 1024**3:
            raise ValueError('Archive exceeds the reviewed Pure extraction budget')
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe archive member: {member.name}')
            resolved = (destination / member.name).resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise ValueError(f'Archive member escapes destination: {member.name}')
        destination.mkdir(parents=True)
        archive.extractall(destination, members=members, filter='data')


def main():
    if RAW.resolve().drive.upper() != 'D:' or not RAW.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('This project requires the explicit D: raw-data destination')
    RAW.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(RAW).free < 2 * 1024**3:
        raise RuntimeError('Require at least 2 GiB free before retrieval/extraction')
    archive = RAW / 'KuaiRand-Pure.tar.gz'
    start = time.perf_counter()
    if not archive.exists():
        partial = RAW / f'KuaiRand-Pure.{uuid.uuid4().hex}.part'
        print(f'Downloading {URL} -> {partial}', flush=True)
        with requests.get(URL, stream=True, timeout=(20, 45)) as response:
            response.raise_for_status()
            received = 0
            with partial.open('xb') as stream:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    stream.write(block)
                    received += len(block)
                    if received // (8 * 1024**2) != (received-len(block)) // (8 * 1024**2):
                        print(f'Downloaded {received / 1024**2:.1f} MiB', flush=True)
        verify_archive(partial, expected_size=SIZE)
        partial.rename(archive) # Windows refuses to replace an existing destination.
    integrity = verify_archive(archive, expected_size=SIZE)
    destination, manifest_path = RAW / 'extracted', RAW / 'manifest.json'
    if destination.exists():
        if not manifest_path.exists():
            raise FileExistsError('Existing extraction has no manifest; refusing to overwrite')
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        for file in previous['files']:
            candidate = (destination / file['path']).resolve()
            if not candidate.is_relative_to(destination.resolve()) or digest_file(candidate) != file['sha256']:
                raise ValueError(f'Existing extracted file failed verification: {file["path"]}')
        print('Verified existing archive and extracted files; no changes', flush=True)
        return
    extract_archive(archive, destination)
    files = [{'path': str(path.relative_to(destination)), 'bytes': path.stat().st_size,
              'sha256': digest_file(path)} for path in sorted(destination.rglob('*')) if path.is_file()]
    result = {'status': 'VERIFIED', 'timestamp': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'source': URL, 'metadata_record': 'https://zenodo.org/records/10439422',
              'record_license': 'cc-by-4.0', 'archive': integrity, 'files': files,
              'extracted_bytes': sum(file['bytes'] for file in files),
              'elapsed_seconds': time.perf_counter()-start}
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
