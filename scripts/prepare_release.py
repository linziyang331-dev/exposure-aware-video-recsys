"""Audit an explicit public allowlist and export a source release without publishing."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {'__pycache__', '.git', '.runtime', 'log_tensorboard', '.pytest_cache'}
TEXT_SUFFIXES = {'.py', '.ps1', '.yaml', '.yml', '.json', '.md', '.csv', '.txt', '.patch'}
SPECIAL_NAMES = {'LICENSE', 'NOTICE', '.gitignore', '.gitattributes'}
SECRET_PATTERNS = [
    re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}'),
    re.compile(r'sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'(?i)[A-Z]:[/\\]+Users[/\\]+[^/\\\s"<>]+'),
]


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Use a nonempty repository-relative POSIX path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {'..', '.'} for p in value.split('/')) or path.as_posix() != value:
        raise ValueError('Unsafe repository path: '+value)
    return Path(*path.parts)


def _regular_path(root, path):
    for part in [path, *path.parents]:
        if part == root:
            break
        if part.is_symlink() or (hasattr(part, 'is_junction') and part.is_junction()):
            raise ValueError('Links and junctions are not public release inputs')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Release input escapes repository')


def collect_files(root, policy):
    root = Path(root).resolve()
    declared = policy['files']+policy['directories']
    if len(set(declared)) != len(declared):
        raise ValueError('Repeated release declaration')
    selected = set()
    for name in declared:
        rel = _relative(name);path = root/rel
        _regular_path(root, path)
        if not path.exists():
            raise ValueError('Missing release input: '+name)
        if name in policy['directories']:
            if not path.is_dir():
                raise ValueError('Expected directory: '+name)
            for current, directories, files in os.walk(path, followlinks=False):
                current = Path(current)
                for entry in directories+files:
                    _regular_path(root, current/entry)
                directories[:] = sorted(d for d in directories if d not in SKIP_PARTS)
                for entry in files:
                    candidate = current/entry
                    if candidate.suffix not in {'.pyc', '.pyo'}:
                        selected.add(candidate.relative_to(root))
        else:
            if not path.is_file():
                raise ValueError('Expected file: '+name)
            selected.add(rel)
    return sorted(selected, key=lambda p:p.as_posix())


def audit_files(root, policy):
    root = Path(root).resolve();files = collect_files(root, policy)
    total = 0;selected = {p.as_posix() for p in files}
    for rel in files:
        path = root/rel;size = path.stat().st_size;total += size
        if size > policy['max_file_bytes']:
            raise ValueError('Oversized public file: '+rel.as_posix())
        if rel.suffix == '.png':
            if not path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('Invalid PNG: '+rel.as_posix())
            continue
        if rel.suffix not in TEXT_SUFFIXES|{'.svg'} and rel.name not in SPECIAL_NAMES:
            raise ValueError('Unsupported public file: '+rel.as_posix())
        try:
            value = path.read_text(encoding='utf-8-sig')
        except UnicodeError as exc:
            raise ValueError('Public text is not UTF-8: '+rel.as_posix()) from exc
        if any(pattern.search(value) for pattern in SECRET_PATTERNS):
            raise ValueError('Potential credential or personal user path: '+rel.as_posix())
        if rel.suffix == '.md':
            # Ignore code fences; command examples are not Markdown links.
            rendered = re.sub(r'```.*?```', '', value, flags=re.S)
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', rendered):
                target = target.strip().strip('<>')
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                destination = (path.parent/unquote(parsed.path)).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError('Markdown link escapes repository: '+rel.as_posix())
                name = destination.relative_to(root).as_posix()
                if name not in selected and not any(p.startswith(name+'/') for p in selected):
                    raise ValueError('Unpublished Markdown link in '+rel.as_posix()+': '+target)
    if total > policy['max_total_bytes']:
        raise ValueError('Release exceeds total size budget')
    return {'status':'PASS','file_count':len(files),'total_bytes':total,
            'files':[p.as_posix() for p in files]}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_release(root, policy, output):
    root = Path(root).resolve();output = Path(output).resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError('Export must not contain or replace source repository')
    for directory in policy['directories']:
        source = (root/_relative(directory)).resolve()
        if output == source or output.is_relative_to(source):
            raise ValueError('Export cannot be inside a published source directory')
    archive = Path(str(output)+'.zip')
    if output.exists() or archive.exists():
        raise FileExistsError('Preserve existing export; choose a new directory')
    audit = audit_files(root, policy)
    records = {name:{'sha256':_digest(root/name),'bytes':(root/name).stat().st_size} for name in audit['files']}
    manifest = {'version':policy['version'],'scope':'public source and aggregate evidence; no datasets or model weights',
                'file_count':len(records),'total_bytes':audit['total_bytes'],'files':records}
    output.mkdir(parents=True)
    for name in records:
        destination = output/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name, destination)
    (output/'release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    verify_release(output)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as zipped:
        for name in sorted([*records,'release_manifest.json']):
            info = zipfile.ZipInfo(name, date_time=(2026,10,29,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED;info.external_attr = 0o100644 << 16
            zipped.writestr(info,(output/name).read_bytes())
    verify_archive(output)
    return manifest


def verify_release(output):
    output = Path(output).resolve()
    manifest = json.loads((output/'release_manifest.json').read_text(encoding='utf-8'))
    files = manifest['files']
    actual = set()
    for current, directories, names in os.walk(output,followlinks=False):
        current = Path(current)
        for name in directories+names:
            _regular_path(output,current/name)
        directories[:] = [d for d in directories if d not in SKIP_PARTS]
        actual.update((current/name).relative_to(output).as_posix() for name in names if not name.endswith(('.pyc','.pyo')))
    if actual != set(files)|{'release_manifest.json'}:
        raise ValueError('Release file set changed')
    total = 0
    for name, record in files.items():
        path = output/_relative(name);_regular_path(output,path)
        if _digest(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('Release file changed: '+name)
        total += record['bytes']
    if manifest['file_count'] != len(files) or manifest['total_bytes'] != total:
        raise ValueError('Release summary changed')
    return manifest


def verify_archive(output):
    output = Path(output).resolve();manifest = verify_release(output)
    expected = set(manifest['files'])|{'release_manifest.json'}
    with zipfile.ZipFile(Path(str(output)+'.zip')) as zipped:
        if set(zipped.namelist()) != expected or len(zipped.namelist()) != len(expected):
            raise ValueError('Archive file set changed')
        for name in expected:
            if zipped.read(name) != (output/name).read_bytes():
                raise ValueError('Archive contents changed: '+name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',default='configs/release_v0.1.0.json')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true')
    group.add_argument('--output-directory')
    group.add_argument('--verify-directory')
    args = parser.parse_args()
    if args.verify_directory:
        result = verify_archive(ROOT/args.verify_directory)
    else:
        policy = json.loads((ROOT/args.policy).read_text(encoding='utf-8'))
        if args.output_directory:
            output = (ROOT/args.output_directory).resolve()
            if output.drive.upper() != 'D:' or not output.is_relative_to(ROOT/'.runtime/releases'):
                raise ValueError('Use an explicit D: project .runtime/releases/<new-name> destination')
            result = export_release(ROOT,policy,output)
        else:
            result = audit_files(ROOT,policy)
    print(json.dumps({key:result[key] for key in ['file_count','total_bytes']},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
