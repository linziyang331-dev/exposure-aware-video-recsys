"""Apply four audited pandas-3 fixes ONLY to the D: project RecBole copy."""
import difflib
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    distribution = metadata.distribution('recbole')
    if distribution.version != '1.2.1':
        raise RuntimeError('Patch is reviewed only for RecBole 1.2.1')
    target = (ROOT / '.runtime/python312-packages').resolve()
    source = Path(distribution.locate_file('recbole/data/dataset/dataset.py')).resolve()
    if not source.is_relative_to(target):
        raise RuntimeError('Refusing to patch RecBole outside the project dependency directory')
    original_bytes = source.read_bytes()
    original = original_bytes.decode('utf-8')
    replacements = [
        ('            df[field].fillna(value="", inplace=True)',
         '            df[field] = df[field].fillna(value="")'),
        ('                    feat[field].fillna(value=0, inplace=True)',
         '                    feat[field] = feat[field].fillna(value=0)'),
        ('                    feat[field].fillna(value=feat[field].mean(), inplace=True)',
         '                    feat[field] = feat[field].fillna(value=feat[field].mean())'),
        ('            value = data[k].values',
         '            value = data[k].to_numpy(copy=True)'),
    ]
    updated = original
    for old, new in replacements:
        if old in updated:
            if updated.count(old) != 1:
                raise RuntimeError(f'Unexpected upstream source: multiple matches for {old.strip()}')
            updated = updated.replace(old, new, 1)
        elif new not in updated:
            raise RuntimeError(f'Unexpected upstream source: missing patch anchor {old.strip()}')
    if updated == original:
        print('RecBole pandas-3 patch already applied; no file changes')
        return
    compile(updated, str(source), 'exec')
    patch = ''.join(difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True),
                                        fromfile='a/recbole/data/dataset/dataset.py',
                                        tofile='b/recbole/data/dataset/dataset.py'))
    patch_file = ROOT / 'patches/recbole-1.2.1-pandas3.patch'
    patch_file.parent.mkdir(parents=True, exist_ok=True)
    patch_file.write_text(patch, encoding='utf-8', newline='\n')
    temporary = source.with_suffix('.py.project-patch-tmp')
    temporary.write_text(updated, encoding='utf-8', newline='\n')
    temporary.replace(source)
    report = {'status': 'APPLIED', 'recbole': distribution.version,
              'file': str(source), 'patch': str(patch_file.relative_to(ROOT)),
              'before_sha256': hashlib.sha256(original_bytes).hexdigest(),
              'after_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'changes': 4}
    (ROOT / 'reports/environment_logs/recbole_pandas_patch.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
