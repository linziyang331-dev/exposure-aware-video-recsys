"""Verify project additions, reused packages, and unchanged global environment."""
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / '.runtime' / 'python312-packages'
    baseline = json.loads((ROOT / 'reports/environment_logs/global_packages_before.json').read_text(encoding='utf-8'))
    global_now = {d.metadata['Name']: d.version for d in metadata.distributions(path=[baseline['global_site_packages']])}
    errors, overrides = [], []
    allowed = {'ray': '2.40.0', 'colorama': '0.4.6'}
    added = list(metadata.distributions(path=[str(target)]))
    for distribution in added:
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
                continue
            try:
                version = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                errors.append(f'{distribution.metadata["Name"]}: missing {requirement}')
                continue
            if not requirement.specifier.contains(version, prereleases=True):
                issue = f'{distribution.metadata["Name"]} requires {requirement}; actual {version}'
                if canonicalize_name(distribution.metadata['Name']) == 'recbole' and allowed.get(canonicalize_name(requirement.name)) == version:
                    overrides.append(issue)
                else:
                    errors.append(issue)
    import torch
    import recbole
    cuda_ok = torch.cuda.is_available()
    backward_ok = False
    if cuda_ok:
        tensor = torch.tensor([1., 2.], device='cuda', requires_grad=True)
        tensor.square().sum().backward()
        backward_ok = torch.equal(tensor.grad, torch.tensor([2., 4.], device='cuda'))
    checks = {
        'python_312': sys.version_info[:2] == (3, 12),
        'old_venv_removed': not (ROOT / '.venv').exists(),
        'global_packages_unchanged': global_now == baseline['packages'],
        'recbole_on_D_project': Path(recbole.__file__).resolve().is_relative_to(target),
        'torch_reused_from_global_environment': Path(torch.__file__).resolve().is_relative_to(Path(baseline['global_site_packages']).resolve()),
        'cuda_available': cuda_ok,
        'cuda_backward_correct': backward_ok,
        'dependency_closure_valid_except_recorded_overrides': not errors,
        'cache_and_temp_on_D': all(Path(os.environ[k]).drive.upper() == 'D:' for k in ['TEMP', 'TMP', 'PIP_CACHE_DIR', 'TORCH_HOME']),
    }
    pip_check = subprocess.run([sys.executable, '-B', '-m', 'pip', 'check'], capture_output=True, text=True, encoding='utf-8')
    (ROOT / 'reports/environment_logs/pip_check_after.txt').write_text(pip_check.stdout + pip_check.stderr, encoding='utf-8')
    result = {
        'timestamp': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks,
        'executable': sys.executable,
        'python': sys.version,
        'torch': torch.__version__,
        'torch_origin': torch.__file__,
        'recbole_origin': recbole.__file__,
        'gpu': torch.cuda.get_device_name(0) if cuda_ok else None,
        'added_packages': {d.metadata['Name']: d.version for d in added},
        'dependency_overrides': overrides,
        'unexpected_dependency_errors': errors,
        'pip_check_exit': pip_check.returncode,
        'pip_check_note': 'Not expected to be globally clean: baseline d2l conflicts and the explicit RecBole overrides are recorded.',
    }
    (ROOT / 'reports/environment_python312_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
