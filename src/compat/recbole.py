"""Narrow compatibility boundary for upstream RecBole 1.2.1.

NumPy aliases are added only to the running process, before creating a Config.
The separate installation-time pandas fix is recorded in patches/; no global
package versions or files are changed by this module.
"""
from pathlib import Path

import numpy as np
import torch


def apply_compatibility():
    """Restore three dtype names removed in NumPy 2 for RecBole initialization."""
    aliases = {'float_': np.float64, 'complex_': np.complex128, 'unicode_': np.str_}
    for name, dtype in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, dtype)


def load_run_checkpoint(path, run_directory, *, device='cpu'):
    """Load a checkpoint produced by our run, including its RecBole Config.

    PyTorch 2.6 defaults to weights_only=True; RecBole stores a Config object.
    This entry point is only for our own generated files under the run directory,
    never for third-party checkpoints or untrusted user input.
    """
    path = Path(path).resolve(strict=True)
    root = Path(run_directory).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Checkpoint must be a file inside the run directory')
    return torch.load(path, map_location=device, weights_only=False)
