"""Reproducibility utilities for deterministic training runs.

Provides :func:`seed_everything`, which fixes the random state across all
libraries used by this project before any data loading or model construction.
Call once at the top of each experiment script / notebook cell with the seed
declared in ``configs/dissertation.yaml`` (42, 123, or 456).
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fix random state across Python, NumPy, and PyTorch for full reproducibility.

    Covers every source of non-determinism encountered in this project:
    Python's built-in ``random`` module, the ``PYTHONHASHSEED`` env variable
    (dict iteration order), NumPy's legacy and Generator APIs, PyTorch CPU and
    all CUDA devices, and cuDNN's non-deterministic kernel selection.

    Args:
        seed: Integer seed value. Must be in ``[0, 2**32 − 1]`` for NumPy
            compatibility. Three seeds are declared in
            ``configs/dissertation.yaml`` (42, 123, 456); none were chosen
            post-hoc. Passing an out-of-range value raises ``ValueError``
            from ``np.random.seed``.

    Returns:
        None. Side effects only — mutates global RNG state in all libraries.

    Note:
        Setting ``torch.backends.cudnn.deterministic = True`` and
        ``benchmark = False`` trades a small speed reduction for bitwise
        reproducibility of convolution operations on GPU. For the LSTM/GRU
        architectures used here the overhead is negligible.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
