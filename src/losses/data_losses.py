"""
Data loss functions — thin wrappers around PyTorch functionals.

Each function takes (pred, target) and returns a scalar loss tensor.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error: ``mean((pred - target)^2)``.

    Args:
        pred: Model predictions, arbitrary shape (e.g. ``[batch, 1]``).
            Typically z-scored next-step log-returns on the normalised scale
            produced by the dataset's scaler.
        target: Ground-truth targets, same shape and scale as ``pred``.

    Returns:
        Scalar non-negative MSE loss on the normalised return scale.
    """
    return F.mse_loss(pred, target)


def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean absolute error (L1): ``mean(|pred - target|)``.

    Args:
        pred: Model predictions, arbitrary shape (e.g. ``[batch, 1]``).
            Typically z-scored next-step log-returns on the normalised scale.
        target: Ground-truth targets, same shape and scale as ``pred``.

    Returns:
        Scalar non-negative MAE loss on the normalised return scale.
    """
    return F.l1_loss(pred, target)


def huber_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Huber (smooth L1) loss — quadratic for small errors, linear for large ones.

    Less sensitive to outlier returns than MSE; uses PyTorch's default
    ``delta=1.0`` threshold (on the normalised scale this corresponds to
    ~1 standard deviation).

    Args:
        pred: Model predictions, arbitrary shape (e.g. ``[batch, 1]``).
            Typically z-scored next-step log-returns on the normalised scale.
        target: Ground-truth targets, same shape and scale as ``pred``.

    Returns:
        Scalar non-negative Huber loss on the normalised return scale.
    """
    return F.smooth_l1_loss(pred, target)
