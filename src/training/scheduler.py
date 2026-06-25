"""Physics lambda warmup scheduler — curriculum learning for PINN constraints."""
from __future__ import annotations

import math


class PhysicsScheduler:
    """Gradually scale physics loss weight from 0 to 1 over warmup_epochs.

    Strategies:
        linear:  scale = epoch / warmup_epochs
        cosine:  scale = 0.5 * (1 - cos(pi * epoch / warmup_epochs))
        step:    scale = 0 if epoch < warmup_epochs else 1
    """

    VALID_STRATEGIES = ("linear", "cosine", "step")

    def __init__(self, warmup_epochs: int = 20, strategy: str = "linear") -> None:
        """Initialise the warmup scheduler.

        Args:
            warmup_epochs: Number of epochs over which the physics scale
                ramps from 0.0 to 1.0. A value of 0 disables warmup
                (scale is always 1.0). Must be a non-negative integer.
            strategy: Warmup curve shape. One of ``"linear"``, ``"cosine"``,
                or ``"step"``. Raises ``ValueError`` for unknown values.

        Raises:
            ValueError: If *strategy* is not in ``VALID_STRATEGIES``.
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown warmup strategy '{strategy}'. "
                f"Choose from {self.VALID_STRATEGIES}"
            )
        self.warmup_epochs = warmup_epochs
        self.strategy = strategy

    def get_scale(self, epoch: int) -> float:
        """Return the physics lambda multiplier for the given epoch.

        Args:
            epoch: Zero-based epoch index (``0`` = first epoch).

        Returns:
            float: Warmup multiplier in ``[0.0, 1.0]``. Returns ``1.0``
            for all epochs >= ``warmup_epochs``.
        """
        if self.warmup_epochs <= 0:
            return 1.0
        if epoch >= self.warmup_epochs:
            return 1.0

        t = epoch / self.warmup_epochs  # progress in [0, 1)

        if self.strategy == "linear":
            return t
        if self.strategy == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * t))
        # step
        return 0.0
