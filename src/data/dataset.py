"""PyTorch Dataset for sliding-window time series with temporal splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesDataset(Dataset):
    """Sliding-window time-series dataset with physics / scaler metadata.

    Exposes the feature window ``x = features[idx : idx+seq_len]`` plus a
    one-step-shifted window ``x_next = features[idx+1 : idx+seq_len+1]`` so
    models that need ``dV/dt`` (Black–Scholes) can compute a second forward
    pass without owning the raw data. ``x_next`` is pad-last at the boundary
    (the final sample reuses its last row) to keep ``__len__`` model-free.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int,
        metadata: dict | None = None,
    ) -> None:
        """Construct a sliding-window dataset.

        Args:
            features: ``(T, num_features)`` scaled feature matrix (all
                columns are ``StandardScaler``-transformed). Dtype is cast
                to ``float32``.
            targets: ``(T,)`` next-step log-return targets **on the raw
                scale** (no scaler is applied to ``y``). Dtype → ``float32``.
            seq_len: Look-back window length.
            metadata: Optional dict. Recognised keys (all optional; only
                those actually passed get forwarded via ``__getitem__``):

                * ``"prices"`` — ``(T,)`` raw close prices.
                * ``"timestamps"`` — ``(T,)`` datetime-like vector.
                * ``"returns"`` — ``(T,)`` raw log-returns.
                * ``"dt"`` — scalar timestep (default ``1/252``).
                * ``"price_mean"`` / ``"price_std"`` — scalar ``StandardScaler``
                  stats for the close-price feature column; consumed by
                  ``BlackScholesConstraint.residual`` to de-normalise ``S``.
                * ``"target_mean"`` / ``"target_std"`` — scalar stats for
                  ``V``. Targets here are **unscaled** log-returns, so
                  these default to ``0.0`` / ``1.0`` in ``from_dataframe``.
                * ``"price_feature_idx"`` — int column index of the price
                  feature within ``features``.
        """
        assert len(features) == len(targets), "features/targets length mismatch"
        assert len(features) > seq_len, "not enough rows for seq_len"

        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.seq_len = seq_len
        self.metadata = metadata or {}

    def __len__(self) -> int:
        """Return the number of valid sliding-window samples.

        Returns:
            ``T - seq_len`` where ``T`` is the number of time steps, so that
            every window ``[idx, idx + seq_len)`` has a valid target at
            ``features[idx + seq_len]``.
        """
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, dict]:
        """Return ``(x, y, meta)`` for a single sliding-window sample.

        Args:
            idx: sample index in ``[0, __len__())``.

        Returns:
            Tuple of:
                * ``x`` — ``(seq_len, num_features)`` ``float32`` tensor,
                  scaled features.
                * ``y`` — ``(1,)`` ``float32`` tensor of the next-step raw
                  log-return at ``features[idx + seq_len]``.
                * ``meta`` — per-sample dict forwarded to the loss. Always
                  contains ``"dt"``. Optional keys (only present when the
                  parent dataset was built with them):

                    - ``"prices"``: ``(seq_len + 1,)`` raw close prices
                      covering ``[idx, idx + seq_len]`` inclusive.
                    - ``"timestamps"``: matching ``(seq_len + 1,)`` array.
                    - ``"returns"``: ``(seq_len,)`` raw log-returns.
                    - ``"features_next"``: ``(seq_len, num_features)``
                      ``float32`` tensor of the window shifted by +1
                      step (padded with ``features[-1]`` at the boundary).
                      Enables a second forward pass for ``predictions_next``
                      in the Black–Scholes residual.
                    - ``"price_mean"`` / ``"price_std"``: scalar float
                      tensors (scaler stats for the price column).
                    - ``"target_mean"`` / ``"target_std"``: scalar float
                      tensors (scaler stats for the target; identity
                      ``(0, 1)`` when targets are unscaled).
                    - ``"price_feature_idx"``: Python ``int`` column index.
        """
        end = idx + self.seq_len
        x = torch.from_numpy(self.features[idx:end])          # (seq_len, F)
        y = torch.tensor([self.targets[end]], dtype=torch.float32)  # (1,)

        # +1-shifted window for BS dV/dt. Pad the last row when the window
        # would extend past the end of the feature matrix (only the final
        # sample hits this branch; it makes V_next ≈ V_current → dV/dt ≈ 0
        # for that one sample, which is acceptable for a forward-difference
        # finite-difference approximation).
        lo_next = idx + 1
        hi_next = end + 1
        if hi_next <= len(self.features):
            x_next = self.features[lo_next:hi_next]
        else:
            last_row = self.features[-1:]
            x_next = np.concatenate(
                [self.features[lo_next:], last_row], axis=0,
            )
        meta: dict = {"features_next": torch.from_numpy(x_next.astype(np.float32))}

        if "prices" in self.metadata:
            meta["prices"] = torch.from_numpy(
                self.metadata["prices"][idx : end + 1].astype(np.float32)
            )
        if "timestamps" in self.metadata:
            meta["timestamps"] = self.metadata["timestamps"][idx : end + 1]
        if "returns" in self.metadata:
            meta["returns"] = torch.from_numpy(
                self.metadata["returns"][idx:end].astype(np.float32)
            )
        meta["dt"] = self.metadata.get("dt", 1.0 / 252)

        # Scaler stats — constants across samples but emitted per-sample so
        # the existing collate_fn path stays uniform.
        for k in ("price_mean", "price_std", "target_mean", "target_std"):
            if k in self.metadata:
                meta[k] = torch.tensor(
                    float(self.metadata[k]), dtype=torch.float32,
                )
        if "price_feature_idx" in self.metadata:
            meta["price_feature_idx"] = int(self.metadata["price_feature_idx"])

        return x, y, meta

    # ------------------------------------------------------------------
    # Factory: DataFrame -> train / val / test
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        seq_len: int = 60,
        target_col: str = "log_return",
        split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
        feature_cols: list[str] | None = None,
    ) -> tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
        """Create train/val/test datasets from a feature DataFrame.

        Pipeline:
          * Temporal split (no shuffling — prevents look-ahead leakage).
          * ``StandardScaler`` fit on **train only**; val/test transform with
            the same scaler.
          * Targets (``target_col``) are kept **raw** (no scaler applied),
            so ``target_mean = 0.0`` and ``target_std = 1.0`` are emitted as
            identity stats in the metadata for the Black–Scholes residual's
            de-normalisation of ``V``.
          * The close-price feature's column index and its scaler stats
            (``scaler.mean_[idx]``, ``scaler.scale_[idx]``) are attached to
            every sample's metadata so ``BlackScholesConstraint.residual``
            can recover the raw-scale spot price ``S``.

        Args:
            df: source DataFrame with feature columns and a ``target_col``.
            seq_len: sliding-window length.
            target_col: column name for ``y`` (default ``"log_return"``).
            split_ratios: ``(train, val, test)`` fractions; must sum to 1.
            feature_cols: explicit feature-column subset. When ``None``,
                everything except ``{Open, High, Low, Close, Volume, Date}``
                is used.

        Returns:
            ``(train_ds, val_ds, test_ds)`` — each a ``TimeSeriesDataset``
            whose ``metadata`` carries the scaler stats above in addition
            to the usual ``prices``/``timestamps``/``returns``/``dt`` keys.
        """
        if feature_cols is None:
            exclude = {"Open", "High", "Low", "Close", "Volume", "Date"}
            feature_cols = [c for c in df.columns if c not in exclude]

        target = df[target_col].values
        feat = df[feature_cols].values
        prices = df["Close"].values if "Close" in df.columns else None
        returns = df["log_return"].values if "log_return" in df.columns else None

        # Locate the column that plays the role of "price" for the
        # Black–Scholes residual's `S = S_norm * price_std + price_mean`
        # inversion. Preference order:
        #   1. `Close`           — raw close (exact inversion; rarely
        #                          present since it's excluded by default).
        #   2. `log_close`       — log-price (ok, inversion recovers ln S).
        #   3. `close_normalized`— 20-day z-score of close from
        #                          `compute_features`; the inversion yields
        #                          the 20-day z-score, not raw S. BS then
        #                          acts as an approximate prior on the
        #                          normalised price dynamics rather than
        #                          the true underlying. This is the typical
        #                          path in this project and is a documented
        #                          approximation (see Documentation/Plans/
        #                          2026-04-18-phase-4-training-layer.md).
        #   4. column 0          — last-resort fallback to keep bs_pinn
        #                          runnable on unfamiliar feature sets.
        for candidate in ("Close", "log_close", "close_normalized"):
            if candidate in feature_cols:
                price_feature_idx = feature_cols.index(candidate)
                break
        else:
            price_feature_idx = 0

        # Timestamps
        if isinstance(df.index, pd.DatetimeIndex):
            timestamps = df.index.values
        elif "Date" in df.columns:
            timestamps = pd.to_datetime(df["Date"]).values
        else:
            timestamps = np.arange(len(df))

        # Temporal split boundaries
        n = len(df)
        t1 = int(n * split_ratios[0])
        t2 = int(n * (split_ratios[0] + split_ratios[1]))

        # Fit scaler on TRAIN only
        scaler = StandardScaler()
        scaler.fit(feat[:t1])
        feat_scaled = scaler.transform(feat)
        price_mean = float(scaler.mean_[price_feature_idx])
        price_std = float(scaler.scale_[price_feature_idx])

        def _build(lo: int, hi: int) -> TimeSeriesDataset:
            meta: dict = {
                "dt": 1.0 / 252,
                # Scaler stats for Black–Scholes de-normalisation.
                "price_mean": price_mean,
                "price_std": price_std,
                "target_mean": 0.0,   # targets are raw log-returns
                "target_std": 1.0,
                "price_feature_idx": price_feature_idx,
            }
            if prices is not None:
                meta["prices"] = prices[lo:hi]
            if timestamps is not None:
                meta["timestamps"] = timestamps[lo:hi]
            if returns is not None:
                meta["returns"] = returns[lo:hi]
            return cls(
                features=feat_scaled[lo:hi],
                targets=target[lo:hi],
                seq_len=seq_len,
                metadata=meta,
            )

        train_ds = _build(0, t1)
        val_ds = _build(t1, t2)
        # When split_ratios[2] == 0 the test slice is empty (t2 == n).
        # Fall back to the val slice so evaluate_on_test() has valid data;
        # reported test metrics then equal val metrics, which is acceptable
        # for λ-sweep runs where selection is based on best_val_loss only.
        test_ds = _build(t2, n) if t2 < n else val_ds

        logger.info(
            "Splits -> train={} val={} test={} (seq_len={}, features={})",
            len(train_ds), len(val_ds), len(test_ds),
            seq_len, len(feature_cols),
        )
        return train_ds, val_ds, test_ds


# ------------------------------------------------------------------
# Collate
# ------------------------------------------------------------------


def collate_fn(
    batch: list[tuple[Tensor, Tensor, dict]],
) -> tuple[Tensor, Tensor, dict]:
    """Stack samples and batch the per-sample metadata dicts.

    Inputs:
        batch: list of ``(x, y, meta)`` triples from
            :meth:`TimeSeriesDataset.__getitem__`. All ``meta`` dicts must
            share the same set of keys (guaranteed when samples come from
            the same dataset).

    Returns:
        ``(x_batch, y_batch, meta_batch)`` where

        * ``x_batch``: ``[B, seq_len, num_features]`` ``float32``.
        * ``y_batch``: ``[B, 1]`` ``float32`` raw log-returns.
        * ``meta_batch``: dict carrying the same keys as ``meta``,
          tensor-stacked along a new batch dim where sensible:

            - ``"prices"``: ``[B, seq_len + 1]``.
            - ``"returns"``: ``[B, seq_len]``.
            - ``"timestamps"``: Python list length ``B`` (not stacked —
              datetime64 tensors aren't first-class in PyTorch).
            - ``"features_next"``: ``[B, seq_len, num_features]``.
            - ``"price_mean" / "price_std" / "target_mean" / "target_std"``:
              ``[B, 1]`` — scaler stats broadcast across the batch dim so
              ``BlackScholesConstraint.residual`` can combine them with
              ``[B, 1]`` prediction / price tensors without manual
              broadcasting.
            - ``"price_feature_idx"``: Python ``int`` (scalar, shared
              across the batch).
            - ``"dt"``: scalar float (shared across the batch).
    """
    xs, ys, metas = zip(*batch)
    x_batch = torch.stack(xs)
    y_batch = torch.stack(ys)

    meta_batch: dict = {}
    sample = metas[0]

    if "prices" in sample:
        meta_batch["prices"] = torch.stack([m["prices"] for m in metas])
    if "returns" in sample:
        meta_batch["returns"] = torch.stack([m["returns"] for m in metas])
    if "timestamps" in sample:
        meta_batch["timestamps"] = [m["timestamps"] for m in metas]
    if "features_next" in sample:
        meta_batch["features_next"] = torch.stack(
            [m["features_next"] for m in metas]
        )

    # Scaler stats → `[B, 1]` so they broadcast with the `[B, 1]` prediction
    # and spot tensors inside the Black–Scholes residual.
    for k in ("price_mean", "price_std", "target_mean", "target_std"):
        if k in sample:
            meta_batch[k] = torch.stack([m[k] for m in metas]).view(-1, 1)
    if "price_feature_idx" in sample:
        # Constant across a batch; keep as a Python int.
        meta_batch["price_feature_idx"] = int(sample["price_feature_idx"])

    meta_batch["dt"] = sample.get("dt", 1.0 / 252)

    return x_batch, y_batch, meta_batch
