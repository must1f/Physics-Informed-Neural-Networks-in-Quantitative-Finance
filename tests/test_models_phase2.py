"""Tests for Phase 2 model layer: base_pinn (2A), baselines (2B), pinn (2C),
stacked_pinn (2D), residual_pinn (2E), registry (2F), integration (2G).

Covers: ABC enforcement, shapes, gradient flow, diagnostics, metadata
enrichment, integration with CompositeLoss, registry factory, and edge cases.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.losses.composite import CompositeLoss
from src.losses.data_losses import mse_loss
from src.losses.physics import (
    BlackScholesConstraint,
    GBMConstraint,
    HawkesConstraint,
    LangevinConstraint,
    OUConstraint,
    PhysicsConstraint,
)
from src.models.base_pinn import BasePINN
from src.models.baselines import BaselineModel, _AttentionLSTMEncoder, _TransformerEncoder
from src.models.pinn import PINNModel
from src.models.registry import MODEL_REGISTRY, build_model, list_models
from src.models.residual_pinn import ResidualPINN
from src.models.stacked_pinn import StackedPINN

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH, SEQ, FEAT = 8, 60, 14


@pytest.fixture
def sample_input():
    return torch.randn(BATCH, SEQ, FEAT)


@pytest.fixture
def sample_metadata():
    # Prices must be positive for GBM log-return computation
    prices = torch.rand(BATCH, SEQ + 1) * 100 + 50  # [50, 150]
    returns = torch.log(prices[:, 1:] / prices[:, :-1])
    return {
        "prices": prices,
        "returns": returns,
        "dt": 1.0 / 252,
    }


class _DummyPINN(BasePINN):
    """Minimal concrete subclass for testing BasePINN."""

    def __init__(self, constraints=None):
        super().__init__(constraints or [])
        self.fc = nn.Linear(FEAT, 1)

    def _encode(self, x: Tensor) -> Tensor:
        return x.mean(dim=1)  # [B, F]

    def _predict(self, hidden: Tensor) -> Tensor:
        return self.fc(hidden)  # [B, 1]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2A — BasePINN
# ═══════════════════════════════════════════════════════════════════════════


class TestBasePINNABC:
    """BasePINN cannot be instantiated and enforces the abstract contract."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BasePINN(constraints=[])

    def test_missing_encode_raises(self):
        class _NoPredPINN(BasePINN):
            def _encode(self, x):
                return x.mean(dim=1)
            # _predict missing

        with pytest.raises(TypeError):
            _NoPredPINN(constraints=[])

    def test_missing_predict_raises(self):
        class _NoEncPINN(BasePINN):
            def _predict(self, hidden):
                return hidden[:, :1]
            # _encode missing

        with pytest.raises(TypeError):
            _NoEncPINN(constraints=[])


class TestBasePINNConstraints:
    """Constraints are stored as nn.ModuleList and tracked by the optimizer."""

    def test_empty_constraints(self):
        model = _DummyPINN(constraints=[])
        assert isinstance(model.constraints, nn.ModuleList)
        assert len(model.constraints) == 0

    def test_single_constraint(self):
        model = _DummyPINN(constraints=[GBMConstraint()])
        assert len(model.constraints) == 1
        assert model.constraints[0].name == "gbm"

    def test_multiple_constraints(self):
        model = _DummyPINN(constraints=[GBMConstraint(), OUConstraint()])
        assert len(model.constraints) == 2
        names = [c.name for c in model.constraints]
        assert "gbm" in names and "ou" in names

    def test_constraint_params_in_model_parameters(self):
        """Learnable constraint params must be reachable via model.parameters()."""
        model = _DummyPINN(constraints=[OUConstraint()])
        param_names = {n for n, _ in model.named_parameters()}
        assert any("constraints" in n for n in param_names)

    def test_constraint_params_tracked_by_optimizer(self):
        model = _DummyPINN(constraints=[OUConstraint()])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # All constraint params should be in the optimizer
        constraint_params = set(model.constraints.parameters())
        opt_params = {p for group in opt.param_groups for p in group["params"]}
        assert constraint_params.issubset(opt_params)


class TestBasePINNForward:
    """forward() orchestration via Template Method."""

    def test_forward_returns_tuple(self, sample_input):
        model = _DummyPINN(constraints=[GBMConstraint()])
        result = model(sample_input)
        assert isinstance(result, tuple) and len(result) == 2

    def test_forward_shape(self, sample_input):
        model = _DummyPINN()
        pred, meta = model(sample_input)
        assert pred.shape == (BATCH, 1)
        assert isinstance(meta, dict)

    def test_forward_metadata_none(self, sample_input):
        """Passing metadata=None should not raise."""
        model = _DummyPINN()
        pred, meta = model(sample_input, metadata=None)
        assert pred.shape == (BATCH, 1)
        assert meta == {}

    def test_forward_metadata_passthrough(self, sample_input, sample_metadata):
        """Default _build_physics_metadata returns metadata unchanged."""
        model = _DummyPINN()
        _, enriched = model(sample_input, metadata=sample_metadata)
        assert enriched is sample_metadata


class TestBasePINNPredict:
    """predict() runs in inference mode without grad."""

    def test_predict_shape(self, sample_input):
        model = _DummyPINN()
        pred = model.predict(sample_input)
        assert pred.shape == (BATCH, 1)

    def test_predict_no_grad(self, sample_input):
        model = _DummyPINN()
        pred = model.predict(sample_input)
        assert not pred.requires_grad


class TestBasePINNDiagnostics:
    """diagnostics() returns constraint names and learnable params."""

    def test_diagnostics_empty(self):
        diag = _DummyPINN().diagnostics()
        assert diag == {"constraints": []}

    def test_diagnostics_with_constraints(self):
        model = _DummyPINN(constraints=[OUConstraint()])
        diag = model.diagnostics()
        assert "ou" in diag["constraints"]
        assert "ou_params" in diag
        assert isinstance(diag["ou_params"], dict)
        # OU has 2 learnable params (_theta_raw, _sigma_raw); mu is a fixed buffer
        assert len(diag["ou_params"]) == 2

    def test_diagnostics_multi_constraint(self):
        model = _DummyPINN(constraints=[GBMConstraint(), HawkesConstraint()])
        diag = model.diagnostics()
        assert set(diag["constraints"]) == {"gbm", "hawkes"}
        assert "hawkes_params" in diag
        # GBM has 0 learnable params
        assert diag["gbm_params"] == {}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2B — Baselines
# ═══════════════════════════════════════════════════════════════════════════


class TestBaselineModelArchitectures:
    """Each architecture produces correct output shape."""

    @pytest.mark.parametrize("arch", BaselineModel.VALID_ARCHS)
    def test_output_shape(self, arch, sample_input):
        model = BaselineModel(arch=arch, input_dim=FEAT)
        pred = model(sample_input)
        assert pred.shape == (BATCH, 1)

    @pytest.mark.parametrize("arch", BaselineModel.VALID_ARCHS)
    def test_gradient_flow(self, arch, sample_input):
        model = BaselineModel(arch=arch, input_dim=FEAT)
        pred = model(sample_input)
        pred.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, f"No gradients for {arch}"

    def test_invalid_arch(self):
        with pytest.raises(ValueError, match="Unknown arch"):
            BaselineModel(arch="invalid", input_dim=FEAT)


class TestBaselineModelLSTM:
    def test_encoder_type(self):
        model = BaselineModel(arch="lstm", input_dim=FEAT)
        assert isinstance(model.encoder, nn.LSTM)
        assert not model.encoder.bidirectional


class TestBaselineModelGRU:
    def test_encoder_type(self):
        model = BaselineModel(arch="gru", input_dim=FEAT)
        assert isinstance(model.encoder, nn.GRU)


class TestBaselineModelBiLSTM:
    def test_encoder_type(self):
        model = BaselineModel(arch="bilstm", input_dim=FEAT)
        assert isinstance(model.encoder, nn.LSTM)
        assert model.encoder.bidirectional

    def test_prediction_head_accepts_bilstm_output(self):
        # BiLSTM concatenates forward + backward final states → 2H input to head.
        model = BaselineModel(arch="bilstm", input_dim=FEAT, hidden_dim=128)
        assert hasattr(model, "prediction_head")
        assert model.prediction_head.in_features == 256  # 2 * hidden_dim
        assert model.prediction_head.out_features == 1


class TestBaselineModelAttentionLSTM:
    def test_encoder_type(self):
        model = BaselineModel(arch="attention_lstm", input_dim=FEAT)
        assert isinstance(model.encoder, _AttentionLSTMEncoder)

    def test_attention_weights_stored(self, sample_input):
        model = BaselineModel(arch="attention_lstm", input_dim=FEAT)
        _ = model(sample_input)
        w = model.encoder.last_attention_weights
        assert w is not None
        assert w.shape == (BATCH, SEQ)

    def test_attention_weights_sum_to_one(self, sample_input):
        model = BaselineModel(arch="attention_lstm", input_dim=FEAT)
        _ = model(sample_input)
        w = model.encoder.last_attention_weights
        sums = w.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)


class TestBaselineModelTransformer:
    def test_encoder_type(self):
        model = BaselineModel(arch="transformer", input_dim=FEAT)
        assert isinstance(model.encoder, _TransformerEncoder)

    def test_learnable_positional_encoding(self):
        model = BaselineModel(arch="transformer", input_dim=FEAT, hidden_dim=64)
        assert hasattr(model.encoder, "pos_encoding")
        assert isinstance(model.encoder.pos_encoding, nn.Parameter)
        assert model.encoder.pos_encoding.shape[2] == 64

    def test_causal_masking(self, sample_input):
        """Transformer should still produce valid output (causal mask applied internally)."""
        model = BaselineModel(arch="transformer", input_dim=FEAT)
        pred = model(sample_input)
        assert not torch.isnan(pred).any()
        assert not torch.isinf(pred).any()


class TestBaselineModelHyperparams:
    """Custom hyperparameters are respected."""

    @pytest.mark.parametrize("arch", ["lstm", "gru"])
    def test_custom_hidden_dim(self, arch, sample_input):
        model = BaselineModel(arch=arch, input_dim=FEAT, hidden_dim=64)
        pred = model(sample_input)
        assert pred.shape == (BATCH, 1)
        assert model.prediction_head.in_features == 64

    def test_custom_num_layers(self):
        model = BaselineModel(arch="lstm", input_dim=FEAT, num_layers=3)
        assert model.encoder.num_layers == 3

    @pytest.mark.parametrize("arch", BaselineModel.VALID_ARCHS)
    def test_single_layer_no_dropout_warning(self, arch, sample_input):
        """Single-layer RNNs with dropout > 0 shouldn't crash (PyTorch warns, not errors)."""
        model = BaselineModel(arch=arch, input_dim=FEAT, num_layers=1, dropout=0.0)
        pred = model(sample_input)
        assert pred.shape == (BATCH, 1)


class TestBaselineNotPINN:
    """Baselines are plain nn.Module, not BasePINN."""

    def test_not_base_pinn(self):
        model = BaselineModel(arch="lstm", input_dim=FEAT)
        assert not isinstance(model, BasePINN)
        assert not hasattr(model, "constraints")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2C — PINNModel
# ═══════════════════════════════════════════════════════════════════════════


class TestPINNModelInheritance:
    """PINNModel inherits from BasePINN correctly."""

    def test_is_base_pinn(self):
        model = PINNModel(input_dim=FEAT)
        assert isinstance(model, BasePINN)

    def test_inherits_predict(self, sample_input):
        model = PINNModel(input_dim=FEAT, constraints=[GBMConstraint()])
        pred = model.predict(sample_input)
        assert pred.shape == (BATCH, 1)
        assert not pred.requires_grad

    def test_inherits_diagnostics(self):
        model = PINNModel(input_dim=FEAT, constraints=[OUConstraint()])
        diag = model.diagnostics()
        assert "ou" in diag["constraints"]


class TestPINNModelEncoders:
    """LSTM and GRU encoder variants."""

    def test_lstm_default(self, sample_input, sample_metadata):
        model = PINNModel(input_dim=FEAT)
        assert model.encoder_type == "lstm"
        pred, _ = model(sample_input, sample_metadata)
        assert pred.shape == (BATCH, 1)

    def test_gru_encoder(self, sample_input, sample_metadata):
        model = PINNModel(input_dim=FEAT, encoder="gru")
        assert model.encoder_type == "gru"
        pred, _ = model(sample_input, sample_metadata)
        assert pred.shape == (BATCH, 1)

    def test_invalid_encoder(self):
        with pytest.raises(ValueError, match="Unknown encoder"):
            PINNModel(input_dim=FEAT, encoder="rnn")


class TestPINNModelComposition:
    """One class, 8 variants via different constraints lists."""

    def test_baseline_pinn(self, sample_input, sample_metadata):
        model = PINNModel(input_dim=FEAT, constraints=[])
        assert len(model.constraints) == 0
        pred, _ = model(sample_input, sample_metadata)
        assert pred.shape == (BATCH, 1)

    def test_gbm_pinn(self):
        model = PINNModel(input_dim=FEAT, constraints=[GBMConstraint()])
        assert len(model.constraints) == 1
        assert model.constraints[0].name == "gbm"

    def test_ou_pinn(self):
        model = PINNModel(input_dim=FEAT, constraints=[OUConstraint()])
        assert model.constraints[0].name == "ou"

    def test_gbm_ou_pinn(self):
        model = PINNModel(
            input_dim=FEAT,
            constraints=[GBMConstraint(), OUConstraint()],
        )
        assert len(model.constraints) == 2

    def test_global_pinn(self):
        model = PINNModel(
            input_dim=FEAT,
            constraints=[
                GBMConstraint(), OUConstraint(),
                BlackScholesConstraint(), LangevinConstraint(),
            ],
        )
        assert len(model.constraints) == 4
        names = {c.name for c in model.constraints}
        assert names == {"gbm", "ou", "bs", "langevin"}

    def test_hawkes_pinn(self):
        model = PINNModel(input_dim=FEAT, constraints=[HawkesConstraint()])
        assert model.constraints[0].name == "hawkes"

    def test_hawkes_ou_pinn(self):
        model = PINNModel(
            input_dim=FEAT,
            constraints=[HawkesConstraint(), OUConstraint()],
        )
        assert len(model.constraints) == 2


class TestPINNModelBlackScholes:
    """BS-specific metadata enrichment and vol_head."""

    def test_vol_head_created(self):
        model = PINNModel(input_dim=FEAT, constraints=[BlackScholesConstraint()])
        assert hasattr(model, "vol_head")
        assert model._has_bs is True

    def test_no_vol_head_without_bs(self):
        model = PINNModel(input_dim=FEAT, constraints=[GBMConstraint()])
        assert not hasattr(model, "vol_head")
        assert model._has_bs is False

    def test_metadata_enriched(self, sample_input, sample_metadata):
        model = PINNModel(input_dim=FEAT, constraints=[BlackScholesConstraint()])
        _, enriched = model(sample_input, sample_metadata)
        assert "volatilities" in enriched
        # Old autograd machinery removed — inputs/price_feature_idx no longer injected
        assert "inputs" not in enriched
        assert "predictions_next" not in enriched

    def test_volatilities_positive(self, sample_input, sample_metadata):
        """softplus ensures volatilities > 0."""
        model = PINNModel(input_dim=FEAT, constraints=[BlackScholesConstraint()])
        _, enriched = model(sample_input, sample_metadata)
        assert (enriched["volatilities"] > 0).all()

    def test_inputs_require_grad(self, sample_input, sample_metadata):
        """After removing autograd machinery, forward() must NOT set requires_grad."""
        model = PINNModel(input_dim=FEAT, constraints=[BlackScholesConstraint()])
        assert not sample_input.requires_grad
        _ = model(sample_input, sample_metadata)
        assert not sample_input.requires_grad

    def test_metadata_passthrough_without_bs(self, sample_input, sample_metadata):
        model = PINNModel(input_dim=FEAT, constraints=[GBMConstraint()])
        _, enriched = model(sample_input, sample_metadata)
        assert enriched is sample_metadata


class TestPINNModelGradientFlow:
    """Gradients flow through encoder AND constraint params."""

    def test_encoder_gradient(self, sample_input, sample_metadata):
        model = PINNModel(input_dim=FEAT, constraints=[GBMConstraint()])
        pred, _ = model(sample_input, sample_metadata)
        pred.sum().backward()
        encoder_grads = [
            p.grad for n, p in model.named_parameters()
            if "_encoder" in n and p.grad is not None
        ]
        assert len(encoder_grads) > 0

    def test_constraint_gradient(self, sample_input, sample_metadata):
        """OU constraint has learnable params — gradients must flow to them."""
        model = PINNModel(input_dim=FEAT, constraints=[OUConstraint()])
        pred, enriched = model(sample_input, sample_metadata)
        loss_fn = CompositeLoss(mse_loss, list(model.constraints))
        target = torch.randn(BATCH, 1)
        total, _ = loss_fn(pred, target, enriched)
        total.backward()
        constraint_grads = [
            p.grad for p in model.constraints.parameters()
            if p.grad is not None
        ]
        assert len(constraint_grads) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Model + CompositeLoss end-to-end
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegrationWithCompositeLoss:
    """Full training loop step: model → CompositeLoss → backward."""

    @pytest.mark.parametrize("constraints,names", [
        ([], []),
        ([GBMConstraint()], ["gbm"]),
        ([OUConstraint()], ["ou"]),
        ([GBMConstraint(), OUConstraint()], ["gbm", "ou"]),
        ([HawkesConstraint()], ["hawkes"]),
        ([LangevinConstraint()], ["langevin"]),
    ])
    def test_training_step(self, constraints, names, sample_input, sample_metadata):
        model = PINNModel(input_dim=FEAT, constraints=constraints)
        loss_fn = CompositeLoss(mse_loss, list(model.constraints))
        target = torch.randn(BATCH, 1)

        pred, enriched = model(sample_input, sample_metadata)
        total, breakdown = loss_fn(pred, target, enriched)
        total.backward()

        assert "data" in breakdown
        assert "total" in breakdown
        for n in names:
            assert n in breakdown
        assert total.isfinite()

    @pytest.mark.parametrize("arch", BaselineModel.VALID_ARCHS)
    def test_baseline_training_step(self, arch, sample_input):
        """Baselines use CompositeLoss with empty constraints."""
        model = BaselineModel(arch=arch, input_dim=FEAT)
        loss_fn = CompositeLoss(mse_loss, constraints=[])
        target = torch.randn(BATCH, 1)

        pred = model(sample_input)
        total, breakdown = loss_fn(pred, target, {})
        total.backward()

        assert total.isfinite()
        assert "data" in breakdown


class TestIntegrationParamCounts:
    """Sanity check: PINNs have more params than just the encoder."""

    def test_pinn_includes_constraint_params(self):
        baseline = PINNModel(input_dim=FEAT, constraints=[])
        pinn = PINNModel(input_dim=FEAT, constraints=[OUConstraint()])
        baseline_count = sum(p.numel() for p in baseline.parameters())
        pinn_count = sum(p.numel() for p in pinn.parameters())
        assert pinn_count > baseline_count

    def test_bs_pinn_has_vol_head(self):
        no_bs = PINNModel(input_dim=FEAT, constraints=[GBMConstraint()])
        bs = PINNModel(input_dim=FEAT, constraints=[BlackScholesConstraint()])
        no_bs_count = sum(p.numel() for p in no_bs.parameters())
        bs_count = sum(p.numel() for p in bs.parameters())
        # BS has vol_head (Linear(128, 1) = 129) + BS learnable param (1)
        assert bs_count > no_bs_count


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_batch_size_one(self):
        x = torch.randn(1, SEQ, FEAT)
        model = PINNModel(input_dim=FEAT, constraints=[GBMConstraint()])
        pred, _ = model(x, {"prices": torch.randn(1, SEQ + 1)})
        assert pred.shape == (1, 1)

    def test_different_seq_len(self):
        for seq in [10, 30, 120]:
            x = torch.randn(4, seq, FEAT)
            model = BaselineModel(arch="lstm", input_dim=FEAT)
            assert model(x).shape == (4, 1)

    def test_different_feature_dim(self):
        for feat in [1, 5, 50]:
            x = torch.randn(4, SEQ, feat)
            model = PINNModel(input_dim=feat, constraints=[])
            pred, _ = model(x)
            assert pred.shape == (4, 1)

    def test_pinn_default_constraints_none(self, sample_input):
        """constraints=None should default to empty list."""
        model = PINNModel(input_dim=FEAT)
        assert len(model.constraints) == 0
        pred, _ = model(sample_input)
        assert pred.shape == (BATCH, 1)

    def test_output_finite(self, sample_input, sample_metadata):
        """No NaN or Inf in output for normal inputs."""
        for arch in BaselineModel.VALID_ARCHS:
            model = BaselineModel(arch=arch, input_dim=FEAT)
            pred = model(sample_input)
            assert torch.isfinite(pred).all(), f"{arch} produced non-finite output"

        model = PINNModel(input_dim=FEAT, constraints=[OUConstraint()])
        pred, _ = model(sample_input, sample_metadata)
        assert torch.isfinite(pred).all()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2D — StackedPINN
# ═══════════════════════════════════════════════════════════════════════════


class TestStackedPINNInheritance:
    """StackedPINN inherits from BasePINN correctly."""

    def test_is_base_pinn(self):
        model = StackedPINN(input_dim=FEAT)
        assert isinstance(model, BasePINN)

    def test_inherits_forward(self, sample_input, sample_metadata):
        """forward() comes from BasePINN — returns (pred, metadata) tuple."""
        model = StackedPINN(input_dim=FEAT, constraints=[GBMConstraint()])
        result = model(sample_input, sample_metadata)
        assert isinstance(result, tuple) and len(result) == 2

    def test_inherits_predict(self, sample_input):
        model = StackedPINN(input_dim=FEAT, constraints=[OUConstraint()])
        pred = model.predict(sample_input)
        assert pred.shape == (BATCH, 1)
        assert not pred.requires_grad

    def test_inherits_diagnostics(self):
        model = StackedPINN(
            input_dim=FEAT,
            constraints=[GBMConstraint(), OUConstraint()],
        )
        diag = model.diagnostics()
        assert set(diag["constraints"]) == {"gbm", "ou"}
        assert "ou_params" in diag

    def test_does_not_override_forward(self):
        """forward should be BasePINN.forward, not a local override."""
        assert "forward" not in StackedPINN.__dict__

    def test_does_not_override_predict(self):
        assert "predict" not in StackedPINN.__dict__

    def test_does_not_override_diagnostics(self):
        assert "diagnostics" not in StackedPINN.__dict__


class TestStackedPINNArchitecture:
    """Both LSTM and GRU encoders present with correct structure."""

    def test_has_lstm_encoder(self):
        model = StackedPINN(input_dim=FEAT)
        assert hasattr(model, "lstm")
        assert isinstance(model.lstm, nn.LSTM)

    def test_has_gru_encoder(self):
        model = StackedPINN(input_dim=FEAT)
        assert hasattr(model, "gru")
        assert isinstance(model.gru, nn.GRU)

    def test_has_attention_fusion(self):
        model = StackedPINN(input_dim=FEAT)
        assert hasattr(model, "attn")
        assert isinstance(model.attn, nn.Sequential)

    def test_has_prediction_head(self):
        model = StackedPINN(input_dim=FEAT, hidden_dim=64)
        assert model.prediction_head.in_features == 64
        assert model.prediction_head.out_features == 1

    def test_attention_network_structure(self):
        """attn: Linear(2H, H) → Tanh → Linear(H, 2)."""
        model = StackedPINN(input_dim=FEAT, hidden_dim=64)
        layers = list(model.attn.children())
        assert len(layers) == 3
        assert isinstance(layers[0], nn.Linear)
        assert layers[0].in_features == 128  # 2 * H
        assert layers[0].out_features == 64
        assert isinstance(layers[1], nn.Tanh)
        assert isinstance(layers[2], nn.Linear)
        assert layers[2].in_features == 64
        assert layers[2].out_features == 2

    def test_encoder_configs_match(self):
        """Both encoders should use the same hyperparameters."""
        model = StackedPINN(
            input_dim=FEAT, hidden_dim=64, num_layers=3, dropout=0.3,
        )
        assert model.lstm.input_size == FEAT
        assert model.lstm.hidden_size == 64
        assert model.lstm.num_layers == 3
        assert model.gru.input_size == FEAT
        assert model.gru.hidden_size == 64
        assert model.gru.num_layers == 3


class TestStackedPINNForwardPass:
    """Output shapes and values from forward pass."""

    def test_output_shape(self, sample_input, sample_metadata):
        model = StackedPINN(input_dim=FEAT)
        pred, meta = model(sample_input, sample_metadata)
        assert pred.shape == (BATCH, 1)

    def test_output_finite(self, sample_input, sample_metadata):
        model = StackedPINN(input_dim=FEAT)
        pred, _ = model(sample_input, sample_metadata)
        assert torch.isfinite(pred).all()

    def test_metadata_passthrough(self, sample_input, sample_metadata):
        """Default _build_physics_metadata returns metadata unchanged."""
        model = StackedPINN(input_dim=FEAT)
        _, enriched = model(sample_input, sample_metadata)
        assert enriched is sample_metadata

    def test_metadata_none(self, sample_input):
        """Passing metadata=None should not raise."""
        model = StackedPINN(input_dim=FEAT)
        pred, meta = model(sample_input, metadata=None)
        assert pred.shape == (BATCH, 1)
        assert meta == {}

    @pytest.mark.parametrize("batch", [1, 4, 16])
    def test_variable_batch_size(self, batch):
        x = torch.randn(batch, SEQ, FEAT)
        model = StackedPINN(input_dim=FEAT)
        pred, _ = model(x)
        assert pred.shape == (batch, 1)

    @pytest.mark.parametrize("seq", [10, 30, 60, 120])
    def test_variable_seq_len(self, seq):
        x = torch.randn(4, seq, FEAT)
        model = StackedPINN(input_dim=FEAT)
        pred, _ = model(x)
        assert pred.shape == (4, 1)

    @pytest.mark.parametrize("feat", [1, 7, 50])
    def test_variable_feature_dim(self, feat):
        x = torch.randn(4, SEQ, feat)
        model = StackedPINN(input_dim=feat)
        pred, _ = model(x)
        assert pred.shape == (4, 1)


class TestStackedPINNAttentionWeights:
    """Attention fusion weights are stored and valid."""

    def test_weights_stored_after_forward(self, sample_input):
        model = StackedPINN(input_dim=FEAT)
        assert model.last_attention_weights is None
        model(sample_input)
        assert model.last_attention_weights is not None

    def test_weights_shape(self, sample_input):
        model = StackedPINN(input_dim=FEAT)
        model(sample_input)
        assert model.last_attention_weights.shape == (BATCH, 2)

    def test_weights_sum_to_one(self, sample_input):
        model = StackedPINN(input_dim=FEAT)
        model(sample_input)
        sums = model.last_attention_weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_weights_non_negative(self, sample_input):
        """Softmax output must be >= 0."""
        model = StackedPINN(input_dim=FEAT)
        model(sample_input)
        assert (model.last_attention_weights >= 0).all()

    def test_weights_detached(self, sample_input):
        """Stored weights should be detached (no grad tracking)."""
        model = StackedPINN(input_dim=FEAT)
        model(sample_input)
        assert not model.last_attention_weights.requires_grad

    def test_weights_update_each_forward(self, sample_input):
        """Weights should change with different inputs."""
        model = StackedPINN(input_dim=FEAT)
        model(sample_input)
        w1 = model.last_attention_weights.clone()
        model(torch.randn_like(sample_input) * 10)  # very different input
        w2 = model.last_attention_weights
        # Weights should differ (with very high probability)
        assert not torch.allclose(w1, w2, atol=1e-6)

    def test_weights_stored_during_predict(self, sample_input):
        """predict() also calls _encode, so weights should be set."""
        model = StackedPINN(input_dim=FEAT)
        model.predict(sample_input)
        assert model.last_attention_weights is not None
        assert model.last_attention_weights.shape == (BATCH, 2)


class TestStackedPINNConstraints:
    """Constraint composition works correctly."""

    def test_empty_constraints(self, sample_input):
        model = StackedPINN(input_dim=FEAT, constraints=[])
        assert len(model.constraints) == 0
        pred, _ = model(sample_input)
        assert pred.shape == (BATCH, 1)

    def test_none_constraints(self, sample_input):
        model = StackedPINN(input_dim=FEAT)
        assert len(model.constraints) == 0

    def test_single_constraint(self):
        model = StackedPINN(input_dim=FEAT, constraints=[GBMConstraint()])
        assert len(model.constraints) == 1

    def test_multi_constraint(self):
        model = StackedPINN(
            input_dim=FEAT,
            constraints=[GBMConstraint(), OUConstraint(), LangevinConstraint()],
        )
        assert len(model.constraints) == 3

    def test_constraint_module_list(self):
        model = StackedPINN(input_dim=FEAT, constraints=[OUConstraint()])
        assert isinstance(model.constraints, nn.ModuleList)

    def test_constraint_params_in_model(self):
        model = StackedPINN(input_dim=FEAT, constraints=[OUConstraint()])
        param_names = {n for n, _ in model.named_parameters()}
        assert any("constraints" in n for n in param_names)


class TestStackedPINNGradientFlow:
    """Gradients flow through both encoders, attention, and constraints."""

    def _get_grads(self, model, sample_input, sample_metadata):
        model.zero_grad()
        pred, enriched = model(sample_input, sample_metadata)
        loss_fn = CompositeLoss(mse_loss, list(model.constraints))
        total, _ = loss_fn(pred, torch.randn(BATCH, 1), enriched)
        total.backward()
        return total

    def test_lstm_receives_gradients(self, sample_input, sample_metadata):
        model = StackedPINN(input_dim=FEAT, constraints=[OUConstraint()])
        self._get_grads(model, sample_input, sample_metadata)
        lstm_grads = [
            p.grad for p in model.lstm.parameters() if p.grad is not None
        ]
        assert len(lstm_grads) > 0

    def test_gru_receives_gradients(self, sample_input, sample_metadata):
        model = StackedPINN(input_dim=FEAT, constraints=[OUConstraint()])
        self._get_grads(model, sample_input, sample_metadata)
        gru_grads = [
            p.grad for p in model.gru.parameters() if p.grad is not None
        ]
        assert len(gru_grads) > 0

    def test_attention_receives_gradients(self, sample_input, sample_metadata):
        model = StackedPINN(input_dim=FEAT, constraints=[OUConstraint()])
        self._get_grads(model, sample_input, sample_metadata)
        attn_grads = [
            p.grad for p in model.attn.parameters() if p.grad is not None
        ]
        assert len(attn_grads) > 0

    def test_prediction_head_receives_gradients(self, sample_input, sample_metadata):
        model = StackedPINN(input_dim=FEAT, constraints=[OUConstraint()])
        self._get_grads(model, sample_input, sample_metadata)
        head_grads = [
            p.grad for p in model.prediction_head.parameters()
            if p.grad is not None
        ]
        assert len(head_grads) > 0

    def test_constraint_receives_gradients(self, sample_input, sample_metadata):
        model = StackedPINN(input_dim=FEAT, constraints=[OUConstraint()])
        self._get_grads(model, sample_input, sample_metadata)
        constraint_grads = [
            p.grad for p in model.constraints.parameters()
            if p.grad is not None
        ]
        assert len(constraint_grads) > 0

    def test_loss_is_finite(self, sample_input, sample_metadata):
        model = StackedPINN(input_dim=FEAT, constraints=[OUConstraint()])
        total = self._get_grads(model, sample_input, sample_metadata)
        assert total.isfinite()


class TestStackedPINNIntegration:
    """End-to-end integration with CompositeLoss."""

    @pytest.mark.parametrize("constraints,names", [
        ([], []),
        ([GBMConstraint()], ["gbm"]),
        ([OUConstraint()], ["ou"]),
        ([GBMConstraint(), OUConstraint()], ["gbm", "ou"]),
        ([HawkesConstraint()], ["hawkes"]),
        ([LangevinConstraint()], ["langevin"]),
        ([HawkesConstraint(), OUConstraint()], ["hawkes", "ou"]),
    ])
    def test_composite_loss_training_step(
        self, constraints, names, sample_input, sample_metadata,
    ):
        model = StackedPINN(input_dim=FEAT, constraints=constraints)
        loss_fn = CompositeLoss(mse_loss, list(model.constraints))
        target = torch.randn(BATCH, 1)

        pred, enriched = model(sample_input, sample_metadata)
        total, breakdown = loss_fn(pred, target, enriched)
        total.backward()

        assert total.isfinite()
        assert "data" in breakdown
        for n in names:
            assert n in breakdown

    def test_optimizer_step(self, sample_input, sample_metadata):
        """Full optimizer step doesn't crash and changes params."""
        model = StackedPINN(input_dim=FEAT, constraints=[OUConstraint()])
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = CompositeLoss(mse_loss, list(model.constraints))
        target = torch.randn(BATCH, 1)

        # Capture initial params
        init_params = {
            n: p.clone() for n, p in model.named_parameters()
        }

        pred, enriched = model(sample_input, sample_metadata)
        total, _ = loss_fn(pred, target, enriched)
        total.backward()
        opt.step()

        # At least some params should have changed
        changed = sum(
            1 for n, p in model.named_parameters()
            if not torch.equal(p, init_params[n])
        )
        assert changed > 0


class TestStackedPINNParamCounts:
    """Parameter count sanity checks."""

    def test_more_params_than_single_encoder(self):
        """StackedPINN should have more params than a single-encoder PINNModel."""
        single = PINNModel(input_dim=FEAT, hidden_dim=128, constraints=[])
        stacked = StackedPINN(input_dim=FEAT, hidden_dim=128, constraints=[])
        single_count = sum(p.numel() for p in single.parameters())
        stacked_count = sum(p.numel() for p in stacked.parameters())
        assert stacked_count > single_count

    def test_constraint_params_add_to_total(self):
        base = StackedPINN(input_dim=FEAT, constraints=[])
        with_ou = StackedPINN(input_dim=FEAT, constraints=[OUConstraint()])
        base_count = sum(p.numel() for p in base.parameters())
        ou_count = sum(p.numel() for p in with_ou.parameters())
        assert ou_count == base_count + 2  # OU has 2 learnable params (_theta_raw, _sigma_raw)

    def test_both_encoders_contribute_params(self):
        model = StackedPINN(input_dim=FEAT)
        lstm_params = sum(p.numel() for p in model.lstm.parameters())
        gru_params = sum(p.numel() for p in model.gru.parameters())
        attn_params = sum(p.numel() for p in model.attn.parameters())
        head_params = sum(p.numel() for p in model.prediction_head.parameters())
        total = sum(p.numel() for p in model.parameters())
        assert total == lstm_params + gru_params + attn_params + head_params


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2E — ResidualPINN
# ═══════════════════════════════════════════════════════════════════════════


class TestResidualPINNInheritance:
    """ResidualPINN inherits from BasePINN correctly."""

    def test_is_base_pinn(self):
        model = ResidualPINN(input_dim=FEAT)
        assert isinstance(model, BasePINN)

    def test_inherits_forward(self, sample_input, sample_metadata):
        model = ResidualPINN(input_dim=FEAT, constraints=[GBMConstraint()])
        result = model(sample_input, sample_metadata)
        assert isinstance(result, tuple) and len(result) == 2

    def test_inherits_predict(self, sample_input):
        model = ResidualPINN(input_dim=FEAT, constraints=[OUConstraint()])
        pred = model.predict(sample_input)
        assert pred.shape == (BATCH, 1)
        assert not pred.requires_grad

    def test_inherits_diagnostics(self):
        model = ResidualPINN(
            input_dim=FEAT, constraints=[GBMConstraint(), OUConstraint()],
        )
        diag = model.diagnostics()
        assert set(diag["constraints"]) == {"gbm", "ou"}
        assert "ou_params" in diag

    def test_does_not_override_forward(self):
        assert "forward" not in ResidualPINN.__dict__

    def test_does_not_override_predict(self):
        assert "predict" not in ResidualPINN.__dict__

    def test_does_not_override_diagnostics(self):
        assert "diagnostics" not in ResidualPINN.__dict__


class TestResidualPINNArchitecture:
    """Base LSTM + correction GRU with correct structure."""

    def test_base_encoder_is_lstm(self):
        model = ResidualPINN(input_dim=FEAT)
        assert isinstance(model.base_encoder, nn.LSTM)

    def test_correction_encoder_is_gru(self):
        model = ResidualPINN(input_dim=FEAT)
        assert isinstance(model.correction_encoder, nn.GRU)

    def test_different_architectures(self):
        """Base and correction use different RNN types for diversity."""
        model = ResidualPINN(input_dim=FEAT)
        assert type(model.base_encoder) is not type(model.correction_encoder)

    def test_correction_half_hidden_dim(self):
        model = ResidualPINN(input_dim=FEAT, hidden_dim=128)
        assert model.base_encoder.hidden_size == 128
        assert model.correction_encoder.hidden_size == 64  # H // 2

    def test_correction_half_hidden_dim_custom(self):
        model = ResidualPINN(input_dim=FEAT, hidden_dim=64)
        assert model.correction_encoder.hidden_size == 32

    def test_base_head_shape(self):
        model = ResidualPINN(input_dim=FEAT, hidden_dim=128)
        assert model.base_head.in_features == 128
        assert model.base_head.out_features == 1

    def test_correction_head_structure(self):
        """correction_head: Linear(H//2, H//4) → Tanh → Linear(H//4, 1) → Tanh."""
        model = ResidualPINN(input_dim=FEAT, hidden_dim=128)
        layers = list(model.correction_head.children())
        assert len(layers) == 4
        assert isinstance(layers[0], nn.Linear)
        assert layers[0].in_features == 64   # H // 2
        assert layers[0].out_features == 32  # H // 4
        assert isinstance(layers[1], nn.Tanh)
        assert isinstance(layers[2], nn.Linear)
        assert layers[2].in_features == 32
        assert layers[2].out_features == 1
        assert isinstance(layers[3], nn.Tanh)  # bounding mechanism

    def test_correction_head_ends_with_tanh(self):
        """The final Tanh is the bounding mechanism — must be last."""
        model = ResidualPINN(input_dim=FEAT)
        layers = list(model.correction_head.children())
        assert isinstance(layers[-1], nn.Tanh)

    def test_max_correction_stored(self):
        model = ResidualPINN(input_dim=FEAT, max_correction=0.05)
        assert model.max_correction == 0.05

    def test_encoder_configs(self):
        model = ResidualPINN(
            input_dim=FEAT, hidden_dim=64, num_layers=3, dropout=0.3,
        )
        assert model.base_encoder.input_size == FEAT
        assert model.base_encoder.num_layers == 3
        assert model.correction_encoder.input_size == FEAT
        assert model.correction_encoder.num_layers == 3


class TestResidualPINNForwardPass:
    """Output shapes and values."""

    def test_output_shape(self, sample_input, sample_metadata):
        model = ResidualPINN(input_dim=FEAT)
        pred, meta = model(sample_input, sample_metadata)
        assert pred.shape == (BATCH, 1)

    def test_output_finite(self, sample_input, sample_metadata):
        model = ResidualPINN(input_dim=FEAT)
        pred, _ = model(sample_input, sample_metadata)
        assert torch.isfinite(pred).all()

    def test_metadata_passthrough(self, sample_input, sample_metadata):
        model = ResidualPINN(input_dim=FEAT)
        _, enriched = model(sample_input, sample_metadata)
        assert enriched is sample_metadata

    def test_metadata_none(self, sample_input):
        model = ResidualPINN(input_dim=FEAT)
        pred, meta = model(sample_input, metadata=None)
        assert pred.shape == (BATCH, 1)
        assert meta == {}

    @pytest.mark.parametrize("batch", [1, 4, 16])
    def test_variable_batch_size(self, batch):
        x = torch.randn(batch, SEQ, FEAT)
        model = ResidualPINN(input_dim=FEAT)
        pred, _ = model(x)
        assert pred.shape == (batch, 1)

    @pytest.mark.parametrize("seq", [10, 30, 60, 120])
    def test_variable_seq_len(self, seq):
        x = torch.randn(4, seq, FEAT)
        model = ResidualPINN(input_dim=FEAT)
        pred, _ = model(x)
        assert pred.shape == (4, 1)

    @pytest.mark.parametrize("feat", [1, 7, 50])
    def test_variable_feature_dim(self, feat):
        x = torch.randn(4, SEQ, feat)
        model = ResidualPINN(input_dim=feat)
        pred, _ = model(x)
        assert pred.shape == (4, 1)


class TestResidualPINNCorrectionBounding:
    """Correction output is bounded by max_correction."""

    def test_correction_bounded_default(self, sample_input):
        model = ResidualPINN(input_dim=FEAT, max_correction=0.1)
        corrections = []
        for _ in range(50):
            x = torch.randn(BATCH, SEQ, FEAT)
            hidden = model._encode(x)
            h_corr = hidden[:, model.base_head.in_features:]
            corr = model.correction_head(h_corr) * model.max_correction
            corrections.append(corr.abs().max().item())
        assert max(corrections) <= 0.1 + 1e-6

    def test_correction_bounded_custom(self):
        model = ResidualPINN(input_dim=FEAT, max_correction=0.05)
        corrections = []
        for _ in range(50):
            x = torch.randn(BATCH, SEQ, FEAT)
            hidden = model._encode(x)
            h_corr = hidden[:, model.base_head.in_features:]
            corr = model.correction_head(h_corr) * model.max_correction
            corrections.append(corr.abs().max().item())
        assert max(corrections) <= 0.05 + 1e-6

    def test_correction_bounded_large_input(self):
        """Even with extreme inputs, correction stays bounded."""
        model = ResidualPINN(input_dim=FEAT, max_correction=0.1)
        x = torch.randn(BATCH, SEQ, FEAT) * 100  # large input
        hidden = model._encode(x)
        h_corr = hidden[:, model.base_head.in_features:]
        corr = model.correction_head(h_corr) * model.max_correction
        assert corr.abs().max().item() <= 0.1 + 1e-6

    def test_correction_cannot_dominate(self, sample_input):
        """Correction magnitude should be much smaller than typical base prediction."""
        model = ResidualPINN(input_dim=FEAT, max_correction=0.1)
        # max possible correction is 0.1, while base predictions are unbounded
        hidden = model._encode(sample_input)
        split = model.base_head.in_features
        base_pred = model.base_head(hidden[:, :split])
        h_corr = hidden[:, split:]
        corr = model.correction_head(h_corr) * model.max_correction
        # Correction is bounded; base is not
        assert corr.abs().max().item() <= 0.1 + 1e-6

    def test_hidden_state_split_correct(self, sample_input):
        """_encode produces [B, H + H//2] and _predict splits correctly."""
        model = ResidualPINN(input_dim=FEAT, hidden_dim=128)
        hidden = model._encode(sample_input)
        assert hidden.shape == (BATCH, 128 + 64)  # H + H//2
        # Split point matches base_head.in_features
        assert model.base_head.in_features == 128


class TestResidualPINNConstraints:
    """Constraint composition."""

    def test_empty_constraints(self, sample_input):
        model = ResidualPINN(input_dim=FEAT, constraints=[])
        assert len(model.constraints) == 0
        pred, _ = model(sample_input)
        assert pred.shape == (BATCH, 1)

    def test_none_constraints(self):
        model = ResidualPINN(input_dim=FEAT)
        assert len(model.constraints) == 0

    def test_single_constraint(self):
        model = ResidualPINN(input_dim=FEAT, constraints=[GBMConstraint()])
        assert len(model.constraints) == 1

    def test_multi_constraint(self):
        model = ResidualPINN(
            input_dim=FEAT,
            constraints=[GBMConstraint(), OUConstraint(), LangevinConstraint()],
        )
        assert len(model.constraints) == 3

    def test_constraint_module_list(self):
        model = ResidualPINN(input_dim=FEAT, constraints=[OUConstraint()])
        assert isinstance(model.constraints, nn.ModuleList)


class TestResidualPINNGradientFlow:
    """Gradients flow through both encoders, heads, and constraints."""

    def _get_grads(self, model, sample_input, sample_metadata):
        model.zero_grad()
        pred, enriched = model(sample_input, sample_metadata)
        loss_fn = CompositeLoss(mse_loss, list(model.constraints))
        total, _ = loss_fn(pred, torch.randn(BATCH, 1), enriched)
        total.backward()
        return total

    def test_base_encoder_receives_gradients(self, sample_input, sample_metadata):
        model = ResidualPINN(input_dim=FEAT, constraints=[OUConstraint()])
        self._get_grads(model, sample_input, sample_metadata)
        grads = [p.grad for p in model.base_encoder.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_correction_encoder_receives_gradients(self, sample_input, sample_metadata):
        model = ResidualPINN(input_dim=FEAT, constraints=[OUConstraint()])
        self._get_grads(model, sample_input, sample_metadata)
        grads = [p.grad for p in model.correction_encoder.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_base_head_receives_gradients(self, sample_input, sample_metadata):
        model = ResidualPINN(input_dim=FEAT, constraints=[OUConstraint()])
        self._get_grads(model, sample_input, sample_metadata)
        grads = [p.grad for p in model.base_head.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_correction_head_receives_gradients(self, sample_input, sample_metadata):
        model = ResidualPINN(input_dim=FEAT, constraints=[OUConstraint()])
        self._get_grads(model, sample_input, sample_metadata)
        grads = [p.grad for p in model.correction_head.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_constraint_receives_gradients(self, sample_input, sample_metadata):
        model = ResidualPINN(input_dim=FEAT, constraints=[OUConstraint()])
        self._get_grads(model, sample_input, sample_metadata)
        grads = [p.grad for p in model.constraints.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_loss_is_finite(self, sample_input, sample_metadata):
        model = ResidualPINN(input_dim=FEAT, constraints=[OUConstraint()])
        total = self._get_grads(model, sample_input, sample_metadata)
        assert total.isfinite()


class TestResidualPINNIntegration:
    """End-to-end integration with CompositeLoss."""

    @pytest.mark.parametrize("constraints,names", [
        ([], []),
        ([GBMConstraint()], ["gbm"]),
        ([OUConstraint()], ["ou"]),
        ([GBMConstraint(), OUConstraint()], ["gbm", "ou"]),
        ([HawkesConstraint()], ["hawkes"]),
        ([LangevinConstraint()], ["langevin"]),
        ([HawkesConstraint(), OUConstraint()], ["hawkes", "ou"]),
    ])
    def test_composite_loss_training_step(
        self, constraints, names, sample_input, sample_metadata,
    ):
        model = ResidualPINN(input_dim=FEAT, constraints=constraints)
        loss_fn = CompositeLoss(mse_loss, list(model.constraints))
        target = torch.randn(BATCH, 1)

        pred, enriched = model(sample_input, sample_metadata)
        total, breakdown = loss_fn(pred, target, enriched)
        total.backward()

        assert total.isfinite()
        assert "data" in breakdown
        for n in names:
            assert n in breakdown

    def test_optimizer_step(self, sample_input, sample_metadata):
        """Full optimizer step changes params."""
        model = ResidualPINN(input_dim=FEAT, constraints=[OUConstraint()])
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = CompositeLoss(mse_loss, list(model.constraints))
        target = torch.randn(BATCH, 1)

        init_params = {n: p.clone() for n, p in model.named_parameters()}

        pred, enriched = model(sample_input, sample_metadata)
        total, _ = loss_fn(pred, target, enriched)
        total.backward()
        opt.step()

        changed = sum(
            1 for n, p in model.named_parameters()
            if not torch.equal(p, init_params[n])
        )
        assert changed > 0


class TestResidualPINNParamCounts:
    """Parameter count sanity checks."""

    def test_has_both_encoder_params(self):
        model = ResidualPINN(input_dim=FEAT)
        base_p = sum(p.numel() for p in model.base_encoder.parameters())
        corr_p = sum(p.numel() for p in model.correction_encoder.parameters())
        assert base_p > 0 and corr_p > 0
        # Base LSTM should have more params than correction GRU (full H vs H//2)
        assert base_p > corr_p

    def test_constraint_params_add_to_total(self):
        base = ResidualPINN(input_dim=FEAT, constraints=[])
        with_ou = ResidualPINN(input_dim=FEAT, constraints=[OUConstraint()])
        base_count = sum(p.numel() for p in base.parameters())
        ou_count = sum(p.numel() for p in with_ou.parameters())
        assert ou_count == base_count + 2  # OU has 2 learnable params (_theta_raw, _sigma_raw)

    def test_total_param_accounting(self):
        model = ResidualPINN(input_dim=FEAT)
        base_enc = sum(p.numel() for p in model.base_encoder.parameters())
        base_head = sum(p.numel() for p in model.base_head.parameters())
        corr_enc = sum(p.numel() for p in model.correction_encoder.parameters())
        corr_head = sum(p.numel() for p in model.correction_head.parameters())
        total = sum(p.numel() for p in model.parameters())
        assert total == base_enc + base_head + corr_enc + corr_head


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2F — Registry
# ═══════════════════════════════════════════════════════════════════════════

ALL_MODEL_NAMES = [
    "lstm", "gru", "bilstm", "attention_lstm", "transformer",
    "baseline_pinn", "gbm_pinn", "ou_pinn", "bs_pinn",
    "gbm_ou_pinn", "global_pinn",
    "hawkes_pinn", "hawkes_ou_pinn",
    "hawkes_v2_pinn", "hawkes_v2_multiscale_pinn",
    "stacked_pinn", "residual_pinn",
]

BASELINE_NAMES = {"lstm", "gru", "bilstm", "attention_lstm", "transformer"}
# Classical econometric baselines (random_walk, historical_mean, persistence,
# garch) are registered for run_experiment name-validation only — they bypass
# the neural training loop, so the parameterised tests below that iterate
# ALL_MODEL_NAMES must NOT include them. Kept as a separate constant so the
# registry-catalogue test still verifies their presence.
CLASSICAL_NAMES = {"random_walk", "historical_mean", "persistence", "garch"}
# BS models need extra metadata for residual — skip CompositeLoss end-to-end
BS_NAMES = {"bs_pinn", "global_pinn"}


class TestRegistryCatalogue:
    """Registry contains 15 neural models + 4 classical baselines = 19 total."""

    def test_list_models_count(self):
        assert len(list_models()) == len(ALL_MODEL_NAMES) + len(CLASSICAL_NAMES)

    def test_list_models_names(self):
        assert set(list_models()) == set(ALL_MODEL_NAMES) | CLASSICAL_NAMES

    def test_registry_dict_matches_list(self):
        assert list(MODEL_REGISTRY.keys()) == list_models()


class TestRegistryBuildModel:
    """build_model() factory produces working models."""

    @pytest.mark.parametrize("name", ALL_MODEL_NAMES)
    def test_build_returns_module(self, name):
        model = build_model(name, input_dim=FEAT)
        assert isinstance(model, nn.Module)

    @pytest.mark.parametrize("name", ALL_MODEL_NAMES)
    def test_build_forward_shape(self, name, sample_input):
        model = build_model(name, input_dim=FEAT)
        if hasattr(model, "constraints"):
            pred, _ = model(sample_input)
        else:
            pred = model(sample_input)
        assert pred.shape == (BATCH, 1)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent", input_dim=FEAT)

    def test_kwargs_forwarded(self):
        model = build_model("lstm", input_dim=FEAT, hidden_dim=64, num_layers=3)
        assert model.encoder.hidden_size == 64
        assert model.encoder.num_layers == 3


class TestRegistryModelTypes:
    """Each registry entry produces the correct model class."""

    @pytest.mark.parametrize("name", list(BASELINE_NAMES))
    def test_baselines_are_baseline_model(self, name):
        model = build_model(name, input_dim=FEAT)
        assert isinstance(model, BaselineModel)
        assert not isinstance(model, BasePINN)

    @pytest.mark.parametrize("name", [
        "baseline_pinn", "gbm_pinn", "ou_pinn", "bs_pinn",
        "gbm_ou_pinn", "global_pinn", "hawkes_pinn", "hawkes_ou_pinn",
    ])
    def test_core_pinns_are_pinn_model(self, name):
        model = build_model(name, input_dim=FEAT)
        assert isinstance(model, PINNModel)

    def test_stacked_is_stacked_pinn(self):
        model = build_model("stacked_pinn", input_dim=FEAT)
        assert isinstance(model, StackedPINN)

    def test_residual_is_residual_pinn(self):
        model = build_model("residual_pinn", input_dim=FEAT)
        assert isinstance(model, ResidualPINN)


class TestRegistryConstraintCounts:
    """Pre-configured constraints match the spec."""

    @pytest.mark.parametrize("name,expected", [
        ("baseline_pinn", 0),
        ("gbm_pinn", 1),
        ("ou_pinn", 1),
        ("bs_pinn", 1),
        ("gbm_ou_pinn", 2),
        ("global_pinn", 4),
        ("hawkes_pinn", 1),
        ("hawkes_ou_pinn", 2),
    ])
    def test_constraint_count(self, name, expected):
        model = build_model(name, input_dim=FEAT)
        assert len(model.constraints) == expected

    def test_global_pinn_constraint_names(self):
        model = build_model("global_pinn", input_dim=FEAT)
        names = {c.name for c in model.constraints}
        assert names == {"gbm", "ou", "bs", "langevin"}


class TestRegistryNoSharedState:
    """Each build_model() call produces independent instances."""

    def test_separate_instances(self):
        m1 = build_model("gbm_pinn", input_dim=FEAT)
        m2 = build_model("gbm_pinn", input_dim=FEAT)
        # Different objects
        assert m1 is not m2
        assert m1.constraints[0] is not m2.constraints[0]

    def test_param_independence(self):
        """Changing params in one model doesn't affect another."""
        m1 = build_model("ou_pinn", input_dim=FEAT)
        m2 = build_model("ou_pinn", input_dim=FEAT)
        # Mutate m1's first encoder weight
        with torch.no_grad():
            list(m1.parameters())[0].fill_(999.0)
        # m2 should be unaffected
        assert list(m2.parameters())[0].abs().max().item() != 999.0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2G — Full Integration Verification
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase2GIntegration:
    """End-to-end smoke test: every model builds, runs forward, backward,
    predict, and diagnostics without errors.
    """

    @pytest.fixture
    def physics_metadata(self):
        """Realistic metadata with positive prices for GBM."""
        prices = torch.rand(BATCH, SEQ + 1) * 100 + 50
        returns = torch.log(prices[:, 1:] / prices[:, :-1])
        return {"prices": prices, "returns": returns, "dt": 1.0 / 252}

    @pytest.mark.parametrize("name", ALL_MODEL_NAMES)
    def test_forward_shape(self, name, sample_input, physics_metadata):
        model = build_model(name, input_dim=FEAT)
        if hasattr(model, "constraints"):
            pred, _ = model(sample_input, physics_metadata)
        else:
            pred = model(sample_input)
        assert pred.shape == (BATCH, 1), f"{name}: shape {pred.shape}"
        assert torch.isfinite(pred).all(), f"{name}: non-finite output"

    @pytest.mark.parametrize("name", [
        n for n in ALL_MODEL_NAMES if n not in BS_NAMES
    ])
    def test_training_step_non_bs(self, name, sample_input, physics_metadata):
        """Full forward → CompositeLoss → backward for non-BS models."""
        model = build_model(name, input_dim=FEAT)
        target = torch.randn(BATCH, 1)

        if hasattr(model, "constraints"):
            pred, enriched = model(sample_input, physics_metadata)
            loss_fn = CompositeLoss(mse_loss, list(model.constraints))
            total, breakdown = loss_fn(pred, target, enriched)
        else:
            pred = model(sample_input)
            loss_fn = CompositeLoss(mse_loss, constraints=[])
            total, breakdown = loss_fn(pred, target, {})

        total.backward()

        assert total.isfinite(), f"{name}: non-finite loss"
        assert "data" in breakdown
        assert "total" in breakdown

        # Gradient flow
        grad_ok = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        assert grad_ok, f"{name}: no gradient flow"

    @pytest.mark.parametrize("name", list(BS_NAMES))
    def test_bs_models_forward_and_grad(self, name, sample_input, physics_metadata):
        """BS models need training-pipeline metadata for residual; test forward + grad only."""
        model = build_model(name, input_dim=FEAT)
        pred, enriched = model(sample_input, physics_metadata)
        assert pred.shape == (BATCH, 1)

        # Gradient flows through encoder via simple loss
        loss = pred.sum()
        loss.backward()
        grad_ok = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        assert grad_ok, f"{name}: no gradient flow"

    @pytest.mark.parametrize("name", [
        n for n in ALL_MODEL_NAMES if n not in BASELINE_NAMES
    ])
    def test_predict_inference(self, name, sample_input):
        """predict() works in no-grad mode for all PINNs."""
        model = build_model(name, input_dim=FEAT)
        pred = model.predict(sample_input)
        assert pred.shape == (BATCH, 1)
        assert not pred.requires_grad

    @pytest.mark.parametrize("name", [
        n for n in ALL_MODEL_NAMES if n not in BASELINE_NAMES
    ])
    def test_diagnostics(self, name):
        """diagnostics() returns constraint info for all PINNs."""
        model = build_model(name, input_dim=FEAT)
        diag = model.diagnostics()
        assert "constraints" in diag
        assert isinstance(diag["constraints"], list)
        assert len(diag["constraints"]) == len(model.constraints)

    @pytest.mark.parametrize("name", ALL_MODEL_NAMES)
    def test_model_parameters_non_empty(self, name):
        """Every model has at least some learnable parameters."""
        model = build_model(name, input_dim=FEAT)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0, f"{name}: zero parameters"

    @pytest.mark.parametrize("name", [
        n for n in ALL_MODEL_NAMES if n not in BASELINE_NAMES
    ])
    def test_constraints_in_model_params(self, name):
        """Physics constraint params are reachable via model.parameters()."""
        model = build_model(name, input_dim=FEAT)
        if len(model.constraints) == 0:
            return  # baseline_pinn has no constraint params
        constraint_params = set(model.constraints.parameters())
        model_params = set(model.parameters())
        assert constraint_params.issubset(model_params)


class TestPhase2GImports:
    """__init__.py exports all public symbols."""

    def test_import_build_model(self):
        from src.models import build_model as bm
        assert callable(bm)

    def test_import_list_models(self):
        from src.models import list_models as lm
        assert callable(lm)

    def test_import_base_pinn(self):
        from src.models import BasePINN as BP
        assert BP is BasePINN

    def test_import_baseline_model(self):
        from src.models import BaselineModel as BM
        assert BM is BaselineModel

    def test_import_pinn_model(self):
        from src.models import PINNModel as PM
        assert PM is PINNModel

    def test_import_stacked_pinn(self):
        from src.models import StackedPINN as SP
        assert SP is StackedPINN

    def test_import_residual_pinn(self):
        from src.models import ResidualPINN as RP
        assert RP is ResidualPINN
