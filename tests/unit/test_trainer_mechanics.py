"""
Unit tests for trainer mechanics: loss computation, gradients, optimizer, and EMA.

Tests verify that the core training loop components work correctly in isolation,
without requiring full model training.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, MagicMock
import math

from tests.utils import (
    assert_loss_valid,
    assert_tensor_finite,
    assert_tensors_close,
    assert_tensor_dtype,
)


# =============================================================================
# TestLossComputation - Verify loss math is correct
# =============================================================================


class TestLossComputation:
    """Verify loss computation matches expected mathematical formulas."""

    def test_mse_loss_computation(self):
        """MSE loss matches torch.nn.functional.mse_loss."""
        pred = torch.randn(4, 1, 64, 64)
        target = torch.randn(4, 1, 64, 64)

        # Manual MSE
        manual_mse = ((pred - target) ** 2).mean()

        # PyTorch MSE
        pytorch_mse = F.mse_loss(pred, target)

        assert_tensors_close(manual_mse, pytorch_mse, name="MSE loss")

    def test_weighted_mse_per_sample(self):
        """Per-sample weighted MSE preserves batch dimension during computation."""
        batch_size = 4
        pred = torch.randn(batch_size, 1, 64, 64)
        target = torch.randn(batch_size, 1, 64, 64)
        weights = torch.rand(batch_size)

        # Per-sample MSE pattern from trainer
        mse_per_sample = ((pred.float() - target.float()) ** 2).flatten(1).mean(1)

        # Verify shape before weighting
        assert mse_per_sample.shape == (batch_size,), \
            f"Per-sample MSE should have shape ({batch_size},), got {mse_per_sample.shape}"

        # Weighted mean
        weighted_mse = (mse_per_sample * weights.float()).mean()

        assert_loss_valid(weighted_mse, "Weighted MSE")

    def test_kl_divergence_computation(self):
        """KL divergence for VAE matches formula: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))."""
        batch_size = 4
        latent_dim = 128

        mu = torch.randn(batch_size, latent_dim)
        logvar = torch.randn(batch_size, latent_dim)

        # KL divergence formula from VAE
        kl_manual = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_mean = kl_manual.mean()

        # Verify shape
        assert kl_manual.shape == (batch_size,), \
            f"Per-sample KL should have shape ({batch_size},), got {kl_manual.shape}"

        # KL should be non-negative (though can be very small)
        assert_loss_valid(kl_mean, "KL divergence")

        # Verify KL is 0 when mu=0 and logvar=0 (standard normal)
        kl_standard = -0.5 * torch.sum(
            1 + torch.zeros(batch_size, latent_dim) -
            torch.zeros(batch_size, latent_dim).pow(2) -
            torch.zeros(batch_size, latent_dim).exp(),
            dim=1
        )
        assert torch.allclose(kl_standard, torch.zeros(batch_size), atol=1e-5), \
            "KL should be 0 for standard normal (mu=0, logvar=0)"

    def test_perceptual_loss_uses_fp32(self):
        """Loss computation must cast BF16 to FP32 before computation (Pitfall #41)."""
        pred_bf16 = torch.randn(4, 3, 64, 64, dtype=torch.bfloat16)
        target_bf16 = torch.randn(4, 3, 64, 64, dtype=torch.bfloat16)

        # CORRECT pattern: cast to FP32 before loss
        loss_fp32 = ((pred_bf16.float() - target_bf16.float()) ** 2).mean()

        assert_tensor_dtype(loss_fp32, torch.float32, "FP32 casted loss")

        # Verify this is different from BF16 computation (due to precision)
        loss_bf16 = ((pred_bf16 - target_bf16) ** 2).mean()
        assert loss_bf16.dtype == torch.bfloat16, "BF16 loss should stay BF16"
        assert loss_fp32.dtype == torch.float32, "FP32 casted loss should be FP32"


# =============================================================================
# TestGradientFlow - Verify gradients flow correctly
# =============================================================================


class TestGradientFlow:
    """Verify gradient computation and flow through model parameters."""

    def test_gradients_reach_model_parameters(self):
        """loss.backward() populates gradients on model parameters."""
        # Simple model
        model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

        # Forward pass
        x = torch.randn(4, 1, 32, 32)
        target = torch.randn(4, 1, 32, 32)
        pred = model(x)
        loss = F.mse_loss(pred, target)

        # Verify no gradients before backward
        for param in model.parameters():
            assert param.grad is None, "Gradients should be None before backward"

        # Backward pass
        loss.backward()

        # Verify gradients exist after backward
        for i, param in enumerate(model.parameters()):
            assert param.grad is not None, f"Gradient for param {i} should exist after backward"
            assert_tensor_finite(param.grad, f"Gradient for param {i}")

    def test_gradient_clipping_respects_max_norm(self):
        """Gradient clipping limits gradient norm to specified max."""
        model = nn.Linear(64, 64)
        x = torch.randn(4, 64)
        target = torch.randn(4, 64)

        # Create large gradients
        pred = model(x)
        loss = ((pred - target) ** 2).sum() * 1000  # Scale up for large gradients
        loss.backward()

        # Clip gradients - returns the original norm BEFORE clipping
        max_norm = 1.0
        original_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Verify original norm was large (gradients existed)
        assert original_norm > max_norm, \
            f"Original gradient norm {original_norm} should be > {max_norm}"

        # Compute actual norm after clipping to verify
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.detach().pow(2).sum()
        clipped_norm = total_norm.sqrt()

        # Verify clipping worked
        assert clipped_norm <= max_norm + 1e-4, \
            f"Clipped gradient norm {clipped_norm} should be <= {max_norm}"

    def test_no_gradients_during_validation(self):
        """torch.no_grad() prevents gradient computation."""
        model = nn.Linear(64, 64)
        x = torch.randn(4, 64)

        with torch.no_grad():
            pred = model(x)
            loss = pred.sum()

        # Verify no gradient computation happened
        assert not pred.requires_grad, "Output should not require grad under no_grad()"

        # Verify we cannot call backward (would fail if tried)
        with pytest.raises(RuntimeError):
            loss.backward()


# =============================================================================
# TestOptimizerState - Verify optimizer behavior
# =============================================================================


class TestOptimizerState:
    """Verify optimizer step, zero_grad, and scheduler behavior."""

    def test_optimizer_step_changes_weights(self):
        """Optimizer step() modifies model weights."""
        model = nn.Linear(64, 64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # Store original weights
        original_weight = model.weight.clone()

        # Forward/backward
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()

        # Step
        optimizer.step()

        # Verify weights changed
        weight_changed = not torch.allclose(model.weight, original_weight)
        assert weight_changed, "Weights should change after optimizer.step()"

    def test_zero_grad_clears_gradients(self):
        """zero_grad() clears all accumulated gradients."""
        model = nn.Linear(64, 64)
        # Use set_to_none=False to ensure gradients are zeroed, not set to None
        optimizer = torch.optim.Adam(model.parameters())

        # Create gradients
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()

        # Verify gradients exist
        assert model.weight.grad is not None, "Gradient should exist after backward"
        assert model.weight.grad.abs().sum() > 0, "Gradient should be non-zero"

        # Zero gradients (with set_to_none=False to keep tensor)
        optimizer.zero_grad(set_to_none=False)

        # Verify gradients cleared (tensor still exists but is all zeros)
        assert model.weight.grad is not None, "Grad tensor should still exist"
        assert model.weight.grad.abs().sum() == 0, "Gradient should be zero after zero_grad"

    def test_zero_grad_set_to_none(self):
        """zero_grad(set_to_none=True) sets gradients to None."""
        model = nn.Linear(64, 64)
        optimizer = torch.optim.Adam(model.parameters())

        # Create gradients
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()

        # Verify gradients exist
        assert model.weight.grad is not None, "Gradient should exist after backward"

        # Zero gradients with set_to_none=True (default in newer PyTorch)
        optimizer.zero_grad(set_to_none=True)

        # Verify gradients are None
        assert model.weight.grad is None, "Gradient should be None with set_to_none=True"

    def test_learning_rate_scheduler_decreases_lr(self):
        """Learning rate scheduler decreases LR over steps."""
        model = nn.Linear(64, 64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        initial_lr = optimizer.param_groups[0]['lr']

        # Step scheduler multiple times
        for _ in range(10):
            scheduler.step()

        final_lr = optimizer.param_groups[0]['lr']

        assert final_lr < initial_lr, \
            f"LR should decrease: {initial_lr} -> {final_lr}"
        assert final_lr == initial_lr * 0.5 * 0.5, \
            f"LR should be {initial_lr * 0.25} after 10 steps with step_size=5, gamma=0.5"


# =============================================================================
# TestEMAWeights - Verify EMA updates
# =============================================================================


class TestEMAWeights:
    """Verify Exponential Moving Average weight updates."""

    def test_ema_update_formula(self):
        """EMA update follows: ema = decay * ema + (1 - decay) * current."""
        decay = 0.99

        # Initialize weights
        ema_weight = torch.randn(64, 64)
        current_weight = torch.randn(64, 64)

        # Manual EMA update
        expected = decay * ema_weight + (1 - decay) * current_weight

        # Verify formula
        ema_updated = decay * ema_weight + (1 - decay) * current_weight
        assert_tensors_close(ema_updated, expected, name="EMA update")

    def test_ema_converges_to_current_with_low_decay(self):
        """With low decay (0.0), EMA immediately equals current weights."""
        decay = 0.0

        ema_weight = torch.randn(64, 64)
        current_weight = torch.randn(64, 64)

        # EMA with decay=0 should immediately become current
        ema_updated = decay * ema_weight + (1 - decay) * current_weight

        assert_tensors_close(ema_updated, current_weight, name="EMA with decay=0")

    def test_ema_ignores_current_with_high_decay(self):
        """With high decay (1.0), EMA ignores current weights entirely."""
        decay = 1.0

        ema_weight = torch.randn(64, 64)
        current_weight = torch.randn(64, 64)

        # EMA with decay=1 should remain unchanged
        ema_updated = decay * ema_weight + (1 - decay) * current_weight

        assert_tensors_close(ema_updated, ema_weight, name="EMA with decay=1")

    def test_ema_gradual_convergence(self):
        """EMA gradually converges to current weights over many updates."""
        decay = 0.9
        current_weight = torch.ones(64, 64)  # Target: all ones
        ema_weight = torch.zeros(64, 64)     # Start: all zeros

        # Apply EMA updates
        for _ in range(100):
            ema_weight = decay * ema_weight + (1 - decay) * current_weight

        # After many updates, should be close to current
        assert torch.allclose(ema_weight, current_weight, atol=0.01), \
            f"EMA should converge to current after many updates, diff={torch.abs(ema_weight - current_weight).max()}"


# =============================================================================
# TestMinSNRWeighting - Verify SNR-based loss weighting
# =============================================================================


class TestMinSNRWeighting:
    """Test Min-SNR loss weighting computation."""

    def test_snr_weights_non_negative(self):
        """SNR weights should be non-negative."""
        # RFlow SNR: t / (1 - t) for t in (0, 1)
        timesteps = torch.linspace(0.01, 0.99, 100)
        snr = timesteps / (1 - timesteps)

        assert (snr >= 0).all(), "SNR should be non-negative"

    def test_snr_increases_with_timestep(self):
        """SNR increases as t approaches 1 (less noise)."""
        timesteps = torch.linspace(0.01, 0.99, 100)
        snr = timesteps / (1 - timesteps)

        # Check monotonically increasing
        diffs = snr[1:] - snr[:-1]
        assert (diffs >= 0).all(), "SNR should increase with timestep"

    def test_min_snr_clipping(self):
        """Min-SNR gamma clipping limits weights."""
        gamma = 5.0
        timesteps = torch.linspace(0.01, 0.99, 100)
        snr = timesteps / (1 - timesteps)

        # Min-SNR weighting: min(SNR, gamma) / SNR
        weights = torch.minimum(snr, torch.tensor(gamma)) / snr

        # Weights should be bounded
        assert (weights >= 0).all(), "Weights should be non-negative"
        assert (weights <= 1).all(), "Weights should be <= 1 with min-SNR"


# =============================================================================
# TestLossScaling - Verify loss scaling patterns
# =============================================================================


class TestLossScaling:
    """Test loss scaling and combination patterns."""

    def test_multi_loss_weighting(self):
        """Multiple losses combine correctly with weights."""
        mse_loss = torch.tensor(0.5)
        perceptual_loss = torch.tensor(0.3)
        kl_loss = torch.tensor(0.1)

        weights = {'mse': 1.0, 'perceptual': 0.1, 'kl': 0.01}

        total = (
            weights['mse'] * mse_loss +
            weights['perceptual'] * perceptual_loss +
            weights['kl'] * kl_loss
        )

        expected = 1.0 * 0.5 + 0.1 * 0.3 + 0.01 * 0.1
        assert_tensors_close(total, torch.tensor(expected), name="Weighted total loss")

    def test_loss_nan_detection(self):
        """NaN losses should be detected and handled."""
        valid_loss = torch.tensor(0.5)
        nan_loss = torch.tensor(float('nan'))

        assert torch.isfinite(valid_loss), "Valid loss should be finite"
        assert not torch.isfinite(nan_loss), "NaN loss should be detected as non-finite"

    def test_loss_inf_detection(self):
        """Infinite losses should be detected."""
        valid_loss = torch.tensor(0.5)
        inf_loss = torch.tensor(float('inf'))

        assert torch.isfinite(valid_loss), "Valid loss should be finite"
        assert not torch.isfinite(inf_loss), "Inf loss should be detected as non-finite"


# =============================================================================
# TestConditioningDropout - Verify CFG dropout for ControlNet
# =============================================================================


class TestConditioningDropout:
    """Test CFG conditioning dropout for ControlNet training."""

    def _apply_conditioning_dropout(
        self,
        conditioning: torch.Tensor,
        batch_size: int,
        dropout_prob: float,
        training: bool = True,
    ) -> torch.Tensor:
        """Standalone implementation of conditioning dropout for testing.

        Mirrors the trainer's _apply_conditioning_dropout method.
        """
        if conditioning is None or dropout_prob <= 0:
            return conditioning

        if not training:
            return conditioning

        # Per-sample dropout mask
        dropout_mask = torch.rand(batch_size, device=conditioning.device)
        keep_mask = (dropout_mask >= dropout_prob).float()

        # Expand to match conditioning dims
        for _ in range(conditioning.dim() - 1):
            keep_mask = keep_mask.unsqueeze(-1)

        return conditioning * keep_mask

    def test_dropout_zeros_entire_samples(self):
        """Dropout zeros entire samples, not individual pixels."""
        torch.manual_seed(42)
        batch_size = 8
        conditioning = torch.ones(batch_size, 1, 64, 64)
        dropout_prob = 0.5

        result = self._apply_conditioning_dropout(
            conditioning, batch_size, dropout_prob
        )

        # Each sample should be either all zeros or all ones
        for i in range(batch_size):
            sample = result[i]
            is_all_zero = (sample == 0).all()
            is_all_one = (sample == 1).all()
            assert is_all_zero or is_all_one, \
                f"Sample {i} should be entirely zero or entirely one, not mixed"

    def test_dropout_respects_probability(self):
        """Dropout rate approximately matches specified probability."""
        torch.manual_seed(123)
        batch_size = 1000
        conditioning = torch.ones(batch_size, 1, 8, 8)
        dropout_prob = 0.15

        result = self._apply_conditioning_dropout(
            conditioning, batch_size, dropout_prob
        )

        # Count dropped samples (all zeros)
        dropped = sum(1 for i in range(batch_size) if (result[i] == 0).all())
        dropout_rate = dropped / batch_size

        # Should be approximately 15% (allow 5% tolerance for randomness)
        assert 0.10 <= dropout_rate <= 0.20, \
            f"Dropout rate {dropout_rate:.2%} should be ~{dropout_prob:.0%}"

    def test_dropout_disabled_when_prob_zero(self):
        """Zero dropout probability leaves conditioning unchanged."""
        conditioning = torch.randn(4, 1, 64, 64)
        original = conditioning.clone()

        result = self._apply_conditioning_dropout(
            conditioning, batch_size=4, dropout_prob=0.0
        )

        assert_tensors_close(result, original, name="Zero dropout")

    def test_dropout_disabled_during_eval(self):
        """Dropout is not applied during evaluation (training=False)."""
        torch.manual_seed(42)
        conditioning = torch.randn(4, 1, 64, 64)
        original = conditioning.clone()

        result = self._apply_conditioning_dropout(
            conditioning, batch_size=4, dropout_prob=0.5, training=False
        )

        assert_tensors_close(result, original, name="Eval mode dropout")

    def test_dropout_works_with_3d_volumes(self):
        """Dropout works correctly with 3D volumes [B, C, D, H, W]."""
        torch.manual_seed(42)
        batch_size = 8
        conditioning = torch.ones(batch_size, 1, 16, 32, 32)  # 3D volume
        dropout_prob = 0.5

        result = self._apply_conditioning_dropout(
            conditioning, batch_size, dropout_prob
        )

        # Verify shape preserved
        assert result.shape == conditioning.shape, \
            f"Shape should be preserved: {conditioning.shape} vs {result.shape}"

        # Each sample should be entirely zero or entirely one
        for i in range(batch_size):
            sample = result[i]
            is_all_zero = (sample == 0).all()
            is_all_one = (sample == 1).all()
            assert is_all_zero or is_all_one, \
                f"3D sample {i} should be entirely zero or entirely one"

    def test_dropout_preserves_non_dropped_values(self):
        """Non-dropped samples retain their exact original values."""
        torch.manual_seed(0)
        batch_size = 4
        conditioning = torch.randn(batch_size, 1, 64, 64)
        original = conditioning.clone()

        result = self._apply_conditioning_dropout(
            conditioning, batch_size, dropout_prob=0.5
        )

        # For non-dropped samples, values should be identical
        for i in range(batch_size):
            if not (result[i] == 0).all():  # Not dropped
                assert_tensors_close(
                    result[i], original[i],
                    name=f"Non-dropped sample {i}"
                )
