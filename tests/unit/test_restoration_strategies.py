"""Unit tests for restoration/bridge diffusion strategies.

Covers IRSDEStrategy, ResfusionStrategy, BridgeStrategy — added late in the
project and initially lacking test coverage (findings L-02 / C-03).

Each strategy is smoke-tested for:
- setup_scheduler does not crash and populates expected attributes
- add_noise returns tensor matching clean_images shape (dtype float)
- sample_timesteps returns batch-sized long tensor in the valid range
- compute_loss returns (scalar_loss, predicted_clean) tuple with the right shapes
- Factory registration (import + constructor)
- For ResFusion: stateless compute_loss (no stored instance state required after add_noise)
"""

from __future__ import annotations

import torch

from medgen.diffusion import (
    BridgeStrategy,
    IRSDEStrategy,
    ResfusionStrategy,
)


# =============================================================================
# IR-SDE
# =============================================================================


class TestIRSDEStrategy:
    def _make(self) -> IRSDEStrategy:
        s = IRSDEStrategy()
        s.setup_scheduler(num_timesteps=12, image_size=16, depth_size=8, spatial_dims=3)
        return s

    def test_setup_scheduler_populates_schedules(self):
        s = self._make()
        assert s.T == 12
        assert s._thetas is not None and s._thetas.numel() == 13  # T+1
        assert s._thetas_cumsum is not None
        # cumsum[0] should be 0 (subtracted-off offset)
        assert torch.isclose(s._thetas_cumsum[0], torch.tensor(0.0))
        assert s._dt > 0

    def test_add_noise_shape(self):
        s = self._make()
        x0 = torch.rand(2, 1, 8, 16, 16)
        degraded = torch.rand(2, 1, 8, 16, 16)
        t = torch.tensor([3, 8], dtype=torch.long)
        x_t = s.add_noise(x0, degraded, t)
        assert x_t.shape == x0.shape
        assert x_t.dtype == torch.float32

    def test_sample_timesteps_in_range(self):
        s = self._make()
        x0 = torch.rand(4, 1, 8, 16, 16)
        t = s.sample_timesteps(x0)
        assert t.shape == (4,)
        assert t.dtype == torch.long
        # IR-SDE uses 1-based timesteps in [1, T]
        assert int(t.min()) >= 1
        assert int(t.max()) <= s.T

    def test_num_timesteps_property(self):
        s = self._make()
        assert s.num_timesteps == 12

    def test_loss_name(self):
        assert IRSDEStrategy().loss_name == 'L1'


# =============================================================================
# ResFusion
# =============================================================================


class TestResfusionStrategy:
    def _make(self) -> ResfusionStrategy:
        s = ResfusionStrategy()
        s.setup_scheduler(num_timesteps=12, image_size=16, depth_size=8, spatial_dims=3)
        return s

    def test_setup_scheduler_populates_alphas(self):
        s = self._make()
        assert s._alphas is not None
        assert s._betas is not None
        assert s._sqrt_alphas_hat is not None
        assert s._sqrt_1m_alphas_hat is not None

    def test_add_noise_shape(self):
        s = self._make()
        x0 = torch.rand(2, 1, 8, 16, 16)
        degraded = torch.rand(2, 1, 8, 16, 16)
        t = torch.tensor([2, 6], dtype=torch.long)
        x_t = s.add_noise(x0, degraded, t)
        assert x_t.shape == x0.shape

    def test_add_noise_output_in_01_range_center_bias(self):
        """Output should broadly live in [0, 1] per the to_01() conversion."""
        s = self._make()
        x0 = torch.rand(1, 1, 8, 16, 16)
        degraded = torch.rand(1, 1, 8, 16, 16)
        t = torch.tensor([5], dtype=torch.long)
        x_t = s.add_noise(x0, degraded, t)
        # Allow some spillover from gaussian noise but centre should be reasonable
        assert x_t.mean() < 1.2 and x_t.mean() > -0.2

    def test_compute_loss_stateless_after_add_noise(self):
        """Regression for H-05: compute_loss must NOT depend on stored instance state.

        We call add_noise once, deliberately clobber any stored state, then call
        compute_loss. It should still produce a finite loss because epsilon/R are
        derived from inputs.
        """
        s = self._make()
        x0 = torch.rand(1, 1, 8, 16, 16)
        degraded = torch.rand(1, 1, 8, 16, 16)
        t = torch.tensor([5], dtype=torch.long)
        x_t = s.add_noise(x0, degraded, t)

        # Explicitly remove any old stored state that might have existed historically
        for attr in ('_last_epsilon', '_last_R'):
            if hasattr(s, attr):
                setattr(s, attr, None)

        pred = torch.randn_like(x0)
        loss, predicted_clean = s.compute_loss(pred, x0, degraded, x_t, t)
        assert torch.is_tensor(loss)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert predicted_clean.shape == x0.shape


# =============================================================================
# Bridge
# =============================================================================


class TestBridgeStrategy:
    def _make(self) -> BridgeStrategy:
        s = BridgeStrategy()
        s.setup_scheduler(num_timesteps=1000, image_size=16, depth_size=8, spatial_dims=3)
        return s

    def test_setup_scheduler(self):
        s = self._make()
        assert s._num_train_timesteps == 1000
        assert s.scheduler is None  # Bridge doesn't use a MONAI scheduler

    def test_add_noise_shape(self):
        s = self._make()
        x0 = torch.rand(2, 1, 8, 16, 16)
        x1 = torch.rand(2, 1, 8, 16, 16)  # noise endpoint
        t = torch.tensor([100, 500], dtype=torch.long)
        x_t = s.add_noise(x0, x1, t)
        assert x_t.shape == x0.shape

    def test_compute_target_is_clean(self):
        """Bridge predicts x₀, so compute_target returns clean_images unchanged."""
        s = self._make()
        x0 = torch.rand(1, 1, 4, 8, 8)
        x1 = torch.rand(1, 1, 4, 8, 8)
        target = s.compute_target(x0, x1)
        assert torch.allclose(target, x0)

    def test_compute_loss_returns_mse_and_predicted(self):
        s = self._make()
        x0 = torch.rand(1, 1, 4, 8, 8)
        x1 = torch.rand(1, 1, 4, 8, 8)
        t = torch.tensor([400], dtype=torch.long)
        x_t = s.add_noise(x0, x1, t)
        pred = x0 + 0.01 * torch.randn_like(x0)
        loss, predicted_clean = s.compute_loss(pred, x0, x1, x_t, t)
        assert torch.is_tensor(loss) and loss.dim() == 0
        assert torch.isfinite(loss)
        assert predicted_clean.shape == x0.shape

    def test_sample_timesteps_in_range(self):
        s = self._make()
        x0 = torch.rand(8, 1, 4, 8, 8)
        t = s.sample_timesteps(x0)
        assert t.shape == (8,)
        # Uniform in [1, num_train_timesteps-1]
        assert int(t.min()) >= 1
        assert int(t.max()) <= s.num_timesteps - 1


# =============================================================================
# Factory registration (C-03 regression)
# =============================================================================


def test_create_strategy_registers_all_five():
    """Regression for C-03: trainer factory must register all strategies
    declared valid by StrategyConfig (ddpm/rflow/bridge/irsde/resfusion).
    """
    from medgen.diffusion import DDPMStrategy, RFlowStrategy
    from medgen.pipeline.diffusion_trainer_base import DiffusionTrainerBase

    strategies = {
        'ddpm': DDPMStrategy,
        'rflow': RFlowStrategy,
        'bridge': BridgeStrategy,
        'irsde': IRSDEStrategy,
        'resfusion': ResfusionStrategy,
    }
    # Minimal stand-in `self` — _create_strategy only uses the name arg for dispatch.
    class _Dummy:
        pass
    dummy = _Dummy()
    for name, expected_type in strategies.items():
        result = DiffusionTrainerBase._create_strategy(dummy, name)
        assert isinstance(result, expected_type), (
            f"Factory returned wrong type for {name}: {type(result).__name__}"
        )


def test_create_strategy_unknown_name_raises():
    from medgen.pipeline.diffusion_trainer_base import DiffusionTrainerBase
    import pytest

    class _Dummy:
        pass

    with pytest.raises(ValueError, match="Unknown strategy"):
        DiffusionTrainerBase._create_strategy(_Dummy(), 'nope')
