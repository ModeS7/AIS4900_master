"""Tests for compression training helpers and ConditioningContext.

Tests cover:
- compute_kl_loss: Pure math, no mocks needed
- prepare_batch: Batch format handling with minimal mocking
- ConditioningContext: Dataclass behavior, properties, helpers
"""
import pytest
import torch
from unittest.mock import Mock
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# TestComputeKlLoss
# ---------------------------------------------------------------------------

class TestComputeKlLoss:
    """Tests for pipeline.compression_training.compute_kl_loss."""

    def test_zero_params_gives_near_zero_kl(self):
        """KL(N(0,1) || N(0,1)) should be ~0."""
        from medgen.pipeline.compression_training import compute_kl_loss
        mean = torch.zeros(2, 4, 8, 8)
        logvar = torch.zeros(2, 4, 8, 8)
        loss = compute_kl_loss(mean, logvar)
        assert loss.ndim == 0  # scalar
        assert abs(loss.item()) < 1e-6

    def test_2d_input_returns_scalar(self):
        """[B, C, H, W] input should return a scalar."""
        from medgen.pipeline.compression_training import compute_kl_loss
        mean = torch.randn(4, 8, 16, 16)
        logvar = torch.randn(4, 8, 16, 16)
        loss = compute_kl_loss(mean, logvar)
        assert loss.ndim == 0

    def test_3d_input_returns_scalar(self):
        """[B, C, D, H, W] input should return a scalar."""
        from medgen.pipeline.compression_training import compute_kl_loss
        mean = torch.randn(2, 4, 8, 8, 8)
        logvar = torch.randn(2, 4, 8, 8, 8)
        loss = compute_kl_loss(mean, logvar)
        assert loss.ndim == 0

    def test_nonzero_mean_increases_kl(self):
        """Non-zero mean should give KL > 0."""
        from medgen.pipeline.compression_training import compute_kl_loss
        mean = torch.ones(2, 4, 8, 8) * 3.0
        logvar = torch.zeros(2, 4, 8, 8)
        loss = compute_kl_loss(mean, logvar)
        assert loss.item() > 0.0

    def test_batch_averaging(self):
        """KL should average over batch, giving same result for repeated samples."""
        from medgen.pipeline.compression_training import compute_kl_loss
        mean_single = torch.randn(1, 4, 8, 8)
        logvar_single = torch.randn(1, 4, 8, 8)
        loss_single = compute_kl_loss(mean_single, logvar_single)

        # Stack same sample 4 times
        mean_batch = mean_single.expand(4, -1, -1, -1)
        logvar_batch = logvar_single.expand(4, -1, -1, -1)
        loss_batch = compute_kl_loss(mean_batch, logvar_batch)

        assert abs(loss_single.item() - loss_batch.item()) < 1e-5


# ---------------------------------------------------------------------------
# TestPrepareBatch
# ---------------------------------------------------------------------------

def _make_trainer_mock(spatial_dims=2, in_channels=1, image_keys=None):
    """Create minimal trainer mock for prepare_batch tests."""
    trainer = Mock()
    trainer.device = torch.device('cpu')
    trainer.spatial_dims = spatial_dims

    cfg = OmegaConf.create({
        'mode': {
            'in_channels': in_channels,
            'image_keys': image_keys or ['t1_pre'],
        }
    })
    trainer.cfg = cfg
    return trainer


class TestPrepareBatch:
    """Tests for pipeline.compression_training.prepare_batch."""

    def test_2d_tuple_returns_both_on_device(self):
        """2D tuple (images, mask) -> both tensors returned."""
        from medgen.pipeline.compression_training import prepare_batch
        trainer = _make_trainer_mock(spatial_dims=2)
        images = torch.randn(2, 1, 64, 64)
        mask = torch.randn(2, 1, 64, 64)
        result_img, result_mask = prepare_batch(trainer, (images, mask))
        assert result_img.shape == (2, 1, 64, 64)
        assert result_mask.shape == (2, 1, 64, 64)

    def test_2d_dict_concatenates_channels(self):
        """2D dict batch -> image_keys concatenated along channel dim."""
        from medgen.pipeline.compression_training import prepare_batch
        trainer = _make_trainer_mock(
            spatial_dims=2, in_channels=2,
            image_keys=['t1_pre', 't1_gd'],
        )
        batch = {
            't1_pre': torch.randn(2, 1, 64, 64),
            't1_gd': torch.randn(2, 1, 64, 64),
            'seg': torch.randn(2, 1, 64, 64),
        }
        images, mask = prepare_batch(trainer, batch)
        assert images.shape == (2, 2, 64, 64)  # 2 channels concatenated
        assert mask.shape == (2, 1, 64, 64)

    def test_2d_tensor_extra_channels_splits(self):
        """2D tensor with extra channels -> split into image + mask."""
        from medgen.pipeline.compression_training import prepare_batch
        trainer = _make_trainer_mock(spatial_dims=2, in_channels=1)
        # 2 channels: 1 image + 1 mask
        tensor = torch.randn(2, 2, 64, 64)
        images, mask = prepare_batch(trainer, tensor)
        assert images.shape == (2, 1, 64, 64)
        assert mask.shape == (2, 1, 64, 64)

    def test_2d_tensor_exact_channels_returns_none_mask(self):
        """2D tensor with exact channels -> (tensor, None)."""
        from medgen.pipeline.compression_training import prepare_batch
        trainer = _make_trainer_mock(spatial_dims=2, in_channels=1)
        tensor = torch.randn(2, 1, 64, 64)
        images, mask = prepare_batch(trainer, tensor)
        assert images.shape == (2, 1, 64, 64)
        assert mask is None

    def test_3d_dict_returns_both(self):
        """3D dict batch -> image and seg returned."""
        from medgen.pipeline.compression_training import prepare_batch
        trainer = _make_trainer_mock(spatial_dims=3)
        batch = {
            'image': torch.randn(1, 1, 32, 32, 32),
            'seg': torch.randn(1, 1, 32, 32, 32),
        }
        images, mask = prepare_batch(trainer, batch)
        assert images.shape == (1, 1, 32, 32, 32)
        assert mask.shape == (1, 1, 32, 32, 32)

    def test_3d_tuple_returns_both(self):
        """3D tuple batch -> image and mask returned."""
        from medgen.pipeline.compression_training import prepare_batch
        trainer = _make_trainer_mock(spatial_dims=3)
        img = torch.randn(1, 1, 32, 32, 32)
        seg = torch.randn(1, 1, 32, 32, 32)
        images, mask = prepare_batch(trainer, (img, seg))
        assert images.shape == (1, 1, 32, 32, 32)
        assert mask.shape == (1, 1, 32, 32, 32)


# ---------------------------------------------------------------------------
# TestConditioningContext
# ---------------------------------------------------------------------------

class TestConditioningContext:
    """Tests for diffusion.conditioning.ConditioningContext."""

    def test_empty_defaults(self):
        """empty() should create context with all None/defaults."""
        from medgen.diffusion.conditioning import ConditioningContext
        ctx = ConditioningContext.empty()
        assert ctx.omega is None
        assert ctx.mode_id is None
        assert ctx.size_bins is None
        assert ctx.bin_maps is None
        assert ctx.image_conditioning is None
        assert ctx.cfg_scale == 1.0
        assert ctx.cfg_scale_end is None
        assert ctx.latent_channels == 1

    def test_use_cfg_property(self):
        """use_cfg is True only when scale > 1.0."""
        from medgen.diffusion.conditioning import ConditioningContext
        assert not ConditioningContext(cfg_scale=1.0).use_cfg
        assert ConditioningContext(cfg_scale=2.0).use_cfg
        assert not ConditioningContext(cfg_scale=0.5).use_cfg

    def test_use_dynamic_cfg_property(self):
        """use_dynamic_cfg requires different end scale."""
        from medgen.diffusion.conditioning import ConditioningContext
        # No end -> not dynamic
        assert not ConditioningContext(cfg_scale=2.0).use_dynamic_cfg
        # Same end -> not dynamic
        assert not ConditioningContext(cfg_scale=2.0, cfg_scale_end=2.0).use_dynamic_cfg
        # Different end -> dynamic
        assert ConditioningContext(cfg_scale=2.0, cfg_scale_end=1.0).use_dynamic_cfg

    def test_get_cfg_scale_at_step_constant(self):
        """Constant CFG returns same scale at every step."""
        from medgen.diffusion.conditioning import ConditioningContext
        ctx = ConditioningContext(cfg_scale=3.0)
        assert ctx.get_cfg_scale_at_step(0, 25) == 3.0
        assert ctx.get_cfg_scale_at_step(12, 25) == 3.0
        assert ctx.get_cfg_scale_at_step(24, 25) == 3.0

    def test_get_cfg_scale_at_step_interpolation(self):
        """Dynamic CFG linearly interpolates between start and end."""
        from medgen.diffusion.conditioning import ConditioningContext
        ctx = ConditioningContext(cfg_scale=4.0, cfg_scale_end=2.0)
        assert ctx.get_cfg_scale_at_step(0, 25) == pytest.approx(4.0)
        assert ctx.get_cfg_scale_at_step(24, 25) == pytest.approx(2.0)
        # Midpoint
        assert ctx.get_cfg_scale_at_step(12, 25) == pytest.approx(3.0, abs=0.1)

    def test_has_embedding_conditioning(self):
        """has_embedding_conditioning checks omega, mode_id, size_bins."""
        from medgen.diffusion.conditioning import ConditioningContext
        assert not ConditioningContext.empty().has_embedding_conditioning
        assert ConditioningContext(omega=torch.randn(2, 5)).has_embedding_conditioning
        assert ConditioningContext(mode_id=torch.tensor([0, 1])).has_embedding_conditioning
        assert ConditioningContext(size_bins=torch.randn(2, 7)).has_embedding_conditioning

    def test_has_spatial_conditioning(self):
        """has_spatial_conditioning checks bin_maps and image_conditioning."""
        from medgen.diffusion.conditioning import ConditioningContext
        assert not ConditioningContext.empty().has_spatial_conditioning
        assert ConditioningContext(bin_maps=torch.randn(2, 3, 64, 64)).has_spatial_conditioning
        assert ConditioningContext(
            image_conditioning=torch.randn(2, 1, 64, 64)
        ).has_spatial_conditioning

    def test_get_uncond_tensors_zeros(self):
        """get_uncond_tensors returns zeros matching shapes."""
        from medgen.diffusion.conditioning import ConditioningContext
        ctx = ConditioningContext(
            size_bins=torch.ones(2, 7),
            bin_maps=torch.ones(2, 3, 64, 64),
            image_conditioning=torch.ones(2, 1, 64, 64),
        )
        uncond = ctx.get_uncond_tensors()
        assert torch.all(uncond['size_bins'] == 0)
        assert uncond['size_bins'].shape == (2, 7)
        assert torch.all(uncond['bin_maps'] == 0)
        assert torch.all(uncond['image_conditioning'] == 0)

    def test_get_uncond_tensors_skips_none(self):
        """get_uncond_tensors omits keys for None fields."""
        from medgen.diffusion.conditioning import ConditioningContext
        ctx = ConditioningContext.empty()
        uncond = ctx.get_uncond_tensors()
        assert len(uncond) == 0

    def test_to_device_returns_new_instance(self):
        """to_device returns a new context (immutable pattern)."""
        from medgen.diffusion.conditioning import ConditioningContext
        ctx = ConditioningContext(
            size_bins=torch.randn(2, 7),
            cfg_scale=3.0,
        )
        moved = ctx.to_device(torch.device('cpu'))
        assert moved is not ctx
        assert moved.cfg_scale == 3.0
        assert moved.size_bins is not None
        assert moved.size_bins.shape == (2, 7)

    def test_with_cfg_returns_new_instance(self):
        """with_cfg returns new context with updated CFG, preserving other fields."""
        from medgen.diffusion.conditioning import ConditioningContext
        ctx = ConditioningContext(
            size_bins=torch.randn(2, 7),
            cfg_scale=1.0,
        )
        new_ctx = ctx.with_cfg(5.0, cfg_scale_end=2.0)
        assert new_ctx is not ctx
        assert new_ctx.cfg_scale == 5.0
        assert new_ctx.cfg_scale_end == 2.0
        assert new_ctx.size_bins is ctx.size_bins  # same tensor ref

    def test_with_latent_channels_returns_new_instance(self):
        """with_latent_channels returns new context with updated channels."""
        from medgen.diffusion.conditioning import ConditioningContext
        ctx = ConditioningContext(latent_channels=1, cfg_scale=2.0)
        new_ctx = ctx.with_latent_channels(4)
        assert new_ctx is not ctx
        assert new_ctx.latent_channels == 4
        assert new_ctx.cfg_scale == 2.0  # preserved

    def test_from_batch_populates_fields(self):
        """from_batch should populate fields from BatchData."""
        from medgen.diffusion.batch_data import BatchData
        from medgen.diffusion.conditioning import ConditioningContext

        bd = BatchData(
            images=torch.randn(4, 1, 64, 64),
            labels=torch.randn(4, 1, 64, 64),
            size_bins=torch.randn(4, 7),
            bin_maps=torch.randn(4, 3, 64, 64),
            mode_id=torch.tensor([0, 1, 0, 1]),
        )
        ctx = ConditioningContext.from_batch(bd, cfg_scale=2.5, latent_channels=4)
        assert ctx.size_bins is bd.size_bins
        assert ctx.bin_maps is bd.bin_maps
        assert ctx.mode_id is bd.mode_id
        assert ctx.image_conditioning is bd.labels
        assert ctx.cfg_scale == 2.5
        assert ctx.latent_channels == 4

    def test_frozen_immutability(self):
        """Frozen dataclass should prevent attribute assignment."""
        from medgen.diffusion.conditioning import ConditioningContext
        ctx = ConditioningContext.empty()
        with pytest.raises(AttributeError):
            ctx.cfg_scale = 5.0
