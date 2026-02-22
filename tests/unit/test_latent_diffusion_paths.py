"""Tests for latent diffusion code paths.

These tests verify latent space diffusion functionality including
the LatentSpace class, config validation, and visualization paths.
"""
import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from omegaconf import OmegaConf


class TestLatentSpace:
    """Tests for LatentSpace and PixelSpace classes."""

    def test_pixel_space_identity_encode(self):
        """Verify PixelSpace.encode() is identity transform."""
        from medgen.diffusion.spaces import PixelSpace

        space = PixelSpace()
        x = torch.randn(2, 1, 64, 64)

        result = space.encode(x)
        assert torch.equal(result, x)

    def test_pixel_space_identity_decode(self):
        """Verify PixelSpace.decode() is identity transform."""
        from medgen.diffusion.spaces import PixelSpace

        space = PixelSpace()
        x = torch.randn(2, 1, 64, 64)

        result = space.decode(x)
        assert torch.equal(result, x)

    def test_pixel_space_scale_factor_is_one(self):
        """Verify PixelSpace reports scale_factor=1."""
        from medgen.diffusion.spaces import PixelSpace

        space = PixelSpace()
        assert space.scale_factor == 1

    def test_pixel_space_latent_channels_is_one(self):
        """Verify PixelSpace reports latent_channels=1."""
        from medgen.diffusion.spaces import PixelSpace

        space = PixelSpace()
        assert space.latent_channels == 1

    def test_pixel_space_get_latent_channels_returns_input(self):
        """Verify PixelSpace.get_latent_channels returns input unchanged."""
        from medgen.diffusion.spaces import PixelSpace

        space = PixelSpace()
        assert space.get_latent_channels(1) == 1
        assert space.get_latent_channels(3) == 3

    def test_latent_space_stores_scale_factor(self):
        """Verify LatentSpace reports correct scale factor."""
        from medgen.diffusion.spaces import LatentSpace

        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=iter([]))

        space = LatentSpace(
            compression_model=mock_model,
            device=torch.device('cpu'),
            scale_factor=8,
            latent_channels=4,
        )

        assert space.scale_factor == 8
        assert space.latent_channels == 4

    def test_latent_space_encode_batch_handles_dict(self):
        """Verify LatentSpace.encode_batch handles dict inputs."""
        from medgen.diffusion.spaces import LatentSpace

        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=iter([]))
        mock_model.encode = Mock(return_value=(torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)))

        space = LatentSpace(
            compression_model=mock_model,
            device=torch.device('cpu'),
            scale_factor=8,
            latent_channels=4,
            compression_type='vae',
            deterministic=True,
        )

        data = {
            't1_pre': torch.randn(2, 1, 64, 64),
            't1_gd': torch.randn(2, 1, 64, 64),
        }

        result = space.encode_batch(data)
        assert isinstance(result, dict)
        assert 't1_pre' in result
        assert 't1_gd' in result


class TestLatentConfigValidation:
    """Tests for latent config validation."""

    def test_validate_latent_config_skips_when_disabled(self):
        """Verify validation skips when latent.enabled=False."""
        from medgen.core.validation import validate_latent_config

        cfg = OmegaConf.create({
            'latent': {'enabled': False},
        })

        errors = validate_latent_config(cfg)
        assert len(errors) == 0

    def test_validate_latent_config_missing_checkpoint(self):
        """Verify error when latent enabled without checkpoint."""
        from medgen.core.validation import validate_latent_config

        cfg = OmegaConf.create({
            'latent': {'enabled': True, 'compression_checkpoint': None},
        })

        errors = validate_latent_config(cfg)
        assert len(errors) == 1
        assert 'checkpoint' in errors[0].lower()

    def test_validate_latent_config_nonexistent_checkpoint(self):
        """Verify error when checkpoint file doesn't exist."""
        from medgen.core.validation import validate_latent_config

        cfg = OmegaConf.create({
            'latent': {
                'enabled': True,
                'compression_checkpoint': '/nonexistent/path/checkpoint.pt',
            },
        })

        errors = validate_latent_config(cfg)
        assert len(errors) == 1
        assert 'not found' in errors[0].lower()

    def test_validate_latent_config_empty_checkpoint_string(self):
        """Verify error when checkpoint is empty string."""
        from medgen.core.validation import validate_latent_config

        cfg = OmegaConf.create({
            'latent': {
                'enabled': True,
                'compression_checkpoint': '',
            },
        })

        errors = validate_latent_config(cfg)
        assert len(errors) == 1
        assert 'checkpoint' in errors[0].lower()


class TestLatentVisualization:
    """Tests for latent space visualization."""

    def test_visualize_samples_3d_logs_latent_before_decode(self):
        """Verify latent samples are logged before decoding for latent space."""
        from medgen.pipeline.visualization import visualize_samples_3d

        trainer = Mock()
        trainer._cached_train_batch = {
            'images': torch.randn(2, 4, 8, 8, 8),
            'labels': None,
            'is_latent': True,
        }
        trainer.mode = Mock()
        trainer.mode.is_conditional = False
        trainer.device = torch.device('cpu')
        trainer.space = Mock()
        trainer.space.scale_factor = 4  # Latent space
        trainer.space.needs_decode = True
        trainer.space.latent_channels = 4
        trainer.space.encode = Mock(return_value=torch.randn(2, 4, 8, 8, 8))
        trainer.space.decode = Mock(return_value=torch.randn(2, 1, 32, 32, 32))
        trainer._gen_metrics_config = None
        trainer._unified_metrics = Mock()
        trainer.use_controlnet = False
        trainer.controlnet_stage1 = False
        trainer.use_size_bin_embedding = False

        # Mock strategy.generate
        trainer.strategy = Mock()
        trainer.strategy.generate = Mock(return_value=torch.randn(2, 4, 8, 8, 8))

        visualize_samples_3d(trainer, Mock(), epoch=0)

        # Should log latent samples before decoding
        trainer._unified_metrics.log_latent_samples.assert_called_once()
        # Should also decode
        trainer.space.decode.assert_called_once()

    def test_visualize_samples_3d_skips_latent_log_for_pixel_space(self):
        """Verify latent log is skipped for pixel space (scale_factor=1)."""
        from medgen.pipeline.visualization import visualize_samples_3d

        trainer = Mock()
        trainer._cached_train_batch = {
            'images': torch.randn(2, 1, 32, 64, 64),
            'labels': None,
        }
        trainer.mode = Mock()
        trainer.mode.is_conditional = False
        trainer.device = torch.device('cpu')
        trainer.space = Mock()
        trainer.space.scale_factor = 1  # Pixel space
        trainer.space.needs_decode = False
        trainer.space.latent_channels = 1
        trainer._gen_metrics_config = None
        trainer._unified_metrics = Mock()
        trainer.use_controlnet = False
        trainer.controlnet_stage1 = False
        trainer.use_size_bin_embedding = False

        # Mock strategy.generate
        trainer.strategy = Mock()
        trainer.strategy.generate = Mock(return_value=torch.randn(2, 1, 32, 64, 64))

        visualize_samples_3d(trainer, Mock(), epoch=0)

        # Should NOT log latent samples (pixel space)
        trainer._unified_metrics.log_latent_samples.assert_not_called()
        # Should NOT call decode (pixel space)
        trainer.space.decode.assert_not_called()


class TestLatentSpaceDetection:
    """Tests for latent space auto-detection."""

    def test_detect_latent_channels_from_attribute(self):
        """Verify latent_channels detection from model attribute."""
        from medgen.diffusion.spaces import LatentSpace

        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=iter([]))
        mock_model.latent_channels = 8

        space = LatentSpace(
            compression_model=mock_model,
            device=torch.device('cpu'),
            scale_factor=8,
            # Don't provide latent_channels - should auto-detect
        )

        assert space.latent_channels == 8

    def test_detect_latent_channels_fallback(self):
        """Verify latent_channels falls back to 4 when not detectable."""
        from medgen.diffusion.spaces import LatentSpace

        mock_model = Mock(spec=[])  # Empty spec - no attributes
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=iter([]))

        space = LatentSpace(
            compression_model=mock_model,
            device=torch.device('cpu'),
            scale_factor=8,
            # Don't provide latent_channels - should fall back to 4
        )

        assert space.latent_channels == 4

    def test_detect_scale_factor_from_config(self):
        """Verify scale_factor detection from model config dict."""
        from medgen.diffusion.spaces import LatentSpace

        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=iter([]))
        mock_model.config = {'spatial_compression_ratio': 32}

        space = LatentSpace(
            compression_model=mock_model,
            device=torch.device('cpu'),
            latent_channels=4,
            compression_type='dcae',
            # Don't provide scale_factor - should auto-detect
        )

        assert space.scale_factor == 32


class TestLatentEncodeDecode:
    """Tests for latent encode/decode operations."""

    def test_latent_space_encode_vae_deterministic(self):
        """Verify VAE deterministic encoding returns mean only."""
        from medgen.diffusion.spaces import LatentSpace

        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=iter([]))
        mu = torch.randn(2, 4, 8, 8)
        logvar = torch.randn(2, 4, 8, 8)
        mock_model.encode = Mock(return_value=(mu, logvar))

        space = LatentSpace(
            compression_model=mock_model,
            device=torch.device('cpu'),
            scale_factor=8,
            latent_channels=4,
            compression_type='vae',
            deterministic=True,
        )

        x = torch.randn(2, 1, 64, 64)
        result = space.encode(x)

        assert torch.equal(result, mu)

    def test_latent_space_encode_dimension_check(self):
        """Verify encode raises error for wrong tensor dimensions."""
        from medgen.diffusion.spaces import LatentSpace

        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.parameters = Mock(return_value=iter([]))

        space = LatentSpace(
            compression_model=mock_model,
            device=torch.device('cpu'),
            scale_factor=8,
            latent_channels=4,
            spatial_dims=2,  # Expects 4D
        )

        # Try to encode 5D tensor (3D volume) when expecting 2D
        x_5d = torch.randn(2, 1, 32, 64, 64)

        with pytest.raises(ValueError, match="expects 4D input"):
            space.encode(x_5d)


class TestSharedDecoderMetricsBug:
    """Tests verifying that quality metrics are honest (not inflated by shared decoder).

    When both prediction and ground truth pass through the same lossy decoder,
    decoder artifacts cancel out and metrics appear artificially perfect.
    The correct approach is to compare decoded predictions against original pixels.
    """

    @staticmethod
    def _lossy_decode(z: torch.Tensor) -> torch.Tensor:
        """Simulate a lossy decoder that adds fixed compression artifacts.

        Uses a deterministic hash-like distortion to mimic quantization effects
        from VQ-VAE codebook lookup. The key property: same input always produces
        the same output, which causes artifacts to cancel in shared decoder comparison.
        """
        # Simulate quantization + reconstruction noise (deterministic per-pixel)
        artifact = 0.1 * torch.sin(z * 20.0) + 0.05 * torch.cos(z * 37.0)
        return (z + artifact).clamp(0, 1)

    def test_shared_decoder_metrics_are_higher_than_honest(self):
        """Verify shared decoder gives inflated metrics vs honest pixel comparison.

        This is the core test: when both prediction and GT go through the same
        decoder, artifacts cancel out and metrics appear better than they really are.
        """
        from medgen.metrics.quality import compute_psnr

        torch.manual_seed(42)
        original_pixels = torch.rand(2, 1, 64, 64)
        gt_latent = original_pixels
        # Near-perfect prediction (simulates low noise timestep)
        pred_latent = gt_latent + 0.01 * torch.randn_like(gt_latent)

        # Shared decoder path: decode(pred) vs decode(gt) — artifacts cancel
        gt_decoded = self._lossy_decode(gt_latent)
        pred_decoded = self._lossy_decode(pred_latent)
        psnr_shared = compute_psnr(pred_decoded, gt_decoded)

        # Honest pixel GT path: decode(pred) vs original — artifacts visible
        psnr_honest = compute_psnr(pred_decoded, original_pixels)

        assert psnr_shared > psnr_honest, (
            f"Shared decoder PSNR ({psnr_shared:.2f} dB) should be higher "
            f"than honest ({psnr_honest:.2f} dB). If equal, decoder has no artifacts."
        )
        # The gap should be substantial (>5 dB)
        gap = psnr_shared - psnr_honest
        assert gap > 3, f"PSNR gap ({gap:.2f} dB) too small to detect shared decoder bug"

    def test_honest_psnr_bounded_by_compression_quality(self):
        """Honest PSNR reflects decoder quality, not prediction quality."""
        from medgen.metrics.quality import compute_psnr

        torch.manual_seed(42)
        original_pixels = torch.rand(2, 1, 64, 64)

        # Even with PERFECT prediction (pred == gt exactly)
        perfect_decoded = self._lossy_decode(original_pixels)
        psnr_perfect = compute_psnr(perfect_decoded, original_pixels)

        # PSNR is bounded by the decoder's reconstruction quality
        # (our lossy_decode has ~10% error, so PSNR ≈ 20-25 dB)
        assert psnr_perfect < 35, (
            f"Even perfect prediction should give PSNR < 35 dB "
            f"due to decoder artifacts, got {psnr_perfect:.2f}"
        )

    def test_shared_decoder_psnr_inflated_to_near_perfect(self):
        """Shared decoder inflates PSNR to near-perfect at low noise."""
        from medgen.metrics.quality import compute_psnr

        torch.manual_seed(42)
        gt_latent = torch.rand(2, 1, 64, 64)
        # Very small prediction error (low noise timestep)
        pred_latent = gt_latent + 0.005 * torch.randn_like(gt_latent)

        gt_decoded = self._lossy_decode(gt_latent)
        pred_decoded = self._lossy_decode(pred_latent)
        psnr_shared = compute_psnr(pred_decoded, gt_decoded)

        # Shared decoder: PSNR should be very high (>35 dB) even though
        # decode(gt) != original_pixels. This is the misleading metric.
        assert psnr_shared > 35, (
            f"Shared decoder with small prediction error should give "
            f"PSNR > 35 dB, got {psnr_shared:.2f}"
        )


class TestLatentStatsVersion:
    """Tests for latent normalization stats versioning."""

    def test_stale_stats_are_recomputed(self, tmp_path):
        """Verify old stats (no version) trigger recomputation."""
        from medgen.data.loaders.latent import (
            LATENT_STATS_VERSION,
            LatentCacheBuilder,
            LatentDataset,
        )

        # Create fake cached latent files with known distribution
        torch.manual_seed(42)
        for i in range(10):
            latent = torch.randn(4, 8, 8, 8) * 0.5 + 0.1  # std≈0.5, mean≈0.1
            torch.save({'latent': latent}, tmp_path / f"sample_{i:04d}.pt")

        # Write metadata with stale stats (no version = old buggy code)
        import json
        metadata = {
            'latent_shift': [0.0, 0.0, 0.0, 0.0],
            'latent_scale': [0.001, 0.001, 0.001, 0.001],  # Buggy: way too small
            'num_samples': 10,
        }
        with open(tmp_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

        # LatentDataset should detect stale stats and recompute
        ds = LatentDataset(str(tmp_path), mode='bravo')

        # Verify stats were recomputed (scale should be ≈0.5, not 0.001)
        assert ds._scale is not None
        for ch_scale in ds._scale.tolist():
            assert ch_scale > 0.1, f"Scale {ch_scale} too small — stale stats not replaced"

        # Verify metadata was updated with version
        with open(tmp_path / 'metadata.json') as f:
            updated = json.load(f)
        assert updated.get('latent_stats_version') == LATENT_STATS_VERSION

    def test_current_version_stats_not_recomputed(self, tmp_path):
        """Verify current-version stats are used directly."""
        from medgen.data.loaders.latent import (
            LATENT_STATS_VERSION,
            LatentDataset,
        )

        # Create one fake latent file (needed for dataset)
        torch.save({'latent': torch.randn(4, 8, 8, 8)}, tmp_path / "sample_0000.pt")

        # Write metadata with current-version stats
        import json
        metadata = {
            'latent_shift': [0.1, 0.2, 0.3, 0.4],
            'latent_scale': [0.5, 0.6, 0.7, 0.8],
            'latent_stats_version': LATENT_STATS_VERSION,
            'num_samples': 1,
        }
        with open(tmp_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

        ds = LatentDataset(str(tmp_path), mode='bravo')

        # Verify the stored stats were used as-is (float32 precision)
        assert torch.allclose(ds._shift, torch.tensor([0.1, 0.2, 0.3, 0.4]))
        assert torch.allclose(ds._scale, torch.tensor([0.5, 0.6, 0.7, 0.8]))

    def test_normalized_latents_have_unit_scale(self, tmp_path):
        """Verify correctly normalized latents have std ≈ 1."""
        from medgen.data.loaders.latent import LatentDataset

        # Create fake latents with known distribution
        torch.manual_seed(42)
        for i in range(20):
            latent = torch.randn(4, 8, 8, 8) * 0.5 + 0.1
            torch.save({'latent': latent}, tmp_path / f"sample_{i:04d}.pt")

        # No metadata → forces fresh computation
        ds = LatentDataset(str(tmp_path), mode='bravo')

        # Load all and check distribution
        all_latents = torch.stack([ds[i]['latent'] for i in range(len(ds))])
        overall_std = all_latents.std().item()

        assert 0.5 < overall_std < 2.0, (
            f"Normalized latent std={overall_std:.3f}, expected ≈1.0"
        )
