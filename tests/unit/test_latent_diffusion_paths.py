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
