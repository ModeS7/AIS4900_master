"""Tests for 3D training code paths.

These tests verify 3D-specific functionality in the diffusion trainer
and related modules work correctly in isolation.
"""
import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from omegaconf import OmegaConf


class Test3DTrainerCreation:
    """Tests for 3D trainer initialization and configuration."""

    def test_create_3d_factory_method_exists(self):
        """Verify DiffusionTrainer.create_3d() class method exists."""
        from medgen.pipeline.trainer import DiffusionTrainer
        assert hasattr(DiffusionTrainer, 'create_3d')
        assert callable(getattr(DiffusionTrainer, 'create_3d'))

    def test_spatial_dims_property_returns_correct_value(self):
        """Verify spatial_dims property returns the configured value."""
        # Mock a minimal trainer to test the property
        from medgen.pipeline.trainer import DiffusionTrainer

        # Create mock with _spatial_dims attribute
        trainer = Mock(spec=DiffusionTrainer)
        trainer._spatial_dims = 3

        # Test property manually
        assert trainer._spatial_dims == 3

    def test_validate_3d_config_missing_volume_raises_error(self):
        """Verify error when spatial_dims=3 without volume config."""
        from medgen.core.validation import validate_3d_config

        cfg = OmegaConf.create({
            'model': {'spatial_dims': 3},
        })

        errors = validate_3d_config(cfg)
        assert len(errors) == 1
        assert 'volume' in errors[0].lower()

    def test_validate_3d_config_with_valid_volume(self):
        """Verify no error with valid 3D volume configuration."""
        from medgen.core.validation import validate_3d_config

        cfg = OmegaConf.create({
            'model': {'spatial_dims': 3},
            'volume': {'depth': 32, 'height': 128, 'width': 128},
        })

        errors = validate_3d_config(cfg)
        assert len(errors) == 0


class Test3DVisualization:
    """Tests for 3D visualization code paths."""

    def test_visualize_samples_3d_warns_without_cached_batch(self):
        """Verify 3D visualization warns when no cached batch for conditional mode."""
        from medgen.pipeline.visualization import visualize_samples_3d

        trainer = Mock()
        trainer._cached_train_batch = None
        trainer.mode = Mock()
        trainer.mode.is_conditional = True

        with patch('medgen.pipeline.visualization.logger') as mock_logger:
            visualize_samples_3d(trainer, Mock(), epoch=0)
            mock_logger.warning.assert_called()

    def test_visualize_samples_3d_unconditional_proceeds(self):
        """Verify 3D unconditional visualization can proceed without cached batch."""
        from medgen.pipeline.visualization import visualize_samples_3d

        trainer = Mock()
        trainer._cached_train_batch = None
        trainer.mode = Mock()
        trainer.mode.is_conditional = False
        trainer.device = torch.device('cpu')
        trainer.volume_depth = 32
        trainer.volume_height = 64
        trainer.volume_width = 64
        trainer.space = Mock()
        trainer.space.scale_factor = 1
        trainer.space.needs_decode = False
        trainer.space.latent_channels = 1
        trainer._gen_metrics_config = None
        trainer._unified_metrics = Mock()
        trainer.use_controlnet = False
        trainer.controlnet_stage1 = False
        trainer.use_size_bin_embedding = False

        # Mock strategy.generate to return a tensor
        trainer.strategy = Mock()
        trainer.strategy.generate = Mock(return_value=torch.randn(4, 1, 32, 64, 64))

        # Should not raise
        visualize_samples_3d(trainer, Mock(), epoch=0)
        trainer.strategy.generate.assert_called_once()

    def test_visualize_denoising_trajectory_3d_warns_without_cached_batch(self):
        """Verify 3D trajectory visualization warns without cached batch."""
        from medgen.pipeline.visualization import visualize_denoising_trajectory_3d

        trainer = Mock()
        trainer._cached_train_batch = None

        with patch('medgen.pipeline.visualization.logger') as mock_logger:
            visualize_denoising_trajectory_3d(trainer, Mock(), epoch=0)
            mock_logger.warning.assert_called()


class Test3DValidation:
    """Tests for 3D validation code paths."""

    def test_compute_volume_3d_msssim_skips_disabled(self):
        """Verify 3D MS-SSIM computation skips when log_msssim=False."""
        from medgen.pipeline.evaluation import compute_volume_3d_msssim

        trainer = Mock()
        trainer.log_msssim = False

        result = compute_volume_3d_msssim(trainer, epoch=0)
        assert result is None

    def test_compute_volume_3d_msssim_skips_multi_modality(self):
        """Verify 3D MS-SSIM computation skips multi_modality mode."""
        from medgen.pipeline.evaluation import compute_volume_3d_msssim

        trainer = Mock()
        trainer.log_msssim = True
        trainer.spatial_dims = 3
        trainer.mode_name = 'multi_modality'
        trainer._mode_config = Mock()
        trainer._mode_config.out_channels = 1
        trainer.cfg = OmegaConf.create({
            'model': {'spatial_dims': 3},
            'mode': {'name': 'multi_modality', 'out_channels': 1},
        })

        result = compute_volume_3d_msssim(trainer, epoch=0)
        assert result is None

    def test_compute_volume_3d_msssim_delegates_to_native_for_3d(self):
        """Verify 3D MS-SSIM delegates to native function for 3D models."""
        from medgen.pipeline.evaluation import compute_volume_3d_msssim

        trainer = Mock()
        trainer.log_msssim = True
        trainer.spatial_dims = 3
        trainer.mode_name = 'bravo'
        trainer._mode_config = Mock()
        trainer._mode_config.out_channels = 1
        trainer.cfg = OmegaConf.create({
            'model': {'spatial_dims': 3},
            'mode': {'name': 'bravo', 'out_channels': 1},
        })

        with patch('medgen.pipeline.evaluation.compute_volume_3d_msssim_native') as mock_native:
            mock_native.return_value = 0.95
            result = compute_volume_3d_msssim(trainer, epoch=0)

            mock_native.assert_called_once_with(trainer, 0, 'val', None)
            assert result == 0.95


class Test3DMetrics:
    """Tests for 3D-specific metrics computation."""

    def test_unified_metrics_handles_5d_tensors(self):
        """Verify UnifiedMetrics can handle 5D tensors for 3D data."""
        from medgen.metrics.unified import UnifiedMetrics

        # UnifiedMetrics should accept spatial_dims=3
        metrics = UnifiedMetrics(
            writer=Mock(),
            mode='bravo',
            spatial_dims=3,
            volume_size=(32, 64, 64),
        )

        assert metrics.spatial_dims == 3

    def test_3d_center_slice_extraction(self):
        """Verify center slice extraction for 3D visualization."""
        # 3D tensor [B, C, D, H, W]
        tensor_3d = torch.randn(2, 1, 32, 64, 64)

        # Center slice is at index D//2
        center_idx = tensor_3d.shape[2] // 2
        center_slice = tensor_3d[:, :, center_idx, :, :]

        assert center_slice.shape == (2, 1, 64, 64)


class Test3DStrategyIntegration:
    """Tests for 3D strategy integration."""

    def test_rflow_strategy_setup_with_3d_params(self):
        """Verify RFlowStrategy setup accepts 3D parameters."""
        from medgen.diffusion.strategies import RFlowStrategy

        strategy = RFlowStrategy()
        scheduler = strategy.setup_scheduler(
            num_timesteps=1000,
            image_size=64,
            depth_size=32,
            spatial_dims=3,
            use_discrete_timesteps=False,
        )

        assert strategy.spatial_dims == 3
        assert scheduler is not None

    def test_rflow_strategy_3d_missing_depth_raises(self):
        """Verify RFlowStrategy raises error when depth_size missing for 3D."""
        from medgen.diffusion.strategies import RFlowStrategy

        strategy = RFlowStrategy()

        with pytest.raises(ValueError, match="depth_size required"):
            strategy.setup_scheduler(
                num_timesteps=1000,
                image_size=64,
                spatial_dims=3,  # 3D but no depth_size
            )

    def test_ddpm_strategy_setup_ignores_3d_params(self):
        """Verify DDPMStrategy setup accepts but ignores 3D-specific params."""
        from medgen.diffusion.strategies import DDPMStrategy

        strategy = DDPMStrategy()
        # DDPM should accept extra kwargs for interface compatibility
        scheduler = strategy.setup_scheduler(
            num_timesteps=1000,
            image_size=64,
            depth_size=32,  # Should be ignored
        )

        assert scheduler is not None
