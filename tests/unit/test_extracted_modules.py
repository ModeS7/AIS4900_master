"""Tests for trainer decomposition modules (Phase 4 extractions).

These tests verify the extracted pipeline modules function correctly
in isolation from the full trainer.
"""
import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from omegaconf import OmegaConf


class TestProfilingModule:
    """Tests for src/medgen/pipeline/profiling.py"""

    def test_get_trainer_type_returns_diffusion(self):
        """Verify trainer type identifier is correct."""
        from medgen.pipeline.profiling import get_trainer_type
        assert get_trainer_type() == "diffusion"

    def test_get_metadata_extra_returns_dict(self):
        """Verify metadata extraction returns expected keys."""
        from medgen.pipeline.profiling import get_metadata_extra

        trainer = Mock()
        trainer.strategy_name = "rflow"
        trainer.mode_name = "bravo"
        trainer.image_size = 256
        trainer.num_timesteps = 1000
        trainer.batch_size = 4
        trainer.learning_rate = 1e-4
        trainer.use_sam = False
        trainer.use_ema = True

        result = get_metadata_extra(trainer)
        assert isinstance(result, dict)
        assert result['strategy'] == 'rflow'
        assert result['mode'] == 'bravo'
        assert result['image_size'] == 256
        assert 'created_at' in result

    def test_get_model_config_unet(self):
        """Verify UNet model config extraction."""
        from medgen.pipeline.profiling import get_model_config

        trainer = Mock()
        trainer.model_type = 'unet'
        trainer.is_transformer = False
        trainer.strategy_name = 'rflow'
        trainer.mode_name = 'bravo'
        trainer.mode.get_model_config.return_value = {
            'in_channels': 2,
            'out_channels': 1,
        }
        trainer.cfg = OmegaConf.create({
            'model': {
                'spatial_dims': 2,
                'channels': [64, 128, 256],
                'attention_levels': [False, True, True],
                'num_res_blocks': 2,
                'num_head_channels': 64,
            }
        })

        result = get_model_config(trainer)
        assert result['model_type'] == 'unet'
        assert result['in_channels'] == 2
        assert result['channels'] == [64, 128, 256]

    def test_get_model_config_transformer(self):
        """Verify transformer model config extraction."""
        from medgen.pipeline.profiling import get_model_config

        trainer = Mock()
        trainer.model_type = 'dit'
        trainer.is_transformer = True
        trainer.strategy_name = 'rflow'
        trainer.mode_name = 'bravo'
        trainer.mode.get_model_config.return_value = {
            'in_channels': 2,
            'out_channels': 1,
        }
        trainer.cfg = OmegaConf.create({
            'model': {
                'spatial_dims': 2,
                'image_size': 256,
                'patch_size': 4,
                'variant': 'DiT-XL/2',
                'mlp_ratio': 4.0,
                'conditioning': 'concat',
                'qk_norm': True,
            }
        })

        result = get_model_config(trainer)
        assert result['model_type'] == 'dit'
        assert result['image_size'] == 256
        assert result['patch_size'] == 4


class TestLossesModule:
    """Tests for src/medgen/pipeline/losses.py"""

    def test_compute_self_conditioning_loss_disabled(self):
        """Verify self-conditioning returns 0 when disabled."""
        from medgen.pipeline.losses import compute_self_conditioning_loss

        trainer = Mock()
        trainer._training_tricks.self_cond.enabled = False

        result = compute_self_conditioning_loss(
            trainer,
            model_input=torch.randn(2, 1, 64, 64),
            timesteps=torch.tensor([100, 200]),
            prediction=torch.randn(2, 1, 64, 64),
        )
        assert isinstance(result, torch.Tensor)
        assert result.item() == 0.0

    def test_compute_min_snr_weighted_mse_shape(self):
        """Verify Min-SNR weighted loss returns scalar."""
        from medgen.pipeline.losses import compute_min_snr_weighted_mse

        trainer = Mock()
        trainer.strategy_name = 'rflow'
        trainer._unified_metrics = Mock()
        trainer._unified_metrics.compute_snr_weights.return_value = torch.ones(2)

        prediction = torch.randn(2, 1, 64, 64)
        images = torch.randn(2, 1, 64, 64)
        noise = torch.randn(2, 1, 64, 64)
        timesteps = torch.tensor([100, 500])

        result = compute_min_snr_weighted_mse(trainer, prediction, images, noise, timesteps)
        assert result.ndim == 0  # Scalar
        assert result.item() > 0  # Positive loss

    def test_compute_min_snr_weighted_mse_dict_images(self):
        """Verify Min-SNR weighted loss handles dict images (dual mode)."""
        from medgen.pipeline.losses import compute_min_snr_weighted_mse

        trainer = Mock()
        trainer.strategy_name = 'rflow'
        trainer._unified_metrics = Mock()
        trainer._unified_metrics.compute_snr_weights.return_value = torch.ones(2)

        images = {
            't1_pre': torch.randn(2, 1, 64, 64),
            't1_gd': torch.randn(2, 1, 64, 64),
        }
        noise = {
            't1_pre': torch.randn(2, 1, 64, 64),
            't1_gd': torch.randn(2, 1, 64, 64),
        }
        prediction = torch.randn(2, 2, 64, 64)
        timesteps = torch.tensor([100, 500])

        result = compute_min_snr_weighted_mse(trainer, prediction, images, noise, timesteps)
        assert result.ndim == 0


class TestTrainingTricksModule:
    """Tests for src/medgen/pipeline/training_tricks.py"""

    def test_apply_timestep_jitter_disabled(self):
        """Verify jitter returns unchanged timesteps when disabled."""
        from medgen.pipeline.training_tricks import apply_timestep_jitter

        trainer = Mock()
        trainer._training_tricks.jitter.enabled = False

        timesteps = torch.tensor([100, 500, 900], dtype=torch.long)
        result = apply_timestep_jitter(trainer, timesteps)
        assert torch.equal(result, timesteps)

    def test_apply_timestep_jitter_preserves_dtype_discrete(self):
        """Verify jitter preserves integer dtype for discrete timesteps."""
        from medgen.pipeline.training_tricks import apply_timestep_jitter

        trainer = Mock()
        trainer._training_tricks.jitter.enabled = True
        trainer._training_tricks.jitter.std = 0.05
        trainer.num_timesteps = 1000

        timesteps = torch.tensor([100, 500, 900], dtype=torch.long)
        result = apply_timestep_jitter(trainer, timesteps)
        assert result.dtype == torch.long

    def test_apply_timestep_jitter_preserves_dtype_continuous(self):
        """Verify jitter preserves float dtype for continuous timesteps."""
        from medgen.pipeline.training_tricks import apply_timestep_jitter

        trainer = Mock()
        trainer._training_tricks.jitter.enabled = True
        trainer._training_tricks.jitter.std = 0.05
        trainer.num_timesteps = 1000

        timesteps = torch.tensor([100.5, 500.2, 900.7], dtype=torch.float32)
        result = apply_timestep_jitter(trainer, timesteps)
        assert result.dtype == torch.float32

    def test_get_curriculum_range_returns_none_when_disabled(self):
        """Verify curriculum returns None when disabled."""
        from medgen.pipeline.training_tricks import get_curriculum_range

        trainer = Mock()
        trainer._training_tricks.curriculum.enabled = False

        result = get_curriculum_range(trainer, epoch=5)
        assert result is None

    def test_get_curriculum_range_interpolates(self):
        """Verify curriculum interpolates range correctly."""
        from medgen.pipeline.training_tricks import get_curriculum_range

        trainer = Mock()
        trainer._training_tricks.curriculum.enabled = True
        trainer._training_tricks.curriculum.warmup_epochs = 100
        trainer._training_tricks.curriculum.min_t_start = 0.0
        trainer._training_tricks.curriculum.min_t_end = 0.0
        trainer._training_tricks.curriculum.max_t_start = 0.3
        trainer._training_tricks.curriculum.max_t_end = 1.0

        # At epoch 0: should be near start
        result = get_curriculum_range(trainer, epoch=0)
        assert result is not None
        assert result[1] == pytest.approx(0.3, abs=0.01)

        # At epoch 100: should be at end
        result = get_curriculum_range(trainer, epoch=100)
        assert result is not None
        assert result[1] == pytest.approx(1.0, abs=0.01)

        # At epoch 50: should be halfway
        result = get_curriculum_range(trainer, epoch=50)
        assert result is not None
        assert result[1] == pytest.approx(0.65, abs=0.01)

    def test_apply_noise_augmentation_disabled(self):
        """Verify noise augmentation returns unchanged noise when disabled."""
        from medgen.pipeline.training_tricks import apply_noise_augmentation

        trainer = Mock()
        trainer._training_tricks.noise_augmentation.enabled = False

        noise = torch.randn(2, 1, 64, 64)
        result = apply_noise_augmentation(trainer, noise)
        assert torch.equal(result, noise)

    def test_apply_conditioning_dropout_no_conditioning(self):
        """Verify dropout returns None when conditioning is None."""
        from medgen.pipeline.training_tricks import apply_conditioning_dropout

        trainer = Mock()
        trainer.controlnet_cfg_dropout_prob = 0.1
        trainer.training = True

        result = apply_conditioning_dropout(trainer, None, batch_size=4)
        assert result is None


class TestValidationModule:
    """Tests for src/medgen/pipeline/validation.py"""

    def test_compute_validation_losses_returns_empty_without_loader(self):
        """Verify validation returns empty dict without val_loader."""
        from medgen.pipeline.validation import compute_validation_losses

        trainer = Mock()
        trainer.val_loader = None

        metrics, worst_batch = compute_validation_losses(trainer, epoch=0)
        assert metrics == {}
        assert worst_batch is None


class TestVisualizationModule:
    """Tests for src/medgen/pipeline/visualization.py"""

    def test_visualize_samples_exits_early_non_main_process(self):
        """Verify visualization returns early on non-main process."""
        from medgen.pipeline.visualization import visualize_samples

        trainer = Mock()
        trainer.is_main_process = False

        # Should return without error and without calling model.eval()
        model = Mock()
        visualize_samples(trainer, model, epoch=0)
        model.eval.assert_not_called()

    def test_visualize_samples_3d_requires_cached_batch(self):
        """Verify 3D visualization logs warning without cached batch."""
        from medgen.pipeline.visualization import visualize_samples_3d

        trainer = Mock()
        trainer._cached_train_batch = None
        trainer.mode = Mock()
        trainer.mode.is_conditional = True

        with patch('medgen.pipeline.visualization.logger') as mock_logger:
            visualize_samples_3d(trainer, Mock(), epoch=0)
            mock_logger.warning.assert_called()


class TestEvaluationModule:
    """Tests for src/medgen/pipeline/evaluation.py"""

    def test_evaluate_test_set_returns_empty_non_main_process(self):
        """Verify test evaluation returns empty dict on non-main process."""
        from medgen.pipeline.evaluation import evaluate_test_set

        trainer = Mock()
        trainer.is_main_process = False

        result = evaluate_test_set(trainer, Mock())
        assert result == {}


class TestProtocols:
    """Tests for src/medgen/diffusion/protocols.py"""

    def test_diffusion_model_protocol_runtime_checkable(self):
        """Verify DiffusionModel protocol is runtime checkable."""
        from medgen.diffusion.protocols import DiffusionModel

        class ValidModel:
            def __call__(self, x, timesteps):
                return x

        model = ValidModel()
        assert isinstance(model, DiffusionModel)

    def test_conditional_model_protocol_runtime_checkable(self):
        """Verify ConditionalDiffusionModel protocol is runtime checkable."""
        from medgen.diffusion.protocols import ConditionalDiffusionModel

        class ValidModel:
            def __call__(self, x, timesteps, omega=None, mode_id=None):
                return x

        model = ValidModel()
        assert isinstance(model, ConditionalDiffusionModel)

    def test_size_bin_model_protocol_requires_attribute(self):
        """Verify SizeBinModel requires size_bin_time_embed attribute."""
        from medgen.diffusion.protocols import SizeBinModel

        class MissingAttribute:
            def __call__(self, x, timesteps, size_bins):
                return x

        class HasAttribute:
            size_bin_time_embed = Mock()

            def __call__(self, x, timesteps, size_bins):
                return x

        assert not isinstance(MissingAttribute(), SizeBinModel)
        assert isinstance(HasAttribute(), SizeBinModel)


class TestConfigValidationExtended:
    """Tests for new config validators in src/medgen/core/validation.py"""

    def test_validate_strategy_mode_compatibility_rflow_discrete_error(self):
        """Verify error when rflow uses discrete timesteps."""
        from medgen.core.validation import validate_strategy_mode_compatibility

        cfg = OmegaConf.create({
            'strategy': {'name': 'rflow', 'use_discrete_timesteps': True},
            'mode': {'name': 'bravo'},
            'training': {},
        })

        errors = validate_strategy_mode_compatibility(cfg)
        assert len(errors) == 1
        assert 'rflow' in errors[0].lower()

    def test_validate_strategy_mode_compatibility_ddpm_continuous_error(self):
        """Verify error when ddpm uses continuous timesteps."""
        from medgen.core.validation import validate_strategy_mode_compatibility

        cfg = OmegaConf.create({
            'strategy': {'name': 'ddpm', 'use_discrete_timesteps': False},
            'mode': {'name': 'bravo'},
            'training': {},
        })

        errors = validate_strategy_mode_compatibility(cfg)
        assert len(errors) == 1
        assert 'ddpm' in errors[0].lower()

    def test_validate_strategy_mode_compatibility_multi_modality_no_embedding(self):
        """Verify error when multi_modality mode lacks mode embedding."""
        from medgen.core.validation import validate_strategy_mode_compatibility

        cfg = OmegaConf.create({
            'strategy': {'name': 'rflow'},
            'mode': {'name': 'multi_modality'},
            'training': {'use_mode_embedding': False},
        })

        errors = validate_strategy_mode_compatibility(cfg)
        assert len(errors) == 1
        assert 'mode_embedding' in errors[0].lower()

    def test_validate_strategy_mode_compatibility_valid(self):
        """Verify no errors for valid strategy/mode combination."""
        from medgen.core.validation import validate_strategy_mode_compatibility

        cfg = OmegaConf.create({
            'strategy': {'name': 'rflow', 'use_discrete_timesteps': False},
            'mode': {'name': 'bravo'},
            'training': {},
        })

        errors = validate_strategy_mode_compatibility(cfg)
        assert len(errors) == 0

    def test_validate_3d_config_requires_volume(self):
        """Verify error when spatial_dims=3 without volume config."""
        from medgen.core.validation import validate_3d_config

        cfg = OmegaConf.create({
            'model': {'spatial_dims': 3},
        })

        errors = validate_3d_config(cfg)
        assert len(errors) == 1
        assert 'volume' in errors[0].lower()

    def test_validate_3d_config_positive_dimensions(self):
        """Verify error when volume dimensions are not positive."""
        from medgen.core.validation import validate_3d_config

        cfg = OmegaConf.create({
            'model': {'spatial_dims': 3},
            'volume': {'depth': 0, 'height': 128, 'width': 128},
        })

        errors = validate_3d_config(cfg)
        assert len(errors) == 1
        assert 'depth' in errors[0].lower()

    def test_validate_3d_config_skips_2d(self):
        """Verify 3D validation skips 2D config."""
        from medgen.core.validation import validate_3d_config

        cfg = OmegaConf.create({
            'model': {'spatial_dims': 2},
        })

        errors = validate_3d_config(cfg)
        assert len(errors) == 0

    def test_validate_latent_config_requires_checkpoint(self):
        """Verify error when latent enabled without checkpoint."""
        from medgen.core.validation import validate_latent_config

        cfg = OmegaConf.create({
            'latent': {'enabled': True, 'compression_checkpoint': None},
        })

        errors = validate_latent_config(cfg)
        assert len(errors) == 1
        assert 'checkpoint' in errors[0].lower()

    def test_validate_latent_config_skips_disabled(self):
        """Verify latent validation skips when disabled."""
        from medgen.core.validation import validate_latent_config

        cfg = OmegaConf.create({
            'latent': {'enabled': False},
        })

        errors = validate_latent_config(cfg)
        assert len(errors) == 0

    def test_validate_regional_logging_seg_mode_error(self):
        """Verify error when regional losses enabled with seg mode."""
        from medgen.core.validation import validate_regional_logging

        cfg = OmegaConf.create({
            'mode': {'name': 'seg'},
            'training': {'logging': {'regional_losses': True}},
        })

        errors = validate_regional_logging(cfg)
        assert len(errors) == 1
        assert 'seg' in errors[0].lower()

    def test_validate_regional_logging_valid_with_bravo(self):
        """Verify no error when regional losses enabled with bravo mode."""
        from medgen.core.validation import validate_regional_logging

        cfg = OmegaConf.create({
            'mode': {'name': 'bravo'},
            'training': {'logging': {'regional_losses': True}},
        })

        errors = validate_regional_logging(cfg)
        assert len(errors) == 0
