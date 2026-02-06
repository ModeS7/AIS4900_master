"""Unit tests for logging, history, visualization, strategy math, and checkpointing helpers.

Modules covered:
- evaluation/evaluation_logging.py
- metrics/unified_history.py
- metrics/unified_logging.py
- metrics/unified_visualization.py (pure functions only)
- diffusion/strategy_ddpm.py (compute_target, compute_predicted_clean)
- diffusion/strategy_rflow.py (compute_target, compute_predicted_clean)
- pipeline/compression_checkpointing.py (pure functions only)
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# evaluation/evaluation_logging.py
# ============================================================================

class TestEvaluationLogging:
    """Tests for log_test_header, log_test_results, log_metrics_to_tensorboard."""

    @patch('medgen.evaluation.evaluation_logging.logger')
    def test_log_test_header(self, mock_logger):
        from medgen.evaluation.evaluation_logging import log_test_header
        log_test_header('best')
        assert mock_logger.info.call_count == 3
        # Second call should contain the label
        args = mock_logger.info.call_args_list[1][0][0]
        assert 'BEST' in args

    @patch('medgen.evaluation.evaluation_logging.logger')
    def test_log_test_results(self, mock_logger):
        from medgen.evaluation.evaluation_logging import log_test_results
        metrics = {'dice': 0.9, 'psnr': 30.0}
        log_test_results(metrics, 'best', n_samples=100)
        # Should log header + dice + psnr
        assert mock_logger.info.call_count >= 2

    def test_log_metrics_to_tensorboard_with_writer(self):
        from medgen.evaluation.evaluation_logging import log_metrics_to_tensorboard
        writer = Mock()
        metrics = {'dice': 0.9, 'psnr': 30.0}
        log_metrics_to_tensorboard(writer, metrics, prefix='test_best', step=0)
        assert writer.add_scalar.call_count == 2

    def test_log_metrics_to_tensorboard_none_writer(self):
        from medgen.evaluation.evaluation_logging import log_metrics_to_tensorboard
        # Should not crash with None writer
        log_metrics_to_tensorboard(None, {'dice': 0.9}, prefix='test_best', step=0)


class TestLogTestPerModality:
    """Tests for log_test_per_modality()."""

    def test_writes_modality_suffix(self):
        from medgen.evaluation.evaluation_logging import log_test_per_modality
        writer = Mock()
        metrics = {'msssim': 0.95, 'psnr': 32.0}
        log_test_per_modality(writer, metrics, prefix='test_best', modality='t1_pre', step=0)
        tags = [call[0][0] for call in writer.add_scalar.call_args_list]
        assert 'test_best/MS-SSIM_t1_pre' in tags
        assert 'test_best/PSNR_t1_pre' in tags

    def test_uses_unified_metrics_when_provided(self):
        from medgen.evaluation.evaluation_logging import log_test_per_modality
        writer = Mock()
        unified = Mock()
        metrics = {'msssim': 0.95}
        log_test_per_modality(
            writer, metrics, prefix='test_best', modality='t1_pre',
            step=0, unified_metrics=unified,
        )
        unified.log_test.assert_called_once()
        # writer should NOT be called when unified_metrics handles it
        writer.add_scalar.assert_not_called()


# ============================================================================
# metrics/unified_logging.py
# ============================================================================

def _make_mock_metrics(**overrides):
    """Create a mock UnifiedMetrics object with common attributes."""
    m = Mock()
    m.writer = overrides.get('writer', Mock())
    m.modality = overrides.get('modality', '')
    m.mode = overrides.get('mode', 'bravo')
    m.uses_image_quality = overrides.get('uses_image_quality', True)
    m.spatial_dims = overrides.get('spatial_dims', 2)
    m.log_prefix = overrides.get('log_prefix', '')
    m.num_timestep_bins = overrides.get('num_timestep_bins', 10)
    m._current_lr = overrides.get('_current_lr', None)
    m._grad_norm_count = overrides.get('_grad_norm_count', 0)
    m._vram_allocated = overrides.get('_vram_allocated', 0)
    m._flops_epoch = overrides.get('_flops_epoch', 0)
    m._codebook_tracker = overrides.get('_codebook_tracker', None)
    m._val_psnr_count = 0
    m._val_msssim_count = 0
    m._val_lpips_count = 0
    m._val_msssim_3d_count = 0
    m._val_dice_count = 0
    m._val_iou_count = 0
    m._regional_tracker = overrides.get('_regional_tracker', None)
    m._val_timesteps = {'counts': [0] * 10, 'sums': [0.0] * 10}
    return m


class TestUnifiedLogging:
    """Tests for log_training, log_validation, log_generation, etc."""

    def test_log_training_writes_losses(self):
        from medgen.metrics.unified_logging import log_training
        m = _make_mock_metrics()
        m._train_losses = {
            'MSE': {'sum': 1.0, 'count': 10},
            'Total': {'sum': 2.0, 'count': 10},
        }
        log_training(m, epoch=5)
        # Should write Loss/MSE_train and Loss/Total_train
        tags = [call[0][0] for call in m.writer.add_scalar.call_args_list]
        assert 'Loss/MSE_train' in tags
        assert 'Loss/Total_train' in tags

    def test_log_validation_writes_quality(self):
        from medgen.metrics.unified_logging import log_validation
        m = _make_mock_metrics(uses_image_quality=True)
        m._val_losses = {}
        m._val_psnr_sum = 30.0
        m._val_psnr_count = 1
        m._val_msssim_sum = 0.95
        m._val_msssim_count = 1
        log_validation(m, epoch=5)
        tags = [call[0][0] for call in m.writer.add_scalar.call_args_list]
        assert 'Validation/PSNR' in tags
        assert 'Validation/MS-SSIM' in tags

    def test_log_generation_writes_results(self):
        from medgen.metrics.unified_logging import log_generation
        m = _make_mock_metrics()
        results = {'FID': 12.5, 'KID_mean': 0.01}
        log_generation(m, epoch=10, results=results)
        tags = [call[0][0] for call in m.writer.add_scalar.call_args_list]
        assert 'Generation/FID' in tags
        assert 'Generation/KID_mean' in tags

    def test_log_test_writes_metrics(self):
        from medgen.metrics.unified_logging import log_test
        m = _make_mock_metrics()
        test_metrics = {'PSNR': 30.0, 'MS-SSIM': 0.95}
        log_test(m, test_metrics, prefix='test_best')
        tags = [call[0][0] for call in m.writer.add_scalar.call_args_list]
        assert 'test_best/PSNR' in tags
        assert 'test_best/MS-SSIM' in tags

    def test_log_test_generation_returns_dict(self):
        from medgen.metrics.unified_logging import log_test_generation
        m = _make_mock_metrics()
        results = {'FID': 12.5, 'KID_mean': 0.01}
        exported = log_test_generation(m, results, prefix='test_best')
        assert isinstance(exported, dict)
        assert 'gen_fid' in exported

    def test_log_sample_images_calls_add_images(self):
        from medgen.metrics.unified_logging import log_sample_images
        m = _make_mock_metrics()
        images = torch.randn(4, 1, 64, 64)
        log_sample_images(m, images, tag='Samples', epoch=5)
        m.writer.add_images.assert_called_once_with('Samples', images, 5)

    def test_log_regularization_loss_writes_scalar(self):
        from medgen.metrics.unified_logging import log_regularization_loss
        m = _make_mock_metrics()
        log_regularization_loss(m, loss_type='KL', weighted_loss=0.01, epoch=5)
        m.writer.add_scalar.assert_called_once_with('Loss/KL_val', 0.01, 5)

    def test_log_codebook_metrics_returns_dict(self):
        from medgen.metrics.unified_logging import log_codebook_metrics
        m = _make_mock_metrics()
        tracker = Mock()
        tracker.log_to_tensorboard.return_value = {'usage': 0.8}
        result = log_codebook_metrics(m, tracker, epoch=5)
        tracker.log_to_tensorboard.assert_called_once()
        assert result == {'usage': 0.8}


# ============================================================================
# metrics/unified_history.py
# ============================================================================

class TestUnifiedHistory:
    """Tests for record_epoch_history, save_json_histories, log_console_summary."""

    def test_record_epoch_no_crash_with_empty_state(self):
        from medgen.metrics.unified_history import record_epoch_history
        m = _make_mock_metrics()
        m._regional_tracker = None
        m._regional_history = {}
        m._timestep_history = {}
        m._timestep_region_history = {}
        m._tr_tumor_sum = [0.0] * 10
        m._tr_tumor_count = [0] * 10
        m._tr_bg_sum = [0.0] * 10
        m._tr_bg_count = [0] * 10
        # Should not raise
        record_epoch_history(m, epoch=0)

    def test_save_json_histories_creates_files(self, tmp_path):
        from medgen.metrics.unified_history import save_json_histories
        m = _make_mock_metrics()
        m._regional_history = {'0': {'tumor': 0.5, 'background': 0.3}}
        m._timestep_history = {'0': {'0.0-0.1': 0.5}}
        m._timestep_region_history = {}
        save_json_histories(m, str(tmp_path))
        assert (tmp_path / 'regional_losses.json').exists()
        assert (tmp_path / 'timestep_losses.json').exists()

    @patch('medgen.metrics.unified_history.logger')
    def test_log_console_summary_calls_logger(self, mock_logger):
        from medgen.metrics.unified_history import log_console_summary
        m = _make_mock_metrics()
        m.get_training_losses.return_value = {'Total': 0.5}
        m.get_validation_metrics.return_value = {'MS-SSIM': 0.95}
        log_console_summary(m, epoch=0, total_epochs=100, elapsed_time=10.0)
        assert mock_logger.info.call_count >= 1

    def test_record_epoch_with_train_losses(self):
        from medgen.metrics.unified_history import record_epoch_history
        m = _make_mock_metrics()
        m._regional_tracker = None
        m._regional_history = {}
        m._timestep_history = {}
        m._timestep_region_history = {}
        m._val_timesteps = {
            'counts': [5] + [0] * 9,
            'sums': [2.5] + [0.0] * 9,
        }
        m._tr_tumor_sum = [0.0] * 10
        m._tr_tumor_count = [0] * 10
        m._tr_bg_sum = [0.0] * 10
        m._tr_bg_count = [0] * 10
        record_epoch_history(m, epoch=0)
        # Timestep history should have an entry
        assert '0' in m._timestep_history


# ============================================================================
# metrics/unified_visualization.py (pure utility functions)
# ============================================================================

class TestUnifiedVisualization:
    """Tests for _extract_center_slice and _extract_multiple_slices."""

    def test_extract_center_slice_5d(self):
        from medgen.metrics.unified_visualization import _extract_center_slice
        m = Mock(spatial_dims=3)
        t = torch.randn(2, 1, 16, 64, 64)
        result = _extract_center_slice(m, t)
        assert result.shape == (2, 1, 64, 64)
        assert torch.equal(result, t[:, :, 8, :, :])

    def test_extract_center_slice_4d_passthrough(self):
        from medgen.metrics.unified_visualization import _extract_center_slice
        m = Mock(spatial_dims=2)
        t = torch.randn(2, 1, 64, 64)
        result = _extract_center_slice(m, t)
        assert result is t

    def test_extract_multiple_slices(self):
        from medgen.metrics.unified_visualization import _extract_multiple_slices
        m = Mock(spatial_dims=3)
        t = torch.randn(1, 1, 16, 64, 64)
        result = _extract_multiple_slices(m, t, num_slices=4)
        assert result.shape[0] == 4
        assert result.shape[-2:] == (64, 64)

    def test_extract_multiple_slices_4d_passthrough(self):
        from medgen.metrics.unified_visualization import _extract_multiple_slices
        m = Mock(spatial_dims=2)
        t = torch.randn(4, 1, 64, 64)
        result = _extract_multiple_slices(m, t, num_slices=4)
        assert result is t


# ============================================================================
# diffusion/strategy_ddpm.py
# ============================================================================

class TestDDPMComputeTarget:
    """Tests for DDPMStrategy.compute_target()."""

    def test_returns_noise(self, ddpm_strategy):
        clean = torch.randn(4, 1, 64, 64)
        noise = torch.randn(4, 1, 64, 64)
        target = ddpm_strategy.compute_target(clean, noise)
        assert torch.equal(target, noise)

    def test_dict_input(self, ddpm_strategy):
        clean = {'a': torch.randn(4, 1, 64, 64), 'b': torch.randn(4, 1, 64, 64)}
        noise = {'a': torch.randn(4, 1, 64, 64), 'b': torch.randn(4, 1, 64, 64)}
        target = ddpm_strategy.compute_target(clean, noise)
        # DDPM target is noise regardless
        assert torch.equal(target['a'], noise['a'])
        assert torch.equal(target['b'], noise['b'])


class TestRFlowComputeTarget:
    """Tests for RFlowStrategy.compute_target()."""

    def test_returns_velocity(self, rflow_strategy_2d):
        clean = torch.randn(4, 1, 64, 64)
        noise = torch.randn(4, 1, 64, 64)
        target = rflow_strategy_2d.compute_target(clean, noise)
        expected = clean - noise
        assert torch.allclose(target, expected)

    def test_dict_input(self, rflow_strategy_2d):
        clean = {'a': torch.randn(4, 1, 64, 64), 'b': torch.randn(4, 1, 64, 64)}
        noise = {'a': torch.randn(4, 1, 64, 64), 'b': torch.randn(4, 1, 64, 64)}
        target = rflow_strategy_2d.compute_target(clean, noise)
        assert torch.allclose(target['a'], clean['a'] - noise['a'])
        assert torch.allclose(target['b'], clean['b'] - noise['b'])


class TestDDPMComputePredictedClean:
    """Tests for DDPMStrategy.compute_predicted_clean()."""

    def test_output_shape(self, ddpm_strategy):
        noisy = torch.randn(4, 1, 64, 64)
        pred = torch.randn(4, 1, 64, 64)
        timesteps = torch.randint(0, 100, (4,))
        result = ddpm_strategy.compute_predicted_clean(noisy, pred, timesteps)
        assert result.shape == noisy.shape

    def test_output_clamped(self, ddpm_strategy):
        noisy = torch.randn(4, 1, 64, 64)
        pred = torch.randn(4, 1, 64, 64)
        timesteps = torch.randint(0, 100, (4,))
        result = ddpm_strategy.compute_predicted_clean(noisy, pred, timesteps)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dict_input(self, ddpm_strategy):
        noisy = {'a': torch.randn(4, 1, 64, 64), 'b': torch.randn(4, 1, 64, 64)}
        pred = torch.randn(4, 2, 64, 64)  # 2 channels for dual
        timesteps = torch.randint(0, 100, (4,))
        result = ddpm_strategy.compute_predicted_clean(noisy, pred, timesteps)
        assert isinstance(result, dict)
        assert 'a' in result and 'b' in result


class TestRFlowComputePredictedClean:
    """Tests for RFlowStrategy.compute_predicted_clean()."""

    def test_output_shape(self, rflow_strategy_2d):
        noisy = torch.randn(4, 1, 64, 64)
        pred = torch.randn(4, 1, 64, 64)
        timesteps = torch.randint(0, 100, (4,))
        result = rflow_strategy_2d.compute_predicted_clean(noisy, pred, timesteps)
        assert result.shape == noisy.shape

    def test_output_clamped(self, rflow_strategy_2d):
        noisy = torch.randn(4, 1, 64, 64)
        pred = torch.randn(4, 1, 64, 64)
        timesteps = torch.randint(0, 100, (4,))
        result = rflow_strategy_2d.compute_predicted_clean(noisy, pred, timesteps)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_identity_at_t0(self, rflow_strategy_2d):
        """At t=0, x_0 = x_t + 0*v = x_t (clamped)."""
        noisy = torch.rand(4, 1, 64, 64)  # already in [0, 1]
        pred = torch.randn(4, 1, 64, 64)
        timesteps = torch.zeros(4)  # t=0
        result = rflow_strategy_2d.compute_predicted_clean(noisy, pred, timesteps)
        # At t=0: x_0 = clamp(x_t + 0*v) = clamp(x_t) = x_t (since x_t in [0,1])
        assert torch.allclose(result, noisy, atol=1e-6)


# ============================================================================
# pipeline/compression_checkpointing.py
# ============================================================================

class TestCheckpointExtraState:
    """Tests for get_checkpoint_extra_state()."""

    def test_returns_dict(self):
        from medgen.pipeline.compression_checkpointing import get_checkpoint_extra_state
        trainer = Mock()
        trainer.disable_gan = True
        trainer.use_constant_lr = False
        trainer.discriminator_raw = None
        result = get_checkpoint_extra_state(trainer)
        assert isinstance(result, dict)
        assert 'disable_gan' in result
        assert 'use_constant_lr' in result

    def test_includes_relevant_flags(self):
        from medgen.pipeline.compression_checkpointing import get_checkpoint_extra_state
        trainer = Mock()
        trainer.disable_gan = False
        trainer.use_constant_lr = True
        trainer.discriminator_raw = Mock()
        trainer.cfg.mode.in_channels = 1
        trainer.disc_num_channels = 64
        trainer.disc_num_layers = 3
        result = get_checkpoint_extra_state(trainer)
        assert result['disable_gan'] is False
        assert result['use_constant_lr'] is True
        assert 'disc_config' in result


class TestLogEpochSummary:
    """Tests for log_epoch_summary()."""

    @patch('medgen.pipeline.compression_checkpointing.logger')
    def test_calls_logger(self, mock_logger):
        from medgen.pipeline.compression_checkpointing import log_epoch_summary
        trainer = Mock()
        avg_losses = {'gen': 0.5, 'recon': 0.3, 'disc': 0.1}
        val_metrics = {'gen': 0.4, 'l1': 0.2, 'msssim': 0.95}
        log_epoch_summary(trainer, epoch=0, total_epochs=100,
                          avg_losses=avg_losses, val_metrics=val_metrics,
                          elapsed_time=10.0)
        assert mock_logger.info.call_count >= 1
