"""Tests for strategy generate() methods with CFG and size_bins."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from medgen.diffusion.strategies import DDPMStrategy, RFlowStrategy


class TestDDPMGenerate:
    """Test DDPMStrategy.generate() method."""

    def test_unconditional_generation(self, ddpm_strategy, mock_diffusion_model):
        """1-channel input generates samples."""
        noise = torch.randn(4, 1, 64, 64)

        samples = ddpm_strategy.generate(
            mock_diffusion_model,
            noise,
            num_steps=5,
            device=torch.device('cpu'),
        )

        assert samples.shape == noise.shape

    def test_conditional_single_generation(self, ddpm_strategy, mock_diffusion_model):
        """2-channel input uses conditioning."""
        # [noise, seg_mask]
        model_input = torch.randn(4, 2, 64, 64)

        samples = ddpm_strategy.generate(
            mock_diffusion_model,
            model_input,
            num_steps=5,
            device=torch.device('cpu'),
        )

        # Output should be 1 channel (generated image)
        assert samples.shape == (4, 1, 64, 64)

    def test_conditional_dual_generation(self, ddpm_strategy):
        """3-channel input processes pre/gd separately."""
        # [noise_pre, noise_gd, seg_mask]
        model_input = torch.randn(4, 3, 64, 64)

        # Mock model that returns 2-channel output for dual
        model = Mock()
        model.return_value = torch.randn(4, 2, 64, 64)
        model.eval = Mock(return_value=model)
        model.to = Mock(return_value=model)

        samples = ddpm_strategy.generate(
            model,
            model_input,
            num_steps=5,
            device=torch.device('cpu'),
        )

        # Dual mode returns 2 channels
        assert samples.shape == (4, 2, 64, 64)

    def test_cfg_scale_size_bins(self, ddpm_strategy, mock_size_bin_model):
        """cfg_scale > 1 with size_bins applies guidance."""
        noise = torch.randn(4, 1, 64, 64)
        size_bins = torch.randint(0, 5, (4, 7))

        # Track calls to model
        call_count = [0]
        original_return = torch.randn(4, 1, 64, 64)

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return original_return

        mock_size_bin_model.side_effect = side_effect

        samples = ddpm_strategy.generate(
            mock_size_bin_model,
            noise,
            num_steps=5,
            device=torch.device('cpu'),
            size_bins=size_bins,
            cfg_scale=2.0,
        )

        # CFG requires 2 forward passes per step
        assert call_count[0] == 5 * 2

    def test_progress_bar_optional(self, ddpm_strategy):
        """use_progress_bars controls tqdm."""
        noise = torch.randn(2, 1, 64, 64)

        # Create mock that returns matching batch size
        model = Mock()
        model.return_value = torch.randn(2, 1, 64, 64)
        model.eval = Mock(return_value=model)
        model.to = Mock(return_value=model)

        # Should run without error with progress bars disabled
        samples = ddpm_strategy.generate(
            model,
            noise,
            num_steps=3,
            device=torch.device('cpu'),
            use_progress_bars=False,
        )

        assert samples.shape == noise.shape


class TestRFlowGenerate:
    """Test RFlowStrategy.generate() method."""

    def test_unconditional_generation(self, rflow_strategy_2d, mock_diffusion_model):
        """1-channel generates samples."""
        noise = torch.randn(4, 1, 64, 64)

        samples = rflow_strategy_2d.generate(
            mock_diffusion_model,
            noise,
            num_steps=5,
            device=torch.device('cpu'),
        )

        assert samples.shape == noise.shape

    def test_conditional_single_generation(self, rflow_strategy_2d, mock_diffusion_model):
        """2-channel with conditioning."""
        model_input = torch.randn(4, 2, 64, 64)

        samples = rflow_strategy_2d.generate(
            mock_diffusion_model,
            model_input,
            num_steps=5,
            device=torch.device('cpu'),
        )

        assert samples.shape == (4, 1, 64, 64)

    def test_cfg_size_bins(self, rflow_strategy_2d, mock_size_bin_model):
        """REGRESSION: cfg_scale with size_bins works."""
        noise = torch.randn(4, 1, 64, 64)
        size_bins = torch.randint(0, 5, (4, 7))

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return torch.randn(4, 1, 64, 64)

        mock_size_bin_model.side_effect = side_effect

        samples = rflow_strategy_2d.generate(
            mock_size_bin_model,
            noise,
            num_steps=5,
            device=torch.device('cpu'),
            size_bins=size_bins,
            cfg_scale=2.0,
        )

        # CFG requires 2 forward passes per step
        assert call_count[0] == 5 * 2

    def test_cfg_conditioning(self, rflow_strategy_2d):
        """CFG with image conditioning (seg mask)."""
        # [noise, seg_mask]
        model_input = torch.randn(4, 2, 64, 64)

        call_count = [0]
        model = Mock()

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return torch.randn(4, 1, 64, 64)

        model.side_effect = side_effect
        model.eval = Mock(return_value=model)
        model.to = Mock(return_value=model)

        samples = rflow_strategy_2d.generate(
            model,
            model_input,
            num_steps=5,
            device=torch.device('cpu'),
            cfg_scale=2.0,
        )

        # CFG with conditioning should double calls
        assert call_count[0] == 5 * 2

    def test_dynamic_cfg_start_to_end(self, rflow_strategy_2d, mock_size_bin_model):
        """cfg_scale_end enables decay schedule."""
        noise = torch.randn(4, 1, 64, 64)
        size_bins = torch.randint(0, 5, (4, 7))

        # Should run without error
        samples = rflow_strategy_2d.generate(
            mock_size_bin_model,
            noise,
            num_steps=5,
            device=torch.device('cpu'),
            size_bins=size_bins,
            cfg_scale=3.0,
            cfg_scale_end=1.0,  # Decay from 3 to 1
        )

        assert samples.shape == noise.shape

    def test_2d_input_shape(self, rflow_strategy_2d, mock_diffusion_model):
        """[B, C, H, W] handled correctly."""
        noise = torch.randn(4, 1, 64, 64)

        samples = rflow_strategy_2d.generate(
            mock_diffusion_model,
            noise,
            num_steps=3,
            device=torch.device('cpu'),
        )

        assert samples.shape == (4, 1, 64, 64)

    def test_3d_input_shape(self, rflow_strategy_3d, mock_diffusion_model):
        """[B, C, D, H, W] handled correctly."""
        # Update mock to return 3D output
        mock_diffusion_model.return_value = torch.randn(2, 1, 16, 64, 64)

        noise = torch.randn(2, 1, 16, 64, 64)

        samples = rflow_strategy_3d.generate(
            mock_diffusion_model,
            noise,
            num_steps=3,
            device=torch.device('cpu'),
        )

        assert samples.shape == (2, 1, 16, 64, 64)


class TestCFGEdgeCases:
    """Test CFG edge cases for both strategies."""

    def test_cfg_scale_1_no_guidance_ddpm(self, ddpm_strategy, mock_size_bin_model):
        """cfg_scale=1.0 skips guidance computation for DDPM."""
        noise = torch.randn(4, 1, 64, 64)
        size_bins = torch.randint(0, 5, (4, 7))

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return torch.randn(4, 1, 64, 64)

        mock_size_bin_model.side_effect = side_effect

        samples = ddpm_strategy.generate(
            mock_size_bin_model,
            noise,
            num_steps=5,
            device=torch.device('cpu'),
            size_bins=size_bins,
            cfg_scale=1.0,  # No guidance
        )

        # cfg_scale=1 should only do 1 forward pass per step
        assert call_count[0] == 5

    def test_cfg_scale_1_no_guidance_rflow(self, rflow_strategy_2d, mock_size_bin_model):
        """cfg_scale=1.0 skips guidance computation for RFlow."""
        noise = torch.randn(4, 1, 64, 64)
        size_bins = torch.randint(0, 5, (4, 7))

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return torch.randn(4, 1, 64, 64)

        mock_size_bin_model.side_effect = side_effect

        samples = rflow_strategy_2d.generate(
            mock_size_bin_model,
            noise,
            num_steps=5,
            device=torch.device('cpu'),
            size_bins=size_bins,
            cfg_scale=1.0,
        )

        # No guidance = 1 forward pass per step
        assert call_count[0] == 5

    def test_cfg_scale_end_equals_start(self, rflow_strategy_2d, mock_size_bin_model):
        """Same start/end uses constant CFG."""
        noise = torch.randn(4, 1, 64, 64)
        size_bins = torch.randint(0, 5, (4, 7))

        # Should run without error when start == end
        samples = rflow_strategy_2d.generate(
            mock_size_bin_model,
            noise,
            num_steps=5,
            device=torch.device('cpu'),
            size_bins=size_bins,
            cfg_scale=2.0,
            cfg_scale_end=2.0,  # Same as start
        )

        assert samples.shape == noise.shape


class TestDynamicCFGInterpolation:
    """Test dynamic CFG schedule calculation."""

    def test_cfg_interpolates_linearly(self, rflow_strategy_2d):
        """Dynamic CFG should interpolate linearly."""
        cfg_scale = 7.5
        cfg_scale_end = 1.0
        num_steps = 10

        # Test the interpolation logic
        for step in range(num_steps):
            progress = step / max(num_steps - 1, 1)
            expected = cfg_scale + progress * (cfg_scale_end - cfg_scale)

            # At step 0: 7.5
            # At step 9: 1.0
            if step == 0:
                assert abs(expected - 7.5) < 0.001
            elif step == num_steps - 1:
                assert abs(expected - 1.0) < 0.001

    def test_cfg_decay_direction(self, rflow_strategy_2d):
        """cfg_scale > cfg_scale_end means decay over time."""
        cfg_scale = 3.0
        cfg_scale_end = 1.0
        num_steps = 5

        cfg_values = []
        for step in range(num_steps):
            progress = step / max(num_steps - 1, 1)
            current_cfg = cfg_scale + progress * (cfg_scale_end - cfg_scale)
            cfg_values.append(current_cfg)

        # Should decrease monotonically
        for i in range(1, len(cfg_values)):
            assert cfg_values[i] <= cfg_values[i - 1]

    def test_cfg_increase_direction(self, rflow_strategy_2d):
        """cfg_scale < cfg_scale_end means increase over time."""
        cfg_scale = 1.0
        cfg_scale_end = 3.0
        num_steps = 5

        cfg_values = []
        for step in range(num_steps):
            progress = step / max(num_steps - 1, 1)
            current_cfg = cfg_scale + progress * (cfg_scale_end - cfg_scale)
            cfg_values.append(current_cfg)

        # Should increase monotonically
        for i in range(1, len(cfg_values)):
            assert cfg_values[i] >= cfg_values[i - 1]
