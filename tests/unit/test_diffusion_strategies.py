"""Tests for diffusion strategies: DDPMStrategy, RFlowStrategy."""

import pytest
import torch
from unittest.mock import Mock, patch

from medgen.diffusion.strategies import (
    DDPMStrategy,
    RFlowStrategy,
    ParsedModelInput,
)


class TestParsedModelInput:
    """Test ParsedModelInput container."""

    def test_all_fields_accessible(self):
        """All fields can be accessed."""
        parsed = ParsedModelInput(
            noisy_images=torch.randn(2, 1, 64, 64),
            noisy_pre=None,
            noisy_gd=None,
            conditioning=torch.randn(2, 1, 64, 64),
            is_dual=False,
        )
        assert parsed.noisy_images is not None
        assert parsed.conditioning is not None
        assert parsed.is_dual is False


class TestParseModelInput:
    """Test _parse_model_input method."""

    def test_1_channel_unconditional(self):
        """1 channel -> unconditional mode."""
        strategy = DDPMStrategy()
        model_input = torch.randn(4, 1, 64, 64)
        parsed = strategy._parse_model_input(model_input)

        assert parsed.noisy_images.shape == (4, 1, 64, 64)
        assert parsed.conditioning is None
        assert parsed.is_dual is False

    def test_2_channels_conditional_single(self):
        """2 channels -> [noise, conditioning]."""
        strategy = DDPMStrategy()
        model_input = torch.randn(4, 2, 64, 64)
        parsed = strategy._parse_model_input(model_input)

        assert parsed.noisy_images.shape == (4, 1, 64, 64)
        assert parsed.conditioning.shape == (4, 1, 64, 64)
        assert parsed.is_dual is False

    def test_3_channels_conditional_dual(self):
        """3 channels -> [noise_pre, noise_gd, conditioning]."""
        strategy = DDPMStrategy()
        model_input = torch.randn(4, 3, 64, 64)
        parsed = strategy._parse_model_input(model_input)

        assert parsed.noisy_pre.shape == (4, 1, 64, 64)
        assert parsed.noisy_gd.shape == (4, 1, 64, 64)
        assert parsed.conditioning.shape == (4, 1, 64, 64)
        assert parsed.is_dual is True

    def test_works_with_5d_input(self):
        """Also works with 3D volumes [B, C, D, H, W]."""
        strategy = DDPMStrategy()
        model_input = torch.randn(2, 2, 16, 64, 64)
        parsed = strategy._parse_model_input(model_input)

        assert parsed.noisy_images.shape == (2, 1, 16, 64, 64)
        assert parsed.conditioning.shape == (2, 1, 16, 64, 64)


class TestCallModel:
    """Test _call_model dispatcher."""

    def test_size_bins_wrapper_detection(self):
        """Detects SizeBinModelWrapper via hasattr."""
        strategy = DDPMStrategy()
        strategy.setup_scheduler(num_timesteps=100, image_size=64)

        # Model with size_bin_time_embed attribute
        model = Mock()
        model.size_bin_time_embed = Mock()
        model.return_value = torch.randn(4, 1, 64, 64)

        model_input = torch.randn(4, 1, 64, 64)
        timesteps = torch.randint(0, 100, (4,))
        size_bins = torch.randint(0, 5, (4, 7))

        result = strategy._call_model(model, model_input, timesteps, None, None, size_bins)

        # Should call with size_bins parameter
        model.assert_called_once()
        call_kwargs = model.call_args[1]
        assert 'size_bins' in call_kwargs
        # Verify actual size_bins value was passed correctly
        assert torch.equal(call_kwargs['size_bins'], size_bins), \
            "Wrong size_bins passed to model"

    def test_omega_mode_id_passed(self):
        """omega and mode_id passed for ScoreAug/ModeEmbed."""
        strategy = DDPMStrategy()
        strategy.setup_scheduler(num_timesteps=100, image_size=64)

        model = Mock(return_value=torch.randn(4, 1, 64, 64))

        model_input = torch.randn(4, 1, 64, 64)
        timesteps = torch.randint(0, 100, (4,))
        omega = torch.rand(4)
        mode_id = torch.zeros(4, dtype=torch.long)

        result = strategy._call_model(model, model_input, timesteps, omega, mode_id, None)

        model.assert_called_once()
        call_kwargs = model.call_args[1]
        assert 'omega' in call_kwargs
        assert 'mode_id' in call_kwargs
        # Verify actual values were passed correctly
        assert torch.allclose(call_kwargs['omega'], omega), \
            "Wrong omega passed to model"
        assert torch.equal(call_kwargs['mode_id'], mode_id), \
            "Wrong mode_id passed to model"

    def test_basic_call_without_conditioning(self):
        """Falls back to model(x=input, timesteps=t)."""
        strategy = DDPMStrategy()
        strategy.setup_scheduler(num_timesteps=100, image_size=64)

        model = Mock(return_value=torch.randn(4, 1, 64, 64))

        model_input = torch.randn(4, 1, 64, 64)
        timesteps = torch.randint(0, 100, (4,))

        result = strategy._call_model(model, model_input, timesteps, None, None, None)

        model.assert_called_once()
        call_kwargs = model.call_args[1]
        assert 'x' in call_kwargs
        assert 'timesteps' in call_kwargs


class TestDDPMStrategy:
    """Test DDPMStrategy implementation."""

    def test_setup_scheduler(self, ddpm_strategy):
        """Creates DDPMScheduler."""
        assert ddpm_strategy.scheduler is not None
        assert ddpm_strategy.scheduler.num_train_timesteps == 100

    def test_sample_timesteps_shape(self, ddpm_strategy):
        """sample_timesteps returns correct shape."""
        images = torch.randn(4, 1, 64, 64)
        timesteps = ddpm_strategy.sample_timesteps(images)

        assert timesteps.shape == (4,)
        assert timesteps.dtype == torch.long
        assert (timesteps >= 0).all()
        assert (timesteps < 100).all()

    def test_sample_timesteps_curriculum(self, ddpm_strategy):
        """Curriculum range restricts timestep sampling."""
        images = torch.randn(4, 1, 64, 64)
        curriculum_range = (0.2, 0.5)  # 20-50% of timesteps
        timesteps = ddpm_strategy.sample_timesteps(images, curriculum_range)

        min_t = int(0.2 * 100)
        max_t = int(0.5 * 100)
        assert (timesteps >= min_t).all()
        assert (timesteps <= max_t).all()

    def test_add_noise_single(self, ddpm_strategy):
        """Noise added via scheduler.add_noise."""
        images = torch.randn(4, 1, 64, 64)
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, 100, (4,))

        noisy = ddpm_strategy.add_noise(images, noise, timesteps)

        assert noisy.shape == images.shape
        # Noisy should be different from original (non-zero timesteps)
        assert not torch.allclose(noisy, images)

    def test_add_noise_formula(self, ddpm_strategy):
        """Verify DDPM noising formula: noisy = sqrt(alpha_t) * x + sqrt(1-alpha_t) * noise."""
        images = torch.randn(4, 1, 64, 64)
        noise = torch.randn_like(images)
        # Use uniform timestep for easier verification
        timestep_val = 50
        timesteps = torch.tensor([timestep_val, timestep_val, timestep_val, timestep_val])

        noisy = ddpm_strategy.add_noise(images, noise, timesteps)

        # Get alpha_cumprod at timestep
        alphas_cumprod = ddpm_strategy.scheduler.alphas_cumprod
        alpha_t = alphas_cumprod[timestep_val]

        # DDPM formula: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
        expected = torch.sqrt(alpha_t) * images + torch.sqrt(1 - alpha_t) * noise

        assert torch.allclose(noisy, expected, atol=1e-5), \
            "DDPM noising doesn't match expected formula"

    def test_add_noise_dict(self, ddpm_strategy):
        """Dict format applies noise to each key."""
        images = {
            'pre': torch.randn(4, 1, 64, 64),
            'gd': torch.randn(4, 1, 64, 64),
        }
        noise = {
            'pre': torch.randn(4, 1, 64, 64),
            'gd': torch.randn(4, 1, 64, 64),
        }
        timesteps = torch.randint(0, 100, (4,))

        noisy = ddpm_strategy.add_noise(images, noise, timesteps)

        assert isinstance(noisy, dict)
        assert 'pre' in noisy
        assert 'gd' in noisy


class TestRFlowStrategy:
    """Test RFlowStrategy implementation."""

    @pytest.mark.parametrize("strategy_fixture,desc", [
        ("rflow_strategy_2d", "2D"),
        ("rflow_strategy_3d", "3D with depth_size"),
    ])
    def test_setup_scheduler(self, strategy_fixture, desc, request):
        """Creates RFlowScheduler for both 2D and 3D."""
        strategy = request.getfixturevalue(strategy_fixture)
        assert strategy.scheduler is not None
        assert strategy.scheduler.num_train_timesteps == 100

    def test_setup_scheduler_requires_depth_for_3d(self):
        """ValueError if 3D without depth_size."""
        strategy = RFlowStrategy()
        with pytest.raises(ValueError, match="depth_size"):
            strategy.setup_scheduler(
                num_timesteps=100,
                image_size=64,
                spatial_dims=3,
                # depth_size missing
            )

    def test_sample_timesteps_shape(self, rflow_strategy_2d):
        """sample_timesteps returns correct shape."""
        images = torch.randn(4, 1, 64, 64)
        timesteps = rflow_strategy_2d.sample_timesteps(images)

        assert timesteps.shape == (4,)
        # RFlow can use continuous timesteps
        assert timesteps.dtype in (torch.float32, torch.float64, torch.long)

    def test_sample_timesteps_curriculum(self, rflow_strategy_2d):
        """Curriculum restricts range."""
        images = torch.randn(4, 1, 64, 64)
        curriculum_range = (0.2, 0.5)
        timesteps = rflow_strategy_2d.sample_timesteps(images, curriculum_range)

        # Check timesteps are in expected range (scaled to num_train_timesteps)
        min_t = 0.2 * 100
        max_t = 0.5 * 100
        assert (timesteps >= min_t - 1).all()  # Allow small tolerance
        assert (timesteps <= max_t + 1).all()

    def test_add_noise_single(self, rflow_strategy_2d):
        """Noise interpolation for RFlow."""
        images = torch.randn(4, 1, 64, 64)
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, 100, (4,))

        noisy = rflow_strategy_2d.add_noise(images, noise, timesteps)

        assert noisy.shape == images.shape

    def test_compute_loss_returns_tuple(self, rflow_strategy_2d):
        """compute_loss returns (loss, predicted_clean)."""
        images = torch.randn(4, 1, 64, 64)
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, 100, (4,))
        noisy = rflow_strategy_2d.add_noise(images, noise, timesteps)

        # Mock velocity prediction
        velocity_pred = torch.randn_like(images)

        loss, predicted_clean = rflow_strategy_2d.compute_loss(
            velocity_pred, images, noise, noisy, timesteps
        )

        assert isinstance(loss, torch.Tensor)
        assert predicted_clean.shape == images.shape


class TestExpandToBroadcast:
    """Test _expand_to_broadcast helper."""

    @pytest.mark.parametrize("t_values,ref_shape,expected_shape,desc", [
        ([0.5, 0.3, 0.7, 0.9], (4, 1, 64, 64), (4, 1, 1, 1), "2D: [B] -> [B, 1, 1, 1]"),
        ([0.5, 0.3], (2, 1, 16, 64, 64), (2, 1, 1, 1, 1), "3D: [B] -> [B, 1, 1, 1, 1]"),
    ])
    def test_expansion(self, t_values, ref_shape, expected_shape, desc):
        """Expansion works for both 4D and 5D tensors."""
        strategy = DDPMStrategy()
        t = torch.tensor(t_values)
        reference = torch.randn(*ref_shape)

        expanded = strategy._expand_to_broadcast(t, reference)

        assert expanded.shape == expected_shape


class TestSliceChannel:
    """Test _slice_channel helper."""

    @pytest.mark.parametrize("input_shape,start,end,expected_shape,desc", [
        ((4, 3, 64, 64), 1, 2, (4, 1, 64, 64), "4D slicing"),
        ((2, 3, 16, 64, 64), 2, 3, (2, 1, 16, 64, 64), "5D slicing"),
    ])
    def test_slicing(self, input_shape, start, end, expected_shape, desc):
        """Channel slicing works for both 4D and 5D."""
        strategy = DDPMStrategy()
        tensor = torch.randn(*input_shape)

        sliced = strategy._slice_channel(tensor, start, end)

        assert sliced.shape == expected_shape
