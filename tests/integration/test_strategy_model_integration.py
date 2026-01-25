"""Integration tests for diffusion strategy + real model interaction.

Tests DDPM and RFlow strategies with actual (tiny) model architectures,
verifying end-to-end training and generation workflows.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DiffusionModelUNet


# Skip all tests if CUDA unavailable
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Integration tests require GPU"
)


class TestDDPMModelIntegration:
    """DDPM strategy with real model end-to-end tests."""

    @pytest.fixture
    def strategy_and_model(self, device):
        """DDPM strategy with tiny DiffusionModelUNet."""
        from medgen.diffusion import DDPMStrategy

        strategy = DDPMStrategy()
        strategy.setup_scheduler(num_timesteps=100, image_size=64)

        model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=2,  # noisy image + conditioning
            out_channels=1,  # noise prediction
            channels=(32, 64),  # Minimum viable sizes
            num_res_blocks=1,
            attention_levels=(False, False),
            norm_num_groups=8,
        ).to(device)

        return strategy, model

    @pytest.mark.gpu
    @pytest.mark.timeout(30)
    def test_training_step_end_to_end(self, strategy_and_model, device, deterministic_seed):
        """Full forward -> loss -> backward training step."""
        strategy, model = strategy_and_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create training data
        torch.manual_seed(deterministic_seed)
        images = torch.rand(2, 1, 64, 64, device=device)
        conditioning = torch.rand(2, 1, 64, 64, device=device)

        # Sample timesteps and add noise
        timesteps = strategy.sample_timesteps(images)
        noise = torch.randn_like(images)
        noisy_images = strategy.add_noise(images, noise, timesteps)

        # Forward pass (DiffusionModelUNet requires timesteps)
        model_input = torch.cat([noisy_images, conditioning], dim=1)
        prediction = model(x=model_input, timesteps=timesteps)

        # Compute loss
        loss, _ = strategy.compute_loss(
            prediction=prediction,
            target_images=images,
            noise=noise,
            noisy_images=noisy_images,
            timesteps=timesteps,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Missing gradient"

    @pytest.mark.gpu
    @pytest.mark.timeout(60)
    def test_generation_produces_valid_output(self, strategy_and_model, device, deterministic_seed):
        """Generation returns finite values with correct shape."""
        strategy, model = strategy_and_model
        model.eval()

        torch.manual_seed(deterministic_seed)

        # Create model input: [noise, conditioning]
        noise = torch.randn(2, 1, 64, 64, device=device)
        conditioning = torch.rand(2, 1, 64, 64, device=device)
        model_input = torch.cat([noise, conditioning], dim=1)

        # Generate (fewer steps for speed)
        with torch.no_grad():
            generated = strategy.generate(
                model=model,
                model_input=model_input,
                num_steps=10,  # Reduced for speed
                device=device,
            )

        # Verify output
        assert generated.shape == (2, 1, 64, 64), f"Unexpected shape: {generated.shape}"
        assert torch.isfinite(generated).all(), "Generation contains NaN/Inf"

    @pytest.mark.gpu
    @pytest.mark.timeout(30)
    def test_timestep_sampling_in_range(self, strategy_and_model, device):
        """Sampled timesteps are within valid range [0, num_timesteps)."""
        strategy, model = strategy_and_model

        images = torch.rand(4, 1, 64, 64, device=device)

        for _ in range(10):  # Test multiple samples
            timesteps = strategy.sample_timesteps(images)

            assert timesteps.shape == (4,), f"Unexpected shape: {timesteps.shape}"
            assert (timesteps >= 0).all(), "Timesteps below 0"
            assert (timesteps < 100).all(), "Timesteps >= num_timesteps"
            assert timesteps.dtype == torch.long, "Timesteps should be integers"


class TestRFlowModelIntegration:
    """RFlow strategy with real model end-to-end tests."""

    @pytest.fixture
    def rflow_strategy_and_model(self, device):
        """RFlow strategy with tiny DiffusionModelUNet."""
        from medgen.diffusion import RFlowStrategy

        strategy = RFlowStrategy()
        strategy.setup_scheduler(
            num_timesteps=100,
            image_size=64,
            spatial_dims=2,
        )

        model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=2,  # noisy image + conditioning
            out_channels=1,  # velocity prediction
            channels=(32, 64),
            num_res_blocks=1,
            attention_levels=(False, False),
            norm_num_groups=8,
        ).to(device)

        return strategy, model

    @pytest.mark.gpu
    @pytest.mark.timeout(30)
    def test_velocity_prediction_training(self, rflow_strategy_and_model, device, deterministic_seed):
        """RFlow training step predicts velocity, not noise."""
        strategy, model = rflow_strategy_and_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create training data
        torch.manual_seed(deterministic_seed)
        images = torch.rand(2, 1, 64, 64, device=device)
        conditioning = torch.rand(2, 1, 64, 64, device=device)

        # Sample timesteps and add noise
        timesteps = strategy.sample_timesteps(images)
        noise = torch.randn_like(images)
        noisy_images = strategy.add_noise(images, noise, timesteps)

        # Forward pass (DiffusionModelUNet requires timesteps)
        model_input = torch.cat([noisy_images, conditioning], dim=1)
        prediction = model(x=model_input, timesteps=timesteps)

        # Compute loss (RFlow uses velocity target: x_0 - x_1)
        loss, predicted_clean = strategy.compute_loss(
            prediction=prediction,
            target_images=images,
            noise=noise,
            noisy_images=noisy_images,
            timesteps=timesteps,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"
        assert predicted_clean.shape == images.shape, "Predicted clean shape mismatch"

    @pytest.mark.gpu
    @pytest.mark.timeout(60)
    def test_generation_with_euler(self, rflow_strategy_and_model, device, deterministic_seed):
        """RFlow Euler integration produces valid samples."""
        strategy, model = rflow_strategy_and_model
        model.eval()

        torch.manual_seed(deterministic_seed)

        # Create model input
        noise = torch.randn(2, 1, 64, 64, device=device)
        conditioning = torch.rand(2, 1, 64, 64, device=device)
        model_input = torch.cat([noise, conditioning], dim=1)

        # Generate with Euler integration
        with torch.no_grad():
            generated = strategy.generate(
                model=model,
                model_input=model_input,
                num_steps=10,
                device=device,
            )

        # Verify output
        assert generated.shape == (2, 1, 64, 64), f"Unexpected shape: {generated.shape}"
        assert torch.isfinite(generated).all(), "Generation contains NaN/Inf"

    @pytest.mark.gpu
    @pytest.mark.timeout(30)
    def test_continuous_timesteps(self, rflow_strategy_and_model, device):
        """RFlow can use continuous (float) timesteps."""
        strategy, model = rflow_strategy_and_model

        images = torch.rand(4, 1, 64, 64, device=device)

        # Sample timesteps
        timesteps = strategy.sample_timesteps(images)

        assert timesteps.shape == (4,), f"Unexpected shape: {timesteps.shape}"
        # RFlow timesteps are scaled to [0, num_timesteps]
        assert (timesteps >= 0).all(), "Timesteps below 0"
        assert (timesteps <= 100).all(), "Timesteps > num_timesteps"


class TestStrategyModelDevice:
    """Tests for device consistency between model and data."""

    @pytest.fixture
    def ddpm_on_device(self, device):
        """DDPM strategy with model on device."""
        from medgen.diffusion import DDPMStrategy

        strategy = DDPMStrategy()
        strategy.setup_scheduler(num_timesteps=100, image_size=64)

        model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=2,
            out_channels=1,
            channels=(32, 64),
            num_res_blocks=1,
            attention_levels=(False, False),
            norm_num_groups=8,
        ).to(device)

        return strategy, model

    @pytest.mark.gpu
    @pytest.mark.timeout(30)
    def test_model_and_noise_same_device(self, ddpm_on_device, device):
        """Model input and noise are on same device."""
        strategy, model = ddpm_on_device

        images = torch.rand(2, 1, 64, 64, device=device)
        noise = torch.randn_like(images)  # Same device as images

        timesteps = strategy.sample_timesteps(images)
        noisy_images = strategy.add_noise(images, noise, timesteps)

        assert noisy_images.device == images.device, "Noisy images on wrong device"
        assert timesteps.device == images.device, "Timesteps on wrong device"

    @pytest.mark.gpu
    @pytest.mark.timeout(60)
    def test_generation_respects_device(self, ddpm_on_device, device):
        """Generated output is on same device as input."""
        strategy, model = ddpm_on_device
        model.eval()

        noise = torch.randn(2, 1, 64, 64, device=device)
        conditioning = torch.rand(2, 1, 64, 64, device=device)
        model_input = torch.cat([noise, conditioning], dim=1)

        with torch.no_grad():
            generated = strategy.generate(
                model=model,
                model_input=model_input,
                num_steps=5,
                device=device,
            )

        # Compare device type (cuda:0 should match cuda)
        assert generated.device.type == device.type, f"Output on {generated.device}, expected {device}"


class Test3DStrategyIntegration:
    """3D strategy + model integration tests (marked slow)."""

    @pytest.fixture
    def rflow_3d_strategy_and_model(self, device):
        """RFlow 3D strategy with tiny 3D DiffusionModelUNet."""
        from medgen.diffusion import RFlowStrategy

        strategy = RFlowStrategy()
        strategy.setup_scheduler(
            num_timesteps=100,
            image_size=32,  # Smaller for 3D
            depth_size=16,
            spatial_dims=3,
        )

        model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            channels=(16, 32),  # Minimal but viable for 3D
            num_res_blocks=1,
            attention_levels=(False, False),
            norm_num_groups=8,
        ).to(device)

        return strategy, model

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_3d_training_step(self, rflow_3d_strategy_and_model, device, deterministic_seed):
        """3D RFlow training step works end-to-end."""
        strategy, model = rflow_3d_strategy_and_model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create 3D data: [B, C, D, H, W]
        torch.manual_seed(deterministic_seed)
        volumes = torch.rand(1, 1, 16, 32, 32, device=device)
        conditioning = torch.rand(1, 1, 16, 32, 32, device=device)

        # Sample timesteps and add noise
        timesteps = strategy.sample_timesteps(volumes)
        noise = torch.randn_like(volumes)
        noisy_volumes = strategy.add_noise(volumes, noise, timesteps)

        # Forward pass (DiffusionModelUNet requires timesteps)
        model_input = torch.cat([noisy_volumes, conditioning], dim=1)
        prediction = model(x=model_input, timesteps=timesteps)

        # Compute loss
        loss, _ = strategy.compute_loss(
            prediction=prediction,
            target_images=volumes,
            noise=noise,
            noisy_images=noisy_volumes,
            timesteps=timesteps,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"

    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(90)
    def test_3d_generation(self, rflow_3d_strategy_and_model, device, deterministic_seed):
        """3D generation produces valid volumes."""
        strategy, model = rflow_3d_strategy_and_model
        model.eval()

        torch.manual_seed(deterministic_seed)

        # Create model input: [noise, conditioning]
        noise = torch.randn(1, 1, 16, 32, 32, device=device)
        conditioning = torch.rand(1, 1, 16, 32, 32, device=device)
        model_input = torch.cat([noise, conditioning], dim=1)

        # Generate (very few steps for 3D)
        with torch.no_grad():
            generated = strategy.generate(
                model=model,
                model_input=model_input,
                num_steps=5,
                device=device,
            )

        # Verify
        assert generated.shape == (1, 1, 16, 32, 32), f"Unexpected shape: {generated.shape}"
        assert torch.isfinite(generated).all(), "Generation contains NaN/Inf"


class TestCurriculumLearning:
    """Tests for curriculum learning timestep restriction."""

    @pytest.fixture
    def ddpm_strategy(self, device):
        """DDPM strategy for curriculum testing."""
        from medgen.diffusion import DDPMStrategy

        strategy = DDPMStrategy()
        strategy.setup_scheduler(num_timesteps=1000, image_size=64)
        return strategy

    @pytest.mark.gpu
    @pytest.mark.timeout(30)
    def test_curriculum_restricts_timesteps(self, ddpm_strategy, device):
        """Curriculum range restricts sampled timesteps."""
        strategy = ddpm_strategy
        images = torch.rand(4, 1, 64, 64, device=device)

        # Restrict to early timesteps (0-50%)
        curriculum_range = (0.0, 0.5)

        for _ in range(10):
            timesteps = strategy.sample_timesteps(images, curriculum_range=curriculum_range)

            # Should be in [0, 500) for 1000 total timesteps
            assert (timesteps >= 0).all(), "Timesteps below curriculum min"
            assert (timesteps < 500).all(), "Timesteps above curriculum max"

    @pytest.mark.gpu
    @pytest.mark.timeout(30)
    def test_curriculum_late_timesteps(self, ddpm_strategy, device):
        """Curriculum can restrict to late timesteps only."""
        strategy = ddpm_strategy
        images = torch.rand(4, 1, 64, 64, device=device)

        # Restrict to late timesteps (50-100%)
        curriculum_range = (0.5, 1.0)

        for _ in range(10):
            timesteps = strategy.sample_timesteps(images, curriculum_range=curriculum_range)

            # Should be in [500, 1000) for 1000 total timesteps
            assert (timesteps >= 500).all(), "Timesteps below curriculum min"
            assert (timesteps < 1000).all(), "Timesteps above curriculum max"


class TestUnconditionalGeneration:
    """Tests for unconditional (1-channel) generation."""

    @pytest.fixture
    def unconditional_setup(self, device):
        """Setup for unconditional generation (no conditioning)."""
        from medgen.diffusion import DDPMStrategy

        strategy = DDPMStrategy()
        strategy.setup_scheduler(num_timesteps=100, image_size=64)

        # Unconditional: 1 channel input, 1 channel output
        model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(32, 64),
            num_res_blocks=1,
            attention_levels=(False, False),
            norm_num_groups=8,
        ).to(device)

        return strategy, model

    @pytest.mark.gpu
    @pytest.mark.timeout(60)
    def test_unconditional_generation(self, unconditional_setup, device, deterministic_seed):
        """Unconditional generation works without conditioning."""
        strategy, model = unconditional_setup
        model.eval()

        torch.manual_seed(deterministic_seed)

        # Just noise, no conditioning
        noise = torch.randn(2, 1, 64, 64, device=device)

        with torch.no_grad():
            generated = strategy.generate(
                model=model,
                model_input=noise,  # No conditioning concatenated
                num_steps=10,
                device=device,
            )

        # Verify
        assert generated.shape == (2, 1, 64, 64), f"Unexpected shape: {generated.shape}"
        assert torch.isfinite(generated).all(), "Generation contains NaN/Inf"
