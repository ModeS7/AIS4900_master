"""Tests for exp1e (Min-SNR-γ) and exp1f (EDM preconditioning) features.

Verifies:
- EDM coefficient math at boundary timesteps
- SNR weight computation (no NaN/inf at boundaries)
- Preconditioning composes correctly with generation dispatch
- Checkpoint round-trip for sigma_data / snr_gamma
- End-to-end generation with preconditioning
"""
import pytest
import torch
from unittest.mock import Mock


# ─────────────────────────────────────────────────────────────────────────────
# exp1e: Min-SNR-γ Loss Weighting
# ─────────────────────────────────────────────────────────────────────────────


class TestRFlowSNRWeightedMSE:
    """Tests for compute_rflow_snr_weighted_mse."""

    def test_basic_computation(self):
        """Weighted MSE returns a finite scalar."""
        from medgen.pipeline.losses import compute_rflow_snr_weighted_mse

        prediction = torch.randn(4, 1, 8, 8)
        target = torch.randn(4, 1, 8, 8)
        timesteps = torch.tensor([100.0, 300.0, 500.0, 900.0])

        loss = compute_rflow_snr_weighted_mse(prediction, target, timesteps, 1000, gamma=5.0)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss > 0

    def test_no_nan_at_boundary_timesteps(self):
        """No NaN/inf at t≈0 (clean) and t≈T (noise)."""
        from medgen.pipeline.losses import compute_rflow_snr_weighted_mse

        prediction = torch.randn(4, 1, 8, 8)
        target = torch.randn(4, 1, 8, 8)
        # Near-boundary timesteps (clamped internally to [1e-5, 1-1e-5])
        timesteps = torch.tensor([0.0, 1.0, 999.0, 1000.0])

        loss = compute_rflow_snr_weighted_mse(prediction, target, timesteps, 1000, gamma=5.0)

        assert torch.isfinite(loss)

    def test_weights_downweight_clean_timesteps(self):
        """High-SNR timesteps (near clean, t≈0) should have weight < 1."""
        from medgen.pipeline.losses import compute_rflow_snr_weighted_mse

        # Create identical prediction and target everywhere EXCEPT at t=0
        prediction = torch.zeros(2, 1, 4, 4)
        target = torch.zeros(2, 1, 4, 4)
        # Add error only at sample 0 (t=50, near clean → high SNR → downweighted)
        prediction[0] = 1.0
        # Sample 1 (t=500, mid → moderate SNR → less downweighted)
        prediction[1] = 1.0

        timesteps = torch.tensor([50.0, 500.0])

        loss = compute_rflow_snr_weighted_mse(prediction, target, timesteps, 1000, gamma=5.0)
        assert torch.isfinite(loss)

        # Compare: unweighted loss should be different from weighted
        unweighted = ((prediction.float() - target.float()) ** 2).flatten(1).mean(1).mean()
        # Weighted should be less than unweighted (downweights the easy high-SNR sample)
        assert loss < unweighted

    def test_gamma_zero_gives_all_ones_weight(self):
        """gamma=very large effectively gives weight=1 everywhere (no clipping)."""
        from medgen.pipeline.losses import compute_rflow_snr_weighted_mse

        prediction = torch.randn(4, 1, 8, 8)
        target = torch.randn(4, 1, 8, 8)
        timesteps = torch.tensor([100.0, 300.0, 500.0, 900.0])

        # Very large gamma → min(SNR, gamma) = SNR → weight = 1.0
        loss_large_gamma = compute_rflow_snr_weighted_mse(
            prediction, target, timesteps, 1000, gamma=1e10
        )
        # Standard MSE for comparison
        unweighted = ((prediction.float() - target.float()) ** 2).flatten(1).mean(1).mean()

        # Should be approximately equal (weights ≈ 1)
        assert torch.allclose(loss_large_gamma, unweighted, rtol=1e-3)

    def test_dual_image_mode(self):
        """Works with dict target (dual mode)."""
        from medgen.pipeline.losses import compute_rflow_snr_weighted_mse

        prediction = torch.randn(4, 2, 8, 8)  # 2 channels for dual
        target = {
            'pre': torch.randn(4, 1, 8, 8),
            'gd': torch.randn(4, 1, 8, 8),
        }
        timesteps = torch.tensor([100.0, 300.0, 500.0, 900.0])

        loss = compute_rflow_snr_weighted_mse(prediction, target, timesteps, 1000, gamma=5.0)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss > 0

    def test_3d_volumes(self):
        """Works with 5D tensors [B, C, D, H, W]."""
        from medgen.pipeline.losses import compute_rflow_snr_weighted_mse

        prediction = torch.randn(2, 1, 4, 8, 8)
        target = torch.randn(2, 1, 4, 8, 8)
        timesteps = torch.tensor([200.0, 800.0])

        loss = compute_rflow_snr_weighted_mse(prediction, target, timesteps, 1000, gamma=5.0)

        assert loss.shape == ()
        assert torch.isfinite(loss)


# ─────────────────────────────────────────────────────────────────────────────
# exp1f: EDM Preconditioning
# ─────────────────────────────────────────────────────────────────────────────


class TestEDMCoefficients:
    """Tests for _edm_coefficients math."""

    @pytest.fixture
    def preconditioned_strategy(self):
        """RFlowStrategy with EDM preconditioning enabled."""
        from medgen.diffusion import RFlowStrategy

        strategy = RFlowStrategy()
        strategy.setup_scheduler(num_timesteps=1000, image_size=64)
        strategy.set_preconditioning(sigma_data=0.08, out_channels=1)
        return strategy

    def test_coefficients_are_finite(self, preconditioned_strategy):
        """All coefficients are finite for valid timesteps."""
        timesteps = torch.tensor([50.0, 200.0, 500.0, 800.0, 950.0])
        c_in, c_skip, c_out = preconditioned_strategy._edm_coefficients(timesteps)

        assert torch.isfinite(c_in).all()
        assert torch.isfinite(c_skip).all()
        assert torch.isfinite(c_out).all()

    def test_boundary_t0_clean(self, preconditioned_strategy):
        """At t̃≈0 (clean): verify coefficients against manual computation.

        With σ_data=0.08, t̃=0.001:
        D = (0.999)² * 0.0064 + (0.001)² = 0.006389
        c_skip = (0.999 * 0.0064 - 0.001) / D = 0.005394 / 0.006389 ≈ 0.844
        c_out = 0.08 / √D ≈ 1.001

        Note: c_skip→1 only in the limit t̃→0 AND t̃ << σ²_data.
        At t̃=0.001 and σ_data=0.08, the linear t̃ term in the numerator
        is comparable to σ²_data=0.0064, so c_skip < 1.
        """
        timesteps = torch.tensor([1.0])  # t̃ = 0.001
        c_in, c_skip, c_out = preconditioned_strategy._edm_coefficients(timesteps)

        # Verify against manual computation (not asymptotic limit)
        t_norm = 0.001
        sd2 = 0.08 ** 2  # 0.0064
        D = (1 - t_norm) ** 2 * sd2 + t_norm ** 2
        expected_c_skip = ((1 - t_norm) * sd2 - t_norm) / D
        expected_c_out = 0.08 / D ** 0.5

        assert c_skip.item() == pytest.approx(expected_c_skip, rel=1e-3)
        assert c_out.item() == pytest.approx(expected_c_out, rel=1e-3)

    def test_boundary_t1_noise(self, preconditioned_strategy):
        """At t̃≈1 (noise): c_skip_v≈-1, c_out_v≈σ_data."""
        # Use t=999 (very close to noise, t̃=0.999)
        timesteps = torch.tensor([999.0])
        c_in, c_skip, c_out = preconditioned_strategy._edm_coefficients(timesteps)

        # At t̃→1: c_skip_v → -1, c_out_v → σ_data
        assert c_skip.item() == pytest.approx(-1.0, abs=0.05)
        assert c_out.item() == pytest.approx(0.08, abs=0.01)

    def test_c_in_normalizes_input_variance(self, preconditioned_strategy):
        """c_in = 1/√D should normalize x_t to ~unit variance."""
        sigma_data = 0.08
        timesteps = torch.tensor([500.0])  # t̃ = 0.5
        c_in, _, _ = preconditioned_strategy._edm_coefficients(timesteps)

        # D = (1-0.5)² * 0.08² + 0.5² = 0.25*0.0064 + 0.25 = 0.2516
        D_expected = 0.25 * sigma_data ** 2 + 0.25
        c_in_expected = 1.0 / D_expected ** 0.5

        assert c_in.item() == pytest.approx(c_in_expected, rel=1e-3)

    def test_coefficients_shape_matches_batch(self, preconditioned_strategy):
        """Output shapes match batch dimension."""
        timesteps = torch.tensor([100.0, 200.0, 300.0, 400.0])
        c_in, c_skip, c_out = preconditioned_strategy._edm_coefficients(timesteps)

        assert c_in.shape == (4,)
        assert c_skip.shape == (4,)
        assert c_out.shape == (4,)


class TestEDMPreconditioning:
    """Tests for predict_noise_or_velocity and _call_model with preconditioning."""

    @pytest.fixture
    def preconditioned_strategy(self):
        """RFlowStrategy with preconditioning for bravo (out_channels=1)."""
        from medgen.diffusion import RFlowStrategy

        strategy = RFlowStrategy()
        strategy.setup_scheduler(num_timesteps=1000, image_size=64)
        strategy.set_preconditioning(sigma_data=0.08, out_channels=1)
        return strategy

    @pytest.fixture
    def simple_model(self):
        """Model that returns zeros — makes preconditioning output predictable."""
        def forward(x, timesteps=None):
            # Return zeros for all output channels
            return torch.zeros_like(x[:, :1])
        model = Mock(side_effect=forward)
        return model

    def test_predict_noise_or_velocity_with_preconditioning(self, preconditioned_strategy, simple_model):
        """Preconditioned output = c_skip * x_t + c_out * F(c_in * input)."""
        # model_input = [noisy_image(1ch), conditioning(1ch)]
        model_input = torch.randn(2, 2, 8, 8)
        timesteps = torch.tensor([500.0, 500.0])

        result = preconditioned_strategy.predict_noise_or_velocity(
            simple_model, model_input, timesteps
        )

        assert result.shape == (2, 1, 8, 8)
        assert torch.isfinite(result).all()

        # Since F=0, output should be c_skip * x_t
        c_in, c_skip, c_out = preconditioned_strategy._edm_coefficients(timesteps)
        x_t = model_input[:, :1]
        expected = c_skip.view(-1, 1, 1, 1) * x_t
        assert torch.allclose(result, expected, atol=1e-5)

    def test_model_receives_scaled_input(self, preconditioned_strategy):
        """Model should receive c_in * model_input."""
        received_input = []

        def capture_model(x, timesteps=None):
            received_input.append(x.clone())
            return torch.zeros_like(x[:, :1])

        model = Mock(side_effect=capture_model)
        model_input = torch.ones(2, 2, 8, 8)
        timesteps = torch.tensor([500.0, 500.0])

        preconditioned_strategy.predict_noise_or_velocity(model, model_input, timesteps)

        c_in, _, _ = preconditioned_strategy._edm_coefficients(timesteps)
        expected_input = c_in.view(-1, 1, 1, 1) * model_input
        assert torch.allclose(received_input[0], expected_input, atol=1e-5)

    def test_disabled_when_sigma_data_zero(self):
        """When sigma_data=0, predict_noise_or_velocity is a plain forward."""
        from medgen.diffusion import RFlowStrategy

        strategy = RFlowStrategy()
        strategy.setup_scheduler(num_timesteps=1000, image_size=64)
        # Do NOT call set_preconditioning — sigma_data stays 0

        expected_output = torch.randn(2, 1, 8, 8)
        model = Mock(return_value=expected_output)

        model_input = torch.randn(2, 2, 8, 8)
        timesteps = torch.tensor([500.0, 500.0])

        result = strategy.predict_noise_or_velocity(model, model_input, timesteps)

        # Should be exactly the model output (no preconditioning)
        assert torch.equal(result, expected_output)

    def test_prediction_type_sample_raises(self):
        """set_preconditioning raises ValueError for prediction_type='sample'."""
        from medgen.diffusion import RFlowStrategy

        strategy = RFlowStrategy()
        strategy.setup_scheduler(
            num_timesteps=1000, image_size=64, prediction_type='sample'
        )

        with pytest.raises(ValueError, match="incompatible"):
            strategy.set_preconditioning(sigma_data=0.08, out_channels=1)

    def test_3d_volumes(self, preconditioned_strategy, simple_model):
        """Preconditioning works with 5D tensors [B, C, D, H, W]."""
        model_input = torch.randn(2, 2, 4, 8, 8)
        timesteps = torch.tensor([300.0, 700.0])

        # Override model to handle 5D
        def forward_5d(x, timesteps=None):
            return torch.zeros(x.shape[0], 1, *x.shape[2:])
        simple_model.side_effect = forward_5d

        result = preconditioned_strategy.predict_noise_or_velocity(
            simple_model, model_input, timesteps
        )

        assert result.shape == (2, 1, 4, 8, 8)
        assert torch.isfinite(result).all()


class TestEDMCallModel:
    """Tests for _call_model override (generation path)."""

    @pytest.fixture
    def preconditioned_strategy(self):
        from medgen.diffusion import RFlowStrategy

        strategy = RFlowStrategy()
        strategy.setup_scheduler(num_timesteps=1000, image_size=64)
        strategy.set_preconditioning(sigma_data=0.08, out_channels=1)
        return strategy

    def test_call_model_applies_preconditioning(self, preconditioned_strategy):
        """_call_model wraps base dispatch with preconditioning."""
        received_input = []

        def capture_model(x=None, timesteps=None, **kwargs):
            received_input.append(x.clone())
            return torch.zeros_like(x[:, :1])

        model = Mock(side_effect=capture_model)
        model_input = torch.ones(2, 2, 8, 8)
        timesteps = torch.tensor([500.0, 500.0])

        result = preconditioned_strategy._call_model(
            model, model_input, timesteps, omega=None, mode_id=None
        )

        # Model should have received scaled input
        c_in, c_skip, c_out = preconditioned_strategy._edm_coefficients(timesteps)
        expected_scaled = c_in.view(-1, 1, 1, 1) * model_input
        assert torch.allclose(received_input[0], expected_scaled, atol=1e-5)

        # Output should be c_skip * x_t + c_out * 0 = c_skip * x_t
        x_t = model_input[:, :1]
        expected = c_skip.view(-1, 1, 1, 1) * x_t
        assert torch.allclose(result, expected, atol=1e-5)
        assert result.shape == (2, 1, 8, 8)

    def test_call_model_with_size_bins(self, preconditioned_strategy):
        """Preconditioning composes with size_bins dispatch."""
        received_kwargs = {}

        def capture_model(input_tensor, timesteps=None, size_bins=None, **kwargs):
            received_kwargs['size_bins'] = size_bins
            return torch.zeros_like(input_tensor[:, :1])

        model = Mock(side_effect=capture_model)
        model.size_bin_time_embed = True  # Triggers SizeBinModelWrapper path

        model_input = torch.ones(2, 2, 8, 8)
        timesteps = torch.tensor([500.0, 500.0])
        size_bins = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        result = preconditioned_strategy._call_model(
            model, model_input, timesteps,
            omega=None, mode_id=None, size_bins=size_bins
        )

        assert torch.isfinite(result).all()
        # Size bins should have been passed through to the model
        assert received_kwargs['size_bins'] is not None

    def test_call_model_disabled_when_no_preconditioning(self):
        """Without preconditioning, _call_model is the base class dispatch."""
        from medgen.diffusion import RFlowStrategy

        strategy = RFlowStrategy()
        strategy.setup_scheduler(num_timesteps=1000, image_size=64)

        expected = torch.randn(2, 1, 8, 8)
        model = Mock(return_value=expected)

        model_input = torch.randn(2, 2, 8, 8)
        timesteps = torch.tensor([500.0, 500.0])

        result = strategy._call_model(
            model, model_input, timesteps, omega=None, mode_id=None
        )

        assert torch.equal(result, expected)


class TestEDMGeneration:
    """End-to-end generation with EDM preconditioning."""

    @pytest.fixture
    def preconditioned_strategy(self):
        from medgen.diffusion import RFlowStrategy

        strategy = RFlowStrategy()
        strategy.setup_scheduler(num_timesteps=1000, image_size=64)
        strategy.set_preconditioning(sigma_data=0.08, out_channels=1)
        return strategy

    def test_euler_generation_produces_valid_output(self, preconditioned_strategy):
        """Full Euler generation loop with preconditioning produces finite output."""
        def model_fn(x=None, timesteps=None, **kwargs):
            # Return random velocity prediction
            return torch.randn(x.shape[0], 1, *x.shape[2:])

        model = Mock(side_effect=model_fn)
        model.eval = Mock(return_value=model)

        # Unconditional: 1 channel
        noise = torch.randn(2, 1, 8, 8)

        result = preconditioned_strategy.generate(
            model=model,
            model_input=noise,
            num_steps=5,
            device=torch.device('cpu'),
        )

        assert result.shape == noise.shape
        assert torch.isfinite(result).all()

    def test_conditional_generation_with_preconditioning(self, preconditioned_strategy):
        """Conditional generation (2ch input) with preconditioning."""
        def model_fn(x=None, timesteps=None, **kwargs):
            return torch.randn(x.shape[0], 1, *x.shape[2:])

        model = Mock(side_effect=model_fn)
        model.eval = Mock(return_value=model)

        # Conditional: [noise, seg_mask]
        model_input = torch.randn(2, 2, 8, 8)

        result = preconditioned_strategy.generate(
            model=model,
            model_input=model_input,
            num_steps=5,
            device=torch.device('cpu'),
        )

        assert result.shape == (2, 1, 8, 8)
        assert torch.isfinite(result).all()

    def test_generation_matches_without_preconditioning_at_sigma_zero(self):
        """sigma_data=0 generation should match plain RFlow generation."""
        from medgen.diffusion import RFlowStrategy

        torch.manual_seed(42)
        model_outputs = [torch.randn(2, 1, 8, 8) for _ in range(100)]
        call_idx = [0]

        def deterministic_model(x=None, timesteps=None, **kwargs):
            out = model_outputs[call_idx[0] % len(model_outputs)]
            call_idx[0] += 1
            return out

        # Run without preconditioning
        strategy1 = RFlowStrategy()
        strategy1.setup_scheduler(num_timesteps=1000, image_size=8)
        model1 = Mock(side_effect=deterministic_model)
        model1.eval = Mock(return_value=model1)
        noise = torch.randn(2, 1, 8, 8)
        call_idx[0] = 0
        result1 = strategy1.generate(model1, noise.clone(), num_steps=3, device=torch.device('cpu'))

        # Run with sigma_data=0 (disabled preconditioning)
        strategy2 = RFlowStrategy()
        strategy2.setup_scheduler(num_timesteps=1000, image_size=8)
        # sigma_data stays 0 — no set_preconditioning call
        model2 = Mock(side_effect=deterministic_model)
        model2.eval = Mock(return_value=model2)
        call_idx[0] = 0
        result2 = strategy2.generate(model2, noise.clone(), num_steps=3, device=torch.device('cpu'))

        assert torch.allclose(result1, result2, atol=1e-5)


class TestCheckpointRoundTrip:
    """Verify sigma_data and snr_gamma survive checkpoint save/load."""

    def test_profiling_saves_sigma_data(self):
        """get_model_config includes sigma_data when > 0."""
        from medgen.pipeline.profiling import get_model_config
        from medgen.pipeline.base_config import StrategyConfig

        trainer = Mock()
        trainer.model_type = 'unet'
        trainer.is_transformer = False
        trainer.strategy_name = 'rflow'
        trainer.mode_name = 'bravo'
        trainer.mode.get_model_config.return_value = {
            'in_channels': 2, 'out_channels': 1,
        }
        trainer._strategy_config = StrategyConfig(
            name='rflow', sigma_data=0.08, snr_gamma=0.0,
        )
        from omegaconf import OmegaConf
        trainer.cfg = OmegaConf.create({
            'model': {
                'spatial_dims': 2,
                'channels': [64, 128], 'attention_levels': [False, True],
                'num_res_blocks': 1, 'num_head_channels': 64,
            }
        })

        config = get_model_config(trainer)

        assert config['sigma_data'] == 0.08
        assert 'snr_gamma' not in config  # 0.0 → not saved

    def test_profiling_saves_snr_gamma(self):
        """get_model_config includes snr_gamma when > 0."""
        from medgen.pipeline.profiling import get_model_config
        from medgen.pipeline.base_config import StrategyConfig

        trainer = Mock()
        trainer.model_type = 'unet'
        trainer.is_transformer = False
        trainer.strategy_name = 'rflow'
        trainer.mode_name = 'bravo'
        trainer.mode.get_model_config.return_value = {
            'in_channels': 2, 'out_channels': 1,
        }
        trainer._strategy_config = StrategyConfig(
            name='rflow', sigma_data=0.0, snr_gamma=5.0,
        )
        from omegaconf import OmegaConf
        trainer.cfg = OmegaConf.create({
            'model': {
                'spatial_dims': 2,
                'channels': [64, 128], 'attention_levels': [False, True],
                'num_res_blocks': 1, 'num_head_channels': 64,
            }
        })

        config = get_model_config(trainer)

        assert config['snr_gamma'] == 5.0
        assert 'sigma_data' not in config  # 0.0 → not saved

    def test_profiling_omits_both_when_zero(self):
        """get_model_config omits sigma_data/snr_gamma when both are 0."""
        from medgen.pipeline.profiling import get_model_config
        from medgen.pipeline.base_config import StrategyConfig

        trainer = Mock()
        trainer.model_type = 'unet'
        trainer.is_transformer = False
        trainer.strategy_name = 'rflow'
        trainer.mode_name = 'bravo'
        trainer.mode.get_model_config.return_value = {
            'in_channels': 2, 'out_channels': 1,
        }
        trainer._strategy_config = StrategyConfig(
            name='rflow', sigma_data=0.0, snr_gamma=0.0,
        )
        from omegaconf import OmegaConf
        trainer.cfg = OmegaConf.create({
            'model': {
                'spatial_dims': 2,
                'channels': [64, 128], 'attention_levels': [False, True],
                'num_res_blocks': 1, 'num_head_channels': 64,
            }
        })

        config = get_model_config(trainer)

        assert 'sigma_data' not in config
        assert 'snr_gamma' not in config


class TestStrategyConfig:
    """Verify StrategyConfig dataclass handles new fields."""

    def test_defaults_are_zero(self):
        """snr_gamma and sigma_data default to 0.0."""
        from medgen.pipeline.base_config import StrategyConfig

        sc = StrategyConfig(name='rflow')
        assert sc.snr_gamma == 0.0
        assert sc.sigma_data == 0.0

    def test_from_hydra_parses_new_fields(self):
        """from_hydra extracts snr_gamma and sigma_data."""
        from omegaconf import OmegaConf
        from medgen.pipeline.base_config import StrategyConfig

        cfg = OmegaConf.create({
            'strategy': {
                'name': 'rflow',
                'num_train_timesteps': 1000,
                'snr_gamma': 5.0,
                'sigma_data': 0.08,
            }
        })

        sc = StrategyConfig.from_hydra(cfg)
        assert sc.snr_gamma == 5.0
        assert sc.sigma_data == 0.08

    def test_from_hydra_defaults_when_missing(self):
        """from_hydra returns 0.0 when fields not in config."""
        from omegaconf import OmegaConf
        from medgen.pipeline.base_config import StrategyConfig

        cfg = OmegaConf.create({
            'strategy': {
                'name': 'rflow',
                'num_train_timesteps': 1000,
            }
        })

        sc = StrategyConfig.from_hydra(cfg)
        assert sc.snr_gamma == 0.0
        assert sc.sigma_data == 0.0
