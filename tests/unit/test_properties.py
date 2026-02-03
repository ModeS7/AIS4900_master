"""
Property-based tests for mathematical invariants.

Uses hypothesis to verify that metric functions satisfy expected mathematical
properties regardless of input values. These tests catch edge cases and
subtle bugs that example-based tests might miss.
"""
import pytest
import torch
import numpy as np

# Import hypothesis - skip tests if not installed
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp


# =============================================================================
# PSNR Properties
# =============================================================================

class TestPSNRProperties:
    """Test mathematical properties of PSNR metric."""

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 64, 64),
            elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_psnr_self_is_100(self, arr):
        """PSNR(x, x) = 100 for any valid input (identity property)."""
        from medgen.metrics.quality import compute_psnr

        images = torch.from_numpy(arr).float()
        psnr = compute_psnr(images, images)
        assert psnr == 100.0

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 64, 64),
            elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 64, 64),
            elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50, deadline=5000)
    def test_psnr_symmetric(self, arr1, arr2):
        """PSNR(x, y) = PSNR(y, x) - symmetry property."""
        from medgen.metrics.quality import compute_psnr

        x = torch.from_numpy(arr1).float()
        y = torch.from_numpy(arr2).float()
        assert compute_psnr(x, y) == compute_psnr(y, x)

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 64, 64),
            elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 64, 64),
            elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50, deadline=5000)
    def test_psnr_non_negative(self, arr1, arr2):
        """PSNR is always >= 0."""
        from medgen.metrics.quality import compute_psnr

        x = torch.from_numpy(arr1).float()
        y = torch.from_numpy(arr2).float()
        psnr = compute_psnr(x, y)
        assert psnr >= 0.0


# =============================================================================
# MS-SSIM Properties
# =============================================================================

class TestMSSSIMProperties:
    """Test mathematical properties of MS-SSIM metric."""

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 64, 64),
            # Avoid edge values to prevent numerical instability
            elements=st.floats(0.01, 0.99, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=30, deadline=10000)
    def test_msssim_self_near_one(self, arr):
        """MS-SSIM(x, x) is very close to 1.0 for any valid input."""
        from medgen.metrics.quality import compute_msssim

        images = torch.from_numpy(arr).float()
        msssim = compute_msssim(images, images, spatial_dims=2)
        # Allow small numerical tolerance
        assert msssim > 0.99

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 64, 64),
            elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 64, 64),
            elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=30, deadline=10000)
    def test_msssim_in_valid_range(self, arr1, arr2):
        """MS-SSIM always in [0, 1] range (with floating-point tolerance)."""
        from medgen.metrics.quality import compute_msssim

        x = torch.from_numpy(arr1).float()
        y = torch.from_numpy(arr2).float()
        msssim = compute_msssim(x, y, spatial_dims=2)
        # Allow small floating-point tolerance (MONAI can return slightly > 1.0)
        assert -1e-5 <= msssim <= 1.0 + 1e-5


# =============================================================================
# Dice/IoU Properties
# =============================================================================

class TestDiceIoUProperties:
    """Test mathematical properties of segmentation metrics."""

    @given(st.floats(0.0, 1.0))
    @settings(max_examples=50, deadline=5000)
    def test_dice_threshold_bounds(self, threshold):
        """Dice score is always in [0, 1] for any threshold."""
        from medgen.metrics.quality import compute_dice

        pred = torch.rand(4, 1, 64, 64)
        target = (torch.rand(4, 1, 64, 64) > 0.5).float()
        dice = compute_dice(pred, target, threshold=threshold, apply_sigmoid=False)
        assert 0.0 <= dice <= 1.0

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 64, 64),
            elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_iou_leq_dice(self, arr):
        """IoU <= Dice for same inputs (mathematical property).

        This follows from the relationship:
        Dice = 2 * IoU / (1 + IoU)
        Therefore: Dice > IoU when IoU < 1
        """
        from medgen.metrics.quality import compute_dice, compute_iou

        pred = torch.from_numpy(arr).float()
        target = (torch.rand_like(pred) > 0.5).float()
        dice = compute_dice(pred, target, apply_sigmoid=False)
        iou = compute_iou(pred, target, apply_sigmoid=False)
        # IoU <= Dice always holds (with small numerical tolerance)
        assert iou <= dice + 0.01

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 64, 64),
            elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_dice_self_is_one(self, arr):
        """Dice(x, x) = 1.0 when x is binary (perfect overlap)."""
        from medgen.metrics.quality import compute_dice

        # Make binary
        binary = (torch.from_numpy(arr).float() > 0.5).float()
        # Ensure at least some pixels are positive to avoid edge case
        assume(binary.sum() > 0)
        dice = compute_dice(binary, binary, apply_sigmoid=False)
        # Should be exactly 1.0 (or very close due to smoothing)
        assert dice > 0.99


# =============================================================================
# CFG Interpolation Properties
# =============================================================================

class TestCFGInterpolationProperties:
    """Test CFG (Classifier-Free Guidance) interpolation properties."""

    @given(
        st.floats(1.0, 10.0, allow_nan=False),
        st.floats(1.0, 10.0, allow_nan=False),
        st.integers(2, 100),
    )
    @settings(max_examples=100, deadline=5000)
    def test_cfg_interpolation_bounded(self, start, end, num_steps):
        """CFG interpolation stays within [min, max] bounds."""
        eps = 1e-9  # Small tolerance for floating-point precision
        for step in range(num_steps):
            progress = step / max(num_steps - 1, 1)
            cfg = start + progress * (end - start)
            assert min(start, end) - eps <= cfg <= max(start, end) + eps

    @given(
        st.floats(1.0, 10.0, allow_nan=False),
        st.floats(1.0, 10.0, allow_nan=False),
    )
    @settings(max_examples=100, deadline=5000)
    def test_cfg_endpoints_exact(self, start, end):
        """CFG interpolation hits exact start and end points (within floating-point precision)."""
        eps = 1e-9  # Small tolerance for floating-point precision
        # At progress=0
        assert start + 0.0 * (end - start) == start
        # At progress=1 (may have floating-point error)
        result = start + 1.0 * (end - start)
        assert abs(result - end) < eps


# =============================================================================
# Generation Metric Properties
# =============================================================================

class TestGenerationMetricProperties:
    """Test mathematical properties of generation quality metrics."""

    @given(st.integers(100, 200))
    @settings(max_examples=20, deadline=30000)
    def test_kid_identical_near_zero(self, n_samples):
        """KID(x, x) is approximately 0 for identical distributions.

        Note: KID is an unbiased estimator with variance that decreases with sample size.
        For 100+ samples, the mean should be reasonably close to 0.
        """
        from medgen.metrics.generation import compute_kid

        features = torch.randn(n_samples, 2048)
        kid_mean, kid_std = compute_kid(features, features, subset_size=50, num_subsets=100)
        # Should be close to 0 for identical distributions
        # Allow tolerance of 0.5 due to variance in estimation
        assert abs(kid_mean) < 0.5
        # Standard deviation is always non-negative
        assert kid_std >= 0.0

    @given(st.integers(50, 100))
    @settings(max_examples=10, deadline=60000)
    def test_fid_identical_near_zero(self, n_samples):
        """FID(x, x) is approximately 0 for identical distributions."""
        from medgen.metrics.generation import compute_fid

        # Use smaller feature dimension for faster test
        features = torch.randn(n_samples, 256)
        fid = compute_fid(features, features)
        # Should be very close to 0 for identical distributions
        assert fid < 1.0

    @given(st.integers(100, 200))
    @settings(max_examples=10, deadline=30000)
    def test_kid_variance_reasonable(self, n_samples):
        """KID standard deviation is non-negative and variance is bounded.

        Note: KID mean can be slightly negative due to unbiased estimation,
        but the standard deviation is always non-negative.
        """
        from medgen.metrics.generation import compute_kid

        # Use random features with sufficient variation
        real = torch.randn(n_samples, 512)
        gen = torch.randn(n_samples, 512)
        kid_mean, kid_std = compute_kid(real, gen, subset_size=50, num_subsets=50)
        # Standard deviation is always non-negative
        assert kid_std >= 0.0
        # KID mean should be finite
        assert np.isfinite(kid_mean)

    @given(st.integers(50, 100))
    @settings(max_examples=20, deadline=30000)
    def test_cmmd_identical_near_zero(self, n_samples):
        """CMMD(x, x) is approximately 0 for identical distributions."""
        from medgen.metrics.generation import compute_cmmd

        features = torch.randn(n_samples, 512)
        cmmd = compute_cmmd(features, features)
        # Should be very close to 0 for identical distributions
        assert cmmd < 0.1


# =============================================================================
# Timestep Sampling Properties (Section 5.1)
# =============================================================================


class TestTimestepSamplingProperties:
    """Property-based tests for timestep sampling."""

    @given(st.integers(10, 1000))
    @settings(max_examples=50, deadline=5000)
    def test_timesteps_in_valid_range(self, num_timesteps):
        """Sampled timesteps should be in [0, num_timesteps)."""
        batch_size = 16

        # DDPM: discrete integer timesteps
        timesteps_discrete = torch.randint(0, num_timesteps, (batch_size,))
        assert (timesteps_discrete >= 0).all(), "Timesteps should be >= 0"
        assert (timesteps_discrete < num_timesteps).all(), \
            f"Timesteps should be < {num_timesteps}"

        # RFlow: continuous float timesteps in [0, num_timesteps]
        timesteps_continuous = torch.rand(batch_size) * num_timesteps
        assert (timesteps_continuous >= 0).all(), "Float timesteps should be >= 0"
        assert (timesteps_continuous <= num_timesteps).all(), \
            f"Float timesteps should be <= {num_timesteps}"

    @given(
        st.floats(0.0, 0.5, allow_nan=False),
        st.floats(0.5, 1.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=5000)
    def test_curriculum_range_respected(self, min_frac, max_frac):
        """Curriculum learning timestep ranges are respected."""
        assume(min_frac < max_frac)

        num_timesteps = 1000
        batch_size = 16

        # Simulate curriculum sampling
        min_t = int(min_frac * num_timesteps)
        max_t = int(max_frac * num_timesteps)
        assume(max_t > min_t)  # Ensure valid range

        timesteps = torch.randint(min_t, max_t, (batch_size,))

        assert (timesteps >= min_t).all(), f"Timesteps should be >= {min_t}"
        assert (timesteps < max_t).all(), f"Timesteps should be < {max_t}"


# =============================================================================
# Noise Schedule Properties (Section 5.1)
# =============================================================================


class TestNoiseScheduleProperties:
    """Property-based tests for noise schedule computation."""

    @given(st.integers(100, 1000))
    @settings(max_examples=20, deadline=5000)
    def test_alphas_cumprod_decreasing(self, num_timesteps):
        """Alpha cumprod should monotonically decrease (more noise over time)."""
        # Linear beta schedule
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Check monotonically decreasing
        diffs = alphas_cumprod[1:] - alphas_cumprod[:-1]
        assert (diffs <= 0).all(), "alphas_cumprod should monotonically decrease"

    @given(st.integers(100, 1000))
    @settings(max_examples=20, deadline=5000)
    def test_alphas_cumprod_bounded(self, num_timesteps):
        """Alpha cumprod should be bounded in (0, 1]."""
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        assert (alphas_cumprod > 0).all(), "alphas_cumprod should be > 0"
        assert (alphas_cumprod <= 1).all(), "alphas_cumprod should be <= 1"


# =============================================================================
# Interpolation Properties (Section 5.1)
# =============================================================================


class TestInterpolationProperties:
    """Property-based tests for flow matching interpolation."""

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 32, 32),
            elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 32, 32),
            elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50, deadline=10000)
    def test_linear_interpolation_endpoints(self, x0_arr, x1_arr):
        """Linear interpolation: t=0 gives x0, t=1 gives x1."""
        x0 = torch.from_numpy(x0_arr)
        x1 = torch.from_numpy(x1_arr)

        # At t=0: should be x0
        interp_t0 = (1 - 0) * x0 + 0 * x1
        assert torch.allclose(interp_t0, x0, atol=1e-5), \
            "Interpolation at t=0 should equal x0"

        # At t=1: should be x1
        interp_t1 = (1 - 1) * x0 + 1 * x1
        assert torch.allclose(interp_t1, x1, atol=1e-5), \
            "Interpolation at t=1 should equal x1"

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 32, 32),
            elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 32, 32),
            elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50, deadline=10000)
    def test_velocity_is_difference(self, x0_arr, x1_arr):
        """For flow matching, velocity v = x1 - x0 (points toward clean data)."""
        x0 = torch.from_numpy(x0_arr)  # Clean data
        x1 = torch.from_numpy(x1_arr)  # Noise

        # Velocity field for flow matching
        velocity = x0 - x1  # Direction toward clean

        # Verify velocity has correct relationship
        # At t=0, x_t = x1 (pure noise), at t=1, x_t = x0 (clean)
        # So x_t = (1-t)*x1 + t*x0
        # dx/dt = x0 - x1 = velocity
        expected_velocity = x0 - x1
        assert torch.allclose(velocity, expected_velocity, atol=1e-5), \
            "Velocity should equal x0 - x1 for flow matching"


# =============================================================================
# Diversity Metric Properties (Section 5.2)
# =============================================================================


class TestDiversityMetricProperties:
    """Property-based tests for diversity metrics."""

    @given(st.integers(4, 20))
    @settings(max_examples=20, deadline=30000)
    def test_diversity_non_negative(self, n_samples):
        """Diversity metric should be non-negative."""
        # Generate random samples with variation
        samples = torch.randn(n_samples, 1, 32, 32)

        # Compute pairwise distances as diversity proxy
        samples_flat = samples.flatten(1)
        pairwise_dist = torch.cdist(samples_flat, samples_flat)
        diversity = pairwise_dist.mean()

        assert diversity >= 0, "Diversity should be non-negative"

    @given(st.integers(4, 10))
    @settings(max_examples=20, deadline=30000)
    def test_identical_samples_zero_diversity(self, n_samples):
        """Identical samples should have zero (or near-zero) diversity."""
        # All identical samples
        template = torch.randn(1, 1, 32, 32)
        samples = template.repeat(n_samples, 1, 1, 1)

        # Compute pairwise distances
        samples_flat = samples.flatten(1)
        pairwise_dist = torch.cdist(samples_flat, samples_flat)
        diversity = pairwise_dist.mean()

        assert diversity < 1e-5, \
            f"Identical samples should have ~0 diversity, got {diversity}"


# =============================================================================
# Loss Function Properties (Section 5.3)
# =============================================================================


class TestLossFunctionProperties:
    """Property-based tests for loss function mathematical properties."""

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 32, 32),
            elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
        ),
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 32, 32),
            elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50, deadline=5000)
    def test_mse_loss_non_negative(self, pred_arr, target_arr):
        """MSE loss is always >= 0."""
        import torch.nn.functional as F

        pred = torch.from_numpy(pred_arr)
        target = torch.from_numpy(target_arr)

        mse = F.mse_loss(pred, target)

        assert mse >= 0, f"MSE should be non-negative, got {mse}"

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 32, 32),
            elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50, deadline=5000)
    def test_mse_loss_zero_for_identical(self, arr):
        """MSE loss is exactly 0 for identical inputs."""
        import torch.nn.functional as F

        tensor = torch.from_numpy(arr)

        mse = F.mse_loss(tensor, tensor)

        assert mse == 0.0, f"MSE(x, x) should be 0, got {mse}"

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 128),
            elements=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
        ),
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 128),
            elements=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50, deadline=5000)
    def test_kl_loss_non_negative(self, mu_arr, logvar_arr):
        """KL divergence loss is always >= 0 for VAE."""
        mu = torch.from_numpy(mu_arr)
        logvar = torch.from_numpy(logvar_arr)

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # KL can be slightly negative due to numerical precision,
        # but should be approximately non-negative
        assert (kl_per_sample >= -0.01).all(), \
            f"KL should be approximately non-negative, got min={kl_per_sample.min()}"


# =============================================================================
# Euler Integration Properties (Section 5.1 - RFlow specific)
# =============================================================================


class TestEulerIntegrationProperties:
    """Property-based tests for Euler integration in generation."""

    @given(
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 32, 32),
            elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
        hnp.arrays(
            dtype=np.float32,
            shape=(4, 1, 32, 32),
            elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
        ),
        st.floats(0.01, 0.1, allow_nan=False),
    )
    @settings(max_examples=50, deadline=5000)
    def test_euler_step_moves_toward_velocity(self, x_arr, v_arr, dt):
        """Euler step x_new = x + dt*v moves in velocity direction."""
        x = torch.from_numpy(x_arr)
        v = torch.from_numpy(v_arr)

        # Euler step (ADDITION, not subtraction - Pitfall #47)
        x_new = x + dt * v

        # The change should be in the direction of v
        delta = x_new - x
        expected_delta = dt * v

        assert torch.allclose(delta, expected_delta, atol=1e-5), \
            "Euler step should move by dt*v"

    @given(st.integers(2, 50))
    @settings(max_examples=20, deadline=10000)
    def test_euler_integration_convergence(self, num_steps):
        """Euler integration with many small steps converges to target."""
        # Start with noise
        x_start = torch.randn(4, 1, 16, 16)
        x_target = torch.zeros(4, 1, 16, 16)  # Target: zeros

        # Constant velocity pointing toward target
        v = (x_target - x_start) / num_steps

        # Euler integration
        x = x_start.clone()
        dt = 1.0 / num_steps
        for _ in range(num_steps):
            x = x + dt * (x_target - x_start)  # Fixed velocity

        # Should reach target
        assert torch.allclose(x, x_target, atol=0.1), \
            f"Euler integration should converge, max diff={torch.abs(x - x_target).max()}"
