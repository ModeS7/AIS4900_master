"""Comprehensive edge case tests for mode embedding.

Tests mode_embed.py and combined_embed.py for edge cases:
1. Empty batch (batch_size=0)
2. Single sample batch (batch_size=1)
3. Large batch (batch_size=128)
4. mode_id with wrong dtype (float instead of long)
5. mode_id on different device than model
6. mode_id with negative values
7. mode_id with values >= MODE_ENCODING_DIM (4)
8. mode_id as 2D tensor instead of 1D
9. Calling forward without setting mode_encoding first
10. Calling set_mode_encoding with wrong shape
11. Mixed batch sizes between consecutive forward calls
12. GPU tensor handling (if CUDA available)
"""

import pytest
import torch
import torch.nn as nn

from medgen.data.mode_embed import (
    MODE_ENCODING_DIM,
    encode_mode_id,
    ModeTimeEmbed,
    ModeEmbedModelWrapper,
)
from medgen.data.combined_embed import (
    CombinedTimeEmbed,
    CombinedModelWrapper,
    create_conditioning_wrapper,
)


# =============================================================================
# Helper fixtures and mock models
# =============================================================================

class MockTimeEmbed(nn.Module):
    """Mock time_embed module that returns input unchanged."""

    def __init__(self, in_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.linear(t_emb)


class MockDiffusionModel(nn.Module):
    """Mock MONAI DiffusionModelUNet for testing wrappers."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.time_embed = MockTimeEmbed(128, embed_dim)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Simplified forward: process time embedding and return convolved input
        t_emb = torch.sin(timesteps.float().unsqueeze(-1) * torch.arange(128, device=x.device))
        _ = self.time_embed(t_emb)
        return self.conv(x)


@pytest.fixture
def mock_time_embed():
    """Create mock time embedding module."""
    return MockTimeEmbed()


@pytest.fixture
def mock_model():
    """Create mock diffusion model."""
    return MockDiffusionModel()


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Test encode_mode_id function
# =============================================================================

class TestEncodeModeId:
    """Tests for encode_mode_id function."""

    def test_empty_batch_none_mode_id(self):
        """Test 1a: Empty batch with mode_id=None."""
        device = torch.device('cpu')
        # batch_size=0 should return empty tensor
        result = encode_mode_id(None, device, batch_size=0)
        assert result.shape == (0, MODE_ENCODING_DIM)
        assert result.device == device

    def test_empty_batch_empty_mode_id(self):
        """Test 1b: Empty batch with empty mode_id tensor."""
        device = torch.device('cpu')
        mode_id = torch.tensor([], dtype=torch.long)
        result = encode_mode_id(mode_id, device, batch_size=0)
        assert result.shape == (0, MODE_ENCODING_DIM)

    def test_single_sample_batch(self):
        """Test 2: Single sample batch."""
        device = torch.device('cpu')
        mode_id = torch.tensor([0], dtype=torch.long)
        result = encode_mode_id(mode_id, device)
        assert result.shape == (1, MODE_ENCODING_DIM)
        assert result[0, 0] == 1.0  # bravo = index 0
        assert result[0, 1:].sum() == 0.0  # other indices are 0

    def test_single_sample_scalar(self):
        """Test 2b: Single sample as scalar tensor."""
        device = torch.device('cpu')
        mode_id = torch.tensor(2, dtype=torch.long)  # Scalar
        result = encode_mode_id(mode_id, device)
        assert result.shape == (1, MODE_ENCODING_DIM)
        assert result[0, 2] == 1.0  # t1_pre = index 2

    def test_large_batch(self):
        """Test 3: Large batch (batch_size=128)."""
        device = torch.device('cpu')
        batch_size = 128
        # Create mixed mode_ids
        mode_id = torch.randint(0, MODE_ENCODING_DIM, (batch_size,), dtype=torch.long)
        result = encode_mode_id(mode_id, device)
        assert result.shape == (batch_size, MODE_ENCODING_DIM)
        # Each row should be one-hot
        assert torch.all(result.sum(dim=1) == 1.0)

    def test_wrong_dtype_float(self):
        """Test 4: mode_id with wrong dtype (float instead of long)."""
        device = torch.device('cpu')
        mode_id = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        # The function uses .long() internally in scatter_, so this should work
        result = encode_mode_id(mode_id, device)
        assert result.shape == (3, MODE_ENCODING_DIM)
        # Should still produce valid one-hot encoding
        assert torch.all(result.sum(dim=1) == 1.0)

    def test_different_device_cpu_to_cuda(self):
        """Test 5: mode_id on different device than target.

        FIXED: encode_mode_id now moves mode_id to target device before scatter_().
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        mode_id = torch.tensor([0, 1], dtype=torch.long, device='cpu')
        target_device = torch.device('cuda')

        # Should work - mode_id is automatically moved to target device
        result = encode_mode_id(mode_id, target_device)
        assert result.device.type == 'cuda'
        assert result.shape == (2, MODE_ENCODING_DIM)

    def test_negative_mode_id(self):
        """Test 6: mode_id with negative values should raise ValueError."""
        device = torch.device('cpu')
        mode_id = torch.tensor([-1, 0, 1], dtype=torch.long)
        with pytest.raises(ValueError, match="Invalid mode_id values"):
            encode_mode_id(mode_id, device)

    def test_mode_id_too_large(self):
        """Test 7: mode_id with values >= MODE_ENCODING_DIM should raise ValueError."""
        device = torch.device('cpu')
        mode_id = torch.tensor([0, 1, 4], dtype=torch.long)  # 4 is out of range
        with pytest.raises(ValueError, match="Invalid mode_id values"):
            encode_mode_id(mode_id, device)

    def test_mode_id_way_too_large(self):
        """Test 7b: mode_id with values way out of range."""
        device = torch.device('cpu')
        mode_id = torch.tensor([0, 100, 1000], dtype=torch.long)
        with pytest.raises(ValueError, match="Invalid mode_id values"):
            encode_mode_id(mode_id, device)

    def test_2d_mode_id(self):
        """Test 8: mode_id as 2D tensor should fail or produce unexpected results."""
        device = torch.device('cpu')
        mode_id = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)  # 2D tensor
        # Current implementation only handles 0D and 1D tensors
        # 2D tensor will cause issues in scatter_
        with pytest.raises((RuntimeError, IndexError)):
            encode_mode_id(mode_id, device)

    def test_all_same_mode(self):
        """Test: All samples have same mode."""
        device = torch.device('cpu')
        mode_id = torch.tensor([2, 2, 2, 2], dtype=torch.long)  # All t1_pre
        result = encode_mode_id(mode_id, device)
        assert result.shape == (4, MODE_ENCODING_DIM)
        assert torch.all(result[:, 2] == 1.0)  # All have t1_pre active
        assert result[:, [0, 1, 3]].sum() == 0.0  # No other modes active

    def test_mixed_modes(self):
        """Test: Mixed modes in batch."""
        device = torch.device('cpu')
        mode_id = torch.tensor([0, 1, 2, 3], dtype=torch.long)  # All different
        result = encode_mode_id(mode_id, device)
        # Check each sample has correct one-hot
        for i in range(4):
            assert result[i, i] == 1.0
            for j in range(4):
                if j != i:
                    assert result[i, j] == 0.0


# =============================================================================
# Test ModeTimeEmbed class
# =============================================================================

class TestModeTimeEmbed:
    """Tests for ModeTimeEmbed class."""

    def test_forward_without_mode_encoding(self, mock_time_embed):
        """Test 9: Calling forward without setting mode_encoding first."""
        embed_dim = 256
        mode_embed = ModeTimeEmbed(mock_time_embed, embed_dim)

        t_emb = torch.randn(4, 128)
        # Should work - mode_encoding is None, so mode_emb is not added
        result = mode_embed(t_emb)
        assert result.shape == (4, embed_dim)

    def test_set_mode_encoding_correct_shape(self, mock_time_embed):
        """Test: set_mode_encoding with correct shape."""
        embed_dim = 256
        mode_embed = ModeTimeEmbed(mock_time_embed, embed_dim)

        mode_encoding = torch.zeros(4, MODE_ENCODING_DIM)
        mode_encoding[0, 0] = 1.0
        mode_embed.set_mode_encoding(mode_encoding)

        t_emb = torch.randn(4, 128)
        result = mode_embed(t_emb)
        assert result.shape == (4, embed_dim)

    def test_set_mode_encoding_wrong_shape(self, mock_time_embed):
        """Test 10: set_mode_encoding with wrong shape.

        FIXED: Now raises ValueError early in set_mode_encoding instead of
        failing late during forward pass.
        """
        embed_dim = 256
        mode_embed = ModeTimeEmbed(mock_time_embed, embed_dim)

        # Wrong feature dimension - now caught early
        wrong_encoding = torch.zeros(4, 8)  # Should be MODE_ENCODING_DIM=4
        with pytest.raises(ValueError, match="mode_encoding.shape"):
            mode_embed.set_mode_encoding(wrong_encoding)

    def test_set_mode_encoding_1d(self, mock_time_embed):
        """Test 10b: set_mode_encoding with 1D tensor.

        FIXED: 1D tensors are now rejected with ValueError, preventing silent
        incorrect behavior where mode embedding would broadcast incorrectly.
        """
        embed_dim = 256
        mode_embed = ModeTimeEmbed(mock_time_embed, embed_dim)

        # 1D instead of 2D - now caught early
        wrong_encoding = torch.zeros(MODE_ENCODING_DIM)
        with pytest.raises(ValueError, match="mode_encoding must be 2D"):
            mode_embed.set_mode_encoding(wrong_encoding)

    def test_mixed_batch_sizes(self, mock_time_embed):
        """Test 11: Mixed batch sizes between consecutive forward calls."""
        embed_dim = 256
        mode_embed = ModeTimeEmbed(mock_time_embed, embed_dim)

        # First call with batch_size=4
        mode_encoding_4 = torch.zeros(4, MODE_ENCODING_DIM)
        mode_encoding_4[:, 0] = 1.0
        mode_embed.set_mode_encoding(mode_encoding_4)

        t_emb_4 = torch.randn(4, 128)
        result_4 = mode_embed(t_emb_4)
        assert result_4.shape == (4, embed_dim)

        # Second call with batch_size=8
        mode_encoding_8 = torch.zeros(8, MODE_ENCODING_DIM)
        mode_encoding_8[:, 1] = 1.0
        mode_embed.set_mode_encoding(mode_encoding_8)

        t_emb_8 = torch.randn(8, 128)
        result_8 = mode_embed(t_emb_8)
        assert result_8.shape == (8, embed_dim)

    def test_batch_size_mismatch(self, mock_time_embed):
        """Test 11b: Batch size mismatch between mode_encoding and t_emb.

        Expected behavior: Should raise RuntimeError due to shape mismatch
        during tensor addition [4, 256] + [8, 256].
        """
        embed_dim = 256
        mode_embed = ModeTimeEmbed(mock_time_embed, embed_dim)

        # Set mode_encoding for batch_size=4
        mode_encoding = torch.zeros(4, MODE_ENCODING_DIM)
        mode_encoding[:, 0] = 1.0
        mode_embed.set_mode_encoding(mode_encoding)

        # Call with batch_size=8 - this will cause shape mismatch
        t_emb = torch.randn(8, 128)
        # PyTorch broadcasting: [4, 256] + [8, 256] fails (non-broadcastable)
        with pytest.raises(RuntimeError):
            mode_embed(t_emb)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_handling(self):
        """Test 12: GPU tensor handling.

        Test that ModeTimeEmbed works correctly with GPU tensors.
        """
        embed_dim = 256
        mock_te = MockTimeEmbed().cuda()
        mode_embed = ModeTimeEmbed(mock_te, embed_dim).cuda()

        t_emb = torch.randn(4, 128, device='cuda')
        mode_encoding = torch.zeros(4, MODE_ENCODING_DIM, device='cuda')
        mode_encoding[:, 0] = 1.0
        mode_embed.set_mode_encoding(mode_encoding)

        result = mode_embed(t_emb)
        assert result.device.type == 'cuda'
        assert result.shape == (4, embed_dim)


# =============================================================================
# Test ModeEmbedModelWrapper class
# =============================================================================

class TestModeEmbedModelWrapper:
    """Tests for ModeEmbedModelWrapper class."""

    def test_empty_batch(self, mock_model):
        """Test 1: Empty batch (batch_size=0)."""
        wrapper = ModeEmbedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(0, 1, 32, 32)  # Empty batch
        timesteps = torch.randint(0, 1000, (0,))
        mode_id = torch.tensor([], dtype=torch.long)

        result = wrapper(x, timesteps, mode_id=mode_id)
        assert result.shape == (0, 1, 32, 32)

    def test_single_sample(self, mock_model):
        """Test 2: Single sample batch."""
        wrapper = ModeEmbedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(1, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (1,))
        mode_id = torch.tensor([0], dtype=torch.long)

        result = wrapper(x, timesteps, mode_id=mode_id)
        assert result.shape == (1, 1, 32, 32)

    def test_large_batch(self, mock_model):
        """Test 3: Large batch (batch_size=128)."""
        wrapper = ModeEmbedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(128, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (128,))
        mode_id = torch.randint(0, 4, (128,), dtype=torch.long)

        result = wrapper(x, timesteps, mode_id=mode_id)
        assert result.shape == (128, 1, 32, 32)

    def test_wrong_dtype_mode_id(self, mock_model):
        """Test 4: mode_id with wrong dtype (float)."""
        wrapper = ModeEmbedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(4, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (4,))
        mode_id = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)

        # Should work - encode_mode_id uses .long() in scatter_
        result = wrapper(x, timesteps, mode_id=mode_id)
        assert result.shape == (4, 1, 32, 32)

    def test_negative_mode_id(self, mock_model):
        """Test 6: mode_id with negative values."""
        wrapper = ModeEmbedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(4, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (4,))
        mode_id = torch.tensor([-1, 0, 1, 2], dtype=torch.long)

        with pytest.raises(ValueError, match="Invalid mode_id values"):
            wrapper(x, timesteps, mode_id=mode_id)

    def test_mode_id_too_large(self, mock_model):
        """Test 7: mode_id with values >= MODE_ENCODING_DIM."""
        wrapper = ModeEmbedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(4, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (4,))
        mode_id = torch.tensor([0, 1, 2, 5], dtype=torch.long)  # 5 is out of range

        with pytest.raises(ValueError, match="Invalid mode_id values"):
            wrapper(x, timesteps, mode_id=mode_id)

    def test_2d_mode_id(self, mock_model):
        """Test 8: mode_id as 2D tensor."""
        wrapper = ModeEmbedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(4, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (4,))
        mode_id = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

        with pytest.raises((RuntimeError, IndexError)):
            wrapper(x, timesteps, mode_id=mode_id)

    def test_no_mode_id(self, mock_model):
        """Test: Forward without mode_id (should work with zeros)."""
        wrapper = ModeEmbedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(4, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (4,))

        result = wrapper(x, timesteps)  # No mode_id
        assert result.shape == (4, 1, 32, 32)

    def test_model_without_time_embed(self):
        """Test: Model without time_embed attribute."""
        model = nn.Linear(10, 10)  # Simple model without time_embed

        with pytest.raises(ValueError, match="Model does not have 'time_embed' attribute"):
            ModeEmbedModelWrapper(model, embed_dim=256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_handling(self):
        """Test 12: GPU tensor handling.

        FIXED: ModeEmbedModelWrapper now automatically moves mode_mlp to
        the same device as the model during initialization.
        """
        model = MockDiffusionModel().cuda()
        wrapper = ModeEmbedModelWrapper(model, embed_dim=256)
        # No longer need manual .cuda() - wrapper handles this automatically

        x = torch.randn(4, 1, 32, 32, device='cuda')
        timesteps = torch.randint(0, 1000, (4,), device='cuda')
        mode_id = torch.tensor([0, 1, 2, 3], dtype=torch.long, device='cuda')

        result = wrapper(x, timesteps, mode_id=mode_id)
        assert result.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mode_id_on_cpu_model_on_gpu(self):
        """Test 5: mode_id on different device than model.

        FIXED: encode_mode_id now moves mode_id to target device before scatter_().
        Also, ModeEmbedModelWrapper now automatically moves mode_mlp to model's device.
        """
        model = MockDiffusionModel().cuda()
        wrapper = ModeEmbedModelWrapper(model, embed_dim=256)
        # No longer need manual .cuda() - wrapper now handles this automatically

        x = torch.randn(4, 1, 32, 32, device='cuda')
        timesteps = torch.randint(0, 1000, (4,), device='cuda')
        mode_id = torch.tensor([0, 1, 2, 3], dtype=torch.long, device='cpu')  # CPU!

        # Should work - mode_id is automatically moved to CUDA
        result = wrapper(x, timesteps, mode_id=mode_id)
        assert result.device.type == 'cuda'
        assert result.shape == (4, 1, 32, 32)


# =============================================================================
# Test CombinedTimeEmbed class
# =============================================================================

class TestCombinedTimeEmbed:
    """Tests for CombinedTimeEmbed class."""

    def test_forward_without_encodings(self, mock_time_embed):
        """Test 9: Calling forward without setting encodings first."""
        embed_dim = 256
        combined = CombinedTimeEmbed(mock_time_embed, embed_dim)

        t_emb = torch.randn(4, 128)
        # omega_encoding buffer is initialized to zeros
        # mode_encoding is None, so no mode embedding added
        result = combined(t_emb)
        assert result.shape == (4, embed_dim)

    def test_set_mode_encoding_wrong_shape(self, mock_time_embed):
        """Test 10: set_mode_encoding with wrong shape.

        FIXED: Now raises ValueError early in set_mode_encoding instead of
        failing late during forward pass.
        """
        embed_dim = 256
        combined = CombinedTimeEmbed(mock_time_embed, embed_dim)

        wrong_encoding = torch.zeros(4, 8)  # Wrong feature dim
        with pytest.raises(ValueError, match="mode_encoding.shape"):
            combined.set_mode_encoding(wrong_encoding)

    def test_set_omega_encoding_wrong_shape(self, mock_time_embed):
        """Test 10b: set_omega_encoding with wrong shape."""
        embed_dim = 256
        combined = CombinedTimeEmbed(mock_time_embed, embed_dim)


        # Wrong shape for omega (should be [1, OMEGA_ENCODING_DIM])
        wrong_encoding = torch.zeros(4, 10)  # Wrong both dims

        # This uses copy_() which will fail if shapes don't match
        with pytest.raises(RuntimeError):
            combined.set_omega_encoding(wrong_encoding)

    def test_mixed_batch_sizes(self, mock_time_embed):
        """Test 11: Mixed batch sizes between forward calls."""
        embed_dim = 256
        combined = CombinedTimeEmbed(mock_time_embed, embed_dim)

        from medgen.data.score_aug import OMEGA_ENCODING_DIM

        # First call with batch_size=4
        omega_enc = torch.zeros(1, OMEGA_ENCODING_DIM)
        mode_enc_4 = torch.zeros(4, MODE_ENCODING_DIM)
        combined.set_encodings(omega_enc, mode_enc_4)

        t_emb_4 = torch.randn(4, 128)
        result_4 = combined(t_emb_4)
        assert result_4.shape == (4, embed_dim)

        # Second call with batch_size=8
        mode_enc_8 = torch.zeros(8, MODE_ENCODING_DIM)
        combined.set_encodings(omega_enc, mode_enc_8)

        t_emb_8 = torch.randn(8, 128)
        result_8 = combined(t_emb_8)
        assert result_8.shape == (8, embed_dim)


# =============================================================================
# Test CombinedModelWrapper class
# =============================================================================

class TestCombinedModelWrapper:
    """Tests for CombinedModelWrapper class."""

    def test_empty_batch(self, mock_model):
        """Test 1: Empty batch.

        FIXED: encode_omega in score_aug.py now handles empty mode_id tensors
        by checking mode_id.numel() > 0 before indexing.
        """
        wrapper = CombinedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(0, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (0,))
        mode_id = torch.tensor([], dtype=torch.long)

        # Should work - empty batch returns empty output
        result = wrapper(x, timesteps, omega=None, mode_id=mode_id)
        assert result.shape == (0, 1, 32, 32)

    def test_single_sample(self, mock_model):
        """Test 2: Single sample batch."""
        wrapper = CombinedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(1, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (1,))
        mode_id = torch.tensor([0], dtype=torch.long)

        result = wrapper(x, timesteps, omega=None, mode_id=mode_id)
        assert result.shape == (1, 1, 32, 32)

    def test_large_batch(self, mock_model):
        """Test 3: Large batch."""
        wrapper = CombinedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(128, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (128,))
        mode_id = torch.randint(0, 4, (128,), dtype=torch.long)

        result = wrapper(x, timesteps, omega=None, mode_id=mode_id)
        assert result.shape == (128, 1, 32, 32)

    def test_with_omega(self, mock_model):
        """Test: Forward with omega conditioning."""
        wrapper = CombinedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(4, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (4,))
        mode_id = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        omega = {'type': 'rot90', 'params': {'k': 1}}

        result = wrapper(x, timesteps, omega=omega, mode_id=mode_id)
        assert result.shape == (4, 1, 32, 32)

    def test_negative_mode_id(self, mock_model):
        """Test 6: Negative mode_id."""
        wrapper = CombinedModelWrapper(mock_model, embed_dim=256)

        x = torch.randn(4, 1, 32, 32)
        timesteps = torch.randint(0, 1000, (4,))
        mode_id = torch.tensor([-1, 0, 1, 2], dtype=torch.long)

        with pytest.raises(ValueError, match="Invalid mode_id values"):
            wrapper(x, timesteps, omega=None, mode_id=mode_id)


# =============================================================================
# Test create_conditioning_wrapper factory
# =============================================================================

class TestCreateConditioningWrapper:
    """Tests for create_conditioning_wrapper factory function."""

    def test_no_conditioning(self, mock_model):
        """Test: No conditioning returns original model."""
        wrapper, name = create_conditioning_wrapper(
            mock_model, use_omega=False, use_mode=False
        )
        assert wrapper is mock_model
        assert name is None

    def test_omega_only(self, mock_model):
        """Test: Omega only returns ScoreAugModelWrapper."""
        wrapper, name = create_conditioning_wrapper(
            mock_model, use_omega=True, use_mode=False, embed_dim=256
        )
        assert name == "omega"
        from medgen.data.score_aug import ScoreAugModelWrapper
        assert isinstance(wrapper, ScoreAugModelWrapper)

    def test_mode_only(self, mock_model):
        """Test: Mode only returns ModeEmbedModelWrapper."""
        wrapper, name = create_conditioning_wrapper(
            mock_model, use_omega=False, use_mode=True, embed_dim=256
        )
        assert name == "mode"
        assert isinstance(wrapper, ModeEmbedModelWrapper)

    def test_both_conditioning(self, mock_model):
        """Test: Both returns CombinedModelWrapper."""
        wrapper, name = create_conditioning_wrapper(
            mock_model, use_omega=True, use_mode=True, embed_dim=256
        )
        assert name == "combined"
        assert isinstance(wrapper, CombinedModelWrapper)


# =============================================================================
# Edge case: Batch size mismatch between mode encoding and input
# =============================================================================

class TestBatchSizeMismatch:
    """Test batch size mismatches between encodings and inputs."""

    def test_mode_encoding_batch_mismatch_broadcast_failure(self, mock_time_embed):
        """Batch size mismatch should cause broadcast failure."""
        embed_dim = 256
        mode_embed = ModeTimeEmbed(mock_time_embed, embed_dim)

        # Set mode_encoding for batch_size=2
        mode_encoding = torch.zeros(2, MODE_ENCODING_DIM)
        mode_encoding[:, 0] = 1.0
        mode_embed.set_mode_encoding(mode_encoding)

        # Try forward with batch_size=4
        t_emb = torch.randn(4, 128)

        # This should fail because:
        # out has shape [4, 256]
        # mode_emb has shape [2, 256]
        # These cannot be added together
        with pytest.raises(RuntimeError):
            mode_embed(t_emb)

    def test_mode_encoding_batch_mismatch_broadcast_works(self, mock_time_embed):
        """Batch size 1 should broadcast successfully."""
        embed_dim = 256
        mode_embed = ModeTimeEmbed(mock_time_embed, embed_dim)

        # Set mode_encoding for batch_size=1
        mode_encoding = torch.zeros(1, MODE_ENCODING_DIM)
        mode_encoding[:, 0] = 1.0
        mode_embed.set_mode_encoding(mode_encoding)

        # Forward with batch_size=4
        t_emb = torch.randn(4, 128)

        # This should work due to broadcasting [1, 256] + [4, 256] -> [4, 256]
        result = mode_embed(t_emb)
        assert result.shape == (4, embed_dim)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
