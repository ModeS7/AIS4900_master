"""Tests for SiT (Scalable Interpolant Transformers) blocks.

Tests cover drop_path function, Attention, CrossAttention, Mlp,
SiTBlock with adaLN-Zero, and FinalLayer. Includes gradient flow
verification to catch silent failures.
"""

import pytest
import torch
import torch.nn as nn

from medgen.models.sit_blocks import (
    drop_path,
    DropPath,
    modulate,
    Attention,
    CrossAttention,
    Mlp,
    SiTBlock,
    FinalLayer,
)


class TestDropPath:
    """Tests for stochastic depth drop_path function."""

    def test_drop_path_formula_correct(self):
        """Surviving paths scaled by 1/keep_prob."""
        x = torch.ones(100, 16, 64)  # Large batch for statistics
        drop_prob = 0.5

        # In training, some samples zeroed, others scaled by 2x
        torch.manual_seed(42)
        output = drop_path(x, drop_prob=drop_prob, training=True)

        # Non-zero outputs should be scaled by 1/(1-0.5) = 2
        non_zero_mask = output.abs().sum(dim=(1, 2)) > 0
        if non_zero_mask.any():
            scaled_values = output[non_zero_mask]
            # Each element should be 2.0 (1.0 scaled by 1/0.5)
            assert scaled_values.mean() == pytest.approx(2.0, rel=0.01), \
                "Surviving paths not scaled correctly"

    def test_drop_path_disabled_in_eval(self):
        """drop_path returns input unchanged in eval mode."""
        x = torch.randn(4, 16, 64)
        output = drop_path(x, drop_prob=0.5, training=False)
        assert torch.equal(x, output), "Eval mode should return input unchanged"

    def test_drop_path_zero_prob_identity(self):
        """drop_prob=0 returns input unchanged."""
        x = torch.randn(4, 16, 64)
        output = drop_path(x, drop_prob=0.0, training=True)
        assert torch.equal(x, output), "Zero drop_prob should be identity"

    def test_drop_path_high_prob_mostly_zeros(self):
        """drop_prob=0.99 returns mostly zeros."""
        # drop_prob=1.0 causes div-by-zero (produces NaN), so test with 0.99
        x = torch.randn(100, 16, 64)
        torch.manual_seed(42)
        output = drop_path(x, drop_prob=0.99, training=True)
        # Most samples should be zeroed
        zero_count = (output.abs().sum(dim=(1, 2)) == 0).sum()
        assert zero_count >= 90, "99% drop_prob should zero most samples"

    def test_drop_path_preserves_shape(self):
        """Output shape matches input shape."""
        x = torch.randn(4, 16, 64)
        output = drop_path(x, drop_prob=0.3, training=True)
        assert output.shape == x.shape

    def test_drop_path_module_wrapper(self):
        """DropPath module wraps function correctly."""
        dp = DropPath(drop_prob=0.3)
        x = torch.randn(4, 16, 64)

        dp.train()
        output_train = dp(x)
        assert output_train.shape == x.shape

        dp.eval()
        output_eval = dp(x)
        assert torch.equal(output_eval, x)


class TestModulate:
    """Tests for adaLN modulation function."""

    def test_modulate_formula(self):
        """Verifies x * (1 + scale) + shift formula."""
        x = torch.ones(2, 4, 8)  # [B, N, D]
        shift = torch.full((2, 8), 0.5)  # [B, D]
        scale = torch.full((2, 8), 1.0)  # [B, D]

        output = modulate(x, shift, scale)

        # Expected: 1 * (1 + 1) + 0.5 = 2.5
        assert output.mean() == pytest.approx(2.5, rel=0.01)

    def test_modulate_shape_preservation(self):
        """Output shape matches input shape."""
        x = torch.randn(2, 64, 256)
        shift = torch.randn(2, 256)
        scale = torch.randn(2, 256)

        output = modulate(x, shift, scale)
        assert output.shape == x.shape


class TestAttention:
    """Tests for multi-head self-attention."""

    def test_attention_output_shape(self):
        """Output shape matches input shape."""
        attn = Attention(dim=256, num_heads=8)
        x = torch.randn(2, 64, 256)  # [B, N, D]
        output = attn(x)
        assert output.shape == x.shape

    def test_attention_gradients_flow(self):
        """Gradients reach qkv weights."""
        attn = Attention(dim=256, num_heads=8)
        x = torch.randn(2, 64, 256, requires_grad=True)
        output = attn(x)
        loss = output.mean()
        loss.backward()
        assert attn.qkv.weight.grad is not None
        assert attn.qkv.weight.grad.abs().sum() > 0

    def test_attention_qk_norm(self):
        """QK-normalization layers exist when enabled."""
        attn = Attention(dim=256, num_heads=8, qk_norm=True)
        assert hasattr(attn, 'q_norm')
        assert hasattr(attn, 'k_norm')

    def test_attention_without_qk_norm(self):
        """No QK-norm layers when disabled."""
        attn = Attention(dim=256, num_heads=8, qk_norm=False)
        assert not hasattr(attn, 'q_norm')

    def test_attention_dimension_divisibility(self):
        """Raises error when dim not divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible"):
            Attention(dim=256, num_heads=7)


class TestCrossAttention:
    """Tests for multi-head cross-attention."""

    def test_cross_attention_output_shape(self):
        """Output shape matches query shape."""
        cross_attn = CrossAttention(dim=256, num_heads=8)
        x = torch.randn(2, 64, 256)  # [B, N, D] query
        context = torch.randn(2, 32, 256)  # [B, M, D] context
        output = cross_attn(x, context)
        assert output.shape == x.shape

    def test_cross_attention_gradients_flow(self):
        """Gradients reach both q and kv projections."""
        cross_attn = CrossAttention(dim=256, num_heads=8)
        x = torch.randn(2, 64, 256, requires_grad=True)
        context = torch.randn(2, 32, 256, requires_grad=True)
        output = cross_attn(x, context)
        loss = output.mean()
        loss.backward()

        assert cross_attn.q.weight.grad is not None
        assert cross_attn.kv.weight.grad is not None

    def test_cross_attention_different_context_lengths(self):
        """Works with various context sequence lengths."""
        cross_attn = CrossAttention(dim=256, num_heads=8)
        x = torch.randn(2, 64, 256)

        for context_len in [1, 16, 64, 128]:
            context = torch.randn(2, context_len, 256)
            output = cross_attn(x, context)
            assert output.shape == x.shape


class TestMlp:
    """Tests for MLP block."""

    def test_mlp_output_shape(self):
        """Output shape matches input shape by default."""
        mlp = Mlp(in_features=256)
        x = torch.randn(2, 64, 256)
        output = mlp(x)
        assert output.shape == x.shape

    def test_mlp_hidden_expansion(self):
        """Default hidden dimension is 4x input."""
        mlp = Mlp(in_features=256)
        # fc1 projects to hidden, fc2 projects back
        assert mlp.fc1.out_features == 256 * 4
        assert mlp.fc2.in_features == 256 * 4

    def test_mlp_custom_dimensions(self):
        """Custom hidden and output dimensions work."""
        mlp = Mlp(in_features=256, hidden_features=512, out_features=128)
        assert mlp.fc1.out_features == 512
        assert mlp.fc2.out_features == 128


class TestSiTBlock:
    """Tests for transformer block with adaLN-Zero."""

    def test_sitblock_forward_shape(self):
        """Output shape matches input shape."""
        block = SiTBlock(hidden_size=256, num_heads=8)
        x = torch.randn(2, 64, 256)  # [B, N, D]
        c = torch.randn(2, 256)       # [B, D] conditioning
        output = block(x, c)
        assert output.shape == x.shape

    def test_sitblock_cross_attention_requires_context(self):
        """ValueError when use_cross_attn=True but context=None."""
        block = SiTBlock(hidden_size=256, num_heads=8, use_cross_attn=True)
        x = torch.randn(2, 64, 256)
        c = torch.randn(2, 256)
        with pytest.raises(ValueError, match="context"):
            block(x, c, context=None)

    def test_sitblock_cross_attention_with_context(self):
        """Cross-attention works when context provided."""
        block = SiTBlock(hidden_size=256, num_heads=8, use_cross_attn=True)
        x = torch.randn(2, 64, 256)
        c = torch.randn(2, 256)
        context = torch.randn(2, 32, 256)  # [B, M, D]
        output = block(x, c, context=context)
        assert output.shape == x.shape

    def test_sitblock_without_cross_attention(self):
        """Works without cross-attention."""
        block = SiTBlock(hidden_size=256, num_heads=8, use_cross_attn=False)
        x = torch.randn(2, 64, 256)
        c = torch.randn(2, 256)
        output = block(x, c)  # No context needed
        assert output.shape == x.shape

    def test_sitblock_gradients_flow_through_adaln_gate(self):
        """Gradients reach adaLN parameters."""
        block = SiTBlock(hidden_size=256, num_heads=8)
        x = torch.randn(2, 64, 256, requires_grad=True)
        c = torch.randn(2, 256, requires_grad=True)
        output = block(x, c)
        loss = output.mean()
        loss.backward()

        # adaLN_modulation should have gradients
        # The modulation is nn.Sequential with [SiLU, Linear]
        linear_layer = block.adaLN_modulation[1]
        assert linear_layer.weight.grad is not None
        assert linear_layer.weight.grad.abs().sum() > 0

    def test_sitblock_drop_path_propagates_gradients(self):
        """Gradients flow through drop_path even with high drop_prob."""
        block = SiTBlock(hidden_size=256, num_heads=8, drop_path=0.5)
        block.train()

        # Run multiple times to ensure we get a valid forward pass
        for seed in range(10):
            torch.manual_seed(seed)
            x = torch.randn(2, 64, 256, requires_grad=True)
            c = torch.randn(2, 256)

            output = block(x, c)
            if output.abs().sum() > 0:  # Non-zero output
                loss = output.mean()
                loss.backward()
                assert x.grad is not None, "Gradients should flow to input"
                break
        else:
            pytest.fail("All drop_path samples were dropped")

    def test_sitblock_adaln_modulation_6_params(self):
        """adaLN produces 6 modulation parameters."""
        block = SiTBlock(hidden_size=256, num_heads=8)
        # adaLN_modulation projects to 6 * hidden_size
        linear_layer = block.adaLN_modulation[1]
        assert linear_layer.out_features == 6 * 256


class TestFinalLayer:
    """Tests for final projection layer."""

    @pytest.mark.parametrize("spatial_dims,patch_size", [(2, 4), (3, 2)])
    def test_final_layer_output_shape(self, spatial_dims, patch_size):
        """Output has correct channel dimension."""
        out_channels = 4
        layer = FinalLayer(
            hidden_size=256, patch_size=patch_size,
            out_channels=out_channels, spatial_dims=spatial_dims
        )
        x = torch.randn(2, 64, 256)
        c = torch.randn(2, 256)
        output = layer(x, c)

        expected_patch_dim = patch_size ** spatial_dims * out_channels
        assert output.shape == (2, 64, expected_patch_dim)

    def test_final_layer_2d_projection(self):
        """2D final layer projects to patch_size^2 * out_channels."""
        layer = FinalLayer(hidden_size=256, patch_size=4, out_channels=4, spatial_dims=2)
        # Linear should project to 4^2 * 4 = 64
        assert layer.linear.out_features == 64

    def test_final_layer_3d_projection(self):
        """3D final layer projects to patch_size^3 * out_channels."""
        layer = FinalLayer(hidden_size=256, patch_size=2, out_channels=4, spatial_dims=3)
        # Linear should project to 2^3 * 4 = 32
        assert layer.linear.out_features == 32

    def test_final_layer_gradients_flow(self):
        """Gradients flow through final layer."""
        layer = FinalLayer(hidden_size=256, patch_size=4, out_channels=4, spatial_dims=2)
        x = torch.randn(2, 64, 256, requires_grad=True)
        c = torch.randn(2, 256, requires_grad=True)
        output = layer(x, c)
        loss = output.mean()
        loss.backward()

        assert layer.linear.weight.grad is not None
        assert layer.adaLN_modulation[1].weight.grad is not None

    def test_final_layer_adaln_2_params(self):
        """Final layer adaLN produces 2 modulation parameters (shift, scale)."""
        layer = FinalLayer(hidden_size=256, patch_size=4, out_channels=4, spatial_dims=2)
        # adaLN_modulation projects to 2 * hidden_size
        linear_layer = layer.adaLN_modulation[1]
        assert linear_layer.out_features == 2 * 256


class TestIntegration:
    """Integration tests for combined components."""

    def test_full_forward_pass(self):
        """Full forward pass through SiTBlock + FinalLayer."""
        block = SiTBlock(hidden_size=256, num_heads=8)
        final = FinalLayer(hidden_size=256, patch_size=4, out_channels=4, spatial_dims=2)

        x = torch.randn(2, 64, 256)
        c = torch.randn(2, 256)

        # Forward through block
        x = block(x, c)

        # Forward through final layer
        output = final(x, c)

        # Should have patch_size^2 * out_channels = 64 channels
        assert output.shape == (2, 64, 64)

    def test_multiple_blocks_gradient_flow(self):
        """Gradients flow through multiple stacked blocks."""
        blocks = nn.ModuleList([
            SiTBlock(hidden_size=256, num_heads=8)
            for _ in range(3)
        ])

        x = torch.randn(2, 64, 256, requires_grad=True)
        c = torch.randn(2, 256)

        for block in blocks:
            x = block(x, c)

        loss = x.mean()
        loss.backward()

        # All blocks should have gradients
        for i, block in enumerate(blocks):
            assert block.adaLN_modulation[1].weight.grad is not None, \
                f"Block {i} has no gradients"
