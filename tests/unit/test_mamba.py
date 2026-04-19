"""Unit tests for LaMamba-Diff architecture (SS2D, WindowAttention, MambaDiff).

These tests run on CPU with the PyTorch fallback path for `selective_scan_fn`.
They cover:
- SS2D forward shape (2D + 3D), cross-scan/cross-merge alignment
- WindowAttention forward shape (2D + 3D)
- MambaDiff end-to-end forward (small config)
- AdaLN-Zero init (final_linear zero'd)
- Gradient checkpointing toggle
- Factory `create_mamba_diff`

We force `MAMBA_CUDA_AVAILABLE = False` in all tests so the suite runs
deterministically on CPU without the CUDA mamba_ssm kernel.
"""

from __future__ import annotations

import pytest
import torch

import medgen.models.mamba_blocks as mb

# Force CPU fallback so tests work without CUDA
mb.MAMBA_CUDA_AVAILABLE = False  # noqa: E402

from medgen.models.mamba_blocks import SS2D, WindowAttention
from medgen.models.mamba_diff import (
    MAMBA_VARIANTS,
    MambaDiff,
    MambaDiffBlock,
    create_mamba_diff,
)


# =============================================================================
# SS2D
# =============================================================================


class TestSS2D:
    def test_ss2d_2d_forward_shape(self):
        m = SS2D(d_model=32, d_state=4, spatial_dims=2)
        x = torch.randn(2, 16, 24, 32)  # [B, H, W, C]
        y = m(x)
        assert y.shape == x.shape

    def test_ss2d_3d_forward_shape(self):
        m = SS2D(d_model=32, d_state=4, spatial_dims=3)
        x = torch.randn(2, 8, 12, 16, 32)  # [B, D, H, W, C]
        y = m(x)
        assert y.shape == x.shape

    def test_ss2d_2d_cross_merge_alignment(self):
        """With an identity SSM (ys == xs), cross_merge should sum 4 aligned copies.

        This is the regression test for the cross_merge inverse-transpose bug
        (C-05 in the code review). If inverse-permutation is correct, every
        direction's output at spatial position (h, w) is the same input value,
        so the sum is 4 * input.
        """
        m = SS2D(d_model=4, d_state=2, spatial_dims=2)
        B, C, H, W = 1, 2, 3, 4
        x = torch.arange(B * C * H * W, dtype=torch.float32).reshape(B, C, H, W)
        xs = m._cross_scan(x)  # [B, 4, C, L]
        y = m._cross_merge(xs, (H, W))  # [B, C, L]
        expected = 4 * x.reshape(B, C, H * W)
        assert torch.allclose(y, expected), "2D cross_merge misaligned directions"

    def test_ss2d_3d_cross_merge_alignment(self):
        m = SS2D(d_model=4, d_state=2, spatial_dims=3)
        B, C, D, H, W = 1, 2, 2, 3, 4
        x = torch.arange(B * C * D * H * W, dtype=torch.float32).reshape(B, C, D, H, W)
        xs = m._cross_scan(x)  # [B, 6, C, L]
        y = m._cross_merge(xs, (D, H, W))  # [B, C, L]
        expected = 6 * x.reshape(B, C, D * H * W)
        assert torch.allclose(y, expected), "3D cross_merge misaligned directions"

    def test_ss2d_2d_gradient_flows(self):
        m = SS2D(d_model=8, d_state=2, spatial_dims=2)
        x = torch.randn(1, 4, 4, 8, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0


# =============================================================================
# WindowAttention
# =============================================================================


class TestWindowAttention:
    def test_window_attention_2d_shape(self):
        wa = WindowAttention(dim=32, num_heads=4, window_size=4, spatial_dims=2)
        x = torch.randn(2, 8, 8, 32)  # [B, H, W, C]
        y = wa(x)
        assert y.shape == x.shape

    def test_window_attention_3d_shape(self):
        wa = WindowAttention(dim=32, num_heads=4, window_size=4, spatial_dims=3)
        x = torch.randn(1, 4, 8, 8, 32)  # [B, D, H, W, C]
        y = wa(x)
        assert y.shape == x.shape

    def test_window_attention_shifted(self):
        """Shifted window variant should produce a different output than unshifted."""
        wa = WindowAttention(dim=32, num_heads=4, window_size=4, spatial_dims=2, shift_size=2)
        x = torch.randn(1, 8, 8, 32)
        y = wa(x)
        assert y.shape == x.shape

    def test_window_attention_padding(self):
        """Non-divisible spatial sizes are padded up to window multiples."""
        wa = WindowAttention(dim=32, num_heads=4, window_size=4, spatial_dims=2)
        x = torch.randn(1, 7, 9, 32)
        y = wa(x)
        assert y.shape == x.shape


# =============================================================================
# MambaDiffBlock
# =============================================================================


class TestMambaDiffBlock:
    def test_block_2d_with_conditioning(self):
        blk = MambaDiffBlock(
            dim=32, num_heads=4, window_size=4,
            ssm_d_state=2, ssm_ratio=2.0, mlp_ratio=2.0, spatial_dims=2,
        )
        x = torch.randn(1, 8, 8, 32)
        c = torch.randn(1, 32)  # per-sample conditioning
        y = blk(x, c)
        assert y.shape == x.shape


# =============================================================================
# MambaDiff
# =============================================================================


class TestMambaDiff:
    def _small_2d(self) -> MambaDiff:
        # Keep the model very small so tests are fast on CPU fallback
        return MambaDiff(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            input_size=16,
            patch_size=2,
            dims=[16, 32, 32, 32],
            depths=[1, 1, 1, 1],
            bottleneck_depth=1,
            window_size=4,
            num_heads=2,
            ssm_d_state=2,
            ssm_ratio=2.0,
            mlp_ratio=2.0,
            skip=2,
        )

    def _small_3d(self) -> MambaDiff:
        return MambaDiff(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            input_size=16,
            depth_size=8,
            patch_size=2,
            dims=[8, 16, 16, 16],
            depths=[1, 1, 1, 1],
            bottleneck_depth=1,
            window_size=4,
            num_heads=2,
            ssm_d_state=2,
            ssm_ratio=2.0,
            mlp_ratio=2.0,
            skip=2,
        )

    def test_mamba_diff_2d_forward_shape(self):
        m = self._small_2d()
        x = torch.randn(1, 1, 16, 16)
        t = torch.tensor([500], dtype=torch.long)
        y = m(x, t)
        assert y.shape == x.shape

    def test_mamba_diff_3d_forward_shape(self):
        m = self._small_3d()
        x = torch.randn(1, 1, 8, 16, 16)
        t = torch.tensor([500], dtype=torch.long)
        y = m(x, t)
        assert y.shape == x.shape

    def test_mamba_diff_2d_deterministic(self):
        """Two forward passes with same input should produce same output (no dropout by default)."""
        m = self._small_2d()
        m.eval()
        x = torch.randn(1, 1, 16, 16)
        t = torch.tensor([500], dtype=torch.long)
        with torch.no_grad():
            y1 = m(x, t)
            y2 = m(x, t)
        assert torch.allclose(y1, y2)

    def test_mamba_diff_gradient_checkpointing_toggle(self):
        m = self._small_2d()
        assert m._use_checkpoint is False
        m.enable_gradient_checkpointing()
        assert m._use_checkpoint is True
        # All stages should flip
        for stage in list(m.encoder_stages) + list(m.decoder_stages) + [m.bottleneck]:
            assert stage.use_checkpoint is True

    def test_mamba_diff_final_linear_zero_init(self):
        """AdaLN-Zero convention: final projection starts at zero → model starts near identity."""
        m = self._small_2d()
        assert torch.all(m.final_linear.weight == 0)
        assert torch.all(m.final_linear.bias == 0)

    def test_mamba_diff_adaln_gate_zero_init(self):
        """AdaLN-Zero: the last linear in every adaLN_modulation block is zero-initialized."""
        m = self._small_2d()
        for stage in list(m.encoder_stages) + list(m.decoder_stages) + [m.bottleneck]:
            for blk in stage.blocks:
                last_linear = blk.adaLN_modulation[1]
                assert torch.all(last_linear.weight == 0)
                assert torch.all(last_linear.bias == 0)

    def test_mamba_diff_backward(self):
        m = self._small_2d()
        x = torch.randn(1, 1, 16, 16, requires_grad=True)
        t = torch.tensor([100], dtype=torch.long)
        y = m(x, t)
        y.mean().backward()
        assert x.grad is not None

    def test_hidden_size_attribute(self):
        """`hidden_size` attribute is required by compile/DDP wrappers."""
        m = self._small_2d()
        assert hasattr(m, 'hidden_size')
        assert m.hidden_size > 0


# =============================================================================
# Factory
# =============================================================================


class TestFactory:
    def test_create_mamba_diff_2d_small(self):
        m = create_mamba_diff(
            variant='S', spatial_dims=2, input_size=16, patch_size=2,
            in_channels=1, out_channels=1,
            dims=[16, 32, 32, 32], depths=[1, 1, 1, 1], bottleneck_depth=1,
            window_size=4, num_heads=2, ssm_d_state=2,
        )
        assert isinstance(m, MambaDiff)

    def test_create_mamba_diff_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            create_mamba_diff(variant='ZZZ', spatial_dims=2)

    def test_mamba_variants_dict_exports_s_b_l_xl(self):
        for v in ['S', 'B', 'L', 'XL']:
            assert v in MAMBA_VARIANTS
            assert 'embed_dim' in MAMBA_VARIANTS[v]
            assert 'num_heads' in MAMBA_VARIANTS[v]
