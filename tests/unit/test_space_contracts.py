"""Tests for DiffusionSpace interface contracts.

Verifies scale_factor, needs_decode, get_latent_channels, and
encode/decode roundtrip for non-latent spaces:
- PixelSpace (identity and rescale modes)
- SpaceToDepthSpace (3D lossless rearrangement)
- WaveletSpace (3D Haar decomposition)

Catches regressions in the space abstraction that would silently corrupt
training or generation by producing wrong channel counts or lossy roundtrips.
"""

import pytest
import torch

from medgen.diffusion.spaces import PixelSpace, SpaceToDepthSpace, WaveletSpace


# ---------------------------------------------------------------------------
# PixelSpace (identity)
# ---------------------------------------------------------------------------


class TestPixelSpaceIdentity:
    """PixelSpace(rescale=False) — pure identity."""

    def test_scale_factor(self):
        assert PixelSpace().scale_factor == 1

    def test_needs_decode_false(self):
        assert PixelSpace().needs_decode is False

    def test_latent_channels(self):
        assert PixelSpace().get_latent_channels(3) == 3

    def test_roundtrip_identity(self):
        space = PixelSpace()
        x = torch.randn(2, 1, 32, 32)
        torch.testing.assert_close(space.decode(space.encode(x)), x)


# ---------------------------------------------------------------------------
# PixelSpace (rescale)
# ---------------------------------------------------------------------------


class TestPixelSpaceRescale:
    """PixelSpace(rescale=True) — [0,1] ↔ [-1,1]."""

    def test_scale_factor_still_1(self):
        assert PixelSpace(rescale=True).scale_factor == 1

    def test_needs_decode_true(self):
        assert PixelSpace(rescale=True).needs_decode is True

    def test_encode_range(self):
        space = PixelSpace(rescale=True)
        x = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 1, 3)
        z = space.encode(x)
        torch.testing.assert_close(z, torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 1, 1, 3))

    def test_roundtrip(self):
        space = PixelSpace(rescale=True)
        x = torch.rand(2, 1, 16, 16)
        torch.testing.assert_close(space.decode(space.encode(x)), x)


# ---------------------------------------------------------------------------
# PixelSpace (shift/scale normalization)
# ---------------------------------------------------------------------------


class TestPixelSpaceShiftScale:
    """PixelSpace with shift/scale brain-only normalization."""

    def test_needs_decode_true(self):
        space = PixelSpace(shift=[0.5], scale=[0.2])
        assert space.needs_decode is True

    def test_encode_conditioning_true(self):
        space = PixelSpace(shift=[0.5], scale=[0.2])
        assert space.encode_conditioning is True

    def test_roundtrip_2d(self):
        space = PixelSpace(shift=[0.3], scale=[0.15])
        x = torch.rand(2, 1, 16, 16)
        torch.testing.assert_close(space.decode(space.encode(x)), x, atol=1e-6, rtol=1e-5)

    def test_roundtrip_3d(self):
        space = PixelSpace(shift=[0.3], scale=[0.15])
        x = torch.rand(1, 1, 8, 16, 16)
        torch.testing.assert_close(space.decode(space.encode(x)), x, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# SpaceToDepthSpace
# ---------------------------------------------------------------------------


class TestSpaceToDepthSpace:
    """SpaceToDepthSpace(2, 2) — 3D lossless rearrangement."""

    def test_scale_factor(self):
        assert SpaceToDepthSpace(2, 2).scale_factor == 2

    def test_needs_decode_true(self):
        assert SpaceToDepthSpace(2, 2).needs_decode is True

    def test_latent_channels(self):
        # 2x2x2 = 8x channel multiplier
        assert SpaceToDepthSpace(2, 2).get_latent_channels(1) == 8
        assert SpaceToDepthSpace(2, 2).get_latent_channels(3) == 24

    def test_encode_shape(self):
        space = SpaceToDepthSpace(2, 2)
        x = torch.randn(1, 1, 16, 32, 32)
        z = space.encode(x)
        assert z.shape == (1, 8, 8, 16, 16)

    def test_roundtrip(self):
        space = SpaceToDepthSpace(2, 2)
        x = torch.randn(1, 1, 16, 32, 32)
        torch.testing.assert_close(space.decode(space.encode(x)), x)

    def test_rejects_4d_input(self):
        space = SpaceToDepthSpace(2, 2)
        with pytest.raises(ValueError, match="5D"):
            space.encode(torch.randn(1, 1, 32, 32))


# ---------------------------------------------------------------------------
# WaveletSpace
# ---------------------------------------------------------------------------


class TestWaveletSpace:
    """WaveletSpace — 3D Haar decomposition."""

    def test_scale_factor(self):
        assert WaveletSpace().scale_factor == 2

    def test_needs_decode_true(self):
        assert WaveletSpace().needs_decode is True

    def test_latent_channels(self):
        assert WaveletSpace().get_latent_channels(1) == 8

    def test_encode_shape(self):
        space = WaveletSpace()
        x = torch.randn(1, 1, 16, 32, 32)
        z = space.encode(x)
        assert z.shape == (1, 8, 8, 16, 16)

    def test_roundtrip(self):
        """Haar wavelet is orthogonal → lossless roundtrip."""
        space = WaveletSpace()
        x = torch.randn(1, 1, 16, 32, 32)
        torch.testing.assert_close(space.decode(space.encode(x)), x)

    def test_rejects_4d_input(self):
        space = WaveletSpace()
        with pytest.raises(ValueError, match="5D"):
            space.encode(torch.randn(1, 1, 32, 32))


# ---------------------------------------------------------------------------
# WaveletSpace with shift/scale normalization
# ---------------------------------------------------------------------------


class TestWaveletSpaceNormalized:
    """WaveletSpace with per-subband normalization."""

    def test_roundtrip_with_normalization(self):
        shift = [0.1, -0.05, 0.02, -0.01, 0.03, -0.02, 0.01, -0.005]
        scale = [0.5, 0.3, 0.2, 0.15, 0.3, 0.2, 0.15, 0.1]
        space = WaveletSpace(shift=shift, scale=scale)
        x = torch.randn(1, 1, 16, 32, 32)
        torch.testing.assert_close(
            space.decode(space.encode(x)), x, atol=1e-5, rtol=1e-4,
        )

    def test_shift_scale_properties(self):
        shift = [0.1] * 8
        scale = [0.5] * 8
        space = WaveletSpace(shift=shift, scale=scale)
        assert space.shift == pytest.approx(shift)
        assert space.scale == pytest.approx(scale)
