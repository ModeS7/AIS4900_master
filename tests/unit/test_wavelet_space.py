"""Tests for WaveletSpace (Haar 3D) diffusion space."""
import pytest
import torch

from medgen.diffusion.spaces import WaveletSpace
from medgen.models.haar_wavelet_3d import HaarForward3D, HaarInverse3D


class TestHaarWavelet3D:
    """Test the raw Haar wavelet forward/inverse ops."""

    def test_haar_forward_inverse_roundtrip(self):
        """Encode then decode should be lossless (exact roundtrip)."""
        fwd = HaarForward3D()
        inv = HaarInverse3D()
        x = torch.randn(2, 1, 16, 32, 32)

        z = fwd(x)
        reconstructed = inv(z)

        assert reconstructed.shape == x.shape
        torch.testing.assert_close(reconstructed, x)

    def test_haar_forward_inverse_roundtrip_large(self):
        """Roundtrip at realistic volume size [2, 1, 160, 256, 256] (shape only)."""
        # Use small proxy with same divisibility to avoid OOM
        fwd = HaarForward3D()
        inv = HaarInverse3D()
        x = torch.randn(1, 1, 160, 64, 64)

        z = fwd(x)
        assert z.shape == (1, 8, 80, 32, 32)

        reconstructed = inv(z)
        torch.testing.assert_close(reconstructed, x)

    def test_haar_shapes(self):
        """Forward transform: [B, C, D, H, W] -> [B, C*8, D/2, H/2, W/2]."""
        fwd = HaarForward3D()

        # Single channel
        x = torch.randn(2, 1, 8, 16, 16)
        z = fwd(x)
        assert z.shape == (2, 8, 4, 8, 8)

        # Multi channel
        x = torch.randn(2, 3, 8, 16, 16)
        z = fwd(x)
        assert z.shape == (2, 24, 4, 8, 8)

    def test_haar_low_freq_is_average(self):
        """Channel 0 (LLL) should approximate the local 2x2x2 average.

        For Haar with 1/sqrt(2) normalization applied 3 times:
        LLL = sum(8 voxels) / (sqrt(2))^3 = sum(8 voxels) / 2*sqrt(2)
        """
        fwd = HaarForward3D()

        # Constant input: all voxels = 1.0
        x = torch.ones(1, 1, 4, 4, 4)
        z = fwd(x)

        # LLL = first C channels = z[:, :1, ...]
        lll = z[:, :1, :, :, :]

        # For constant input, LLL should be 1 * (sqrt(2))^3 = 2*sqrt(2)
        # because: low = (even + odd) / sqrt(2) applied 3 times
        # After 1st split (W): low = (1+1)/sqrt(2) = sqrt(2)
        # After 2nd split (H): low = (sqrt(2)+sqrt(2))/sqrt(2) = 2
        # After 3rd split (D): low = (2+2)/sqrt(2) = 2*sqrt(2)
        expected_value = 2 * (2 ** 0.5)
        torch.testing.assert_close(
            lll,
            torch.full_like(lll, expected_value),
        )

        # All detail coefficients should be 0 for constant input
        details = z[:, 1:, :, :, :]
        torch.testing.assert_close(details, torch.zeros_like(details))

    def test_haar_energy_preservation(self):
        """||forward(x)||^2 == ||x||^2 (orthogonal transform)."""
        fwd = HaarForward3D()
        x = torch.randn(2, 1, 8, 16, 16)
        z = fwd(x)

        energy_input = (x ** 2).sum()
        energy_output = (z ** 2).sum()
        torch.testing.assert_close(energy_output, energy_input)

    def test_haar_energy_preservation_multichannel(self):
        """Energy preservation for multi-channel input."""
        fwd = HaarForward3D()
        x = torch.randn(2, 3, 8, 16, 16)
        z = fwd(x)

        energy_input = (x ** 2).sum()
        energy_output = (z ** 2).sum()
        torch.testing.assert_close(energy_output, energy_input)


class TestWaveletSpace:
    """Test WaveletSpace encode/decode and properties."""

    def test_wavelet_space_bravo_shapes(self):
        """In bravo mode: image(8ch) + seg(8ch) = 16ch input to UNet."""
        space = WaveletSpace()

        image = torch.randn(1, 1, 8, 16, 16)
        seg = torch.randn(1, 1, 8, 16, 16)

        encoded_image = space.encode(image)
        encoded_seg = space.encode(seg)

        assert encoded_image.shape == (1, 8, 4, 8, 8)
        assert encoded_seg.shape == (1, 8, 4, 8, 8)

        # Concatenate along channel dim (what mode.format_model_input does)
        model_input = torch.cat([encoded_image, encoded_seg], dim=1)
        assert model_input.shape == (1, 16, 4, 8, 8)

    def test_wavelet_space_get_latent_channels(self):
        """get_latent_channels always multiplies by 8."""
        space = WaveletSpace()
        assert space.get_latent_channels(1) == 8
        assert space.get_latent_channels(2) == 16
        assert space.get_latent_channels(3) == 24

    def test_wavelet_space_properties(self):
        """Check scale_factor, depth_scale_factor, and latent_channels."""
        space = WaveletSpace()
        assert space.scale_factor == 2
        assert space.depth_scale_factor == 2
        assert space.latent_channels == 8

    def test_wavelet_encode_rejects_4d(self):
        """4D input (2D images) should raise ValueError."""
        space = WaveletSpace()
        x = torch.randn(2, 1, 256, 256)

        with pytest.raises(ValueError, match="5D"):
            space.encode(x)

    def test_wavelet_decode_rejects_4d(self):
        """4D input should raise ValueError on decode too."""
        space = WaveletSpace()
        z = torch.randn(2, 8, 128, 128)

        with pytest.raises(ValueError, match="5D"):
            space.decode(z)

    def test_wavelet_divisibility_check(self):
        """Odd dimension should raise ValueError."""
        space = WaveletSpace()
        # W=15 is not divisible by 2
        x = torch.randn(1, 1, 8, 16, 15)

        with pytest.raises(ValueError):
            space.encode(x)

    def test_wavelet_encode_batch_dict(self):
        """encode_batch handles dict of tensors (inherited from DiffusionSpace)."""
        space = WaveletSpace()
        batch = {
            'image': torch.randn(1, 1, 8, 16, 16),
            'seg': torch.randn(1, 1, 8, 16, 16),
        }

        encoded = space.encode_batch(batch)
        assert isinstance(encoded, dict)
        assert encoded['image'].shape == (1, 8, 4, 8, 8)
        assert encoded['seg'].shape == (1, 8, 4, 8, 8)

    def test_wavelet_encode_decode_roundtrip(self):
        """Full WaveletSpace roundtrip through encode/decode."""
        space = WaveletSpace()
        x = torch.randn(2, 1, 8, 16, 16)

        z = space.encode(x)
        assert z.shape == (2, 8, 4, 8, 8)

        reconstructed = space.decode(z)
        assert reconstructed.shape == x.shape
        torch.testing.assert_close(reconstructed, x)
