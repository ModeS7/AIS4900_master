"""Tests for WaveletSpace (Haar 3D) diffusion space and rescaling."""
import pytest
import torch

from medgen.diffusion.spaces import PixelSpace, SpaceToDepthSpace, WaveletSpace
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

    def test_wavelet_no_args_backward_compat(self):
        """WaveletSpace() with no args works identically to before."""
        space = WaveletSpace()
        assert space._shift is None
        assert space._scale is None

        x = torch.randn(2, 1, 8, 16, 16)
        z = space.encode(x)
        reconstructed = space.decode(z)
        torch.testing.assert_close(reconstructed, x)

    def test_wavelet_normalized_roundtrip(self):
        """encode(decode(x)) still reconstructs x exactly with normalization."""
        shift = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        scale = [2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        space = WaveletSpace(shift=shift, scale=scale)

        x = torch.randn(2, 1, 8, 16, 16)
        z = space.encode(x)
        reconstructed = space.decode(z)
        assert reconstructed.shape == x.shape
        torch.testing.assert_close(reconstructed, x)

    def test_wavelet_normalization_changes_values(self):
        """Normalization should actually change the encoded values."""
        x = torch.randn(2, 1, 8, 16, 16)

        space_raw = WaveletSpace()
        z_raw = space_raw.encode(x)

        shift = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        scale = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        space_norm = WaveletSpace(shift=shift, scale=scale)
        z_norm = space_norm.encode(x)

        # Channel 0 (LLL) should differ due to shift=1, scale=2
        assert not torch.allclose(z_raw[:, 0], z_norm[:, 0])
        # Other channels should match (shift=0, scale=1 = identity)
        torch.testing.assert_close(z_raw[:, 1:], z_norm[:, 1:])


class _TensorDataset(torch.utils.data.Dataset):
    """Dataset that returns raw tensors (not tuples like TensorDataset)."""
    def __init__(self, data: torch.Tensor):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]


class TestWaveletSubbandStats:
    """Test WaveletSpace.compute_subband_stats."""

    def test_compute_subband_stats_returns_8_values(self):
        """Stats should have 8 entries (one per subband)."""
        dataset = _TensorDataset(torch.randn(10, 1, 8, 16, 16))
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        stats = WaveletSpace.compute_subband_stats(loader, max_samples=10)

        assert 'wavelet_shift' in stats
        assert 'wavelet_scale' in stats
        assert len(stats['wavelet_shift']) == 8
        assert len(stats['wavelet_scale']) == 8

    def test_compute_subband_stats_scales_positive(self):
        """All scale values must be positive."""
        dataset = _TensorDataset(torch.randn(10, 1, 8, 16, 16))
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        stats = WaveletSpace.compute_subband_stats(loader, max_samples=10)

        for s in stats['wavelet_scale']:
            assert s > 0

    def test_compute_subband_stats_known_data(self):
        """Verify stats on constant data (all subbands known analytically)."""
        # Constant-value volumes: LLL gets all energy, detail subbands are 0
        x = torch.ones(20, 1, 8, 16, 16) * 3.0
        dataset = _TensorDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        stats = WaveletSpace.compute_subband_stats(loader, max_samples=20)

        # LLL shift should be 3.0 * 2*sqrt(2) (Haar scaling for constant input)
        expected_lll = 3.0 * 2 * (2 ** 0.5)
        assert abs(stats['wavelet_shift'][0] - expected_lll) < 0.01

        # Detail subbands should have shift ~0
        for i in range(1, 8):
            assert abs(stats['wavelet_shift'][i]) < 0.01

        # With constant data, std should be clamped to 1e-6 (no variance)
        for i in range(8):
            assert abs(stats['wavelet_scale'][i] - 1e-6) < 1e-7

    def test_compute_subband_stats_max_samples(self):
        """max_samples caps the number of processed samples."""
        dataset = _TensorDataset(torch.randn(100, 1, 8, 16, 16))
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        stats = WaveletSpace.compute_subband_stats(loader, max_samples=5)
        assert len(stats['wavelet_shift']) == 8

    def test_compute_subband_stats_dict_batch(self):
        """Stats computation works with dict-format batches."""
        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10
            def __getitem__(self, idx):
                return {'image': torch.randn(1, 8, 16, 16)}

        loader = torch.utils.data.DataLoader(DictDataset(), batch_size=4)
        stats = WaveletSpace.compute_subband_stats(loader, max_samples=10)

        assert len(stats['wavelet_shift']) == 8
        assert len(stats['wavelet_scale']) == 8

    def test_compute_subband_stats_rescale(self):
        """Stats with rescale=True should differ from rescale=False."""
        x = torch.rand(20, 1, 8, 16, 16)  # [0,1] data
        dataset = _TensorDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        stats_no = WaveletSpace.compute_subband_stats(loader, max_samples=20, rescale=False)
        stats_yes = WaveletSpace.compute_subband_stats(loader, max_samples=20, rescale=True)

        # LLL shift should differ: rescale maps [0,1] -> [-1,1]
        assert abs(stats_no['wavelet_shift'][0] - stats_yes['wavelet_shift'][0]) > 0.1


class TestRescalePixelSpace:
    """Test PixelSpace rescaling."""

    def test_pixel_space_no_rescale_backward_compat(self):
        """PixelSpace() with no args works identically to before."""
        space = PixelSpace()
        assert not space.needs_decode
        assert space.scale_factor == 1

        x = torch.rand(2, 1, 64, 64)
        assert torch.equal(space.encode(x), x)
        assert torch.equal(space.decode(x), x)

    def test_pixel_space_rescale_roundtrip(self):
        """decode(encode(x)) recovers original [0,1] data."""
        space = PixelSpace(rescale=True)
        x = torch.rand(2, 1, 64, 64)

        z = space.encode(x)
        reconstructed = space.decode(z)
        torch.testing.assert_close(reconstructed, x)

    def test_pixel_space_rescale_output_range(self):
        """encode() maps [0,1] to [-1,1]."""
        space = PixelSpace(rescale=True)
        x = torch.rand(2, 1, 64, 64)  # [0, 1]

        z = space.encode(x)
        assert z.min() >= -1.0
        assert z.max() <= 1.0

        # Boundaries
        zeros = space.encode(torch.zeros(1, 1, 4, 4))
        ones = space.encode(torch.ones(1, 1, 4, 4))
        torch.testing.assert_close(zeros, torch.full_like(zeros, -1.0))
        torch.testing.assert_close(ones, torch.ones_like(ones))

    def test_pixel_space_rescale_needs_decode(self):
        """needs_decode is True when rescale is enabled."""
        assert not PixelSpace().needs_decode
        assert not PixelSpace(rescale=False).needs_decode
        assert PixelSpace(rescale=True).needs_decode

    def test_pixel_space_rescale_encode_batch(self):
        """encode_batch/decode_batch work with rescale (inherited from base)."""
        space = PixelSpace(rescale=True)
        batch = {
            'image': torch.rand(2, 1, 32, 32),
            'seg': torch.rand(2, 1, 32, 32),
        }

        encoded = space.encode_batch(batch)
        assert encoded['image'].min() >= -1.0
        assert encoded['image'].max() <= 1.0

        decoded = space.decode_batch(encoded)
        torch.testing.assert_close(decoded['image'], batch['image'])
        torch.testing.assert_close(decoded['seg'], batch['seg'])


class TestRescaleWaveletSpace:
    """Test WaveletSpace rescaling."""

    def test_wavelet_rescale_roundtrip(self):
        """decode(encode(x)) recovers original [0,1] data with rescale."""
        space = WaveletSpace(rescale=True)
        x = torch.rand(2, 1, 8, 16, 16)

        z = space.encode(x)
        reconstructed = space.decode(z)
        torch.testing.assert_close(reconstructed, x)

    def test_wavelet_rescale_with_normalization_roundtrip(self):
        """Roundtrip with both rescale and subband normalization."""
        shift = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        scale = [2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        space = WaveletSpace(shift=shift, scale=scale, rescale=True)

        x = torch.rand(2, 1, 8, 16, 16)
        z = space.encode(x)
        reconstructed = space.decode(z)
        torch.testing.assert_close(reconstructed, x)

    def test_wavelet_rescale_changes_coefficients(self):
        """Rescale should change wavelet coefficients vs no rescale."""
        x = torch.rand(2, 1, 8, 16, 16)

        space_no = WaveletSpace(rescale=False)
        space_yes = WaveletSpace(rescale=True)

        z_no = space_no.encode(x)
        z_yes = space_yes.encode(x)

        # Coefficients should differ
        assert not torch.allclose(z_no, z_yes)

    def test_wavelet_rescale_needs_decode(self):
        """needs_decode is always True for WaveletSpace (scale_factor=2)."""
        # WaveletSpace always has scale_factor=2, so needs_decode is True
        # regardless of rescale
        assert WaveletSpace().needs_decode
        assert WaveletSpace(rescale=True).needs_decode
        assert WaveletSpace(rescale=False).needs_decode

    def test_wavelet_no_rescale_backward_compat(self):
        """WaveletSpace() without rescale works identically to before."""
        space = WaveletSpace()
        # Default rescale is False
        assert not space._rescale

        x = torch.randn(2, 1, 8, 16, 16)
        z = space.encode(x)
        reconstructed = space.decode(z)
        torch.testing.assert_close(reconstructed, x)


class TestNeedsDecode:
    """Test needs_decode property across all space types."""

    def test_pixel_space_default(self):
        assert not PixelSpace().needs_decode

    def test_pixel_space_rescale(self):
        assert PixelSpace(rescale=True).needs_decode

    def test_wavelet_space(self):
        assert WaveletSpace().needs_decode

    def test_space_to_depth(self):
        assert SpaceToDepthSpace().needs_decode
