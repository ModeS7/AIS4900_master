"""Tests for lossless binary mask encoding/decoding.

Tests verify that encode/decode are perfect inverses and that
latent shapes match specifications for all supported formats.
"""

import pytest
import torch

from medgen.data.lossless_mask_codec import (
    encode_mask_lossless,
    decode_mask_lossless,
    get_latent_shape,
    FORMATS,
)


class TestRoundTripIdentity:
    """Tests that encode/decode are perfect inverses."""

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_single_mask_round_trip(self, format):
        """Single 256x256 mask encodes and decodes identically."""
        torch.manual_seed(42)
        mask = torch.randint(0, 2, (256, 256), dtype=torch.float32)
        latent = encode_mask_lossless(mask, format=format)
        decoded = decode_mask_lossless(latent, format=format)
        assert torch.equal(mask, decoded), f"Round-trip failed for {format}"

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_batch_round_trip(self, format):
        """Batch [B,1,256,256] encodes/decodes with ordering preserved."""
        torch.manual_seed(42)
        masks = torch.randint(0, 2, (4, 1, 256, 256), dtype=torch.float32)
        latent = encode_mask_lossless(masks, format=format)
        decoded = decode_mask_lossless(latent, format=format)
        assert torch.equal(masks, decoded), f"Batch round-trip failed for {format}"

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_random_pattern_round_trip(self, format):
        """Random sparse masks with various densities round-trip correctly."""
        for seed in range(5):
            torch.manual_seed(seed)
            # Create mask with varying density
            density = 0.1 + 0.2 * seed  # 10% to 90% density
            mask = (torch.rand(256, 256) < density).float()
            latent = encode_mask_lossless(mask, format=format)
            decoded = decode_mask_lossless(latent, format=format)
            assert torch.equal(mask, decoded), \
                f"Random pattern failed for {format} with density {density}"


class TestLatentShapes:
    """Tests that latent shapes match specifications."""

    def test_format_shapes_match_spec(self):
        """Each format produces documented latent shape."""
        expected = {
            'f32': (32, 8, 8),     # 32 channels, 8x8 spatial
            'f64': (128, 4, 4),   # 128 channels, 4x4 spatial
            'f128': (512, 2, 2),  # 512 channels, 2x2 spatial
            'k8x8': (2, 32, 32),  # 2 channels, 32x32 spatial
        }
        for format, shape in expected.items():
            assert get_latent_shape(format) == shape, \
                f"get_latent_shape({format}) returned wrong shape"

    def test_single_mask_output_shape(self):
        """Single mask [256,256] outputs [C,S,S]."""
        expected_shapes = {
            'f32': (32, 8, 8),
            'f64': (128, 4, 4),
            'f128': (512, 2, 2),
            'k8x8': (2, 32, 32),
        }
        mask = torch.zeros(256, 256)
        for format, shape in expected_shapes.items():
            latent = encode_mask_lossless(mask, format=format)
            assert latent.shape == shape, \
                f"Single mask latent shape wrong for {format}"

    def test_batch_shape_preserved(self):
        """Input [B,1,256,256] outputs [B,C,S,S]."""
        B = 4
        masks = torch.randint(0, 2, (B, 1, 256, 256), dtype=torch.float32)

        for format, (spatial, channels, _) in FORMATS.items():
            latent = encode_mask_lossless(masks, format=format)
            assert latent.shape[0] == B, f"Batch dim lost for {format}"
            assert latent.shape[1:] == (channels, spatial, spatial), \
                f"Latent shape wrong for {format}"


class TestEdgeCases:
    """Tests boundary conditions."""

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_all_zeros_mask(self, format):
        """All-black mask round-trips correctly."""
        mask = torch.zeros(256, 256)
        decoded = decode_mask_lossless(encode_mask_lossless(mask, format), format)
        assert torch.equal(mask, decoded), f"All-zeros failed for {format}"

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_all_ones_mask(self, format):
        """All-white mask round-trips correctly."""
        mask = torch.ones(256, 256)
        decoded = decode_mask_lossless(encode_mask_lossless(mask, format), format)
        assert torch.equal(mask, decoded), f"All-ones failed for {format}"

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_sparse_mask_single_pixel(self, format):
        """Mask with only 1 pixel set round-trips."""
        mask = torch.zeros(256, 256)
        mask[128, 128] = 1.0
        decoded = decode_mask_lossless(encode_mask_lossless(mask, format), format)
        assert torch.equal(mask, decoded), f"Single-pixel failed for {format}"

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_corner_pixels(self, format):
        """Corner pixels encode/decode correctly."""
        mask = torch.zeros(256, 256)
        # Set all four corners
        mask[0, 0] = 1.0
        mask[0, 255] = 1.0
        mask[255, 0] = 1.0
        mask[255, 255] = 1.0
        decoded = decode_mask_lossless(encode_mask_lossless(mask, format), format)
        assert torch.equal(mask, decoded), f"Corner pixels failed for {format}"

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_checkerboard_pattern(self, format):
        """Checkerboard pattern (maximum bit alternation) round-trips."""
        mask = torch.zeros(256, 256)
        mask[::2, ::2] = 1.0  # Even rows, even cols
        mask[1::2, 1::2] = 1.0  # Odd rows, odd cols
        decoded = decode_mask_lossless(encode_mask_lossless(mask, format), format)
        assert torch.equal(mask, decoded), f"Checkerboard failed for {format}"

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_diagonal_stripe_pattern(self, format):
        """Diagonal stripe pattern round-trips."""
        mask = torch.zeros(256, 256)
        for i in range(256):
            mask[i, i] = 1.0  # Main diagonal
            if i > 0:
                mask[i-1, i] = 1.0  # One pixel above
        decoded = decode_mask_lossless(encode_mask_lossless(mask, format), format)
        assert torch.equal(mask, decoded), f"Diagonal stripe failed for {format}"


class TestBatchOrdering:
    """Tests that batch ordering is preserved."""

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_batch_elements_not_swapped(self, format):
        """Individual batch elements maintain their identity."""
        torch.manual_seed(42)
        B = 4
        masks = []
        for i in range(B):
            # Create distinct masks for each batch element
            mask = torch.zeros(1, 256, 256)
            mask[0, i*50:(i+1)*50, i*50:(i+1)*50] = 1.0
            masks.append(mask)
        masks = torch.cat(masks, dim=0).unsqueeze(1)  # [B, 1, 256, 256]

        latent = encode_mask_lossless(masks, format=format)
        decoded = decode_mask_lossless(latent, format=format)

        # Check each element individually
        for i in range(B):
            assert torch.equal(masks[i], decoded[i]), \
                f"Batch element {i} corrupted for {format}"


class TestDeviceHandling:
    """Tests for device compatibility."""

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_cpu_tensor_works(self, format):
        """CPU tensors work correctly."""
        mask = torch.randint(0, 2, (256, 256), dtype=torch.float32, device='cpu')
        latent = encode_mask_lossless(mask, format=format)
        decoded = decode_mask_lossless(latent, format=format)
        assert latent.device == mask.device
        assert decoded.device == mask.device
        assert torch.equal(mask, decoded)

    @pytest.mark.gpu
    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_cuda_tensor_works(self, format):
        """CUDA tensors work correctly."""
        mask = torch.randint(0, 2, (256, 256), dtype=torch.float32, device='cuda')
        latent = encode_mask_lossless(mask, format=format)
        decoded = decode_mask_lossless(latent, format=format)
        assert latent.device == mask.device
        assert decoded.device == mask.device
        assert torch.equal(mask, decoded)


class TestErrorCases:
    """Tests error handling."""

    def test_wrong_input_shape_raises(self):
        """Non-256x256 mask raises assertion."""
        mask = torch.zeros(128, 128)
        with pytest.raises(AssertionError):
            encode_mask_lossless(mask, 'f32')

    def test_wrong_input_shape_3d_raises(self):
        """3D mask without batch dim raises assertion."""
        mask = torch.zeros(1, 256, 256)  # Missing batch dim for 4D path
        with pytest.raises(AssertionError):
            encode_mask_lossless(mask, 'f32')

    def test_invalid_format_raises(self):
        """Unknown format raises KeyError."""
        mask = torch.zeros(256, 256)
        with pytest.raises(KeyError):
            encode_mask_lossless(mask, 'invalid')

    def test_wrong_latent_shape_decode_raises(self):
        """Wrong latent shape raises assertion."""
        latent = torch.zeros(32, 4, 4)  # Wrong spatial for f32
        with pytest.raises(AssertionError):
            decode_mask_lossless(latent, 'f32')


class TestBinarization:
    """Tests that input is properly binarized."""

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_non_binary_input_binarized(self, format):
        """Non-binary values are thresholded at 0.5."""
        mask = torch.rand(256, 256)  # Values in [0, 1]
        latent = encode_mask_lossless(mask, format=format)
        decoded = decode_mask_lossless(latent, format=format)

        # Decoded should match binarized version of input
        expected = (mask > 0.5).float()
        assert torch.equal(expected, decoded), \
            f"Binarization mismatch for {format}"

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_threshold_exactly_half(self, format):
        """Values exactly at 0.5 are treated as 0."""
        mask = torch.full((256, 256), 0.5)
        decoded = decode_mask_lossless(encode_mask_lossless(mask, format), format)
        assert (decoded == 0).all(), f"Threshold boundary wrong for {format}"

    @pytest.mark.parametrize("format", ['f32', 'f64', 'f128', 'k8x8'])
    def test_slightly_above_threshold(self, format):
        """Values slightly above 0.5 are treated as 1."""
        mask = torch.full((256, 256), 0.500001)
        decoded = decode_mask_lossless(encode_mask_lossless(mask, format), format)
        assert (decoded == 1).all(), f"Threshold boundary wrong for {format}"
