"""Tests for SpaceToDepthSpace diffusion space."""
import pytest
import torch

from medgen.diffusion.spaces import SpaceToDepthSpace


class TestSpaceToDepthSpace:
    """Test SpaceToDepthSpace encode/decode and properties."""

    def test_encode_decode_roundtrip(self):
        """Encode then decode should be lossless (exact roundtrip)."""
        space = SpaceToDepthSpace(spatial_factor=2, depth_factor=2)
        x = torch.randn(2, 1, 160, 256, 256)

        encoded = space.encode(x)
        assert encoded.shape == (2, 8, 80, 128, 128)

        decoded = space.decode(encoded)
        assert decoded.shape == x.shape
        torch.testing.assert_close(decoded, x)

    def test_bravo_mode_shapes(self):
        """In bravo mode: image(8ch) + seg(8ch) = 16ch input to UNet."""
        space = SpaceToDepthSpace(spatial_factor=2, depth_factor=2)

        image = torch.randn(1, 1, 160, 256, 256)
        seg = torch.randn(1, 1, 160, 256, 256)

        encoded_image = space.encode(image)
        encoded_seg = space.encode(seg)

        assert encoded_image.shape == (1, 8, 80, 128, 128)
        assert encoded_seg.shape == (1, 8, 80, 128, 128)

        # Concatenate along channel dim (what mode.format_model_input does)
        model_input = torch.cat([encoded_image, encoded_seg], dim=1)
        assert model_input.shape == (1, 16, 80, 128, 128)

    def test_get_latent_channels(self):
        """get_latent_channels multiplies by channel_multiplier."""
        space = SpaceToDepthSpace(spatial_factor=2, depth_factor=2)
        assert space.get_latent_channels(1) == 8
        assert space.get_latent_channels(2) == 16
        assert space.get_latent_channels(3) == 24

    def test_scale_factor_properties(self):
        """Check scale_factor, depth_scale_factor, and latent_channels."""
        space = SpaceToDepthSpace(spatial_factor=2, depth_factor=2)
        assert space.scale_factor == 2
        assert space.depth_scale_factor == 2
        assert space.latent_channels == 8

    def test_encode_rejects_4d(self):
        """4D input (2D images) should raise ValueError."""
        space = SpaceToDepthSpace(spatial_factor=2, depth_factor=2)
        x = torch.randn(2, 1, 256, 256)

        with pytest.raises(ValueError, match="5D"):
            space.encode(x)

    def test_decode_rejects_4d(self):
        """4D input should raise ValueError on decode too."""
        space = SpaceToDepthSpace(spatial_factor=2, depth_factor=2)
        z = torch.randn(2, 8, 128, 128)

        with pytest.raises(ValueError, match="5D"):
            space.decode(z)

    def test_divisibility_check(self):
        """PixelUnshuffle3D raises ValueError if dims not divisible."""
        space = SpaceToDepthSpace(spatial_factor=2, depth_factor=2)
        # D=151 is not divisible by 2
        x = torch.randn(1, 1, 151, 256, 256)

        with pytest.raises(ValueError):
            space.encode(x)

    def test_spatial_only(self):
        """depth_factor=1 gives spatial-only rearrangement."""
        space = SpaceToDepthSpace(spatial_factor=2, depth_factor=1)

        assert space.latent_channels == 4  # 2*2*1
        assert space.scale_factor == 2
        assert space.depth_scale_factor == 1

        x = torch.randn(2, 1, 160, 256, 256)
        encoded = space.encode(x)
        # Depth unchanged, spatial halved, channels 4x
        assert encoded.shape == (2, 4, 160, 128, 128)

        decoded = space.decode(encoded)
        torch.testing.assert_close(decoded, x)

    def test_encode_batch_dict(self):
        """encode_batch handles dict of tensors (inherited from DiffusionSpace)."""
        space = SpaceToDepthSpace(spatial_factor=2, depth_factor=2)
        batch = {
            'image': torch.randn(1, 1, 8, 16, 16),
            'seg': torch.randn(1, 1, 8, 16, 16),
        }

        encoded = space.encode_batch(batch)
        assert isinstance(encoded, dict)
        assert encoded['image'].shape == (1, 8, 4, 8, 8)
        assert encoded['seg'].shape == (1, 8, 4, 8, 8)

    def test_decode_batch_dict(self):
        """decode_batch handles dict of tensors (inherited from DiffusionSpace)."""
        space = SpaceToDepthSpace(spatial_factor=2, depth_factor=2)
        batch = {
            'image': torch.randn(1, 8, 4, 8, 8),
            'seg': torch.randn(1, 8, 4, 8, 8),
        }

        decoded = space.decode_batch(batch)
        assert isinstance(decoded, dict)
        assert decoded['image'].shape == (1, 1, 8, 16, 16)
        assert decoded['seg'].shape == (1, 1, 8, 16, 16)
