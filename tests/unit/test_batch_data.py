"""Tests for BatchData standardization."""
import pytest
import torch
from medgen.diffusion.batch_data import BatchData


class TestBatchDataFromRaw:
    """Test BatchData.from_raw() handles all formats."""

    def test_tensor_input(self):
        """Plain tensor becomes images."""
        data = torch.randn(4, 1, 64, 64)
        batch = BatchData.from_raw(data)
        assert batch.images.shape == (4, 1, 64, 64)
        assert batch.labels is None
        assert batch.size_bins is None

    def test_2tuple_with_labels(self):
        """(images, labels) - labels is multi-dimensional."""
        images = torch.randn(4, 1, 64, 64)
        labels = torch.randn(4, 1, 64, 64)
        batch = BatchData.from_raw((images, labels))
        assert batch.images.shape == (4, 1, 64, 64)
        assert batch.labels.shape == (4, 1, 64, 64)
        assert batch.size_bins is None

    def test_2tuple_with_size_bins(self):
        """(seg, size_bins) - size_bins is 1D."""
        seg = torch.randn(4, 1, 64, 64)
        size_bins = torch.tensor([1, 0, 2, 1])
        batch = BatchData.from_raw((seg, size_bins))
        assert batch.images.shape == (4, 1, 64, 64)
        assert batch.size_bins.shape == (4,)
        assert batch.labels is None

    def test_3tuple_seg_conditioned_input(self):
        """(seg, size_bins, bin_maps) - seg_conditioned_input mode."""
        seg = torch.randn(4, 1, 64, 64)
        size_bins = torch.tensor([1, 0, 2, 1])
        bin_maps = torch.randn(4, 5, 64, 64)
        batch = BatchData.from_raw((seg, size_bins, bin_maps))
        assert batch.images.shape == (4, 1, 64, 64)
        assert batch.size_bins.shape == (4,)
        assert batch.bin_maps.shape == (4, 5, 64, 64)

    def test_3tuple_multi_modality(self):
        """(image, seg, mode_id) - multi_modality mode."""
        image = torch.randn(4, 1, 64, 64)
        seg = torch.randn(4, 1, 64, 64)
        mode_id = torch.tensor([0, 1, 0, 2])
        batch = BatchData.from_raw((image, seg, mode_id))
        assert batch.images.shape == (4, 1, 64, 64)
        assert batch.labels.shape == (4, 1, 64, 64)
        assert batch.mode_id.shape == (4,)

    def test_dict_input(self):
        """Dict format with explicit keys."""
        data = {
            'images': torch.randn(4, 1, 64, 64),
            'labels': torch.randn(4, 1, 64, 64),
            'size_bins': torch.tensor([1, 0, 2, 1]),
        }
        batch = BatchData.from_raw(data)
        assert batch.images.shape == (4, 1, 64, 64)
        assert batch.labels.shape == (4, 1, 64, 64)
        assert batch.size_bins.shape == (4,)

    def test_dict_with_latent_keys(self):
        """Dict format with latent keys (used in latent diffusion)."""
        data = {
            'latent': torch.randn(4, 4, 16, 16),
            'latent_seg': torch.randn(4, 1, 16, 16),
        }
        batch = BatchData.from_raw(data)
        assert batch.images.shape == (4, 4, 16, 16)
        assert batch.labels.shape == (4, 1, 16, 16)

    def test_invalid_tuple_length(self):
        """Tuples with wrong length raise ValueError."""
        with pytest.raises(ValueError, match="Unexpected tuple length"):
            BatchData.from_raw((1, 2, 3, 4))

    def test_invalid_type(self):
        """Unknown types raise ValueError."""
        with pytest.raises(ValueError, match="Unknown batch format"):
            BatchData.from_raw("invalid")

    def test_list_works_like_tuple(self):
        """Lists are handled the same as tuples."""
        images = torch.randn(4, 1, 64, 64)
        labels = torch.randn(4, 1, 64, 64)
        batch = BatchData.from_raw([images, labels])
        assert batch.images.shape == (4, 1, 64, 64)
        assert batch.labels.shape == (4, 1, 64, 64)


class TestBatchDataToDevice:
    """Test BatchData.to_device() moves all tensors."""

    def test_to_device_cpu(self):
        """Moving to CPU works."""
        batch = BatchData(
            images=torch.randn(2, 1, 32, 32),
            labels=torch.randn(2, 1, 32, 32),
            size_bins=torch.tensor([0, 1]),
        )
        moved = batch.to_device(torch.device('cpu'))
        assert moved.images.device.type == 'cpu'
        assert moved.labels.device.type == 'cpu'
        assert moved.size_bins.device.type == 'cpu'

    def test_to_device_handles_none(self):
        """None fields are preserved as None."""
        batch = BatchData(images=torch.randn(2, 1, 32, 32))
        moved = batch.to_device(torch.device('cpu'))
        assert moved.images is not None
        assert moved.labels is None
        assert moved.size_bins is None
        assert moved.bin_maps is None
        assert moved.mode_id is None


class TestBatchData3D:
    """Test BatchData works with 3D volumes."""

    def test_3d_tensor(self):
        """3D volume tensor."""
        data = torch.randn(2, 1, 32, 64, 64)  # [B, C, D, H, W]
        batch = BatchData.from_raw(data)
        assert batch.images.shape == (2, 1, 32, 64, 64)

    def test_3d_tuple_with_labels(self):
        """3D (volume, seg) tuple."""
        volume = torch.randn(2, 1, 32, 64, 64)
        seg = torch.randn(2, 1, 32, 64, 64)
        batch = BatchData.from_raw((volume, seg))
        assert batch.images.shape == (2, 1, 32, 64, 64)
        assert batch.labels.shape == (2, 1, 32, 64, 64)
