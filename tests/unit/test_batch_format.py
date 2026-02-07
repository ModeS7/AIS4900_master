"""Tests for unified batch format.

Tests that all datasets and mode prepare_batch methods use dict format consistently.
"""
import pytest
import torch
from torch.utils.data import DataLoader

from medgen.data.loaders.base import dict_collate_fn, DictDatasetWrapper


class TestDictCollate:
    """Test dict_collate_fn handles all cases correctly."""

    def test_stacks_tensors(self):
        """Tensors should be stacked along batch dimension."""
        batch = [
            {'image': torch.randn(1, 64, 64), 'seg': torch.randn(1, 64, 64)},
            {'image': torch.randn(1, 64, 64), 'seg': torch.randn(1, 64, 64)},
        ]
        result = dict_collate_fn(batch)
        assert result['image'].shape == (2, 1, 64, 64)
        assert result['seg'].shape == (2, 1, 64, 64)

    def test_handles_none_values(self):
        """None values should be kept as None."""
        batch = [
            {'image': torch.randn(1, 64, 64), 'seg': None},
            {'image': torch.randn(1, 64, 64), 'seg': None},
        ]
        result = dict_collate_fn(batch)
        assert result['image'].shape == (2, 1, 64, 64)
        assert result['seg'] is None

    def test_handles_optional_keys(self):
        """Optional keys like size_bins should stack correctly."""
        batch = [
            {'image': torch.randn(1, 64, 64), 'size_bins': torch.tensor([1, 2, 3])},
            {'image': torch.randn(1, 64, 64), 'size_bins': torch.tensor([4, 5, 6])},
        ]
        result = dict_collate_fn(batch)
        assert result['size_bins'].shape == (2, 3)

    def test_handles_mode_id(self):
        """mode_id should be stacked correctly."""
        batch = [
            {'image': torch.randn(1, 64, 64), 'seg': torch.randn(1, 64, 64), 'mode_id': torch.tensor(0)},
            {'image': torch.randn(1, 64, 64), 'seg': torch.randn(1, 64, 64), 'mode_id': torch.tensor(1)},
        ]
        result = dict_collate_fn(batch)
        assert result['mode_id'].shape == (2,)


    def test_warns_on_missing_key(self, caplog):
        """Should warn when a key is present in some but not all samples."""
        import logging
        batch = [
            {'image': torch.randn(1, 64, 64), 'seg': torch.randn(1, 64, 64)},
            {'image': torch.randn(1, 64, 64)},  # Missing 'seg'
        ]
        with caplog.at_level(logging.WARNING, logger="medgen.data.loaders.base"):
            result = dict_collate_fn(batch)

        assert result['image'].shape == (2, 1, 64, 64)
        assert result['seg'].shape == (1, 1, 64, 64)  # Only 1 had seg
        assert any("key 'seg' present in 1/2 samples" in msg for msg in caplog.messages)


class TestDictDatasetWrapperPassthrough:
    """Test DictDatasetWrapper passes through dict format unchanged."""

    def test_dict_passthrough(self):
        """Dict items should pass through unchanged."""
        from monai.data import Dataset as MonaiDataset

        # Dataset that returns dicts
        data = [
            {'image': torch.randn(1, 64, 64), 'seg': torch.randn(1, 64, 64), 'mode_id': torch.tensor(i)}
            for i in range(3)
        ]
        raw_ds = MonaiDataset(data)

        wrapped = DictDatasetWrapper(raw_ds, output_format='multi', spatial_dims=2)

        item = wrapped[0]
        assert isinstance(item, dict)
        assert 'image' in item
        assert 'seg' in item
        assert 'mode_id' in item

    def test_dict_validates_image_key(self):
        """Dict without 'image' key should raise ValueError."""
        from monai.data import Dataset as MonaiDataset

        # Dataset that returns dict without 'image' key
        data = [{'seg': torch.randn(1, 64, 64)} for _ in range(3)]
        raw_ds = MonaiDataset(data)

        wrapped = DictDatasetWrapper(raw_ds, output_format='seg', spatial_dims=2)

        with pytest.raises(ValueError, match="must have 'image' key"):
            wrapped[0]


class TestPrepareBatchSimplified:
    """Test that prepare_batch methods work with dict format."""

    def test_multi_modality_accepts_dict(self):
        """MultiModalityMode.prepare_batch should accept dict format."""
        from medgen.diffusion.modes import MultiModalityMode

        mode = MultiModalityMode(image_keys=['bravo', 'flair', 't1_pre', 't1_gd'])
        batch = {
            'image': torch.randn(4, 1, 64, 64),
            'seg': torch.randn(4, 1, 64, 64),
            'mode_id': torch.tensor([0, 1, 2, 3]),
        }
        device = torch.device('cpu')

        result = mode.prepare_batch(batch, device)

        assert 'images' in result
        assert 'labels' in result
        assert 'mode_id' in result
        assert result['images'].shape == (4, 1, 64, 64)
        assert result['labels'].shape == (4, 1, 64, 64)
        assert result['mode_id'].shape == (4,)

    def test_seg_conditioned_accepts_dict(self):
        """SegmentationConditionedMode.prepare_batch should accept dict format."""
        from medgen.diffusion.modes import SegmentationConditionedMode

        mode = SegmentationConditionedMode()
        batch = {
            'image': torch.randn(4, 1, 64, 64),
            'size_bins': torch.randint(0, 5, (4, 6)),
        }
        device = torch.device('cpu')

        result = mode.prepare_batch(batch, device)

        assert 'images' in result
        assert 'size_bins' in result
        assert result['labels'] is None
        assert result['images'].shape == (4, 1, 64, 64)
        assert result['size_bins'].shape == (4, 6)

    def test_seg_conditioned_input_accepts_dict(self):
        """SegmentationConditionedInputMode.prepare_batch should accept dict format."""
        from medgen.diffusion.modes import SegmentationConditionedInputMode

        mode = SegmentationConditionedInputMode()
        batch = {
            'image': torch.randn(4, 1, 64, 64),
            'size_bins': torch.randint(0, 5, (4, 7)),
            'bin_maps': torch.randn(4, 7, 64, 64),
        }
        device = torch.device('cpu')

        result = mode.prepare_batch(batch, device)

        assert 'images' in result
        assert 'size_bins' in result
        assert 'bin_maps' in result
        assert result['images'].shape == (4, 1, 64, 64)
        assert result['bin_maps'].shape == (4, 7, 64, 64)

    def test_seg_conditioned_input_without_bin_maps(self):
        """SegmentationConditionedInputMode should handle missing bin_maps."""
        from medgen.diffusion.modes import SegmentationConditionedInputMode

        mode = SegmentationConditionedInputMode()
        batch = {
            'image': torch.randn(4, 1, 64, 64),
            'size_bins': torch.randint(0, 5, (4, 7)),
        }
        device = torch.device('cpu')

        result = mode.prepare_batch(batch, device)

        assert result['bin_maps'] is None


class TestPrepareBatchDimValidation:
    """Test that prepare_batch validates tensor dimensionality."""

    # --- SegmentationMode ---

    def test_seg_mode_accepts_4d_dict(self):
        """SegmentationMode should accept 4D [B,C,H,W] dict batch."""
        from medgen.diffusion.modes import SegmentationMode

        mode = SegmentationMode()
        batch = {'image': torch.randn(2, 1, 64, 64)}
        result = mode.prepare_batch(batch, torch.device('cpu'))

        assert result['images'].shape == (2, 1, 64, 64)
        assert result['labels'] is None

    def test_seg_mode_accepts_5d_dict(self):
        """SegmentationMode should accept 5D [B,C,D,H,W] dict batch."""
        from medgen.diffusion.modes import SegmentationMode

        mode = SegmentationMode()
        batch = {'image': torch.randn(2, 1, 16, 64, 64)}
        result = mode.prepare_batch(batch, torch.device('cpu'))

        assert result['images'].shape == (2, 1, 16, 64, 64)

    def test_seg_mode_rejects_3d_dict(self):
        """SegmentationMode should reject 3D tensor in dict batch."""
        from medgen.diffusion.modes import SegmentationMode

        mode = SegmentationMode()
        batch = {'image': torch.randn(2, 64, 64)}

        with pytest.raises(ValueError, match="expected 4D or 5D"):
            mode.prepare_batch(batch, torch.device('cpu'))

    # --- ConditionalSingleMode ---

    def test_cond_single_accepts_4d_dict(self):
        """ConditionalSingleMode should accept 4D dict with seg."""
        from medgen.diffusion.modes import ConditionalSingleMode

        mode = ConditionalSingleMode()
        batch = {
            'image': torch.randn(2, 1, 64, 64),
            'seg': torch.randn(2, 1, 64, 64),
        }
        result = mode.prepare_batch(batch, torch.device('cpu'))

        assert result['images'].shape == (2, 1, 64, 64)
        assert result['labels'].shape == (2, 1, 64, 64)

    def test_cond_single_accepts_5d_dict(self):
        """ConditionalSingleMode should accept 5D dict with seg."""
        from medgen.diffusion.modes import ConditionalSingleMode

        mode = ConditionalSingleMode()
        batch = {
            'image': torch.randn(2, 1, 16, 64, 64),
            'seg': torch.randn(2, 1, 16, 64, 64),
        }
        result = mode.prepare_batch(batch, torch.device('cpu'))

        assert result['images'].shape == (2, 1, 16, 64, 64)
        assert result['labels'].shape == (2, 1, 16, 64, 64)

    def test_cond_single_rejects_bad_image_dim(self):
        """ConditionalSingleMode should reject 3D image tensor."""
        from medgen.diffusion.modes import ConditionalSingleMode

        mode = ConditionalSingleMode()
        batch = {'image': torch.randn(2, 64, 64)}

        with pytest.raises(ValueError, match="expected 4D or 5D image"):
            mode.prepare_batch(batch, torch.device('cpu'))

    def test_cond_single_rejects_bad_seg_dim(self):
        """ConditionalSingleMode should reject 3D seg tensor."""
        from medgen.diffusion.modes import ConditionalSingleMode

        mode = ConditionalSingleMode()
        batch = {
            'image': torch.randn(2, 1, 64, 64),
            'seg': torch.randn(2, 64, 64),
        }

        with pytest.raises(ValueError, match="expected 4D or 5D seg"):
            mode.prepare_batch(batch, torch.device('cpu'))

    # --- ConditionalDualMode ---

    def test_cond_dual_accepts_4d_dict(self):
        """ConditionalDualMode should accept 4D dict batch."""
        from medgen.diffusion.modes import ConditionalDualMode

        mode = ConditionalDualMode()
        batch = {
            't1_pre': torch.randn(2, 1, 64, 64),
            't1_gd': torch.randn(2, 1, 64, 64),
            'seg': torch.randn(2, 1, 64, 64),
        }
        result = mode.prepare_batch(batch, torch.device('cpu'))

        assert result['images']['t1_pre'].shape == (2, 1, 64, 64)
        assert result['images']['t1_gd'].shape == (2, 1, 64, 64)
        assert result['labels'].shape == (2, 1, 64, 64)

    def test_cond_dual_accepts_5d_dict(self):
        """ConditionalDualMode should accept 5D dict batch."""
        from medgen.diffusion.modes import ConditionalDualMode

        mode = ConditionalDualMode()
        batch = {
            't1_pre': torch.randn(2, 1, 16, 64, 64),
            't1_gd': torch.randn(2, 1, 16, 64, 64),
            'seg': torch.randn(2, 1, 16, 64, 64),
        }
        result = mode.prepare_batch(batch, torch.device('cpu'))

        assert result['images']['t1_pre'].shape == (2, 1, 16, 64, 64)
        assert result['labels'].shape == (2, 1, 16, 64, 64)

    def test_cond_dual_rejects_bad_image_dim(self):
        """ConditionalDualMode should reject 3D image tensor."""
        from medgen.diffusion.modes import ConditionalDualMode

        mode = ConditionalDualMode()
        batch = {
            't1_pre': torch.randn(2, 64, 64),  # 3D - bad
            't1_gd': torch.randn(2, 1, 64, 64),
        }

        with pytest.raises(ValueError, match="expected 4D or 5D tensor for 't1_pre'"):
            mode.prepare_batch(batch, torch.device('cpu'))

    def test_cond_dual_rejects_bad_seg_dim(self):
        """ConditionalDualMode should reject 3D seg tensor."""
        from medgen.diffusion.modes import ConditionalDualMode

        mode = ConditionalDualMode()
        batch = {
            't1_pre': torch.randn(2, 1, 64, 64),
            't1_gd': torch.randn(2, 1, 64, 64),
            'seg': torch.randn(2, 64, 64),  # 3D - bad
        }

        with pytest.raises(ValueError, match="expected 4D or 5D seg"):
            mode.prepare_batch(batch, torch.device('cpu'))


class TestBatchDataFromDict:
    """Test BatchData.from_raw() with new dict format."""

    def test_dict_with_image_key(self):
        """Dict with 'image' key should work."""
        from medgen.diffusion.batch_data import BatchData

        data = {
            'image': torch.randn(4, 1, 64, 64),
            'seg': torch.randn(4, 1, 64, 64),
        }
        batch = BatchData.from_raw(data)

        assert batch.images.shape == (4, 1, 64, 64)
        assert batch.labels.shape == (4, 1, 64, 64)

    def test_dict_with_images_key(self):
        """Dict with 'images' key should also work (backwards compat)."""
        from medgen.diffusion.batch_data import BatchData

        data = {
            'images': torch.randn(4, 1, 64, 64),
            'labels': torch.randn(4, 1, 64, 64),
        }
        batch = BatchData.from_raw(data)

        assert batch.images.shape == (4, 1, 64, 64)
        assert batch.labels.shape == (4, 1, 64, 64)

    def test_dict_with_mode_id(self):
        """Dict with mode_id should preserve it."""
        from medgen.diffusion.batch_data import BatchData

        data = {
            'image': torch.randn(4, 1, 64, 64),
            'seg': torch.randn(4, 1, 64, 64),
            'mode_id': torch.tensor([0, 1, 2, 3]),
        }
        batch = BatchData.from_raw(data)

        assert batch.mode_id.shape == (4,)

    def test_tuple_warns_deprecation(self):
        """Tuple format should emit deprecation warning."""
        from medgen.diffusion.batch_data import BatchData

        data = (torch.randn(4, 1, 64, 64), torch.randn(4, 1, 64, 64))

        with pytest.warns(DeprecationWarning, match="Tuple batch format is deprecated"):
            batch = BatchData.from_raw(data)

        assert batch.images.shape == (4, 1, 64, 64)


class TestDatasetDictFormat:
    """Test that actual dataset classes return dict format."""

    def test_multi_diffusion_dataset_returns_dict(self):
        """MultiDiffusionDataset should return dict."""
        import numpy as np
        from medgen.data.loaders.datasets import MultiDiffusionDataset

        # Create minimal samples
        samples = [
            (np.random.randn(1, 64, 64).astype(np.float32),
             np.random.randn(1, 64, 64).astype(np.float32),
             i % 4)
            for i in range(4)
        ]

        dataset = MultiDiffusionDataset(samples)
        item = dataset[0]

        assert isinstance(item, dict)
        assert 'image' in item
        assert 'seg' in item
        assert 'mode_id' in item
        assert isinstance(item['image'], torch.Tensor)
        assert isinstance(item['seg'], torch.Tensor)
        assert isinstance(item['mode_id'], torch.Tensor)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
