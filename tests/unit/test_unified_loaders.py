"""Unit tests for unified loader factory.

Tests that:
1. 2D loaders return dict format
2. 3D loaders return dict format
3. Dict keys are consistent between 2D and 3D
4. DictDatasetWrapper converts correctly
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from medgen.data.loaders.base import (
    BaseDiffusionDataset,
    BaseDiffusionDataset2D,
    BaseDiffusionDataset3D,
    DictDatasetWrapper,
    dict_collate_fn,
    validate_batch_format,
)


class TestBaseDiffusionDataset:
    """Test base dataset classes."""

    def test_cannot_instantiate_abstract(self):
        """Base class is abstract."""
        with pytest.raises(TypeError):
            BaseDiffusionDataset()

    def test_2d_subclass_spatial_dims(self):
        """2D subclass returns spatial_dims=2."""
        class TestDataset(BaseDiffusionDataset2D):
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return {'image': torch.randn(1, 64, 64)}

        ds = TestDataset()
        assert ds.spatial_dims == 2

    def test_3d_subclass_spatial_dims(self):
        """3D subclass returns spatial_dims=3."""
        class TestDataset(BaseDiffusionDataset3D):
            def __len__(self):
                return 1
            def __getitem__(self, idx):
                return {'image': torch.randn(1, 32, 64, 64)}

        ds = TestDataset()
        assert ds.spatial_dims == 3


class TestDictDatasetWrapper:
    """Test DictDatasetWrapper conversion."""

    def test_seg_mode_tensor(self):
        """Seg mode: tensor -> dict with 'image' key."""
        raw_data = [torch.randn(1, 64, 64) for _ in range(3)]
        from monai.data import Dataset as MonaiDataset
        raw_ds = MonaiDataset(raw_data)

        wrapped = DictDatasetWrapper(raw_ds, output_format='seg', spatial_dims=2)

        item = wrapped[0]
        assert isinstance(item, dict)
        assert 'image' in item
        assert item['image'].shape == (1, 64, 64)

    def test_bravo_mode_stacked(self):
        """Bravo mode: [2, H, W] tensor -> {'image': [1, H, W], 'seg': [1, H, W]}."""
        raw_data = [torch.randn(2, 64, 64) for _ in range(3)]
        from monai.data import Dataset as MonaiDataset
        raw_ds = MonaiDataset(raw_data)

        wrapped = DictDatasetWrapper(raw_ds, output_format='bravo', spatial_dims=2)

        item = wrapped[0]
        assert isinstance(item, dict)
        assert 'image' in item
        assert 'seg' in item
        assert item['image'].shape == (1, 64, 64)
        assert item['seg'].shape == (1, 64, 64)

    def test_dual_mode_stacked(self):
        """Dual mode: [3, H, W] tensor -> {'image': [2, H, W], 'seg': [1, H, W]}."""
        raw_data = [torch.randn(3, 64, 64) for _ in range(3)]
        from monai.data import Dataset as MonaiDataset
        raw_ds = MonaiDataset(raw_data)

        wrapped = DictDatasetWrapper(raw_ds, output_format='dual', spatial_dims=2)

        item = wrapped[0]
        assert isinstance(item, dict)
        assert 'image' in item
        assert 'seg' in item
        assert item['image'].shape == (2, 64, 64)
        assert item['seg'].shape == (1, 64, 64)

    def test_multi_mode_tuple(self):
        """Multi mode: (image, seg, mode_id) tuple -> dict."""
        raw_data = [
            (torch.randn(1, 64, 64), torch.ones(1, 64, 64), i % 4)
            for i in range(3)
        ]
        from monai.data import Dataset as MonaiDataset
        raw_ds = MonaiDataset(raw_data)

        wrapped = DictDatasetWrapper(raw_ds, output_format='multi', spatial_dims=2)

        item = wrapped[0]
        assert isinstance(item, dict)
        assert 'image' in item
        assert 'seg' in item
        assert 'mode_id' in item
        assert isinstance(item['mode_id'], torch.Tensor)

    def test_seg_conditioned_mode_tuple(self):
        """Seg_conditioned mode: (seg, size_bins) tuple -> dict."""
        raw_data = [
            (torch.randn(1, 64, 64), torch.zeros(9))
            for _ in range(3)
        ]
        from monai.data import Dataset as MonaiDataset
        raw_ds = MonaiDataset(raw_data)

        wrapped = DictDatasetWrapper(raw_ds, output_format='seg_conditioned', spatial_dims=2)

        item = wrapped[0]
        assert isinstance(item, dict)
        assert 'image' in item
        assert 'size_bins' in item

    def test_numpy_conversion(self):
        """Numpy arrays should be converted to tensors."""
        raw_data = [np.random.randn(1, 64, 64).astype(np.float32) for _ in range(3)]
        from monai.data import Dataset as MonaiDataset
        raw_ds = MonaiDataset(raw_data)

        wrapped = DictDatasetWrapper(raw_ds, output_format='seg', spatial_dims=2)

        item = wrapped[0]
        assert isinstance(item['image'], torch.Tensor)
        assert item['image'].dtype == torch.float32


class TestDictCollateFn:
    """Test dict collate function."""

    def test_collate_tensors(self):
        """Tensors should be stacked along batch dimension."""
        batch = [
            {'image': torch.randn(1, 64, 64), 'seg': torch.randn(1, 64, 64)},
            {'image': torch.randn(1, 64, 64), 'seg': torch.randn(1, 64, 64)},
        ]

        collated = dict_collate_fn(batch)

        assert collated['image'].shape == (2, 1, 64, 64)
        assert collated['seg'].shape == (2, 1, 64, 64)

    def test_collate_with_none(self):
        """None values should be handled gracefully."""
        batch = [
            {'image': torch.randn(1, 64, 64), 'seg': None},
            {'image': torch.randn(1, 64, 64), 'seg': None},
        ]

        collated = dict_collate_fn(batch)

        assert collated['image'].shape == (2, 1, 64, 64)
        assert collated['seg'] is None

    def test_collate_non_tensors(self):
        """Non-tensor values should be collected as list."""
        batch = [
            {'image': torch.randn(1, 64, 64), 'patient': 'patient_0'},
            {'image': torch.randn(1, 64, 64), 'patient': 'patient_1'},
        ]

        collated = dict_collate_fn(batch)

        assert collated['image'].shape == (2, 1, 64, 64)
        assert collated['patient'] == ['patient_0', 'patient_1']

    def test_collate_empty_batch(self):
        """Empty batch should return empty dict."""
        assert dict_collate_fn([]) == {}


class TestValidateBatchFormat:
    """Test batch format validation."""

    def test_dict_passthrough(self):
        """Dict batches should pass through unchanged."""
        batch = {'image': torch.randn(4, 1, 64, 64)}
        result = validate_batch_format(batch)
        assert result is batch

    def test_tuple_conversion(self):
        """Tuple batches should be converted to dict."""
        batch = (torch.randn(4, 1, 64, 64), torch.randn(4, 1, 64, 64))

        with pytest.warns(DeprecationWarning):
            result = validate_batch_format(batch)

        assert isinstance(result, dict)
        assert 'image' in result
        assert 'seg' in result

    def test_tensor_conversion(self):
        """Tensor batches should be converted to dict."""
        batch = torch.randn(4, 1, 64, 64)

        with pytest.warns(DeprecationWarning):
            result = validate_batch_format(batch)

        assert isinstance(result, dict)
        assert 'image' in result

    def test_invalid_format(self):
        """Unknown format should raise ValueError."""
        with pytest.raises(ValueError):
            validate_batch_format("invalid")


class TestOutputFormatMapping:
    """Test compression mode to output format mapping."""

    def test_seg_compression_format(self):
        from medgen.data.loaders.unified import _get_compression_output_format
        assert _get_compression_output_format('seg_compression') == 'compression_seg'

    def test_single_modality_formats(self):
        from medgen.data.loaders.unified import _get_compression_output_format
        for mode in ['bravo', 't1_pre', 't1_gd', 'flair', 'seg']:
            assert _get_compression_output_format(mode) == 'compression_single'

    def test_dual_format(self):
        from medgen.data.loaders.unified import _get_compression_output_format
        assert _get_compression_output_format('dual') == 'compression_dual'

    def test_multi_modality_format(self):
        from medgen.data.loaders.unified import _get_compression_output_format
        assert _get_compression_output_format('multi_modality') == 'compression_multi'


class TestDictDatasetWrapperCompression:
    """Test DictDatasetWrapper with compression output formats."""

    def _make_mock_dataset(self, item):
        """Create mock dataset returning given item."""
        mock = Mock()
        mock.__len__ = Mock(return_value=10)
        mock.__getitem__ = Mock(return_value=item)
        return mock

    def test_compression_seg_tensor_input(self):
        """compression_seg: tensor -> {'image': tensor}"""
        dataset = self._make_mock_dataset(torch.randn(1, 128, 128))
        wrapper = DictDatasetWrapper(dataset, output_format='compression_seg', spatial_dims=2)
        batch = wrapper[0]

        assert isinstance(batch, dict)
        assert 'image' in batch
        assert batch['image'].shape == (1, 128, 128)
        assert 'seg' not in batch or batch.get('seg') is None

    def test_compression_single_tensor_input(self):
        """compression_single: tensor -> {'image': tensor}"""
        dataset = self._make_mock_dataset(torch.randn(1, 128, 128))
        wrapper = DictDatasetWrapper(dataset, output_format='compression_single', spatial_dims=2)
        batch = wrapper[0]

        assert isinstance(batch, dict)
        assert 'image' in batch
        assert batch['image'].shape == (1, 128, 128)

    def test_compression_single_tuple_input(self):
        """compression_single: (image, seg) tuple -> {'image': ..., 'seg': ...}"""
        dataset = self._make_mock_dataset((
            torch.randn(1, 128, 128),
            torch.randn(1, 128, 128),
        ))
        wrapper = DictDatasetWrapper(dataset, output_format='compression_single', spatial_dims=2)
        batch = wrapper[0]

        assert isinstance(batch, dict)
        assert 'image' in batch
        assert 'seg' in batch
        assert batch['image'].shape == (1, 128, 128)
        assert batch['seg'].shape == (1, 128, 128)

    def test_compression_dual_tensor_input(self):
        """compression_dual: 2-channel tensor -> {'image': tensor}"""
        dataset = self._make_mock_dataset(torch.randn(2, 128, 128))
        wrapper = DictDatasetWrapper(dataset, output_format='compression_dual', spatial_dims=2)
        batch = wrapper[0]

        assert isinstance(batch, dict)
        assert 'image' in batch
        assert batch['image'].shape == (2, 128, 128)

    def test_compression_dual_tuple_input(self):
        """compression_dual: (dual, seg) tuple -> {'image': dual, 'seg': seg}"""
        dataset = self._make_mock_dataset((
            torch.randn(2, 128, 128),
            torch.randn(1, 128, 128),
        ))
        wrapper = DictDatasetWrapper(dataset, output_format='compression_dual', spatial_dims=2)
        batch = wrapper[0]

        assert isinstance(batch, dict)
        assert batch['image'].shape == (2, 128, 128)
        assert batch['seg'].shape == (1, 128, 128)

    def test_compression_multi_numpy_input(self):
        """compression_multi: numpy array -> {'image': tensor}"""
        dataset = self._make_mock_dataset(np.random.randn(1, 128, 128).astype(np.float32))
        wrapper = DictDatasetWrapper(dataset, output_format='compression_multi', spatial_dims=2)
        batch = wrapper[0]

        assert isinstance(batch, dict)
        assert 'image' in batch
        assert isinstance(batch['image'], torch.Tensor)


class TestCreateDataloaderRouting:
    """Test unified create_dataloader routing."""

    @patch('medgen.data.loaders.unified.create_diffusion_dataloader')
    def test_routes_to_diffusion(self, mock_diffusion):
        """task='diffusion' routes to diffusion loader."""
        from omegaconf import OmegaConf
        from medgen.data.loaders.unified import create_dataloader

        mock_diffusion.return_value = (Mock(), Mock())
        cfg = OmegaConf.create({'training': {'batch_size': 16, 'augment': False}})

        create_dataloader(cfg, task='diffusion', mode='bravo', spatial_dims=2)
        mock_diffusion.assert_called_once()

    @patch('medgen.data.loaders.unified._create_compression_dataloader')
    def test_routes_to_compression(self, mock_compression):
        """task='compression' routes to compression loader."""
        from omegaconf import OmegaConf
        from medgen.data.loaders.unified import create_dataloader

        mock_compression.return_value = (Mock(), Mock())
        cfg = OmegaConf.create({'training': {'batch_size': 16, 'augment': False}})

        create_dataloader(cfg, task='compression', mode='bravo', spatial_dims=2)
        mock_compression.assert_called_once()

    def test_invalid_task_raises(self):
        """Invalid task raises ValueError."""
        from omegaconf import OmegaConf
        from medgen.data.loaders.unified import create_dataloader

        cfg = OmegaConf.create({'training': {'batch_size': 16}})

        with pytest.raises(ValueError, match="Unknown task"):
            create_dataloader(cfg, task='invalid', mode='bravo', spatial_dims=2)

    def test_invalid_spatial_dims_raises(self):
        """Invalid spatial_dims raises ValueError."""
        from omegaconf import OmegaConf
        from medgen.data.loaders.unified import create_dataloader

        cfg = OmegaConf.create({'training': {'batch_size': 16}})

        with pytest.raises(ValueError, match="spatial_dims"):
            create_dataloader(cfg, task='diffusion', mode='bravo', spatial_dims=4)


class TestBatchFormatConsistency:
    """Ensure all loaders return consistent dict format."""

    def _validate_batch(self, batch: dict):
        """Validate batch has required format."""
        assert isinstance(batch, dict), f"Expected dict, got {type(batch)}"
        assert 'image' in batch, "Missing 'image' key"
        assert isinstance(batch['image'], torch.Tensor), "image must be tensor"

        if 'seg' in batch and batch['seg'] is not None:
            assert isinstance(batch['seg'], torch.Tensor), "seg must be tensor"

        if 'mode_id' in batch and batch['mode_id'] is not None:
            assert isinstance(batch['mode_id'], torch.Tensor), "mode_id must be tensor"

    def test_diffusion_modes_return_dict(self):
        """All diffusion output formats return valid dict."""
        from monai.data import Dataset as MonaiDataset

        for mode in ['seg', 'bravo', 'dual', 'multi', 'seg_conditioned']:
            # Mock appropriate return values for each mode
            if mode == 'seg':
                data = [torch.randn(1, 128, 128) for _ in range(3)]
            elif mode == 'bravo':
                data = [torch.randn(2, 128, 128) for _ in range(3)]
            elif mode == 'dual':
                data = [torch.randn(3, 128, 128) for _ in range(3)]
            elif mode == 'multi':
                data = [(torch.randn(1, 128, 128), torch.randn(1, 128, 128), i) for i in range(3)]
            else:  # seg_conditioned
                data = [(torch.randn(1, 128, 128), torch.zeros(9)) for _ in range(3)]

            dataset = MonaiDataset(data)
            wrapper = DictDatasetWrapper(dataset, output_format=mode, spatial_dims=2)
            batch = wrapper[0]
            self._validate_batch(batch)

    def test_compression_modes_return_dict(self):
        """All compression output formats return valid dict."""
        from monai.data import Dataset as MonaiDataset

        for mode in ['compression_seg', 'compression_single', 'compression_dual', 'compression_multi']:
            if mode == 'compression_dual':
                data = [torch.randn(2, 128, 128) for _ in range(3)]
            else:
                data = [torch.randn(1, 128, 128) for _ in range(3)]

            dataset = MonaiDataset(data)
            wrapper = DictDatasetWrapper(dataset, output_format=mode, spatial_dims=2)
            batch = wrapper[0]
            self._validate_batch(batch)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
