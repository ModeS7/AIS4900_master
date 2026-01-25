"""
Unit tests for data loading mechanics: batch construction, collation, workers.

Tests verify that DataLoader configuration and collation work correctly
without requiring real datasets.
"""

import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Any, Optional

from tests.utils import (
    assert_tensor_shape,
    assert_tensor_dtype,
    assert_batch_shape,
)


# =============================================================================
# Helper Classes for Testing
# =============================================================================


class SimpleIndexDataset(Dataset):
    """Dataset returning index values for shuffle testing."""

    def __init__(self, size: int = 100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.tensor(idx)


class DictDataset(Dataset):
    """Dataset returning dict samples."""

    def __init__(self, size: int = 20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        torch.manual_seed(idx)  # Reproducible per-item
        return {
            'image': torch.rand(1, 32, 32),
            'seg': (torch.rand(1, 32, 32) > 0.5).float(),
            'patient_id': f'patient_{idx}',
        }


class TupleDataset(Dataset):
    """Dataset returning tuple samples."""

    def __init__(self, size: int = 20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        return (
            torch.rand(1, 32, 32),  # image
            (torch.rand(1, 32, 32) > 0.5).float(),  # seg
        )


class OptionalKeysDataset(Dataset):
    """Dataset with some keys present only in some samples."""

    def __init__(self, size: int = 20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        sample = {
            'image': torch.rand(1, 32, 32),
        }
        # Only include seg for even indices
        if idx % 2 == 0:
            sample['seg'] = (torch.rand(1, 32, 32) > 0.5).float()
        return sample


# =============================================================================
# TestBatchConstruction - DataLoader batch mechanics
# =============================================================================


class TestBatchConstruction:
    """Test DataLoader batch construction behavior."""

    def test_batch_size_matches_config(self):
        """DataLoader returns batches matching configured batch_size."""
        dataset = SimpleIndexDataset(size=100)
        batch_size = 8

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in loader:
            # Last batch may be smaller if not drop_last
            assert batch.shape[0] <= batch_size
            break

        # First batch should be exactly batch_size
        first_batch = next(iter(loader))
        assert first_batch.shape[0] == batch_size, \
            f"First batch size should be {batch_size}, got {first_batch.shape[0]}"

    def test_drop_last_excludes_partial_batch(self):
        """drop_last=True excludes the final incomplete batch."""
        dataset = SimpleIndexDataset(size=105)  # 13 batches of 8 + partial
        batch_size = 8

        # Without drop_last: includes partial batch
        loader_keep = DataLoader(dataset, batch_size=batch_size, drop_last=False)
        batches_keep = list(loader_keep)
        assert batches_keep[-1].shape[0] < batch_size, "Last batch should be partial"

        # With drop_last: excludes partial batch
        loader_drop = DataLoader(dataset, batch_size=batch_size, drop_last=True)
        batches_drop = list(loader_drop)
        for batch in batches_drop:
            assert batch.shape[0] == batch_size, \
                f"All batches should be full with drop_last=True, got {batch.shape[0]}"

        assert len(batches_drop) < len(batches_keep), \
            "drop_last=True should have fewer batches"

    def test_shuffle_changes_order(self):
        """shuffle=True changes the order of samples between epochs."""
        dataset = SimpleIndexDataset(size=20)
        batch_size = 5

        # Get order without shuffle
        loader_no_shuffle = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        order1 = torch.cat(list(loader_no_shuffle)).tolist()

        # Get order with shuffle (use generator for reproducibility)
        g = torch.Generator()
        g.manual_seed(42)
        loader_shuffle = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=g
        )
        order2 = torch.cat(list(loader_shuffle)).tolist()

        # Orders should be different
        assert order1 != order2, "Shuffle should change order"

        # But contain same elements
        assert sorted(order1) == sorted(order2), "Shuffle should preserve all elements"


# =============================================================================
# TestCollation - Dict and Tuple collation
# =============================================================================


class TestCollation:
    """Test collation of different sample formats."""

    def test_dict_collation_stacks_tensors(self):
        """Dict samples have tensors stacked correctly."""
        dataset = DictDataset(size=10)
        batch_size = 4

        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dict_collate_fn)
        batch = next(iter(loader))

        assert isinstance(batch, dict), "Collated batch should be dict"
        assert 'image' in batch, "Batch should have 'image' key"
        assert 'seg' in batch, "Batch should have 'seg' key"

        assert_batch_shape(batch['image'], batch_size, 1, spatial_dims=2, name="image")
        assert_batch_shape(batch['seg'], batch_size, 1, spatial_dims=2, name="seg")

    def test_tuple_collation_preserves_structure(self):
        """Tuple samples maintain structure after collation."""
        dataset = TupleDataset(size=10)
        batch_size = 4

        loader = DataLoader(dataset, batch_size=batch_size)
        batch = next(iter(loader))

        assert isinstance(batch, (list, tuple)), "Collated tuple should remain sequence"
        assert len(batch) == 2, "Batch should have 2 elements (image, seg)"

        image, seg = batch
        assert_batch_shape(image, batch_size, 1, spatial_dims=2, name="image")
        assert_batch_shape(seg, batch_size, 1, spatial_dims=2, name="seg")

    def test_collation_handles_none_values(self):
        """Collation handles None values gracefully."""
        # Dataset that may return None for some keys
        class NoneValueDataset(Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, idx):
                return {
                    'image': torch.rand(1, 32, 32),
                    'optional': None if idx % 2 == 0 else torch.rand(1),
                }

        dataset = NoneValueDataset()
        loader = DataLoader(dataset, batch_size=4, collate_fn=dict_collate_fn_none_safe)
        batch = next(iter(loader))

        assert 'image' in batch
        assert 'optional' in batch
        # Optional should be a list with mixed None values
        assert batch['optional'] is None or len(batch['optional']) == 4


# =============================================================================
# TestWorkerProcesses - Multi-process loading
# =============================================================================


class TestWorkerProcesses:
    """Test DataLoader with different num_workers configurations."""

    def test_num_workers_zero_works(self):
        """num_workers=0 loads data in main process."""
        dataset = DictDataset(size=10)

        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=0,
            collate_fn=dict_collate_fn
        )

        # Should work without multiprocessing
        batch = next(iter(loader))
        assert 'image' in batch
        assert batch['image'].shape[0] == 4

    def test_num_workers_positive_works(self):
        """num_workers>0 loads data in subprocesses."""
        dataset = DictDataset(size=20)

        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=2,
            collate_fn=dict_collate_fn,
            persistent_workers=False,  # Avoid issues in test
        )

        # Should work with multiprocessing
        batch = next(iter(loader))
        assert 'image' in batch
        assert batch['image'].shape[0] == 4


# =============================================================================
# TestDatasetFormats - Format conversion and validation
# =============================================================================


class TestDatasetFormats:
    """Test dataset format utilities."""

    def test_dict_wrapper_converts_tuple_to_dict(self):
        """DictDatasetWrapper converts tuple datasets to dict format."""
        tuple_dataset = TupleDataset(size=10)

        # Wrap to produce dicts
        wrapper = DictDatasetWrapper(tuple_dataset, keys=['image', 'seg'])

        sample = wrapper[0]
        assert isinstance(sample, dict), "Wrapper should return dict"
        assert 'image' in sample
        assert 'seg' in sample
        assert sample['image'].shape == (1, 32, 32)

    def test_validate_sample_catches_invalid_shapes(self):
        """Sample validation catches incorrect tensor shapes."""
        # Valid 2D sample
        valid_sample = {
            'image': torch.rand(1, 32, 32),
            'seg': torch.rand(1, 32, 32),
        }
        assert validate_sample_shape(valid_sample, spatial_dims=2)

        # Invalid: wrong channel dim
        invalid_sample = {
            'image': torch.rand(32, 32),  # Missing channel
            'seg': torch.rand(1, 32, 32),
        }
        assert not validate_sample_shape(invalid_sample, spatial_dims=2)

        # Invalid: 3D when expecting 2D
        invalid_3d = {
            'image': torch.rand(1, 16, 32, 32),  # 3D
            'seg': torch.rand(1, 32, 32),
        }
        assert not validate_sample_shape(invalid_3d, spatial_dims=2)

    def test_required_vs_optional_keys(self):
        """Validate required vs optional keys in samples."""
        required_keys = {'image'}
        optional_keys = {'seg', 'mode_id', 'size_bins'}

        # Valid: has required, missing optional
        valid = {'image': torch.rand(1, 32, 32)}
        assert validate_keys(valid, required_keys, optional_keys)

        # Valid: has required and some optional
        valid_with_opt = {
            'image': torch.rand(1, 32, 32),
            'seg': torch.rand(1, 32, 32),
        }
        assert validate_keys(valid_with_opt, required_keys, optional_keys)

        # Invalid: missing required
        invalid = {'seg': torch.rand(1, 32, 32)}
        assert not validate_keys(invalid, required_keys, optional_keys)


# =============================================================================
# Helper Functions (would be in medgen.data.loaders.common)
# =============================================================================


def dict_collate_fn(batch):
    """Collate dict samples by stacking tensors."""
    result = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            # Non-tensor values (e.g., strings) stay as list
            result[key] = values

    return result


def dict_collate_fn_none_safe(batch):
    """Collate dict samples, handling None values."""
    result = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]
        if all(v is None for v in values):
            result[key] = None
        elif all(isinstance(v, torch.Tensor) for v in values if v is not None):
            # Stack only non-None tensors, or return None if mixed
            non_none = [v for v in values if v is not None]
            if len(non_none) == len(values):
                result[key] = torch.stack(values)
            else:
                result[key] = values  # Keep as list with None
        else:
            result[key] = values

    return result


class DictDatasetWrapper(Dataset):
    """Wrapper to convert tuple datasets to dict format."""

    def __init__(self, dataset, keys):
        self.dataset = dataset
        self.keys = keys

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, dict):
            return item
        return dict(zip(self.keys, item))


def validate_sample_shape(sample: Dict[str, Any], spatial_dims: int) -> bool:
    """Validate sample tensor shapes."""
    expected_ndim = 1 + spatial_dims  # C + spatial dims

    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if value.ndim != expected_ndim:
                return False
    return True


def validate_keys(
    sample: Dict[str, Any],
    required: set,
    optional: set,
) -> bool:
    """Validate sample has required keys."""
    sample_keys = set(sample.keys())
    return required.issubset(sample_keys)


# =============================================================================
# TestBatchSampler - Custom sampling patterns
# =============================================================================


class TestBatchSampler:
    """Test custom batch sampling strategies."""

    def test_sequential_sampler_order(self):
        """SequentialSampler returns indices in order."""
        from torch.utils.data import SequentialSampler

        dataset = SimpleIndexDataset(size=10)
        sampler = SequentialSampler(dataset)

        indices = list(sampler)
        assert indices == list(range(10)), "SequentialSampler should return 0..N-1"

    def test_random_sampler_all_indices(self):
        """RandomSampler returns all indices exactly once."""
        from torch.utils.data import RandomSampler

        dataset = SimpleIndexDataset(size=10)
        sampler = RandomSampler(dataset)

        indices = list(sampler)
        assert sorted(indices) == list(range(10)), \
            "RandomSampler should return each index exactly once"

    def test_subset_sampler(self):
        """SubsetRandomSampler only samples from specified indices."""
        from torch.utils.data import SubsetRandomSampler

        subset_indices = [0, 2, 4, 6, 8]
        sampler = SubsetRandomSampler(subset_indices)

        indices = list(sampler)
        assert sorted(indices) == sorted(subset_indices), \
            "SubsetRandomSampler should only return specified indices"
