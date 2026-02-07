"""Base classes for unified 2D/3D diffusion data loading.

This module defines abstract base classes that ensure consistent return formats
across 2D and 3D datasets. All diffusion datasets should return dictionaries
with standardized keys.

Standardized Batch Format:
    All datasets return Dict[str, Any] with these keys:
    - 'image': torch.Tensor [B, C, H, W] or [B, C, D, H, W]
        The image(s) to generate/denoise
    - 'seg': torch.Tensor [B, 1, H, W] or [B, 1, D, H, W] (optional)
        Segmentation mask for conditioning
    - 'mode_id': torch.Tensor [B] (optional)
        Mode identifier for multi-modality training
    - 'size_bins': torch.Tensor [B, num_bins] (optional)
        Tumor size bin counts for seg_conditioned mode
    - 'patient_id': str (optional)
        Patient identifier for tracking

This standardization eliminates the need for isinstance checks in trainers
and ensures consistent behavior between 2D and 3D training.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDiffusionDataset(Dataset, ABC):
    """Abstract base class for all diffusion training datasets.

    Subclasses must implement __len__ and __getitem__ that returns a dict.

    This base class enforces a consistent return format across 2D and 3D
    datasets, eliminating the need for batch format detection in trainers.

    Example:
        >>> class MyDataset(BaseDiffusionDataset):
        ...     def __getitem__(self, idx):
        ...         return {
        ...             'image': torch.randn(1, 256, 256),  # Required
        ...             'seg': torch.ones(1, 256, 256),     # Optional
        ...         }
    """

    # Keys that are always present (or None)
    REQUIRED_KEYS = {'image'}
    # Keys that may be present depending on mode
    OPTIONAL_KEYS = {'seg', 'mode_id', 'size_bins', 'patient_id', 'modality'}

    @property
    @abstractmethod
    def spatial_dims(self) -> int:
        """Return spatial dimensions (2 or 3)."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns:
            Dict with at minimum 'image' key containing the image tensor.
            Shape: [C, H, W] for 2D or [C, D, H, W] for 3D.
            Additional keys (seg, mode_id, etc.) as available.
        """
        pass

    def validate_sample(self, sample: dict[str, Any]) -> None:
        """Validate that a sample has the correct format.

        Raises:
            ValueError: If sample format is invalid.
        """
        if not isinstance(sample, dict):
            raise ValueError(
                f"Dataset must return dict, got {type(sample).__name__}. "
                "All diffusion datasets should return dict format."
            )

        if 'image' not in sample:
            raise ValueError(
                "Dataset sample must contain 'image' key. "
                f"Got keys: {list(sample.keys())}"
            )

        image = sample['image']
        if not isinstance(image, torch.Tensor):
            raise ValueError(
                f"'image' must be torch.Tensor, got {type(image).__name__}"
            )

        expected_dims = self.spatial_dims + 1  # +1 for channel dim
        if image.dim() != expected_dims:
            raise ValueError(
                f"{self.spatial_dims}D dataset 'image' should have {expected_dims} dims "
                f"(C + spatial), got {image.dim()} dims with shape {tuple(image.shape)}"
            )


class BaseDiffusionDataset2D(BaseDiffusionDataset):
    """Base class for 2D diffusion datasets.

    Returns images with shape [C, H, W].
    """

    @property
    def spatial_dims(self) -> int:
        return 2


class BaseDiffusionDataset3D(BaseDiffusionDataset):
    """Base class for 3D diffusion datasets.

    Returns images with shape [C, D, H, W].
    """

    @property
    def spatial_dims(self) -> int:
        return 3


def dict_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for dict-format batches.

    Stacks tensors along batch dimension, keeps non-tensors as lists.

    Args:
        batch: List of sample dicts from dataset.

    Returns:
        Collated dict with batched tensors.

    Example:
        >>> samples = [{'image': torch.randn(1, 64, 64)} for _ in range(4)]
        >>> collated = dict_collate_fn(samples)
        >>> collated['image'].shape
        torch.Size([4, 1, 64, 64])
    """
    if not batch:
        return {}

    collated = {}
    keys = batch[0].keys()

    for key in keys:
        # Collect values where key exists (even if value is None)
        values = [sample[key] for sample in batch if key in sample]

        if len(values) != len(batch):
            logger.warning(
                f"dict_collate_fn: key '{key}' present in {len(values)}/{len(batch)} "
                f"samples. Batch dimension will be {len(values)} instead of {len(batch)}."
            )

        if not values:
            continue  # Key absent from all samples

        # Check if all values are None
        if all(v is None for v in values):
            collated[key] = None
            continue

        # Filter out None values for stacking
        non_none_values = [v for v in values if v is not None]
        if not non_none_values:
            collated[key] = None
            continue

        values = non_none_values

        # Stack tensors, keep non-tensors as list
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values

    return collated


class DictDatasetWrapper(Dataset):
    """Wrapper that converts tuple/tensor dataset output to dict format.

    This wrapper allows existing datasets that return tuples or raw tensors
    to be used with the new unified dict format without modifying them.

    Args:
        dataset: The underlying dataset to wrap.
        keys: List of keys for dict output. For tensors, use ['image'].
              For tuples, keys map to each element.
        spatial_dims: 2 or 3, used for validation.

    Example:
        >>> raw_dataset = extract_slices_dual(merged, has_seg=True)
        >>> # Old: raw_dataset[i] returns ndarray [C, H, W]
        >>>
        >>> wrapped = DictDatasetWrapper(raw_dataset, keys=['image', 'seg'])
        >>> # New: wrapped[i] returns {'image': tensor[1, H, W], 'seg': tensor[1, H, W]}
    """

    def __init__(
        self,
        dataset: Dataset,
        output_format: str = 'bravo',  # 'seg', 'bravo', 'dual', 'multi', 'seg_conditioned'
        spatial_dims: int = 2,
    ):
        self.dataset = dataset
        self.output_format = output_format
        self._spatial_dims = spatial_dims

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]

        # Already a dict - validate and return
        if isinstance(item, dict):
            return self._validate_dict(item)

        # Convert numpy to tensor
        if isinstance(item, np.ndarray):
            item = torch.from_numpy(item.copy()).float()

        # Handle different output formats
        if self.output_format == 'seg':
            # seg mode: single channel segmentation mask
            if isinstance(item, torch.Tensor):
                return {'image': item}
            else:
                return {'image': torch.from_numpy(np.array(item)).float()}

        elif self.output_format == 'bravo':
            # bravo mode: [image, seg] stacked in tensor
            # Input is [2, H, W] tensor with channel 0=bravo, channel 1=seg
            if isinstance(item, torch.Tensor):
                if item.shape[0] == 2:
                    return {
                        'image': item[0:1],  # [1, H, W]
                        'seg': item[1:2],    # [1, H, W]
                    }
                else:
                    return {'image': item}
            else:
                item = torch.from_numpy(np.array(item)).float()
                return {'image': item}

        elif self.output_format == 'dual':
            # dual mode: [t1_pre, t1_gd, seg] stacked
            # Input is [3, H, W] tensor
            if isinstance(item, torch.Tensor):
                if item.shape[0] == 3:
                    return {
                        'image': item[0:2],  # [2, H, W] - both image channels
                        'seg': item[2:3],    # [1, H, W]
                    }
                else:
                    return {'image': item}
            else:
                item = torch.from_numpy(np.array(item)).float()
                return {'image': item}

        elif self.output_format == 'multi':
            # multi mode: (image, seg, mode_id) tuple
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                image, seg, mode_id = item[0], item[1], item[2]
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image.copy()).float()
                if isinstance(seg, np.ndarray):
                    seg = torch.from_numpy(seg.copy()).long()  # Categorical mask
                return {
                    'image': image,
                    'seg': seg,
                    'mode_id': mode_id if isinstance(mode_id, torch.Tensor) else torch.tensor(mode_id),
                }
            else:
                return {'image': torch.from_numpy(np.array(item)).float()}

        elif self.output_format == 'seg_conditioned':
            # seg_conditioned mode: (seg, size_bins) tuple
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                seg, size_bins = item[0], item[1]
                if isinstance(seg, np.ndarray):
                    seg = torch.from_numpy(seg.copy()).long()  # Categorical mask
                if isinstance(size_bins, np.ndarray):
                    size_bins = torch.from_numpy(size_bins.copy()).long()  # Discrete counts
                return {
                    'image': seg,  # seg is the image to generate
                    'size_bins': size_bins,
                }
            else:
                return {'image': torch.from_numpy(np.array(item)).float()}

        elif self.output_format == 'seg_conditioned_input':
            # seg_conditioned_input mode: (seg, size_bins, bin_maps) tuple
            # bin_maps are spatial conditioning maps for input channel conditioning
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                seg, size_bins, bin_maps = item[0], item[1], item[2]
                if isinstance(seg, np.ndarray):
                    seg = torch.from_numpy(seg.copy()).long()  # Categorical mask
                if isinstance(size_bins, np.ndarray):
                    size_bins = torch.from_numpy(size_bins.copy()).long()  # Discrete counts
                if isinstance(bin_maps, np.ndarray):
                    bin_maps = torch.from_numpy(bin_maps.copy()).float()  # Continuous spatial
                return {
                    'image': seg,  # seg is the image to generate
                    'size_bins': size_bins,
                    'bin_maps': bin_maps,  # [num_bins, H, W] spatial maps
                }
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                # Fallback: no bin_maps provided
                seg, size_bins = item[0], item[1]
                if isinstance(seg, np.ndarray):
                    seg = torch.from_numpy(seg.copy()).long()  # Categorical mask
                if isinstance(size_bins, np.ndarray):
                    size_bins = torch.from_numpy(size_bins.copy()).long()  # Discrete counts
                return {
                    'image': seg,
                    'size_bins': size_bins,
                }
            else:
                return {'image': torch.from_numpy(np.array(item)).float()}

        # --- Compression output formats ---
        # These formats are for VAE/VQVAE/DC-AE training where we don't concatenate
        # seg with images for conditioning, but may want seg for regional metrics.

        elif self.output_format == 'compression_seg':
            # DC-AE seg compression: the seg mask IS the image to reconstruct
            if isinstance(item, torch.Tensor):
                return {'image': item}
            return {'image': self._to_tensor(item)}

        elif self.output_format in ('compression_single', 'compression_multi'):
            # Single/multi modality: image + optional aux seg for metrics
            if isinstance(item, torch.Tensor):
                return {'image': item}
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                return {
                    'image': self._to_tensor(item[0]),
                    'seg': self._to_tensor_field(item[1], 'seg'),  # Categorical mask
                }
            return {'image': self._to_tensor(item)}

        elif self.output_format == 'compression_dual':
            # Dual modality: 2-channel image + optional aux seg
            if isinstance(item, torch.Tensor):
                return {'image': item}
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                return {
                    'image': self._to_tensor(item[0]),
                    'seg': self._to_tensor_field(item[1], 'seg'),  # Categorical mask
                }
            return {'image': self._to_tensor(item)}

        # Fallback: just wrap in dict
        if isinstance(item, torch.Tensor):
            return {'image': item}
        else:
            return {'image': torch.from_numpy(np.array(item)).float()}

    def _to_tensor(self, item: Any) -> torch.Tensor:
        """Convert item to tensor (float dtype)."""
        if isinstance(item, torch.Tensor):
            return item
        elif isinstance(item, np.ndarray):
            return torch.from_numpy(item.copy()).float()
        else:
            return torch.from_numpy(np.array(item)).float()

    def _to_tensor_field(self, item: Any, field_name: str) -> torch.Tensor:
        """Convert item to tensor with field-appropriate dtype.

        Args:
            item: numpy array or other convertible type
            field_name: name of the field (used to determine dtype)

        Returns:
            Tensor with appropriate dtype (long for categorical, float for continuous)
        """
        # Fields that should be integer dtype (categorical/discrete values)
        INTEGER_FIELDS = {'seg', 'size_bins', 'labels', 'mode_id'}

        if isinstance(item, torch.Tensor):
            return item
        elif isinstance(item, np.ndarray):
            tensor = torch.from_numpy(item.copy())
        else:
            tensor = torch.from_numpy(np.array(item))

        if field_name in INTEGER_FIELDS:
            return tensor.long()
        return tensor.float()

    def _validate_dict(self, item: dict[str, Any]) -> dict[str, Any]:
        """Validate dict has required 'image' key and convert tensors.

        Args:
            item: Dict from underlying dataset.

        Returns:
            Validated dict with all numpy arrays converted to tensors.

        Raises:
            ValueError: If dict is missing required 'image' key.
        """
        if 'image' not in item:
            raise ValueError(f"Dict batch must have 'image' key, got: {list(item.keys())}")

        # Ensure all tensors (convert numpy arrays with field-aware dtype)
        result = {}
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                result[k] = self._to_tensor_field(v, k)
            else:
                result[k] = v
        return result

    @property
    def spatial_dims(self) -> int:
        return self._spatial_dims


def validate_batch_format(batch: Any) -> dict[str, Any]:
    """Validate and normalize batch to dict format.

    For backward compatibility, converts old tuple/tensor formats to dict.
    Issues a deprecation warning when conversion is needed.

    Args:
        batch: Batch from dataloader (dict, tuple, or tensor).

    Returns:
        Normalized dict-format batch.

    Raises:
        ValueError: If batch format cannot be determined.
    """
    if isinstance(batch, dict):
        return batch

    # Legacy tuple format: (image, seg) or (image, seg, mode_id) etc.
    if isinstance(batch, (tuple, list)):
        import warnings
        warnings.warn(
            "Tuple batch format is deprecated. Update your dataset to return dict. "
            "See medgen.data.loaders.base for the expected format.",
            DeprecationWarning,
            stacklevel=3,
        )

        if len(batch) == 1:
            return {'image': batch[0]}
        elif len(batch) == 2:
            return {'image': batch[0], 'seg': batch[1]}
        elif len(batch) == 3:
            return {'image': batch[0], 'seg': batch[1], 'mode_id': batch[2]}
        else:
            return {'image': batch[0], 'seg': batch[1], 'extra': batch[2:]}

    # Legacy tensor format: single image tensor
    if isinstance(batch, torch.Tensor):
        import warnings
        warnings.warn(
            "Tensor batch format is deprecated. Update your dataset to return dict.",
            DeprecationWarning,
            stacklevel=3,
        )
        return {'image': batch}

    raise ValueError(
        f"Unknown batch format: {type(batch).__name__}. "
        "Expected dict, tuple, or tensor."
    )
