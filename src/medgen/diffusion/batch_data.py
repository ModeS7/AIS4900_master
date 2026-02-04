"""Standardized batch data structure for all training modes."""
import warnings
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class BatchData:
    """Standardized batch data structure.

    All modes unpack to this format, eliminating scattered tuple handling.
    """
    images: Tensor
    labels: Tensor | None = None
    size_bins: Tensor | None = None
    bin_maps: Tensor | None = None
    mode_id: Tensor | None = None

    @classmethod
    def from_raw(cls, data: Tensor | tuple | list | dict) -> 'BatchData':
        """Convert any batch format to standardized BatchData.

        Handles:
        - Tensor: Simple image batch
        - Dict: {'image': ..., 'seg': ..., ...} or {'images': ..., 'labels': ..., ...}
        - 2-tuple: (images, labels) or (seg, size_bins) [deprecated]
        - 3-tuple: (seg, size_bins, bin_maps) or (image, seg, mode_id) [deprecated]
        """
        if isinstance(data, Tensor):
            return cls(images=data)

        if isinstance(data, dict):
            # Support both 'image'/'seg' and 'images'/'labels' keys
            # Use explicit None checks to avoid tensor boolean ambiguity
            images = data.get('image')
            if images is None:
                images = data.get('images')
            if images is None:
                images = data.get('latent')

            labels = data.get('seg')
            if labels is None:
                labels = data.get('labels')
            if labels is None:
                labels = data.get('latent_seg')

            return cls(
                images=images,
                labels=labels,
                size_bins=data.get('size_bins'),
                bin_maps=data.get('bin_maps'),
                mode_id=data.get('mode_id'),
            )

        # Deprecated tuple format - keep for backwards compatibility
        if isinstance(data, (tuple, list)):
            warnings.warn(
                "Tuple batch format is deprecated. Update dataloader to return dict.",
                DeprecationWarning,
                stacklevel=2
            )
            if len(data) == 2:
                first, second = data

                # Validate types before calling .dim()
                if not isinstance(first, Tensor):
                    raise TypeError(
                        f"BatchData.from_raw: first element is {type(first).__name__}, expected Tensor"
                    )
                if not isinstance(second, Tensor):
                    raise TypeError(
                        f"BatchData.from_raw: second element is {type(second).__name__}, expected Tensor"
                    )

                # Distinguish (images, labels) from (seg, size_bins)
                if second.dim() == 1:
                    # size_bins is 1D
                    return cls(images=first, size_bins=second)
                else:
                    # labels/seg is multi-dimensional
                    return cls(images=first, labels=second)

            elif len(data) == 3:
                first, second, third = data

                # Validate types before calling .dim()
                for i, elem in enumerate([first, second, third]):
                    if not isinstance(elem, Tensor):
                        raise TypeError(
                            f"BatchData.from_raw: element {i} is {type(elem).__name__}, expected Tensor"
                        )

                if second.dim() == 1:
                    # (seg, size_bins, bin_maps)
                    return cls(images=first, size_bins=second, bin_maps=third)
                else:
                    # (image, seg, mode_id)
                    return cls(images=first, labels=second, mode_id=third)

            else:
                raise ValueError(f"Unexpected tuple length: {len(data)}")

        raise ValueError(f"Unknown batch format: {type(data)}")

    def to_device(self, device: torch.device) -> 'BatchData':
        """Move all tensors to device."""
        return BatchData(
            images=self.images.to(device) if self.images is not None else None,
            labels=self.labels.to(device) if self.labels is not None else None,
            size_bins=self.size_bins.to(device) if self.size_bins is not None else None,
            bin_maps=self.bin_maps.to(device) if self.bin_maps is not None else None,
            mode_id=self.mode_id.to(device) if self.mode_id is not None else None,
        )
