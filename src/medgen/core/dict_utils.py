"""Dictionary utilities for safe key access with fallbacks."""

from typing import Any, TypeVar

T = TypeVar('T')

# Standard batch key aliases (canonical order)
IMAGE_KEYS = ('image', 'images', 'volume', 'latent')
MASK_KEYS = ('seg', 'mask', 'labels', 'seg_mask', 'latent_seg')
PATIENT_KEYS = ('patient_id', 'patient')


def get_with_fallbacks(d: dict, *keys: str, default: T | None = None) -> T | Any:
    """Get value from dict trying multiple keys in order.

    Args:
        d: Dictionary to search.
        *keys: Keys to try in order.
        default: Default if no key found.

    Returns:
        First found value or default.

    Example:
        >>> get_with_fallbacks(batch, 'image', 'images')
        >>> get_with_fallbacks(batch, 'seg', 'mask', 'labels', default=None)
    """
    for key in keys:
        if key in d:
            return d[key]
    return default
