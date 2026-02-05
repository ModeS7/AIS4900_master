"""Tests for dict_utils module."""

import pytest
from medgen.core.dict_utils import get_with_fallbacks, IMAGE_KEYS, MASK_KEYS, PATIENT_KEYS


class TestGetWithFallbacks:
    """Tests for get_with_fallbacks function."""

    def test_first_key_found(self):
        """First key is found and returned."""
        d = {'image': 1, 'images': 2}
        assert get_with_fallbacks(d, 'image', 'images') == 1

    def test_second_key_found(self):
        """First key missing, second key found."""
        d = {'images': 2}
        assert get_with_fallbacks(d, 'image', 'images') == 2

    def test_no_key_returns_default(self):
        """No key found returns default."""
        d = {'other': 3}
        assert get_with_fallbacks(d, 'image', 'images') is None
        assert get_with_fallbacks(d, 'image', 'images', default=-1) == -1

    def test_empty_dict(self):
        """Empty dict returns default."""
        assert get_with_fallbacks({}, 'a', 'b', default='x') == 'x'

    def test_triple_fallback(self):
        """Third key found after two misses."""
        d = {'labels': 3}
        assert get_with_fallbacks(d, 'seg', 'mask', 'labels') == 3

    def test_none_value_is_returned(self):
        """Key with None value is returned (not skipped)."""
        d = {'image': None, 'images': 2}
        assert get_with_fallbacks(d, 'image', 'images') is None

    def test_falsy_value_is_returned(self):
        """Falsy values (0, False, '') are returned correctly."""
        d = {'count': 0}
        assert get_with_fallbacks(d, 'count', default=10) == 0

        d = {'flag': False}
        assert get_with_fallbacks(d, 'flag', default=True) is False

    def test_single_key(self):
        """Single key lookup works."""
        d = {'x': 42}
        assert get_with_fallbacks(d, 'x') == 42
        assert get_with_fallbacks(d, 'y') is None


class TestStandardKeyConstants:
    """Tests for standard key constant tuples."""

    def test_image_keys_order(self):
        """IMAGE_KEYS has canonical order."""
        assert IMAGE_KEYS == ('image', 'images', 'volume', 'latent')

    def test_mask_keys_order(self):
        """MASK_KEYS has canonical order."""
        assert MASK_KEYS == ('seg', 'mask', 'labels', 'seg_mask', 'latent_seg')

    def test_patient_keys_order(self):
        """PATIENT_KEYS has canonical order."""
        assert PATIENT_KEYS == ('patient_id', 'patient')

    def test_can_use_constants_with_function(self):
        """Constants work with get_with_fallbacks via unpacking."""
        d = {'volume': 'vol_data'}
        assert get_with_fallbacks(d, *IMAGE_KEYS) == 'vol_data'

        d = {'labels': 'label_data'}
        assert get_with_fallbacks(d, *MASK_KEYS) == 'label_data'
