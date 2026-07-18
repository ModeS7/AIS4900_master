"""Tests for non-mutating generated-image conditioning-mask QC."""

import numpy as np
import pytest

from medgen.metrics.brain_mask import evaluate_conditioning_brain_containment


def test_conditioning_inside_main_brain_is_accepted_without_mask_mutation():
    image = np.zeros((24, 24, 24), dtype=np.float32)
    image[4:20, 4:20, 4:20] = 0.8
    seg = np.zeros_like(image)
    seg[10:12, 10:12, 10:12] = 1.0
    original = seg.copy()

    result = evaluate_conditioning_brain_containment(
        image,
        seg,
        margin_mm=3.0,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result['valid'] is True
    assert result['max_distance_mm'] == 0.0
    np.testing.assert_array_equal(seg, original)


def test_lesion_supported_only_by_disconnected_tissue_island_is_rejected():
    image = np.zeros((32, 32, 32), dtype=np.float32)
    image[3:17, 3:17, 3:17] = 0.8
    image[25:29, 25:29, 25:29] = 0.8
    seg = np.zeros_like(image)
    seg[26:28, 26:28, 26:28] = 1.0

    result = evaluate_conditioning_brain_containment(
        image,
        seg,
        margin_mm=3.0,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result['valid'] is False
    assert result['max_distance_mm'] > 3.0


def test_pca_support_prevents_generated_protrusion_from_self_validating():
    image = np.zeros((32, 32, 32), dtype=np.float32)
    image[4:20, 4:20, 4:20] = 0.8
    image[12:15, 20:29, 12:15] = 0.8
    seg = np.zeros_like(image)
    seg[13, 27, 13] = 1.0

    direct = evaluate_conditioning_brain_containment(
        image,
        seg,
        margin_mm=3.0,
    )

    class TrainingSupportStub:
        @staticmethod
        def reconstruct_support(_brain_mask: np.ndarray) -> np.ndarray:
            support = np.zeros_like(image, dtype=bool)
            support[4:20, 4:20, 4:20] = True
            return support

    constrained = evaluate_conditioning_brain_containment(
        image,
        seg,
        margin_mm=3.0,
        brain_support_pca=TrainingSupportStub(),
    )

    assert direct['valid'] is True
    assert constrained['valid'] is False
    assert constrained['max_distance_mm'] == pytest.approx(8.0)


def test_physical_margin_accepts_a_near_boundary_lesion():
    image = np.zeros((20, 20, 20), dtype=np.float32)
    image[4:16, 4:16, 4:16] = 0.8
    seg = np.zeros_like(image)
    seg[16, 10, 10] = 1.0

    accepted = evaluate_conditioning_brain_containment(
        image,
        seg,
        margin_mm=1.0,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )
    rejected = evaluate_conditioning_brain_containment(
        image,
        seg,
        margin_mm=0.5,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
    )

    assert accepted['valid'] is True
    assert accepted['max_distance_mm'] == pytest.approx(1.0)
    assert rejected['valid'] is False


def test_empty_main_brain_fails_closed():
    image = np.zeros((12, 12, 12), dtype=np.float32)
    seg = np.zeros_like(image)
    seg[5, 5, 5] = 1.0

    result = evaluate_conditioning_brain_containment(image, seg, margin_mm=3.0)

    assert result['valid'] is False
    assert result['max_distance_mm'] is None


def test_invalid_geometry_and_parameters_are_rejected():
    image = np.zeros((8, 8, 8), dtype=np.float32)
    seg = np.zeros_like(image)

    with pytest.raises(ValueError, match='does not match'):
        evaluate_conditioning_brain_containment(image, seg[:, :, :-1])
    with pytest.raises(ValueError, match='non-negative'):
        evaluate_conditioning_brain_containment(image, seg, margin_mm=-1.0)
    with pytest.raises(ValueError, match='three positive'):
        evaluate_conditioning_brain_containment(
            image,
            seg,
            voxel_spacing_mm=(1.0, 0.0, 1.0),
        )
