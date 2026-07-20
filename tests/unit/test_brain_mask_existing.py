import os
import stat
from pathlib import Path

import nibabel as nib
import numpy as np

from medgen.metrics.brain_mask import create_brain_mask
from medgen.scripts.brain_mask_existing import mask_one_case


def test_in_place_masking_is_atomic_and_breaks_hard_link(tmp_path: Path) -> None:
    case = tmp_path / "00003"
    case.mkdir()
    bravo = case / "bravo.nii.gz"
    seg = case / "seg.nii.gz"
    retained_source = tmp_path / "retained_source.nii.gz"

    volume = np.full((16, 16, 8), 0.01, dtype=np.float32)
    volume[5:11, 5:11, 2:6] = 1.0
    segmentation = np.zeros_like(volume)
    segmentation[7:9, 7:9, 3:5] = 1.0
    affine = np.diag([0.9375, 0.9375, 1.0, 1.0])
    nib.save(nib.Nifti1Image(volume, affine), bravo)
    nib.save(nib.Nifti1Image(segmentation, affine), seg)
    os.chmod(bravo, 0o640)
    os.link(bravo, retained_source)

    original_inode = bravo.stat().st_ino
    original_seg = seg.read_bytes()
    expected_support = create_brain_mask(volume, threshold=0.05, fill_holes=True, dilate_pixels=2)
    expected = volume * expected_support.astype(volume.dtype)

    assert mask_one_case(
        case,
        case,
        threshold=0.05,
        dilate_pixels=2,
        overwrite=False,
        in_place=True,
        expected_shape=volume.shape,
    )

    masked = np.asarray(nib.load(bravo).dataobj, dtype=np.float32)
    retained = np.asarray(nib.load(retained_source).dataobj, dtype=np.float32)
    assert np.array_equal(masked, expected)
    assert np.array_equal(retained, volume)
    assert bravo.stat().st_ino != original_inode
    assert retained_source.stat().st_ino == original_inode
    assert stat.S_IMODE(bravo.stat().st_mode) == 0o640
    assert seg.read_bytes() == original_seg

    masked_inode = bravo.stat().st_ino
    assert mask_one_case(
        case,
        case,
        threshold=0.05,
        dilate_pixels=2,
        overwrite=False,
        in_place=True,
        expected_shape=volume.shape,
    )
    assert bravo.stat().st_ino == masked_inode


def test_in_place_masking_rejects_symlinked_bravo(tmp_path: Path) -> None:
    case = tmp_path / "00003"
    case.mkdir()
    real_bravo = tmp_path / "real_bravo.nii.gz"
    volume = np.ones((4, 4, 4), dtype=np.float32)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), real_bravo)
    (case / "bravo.nii.gz").symlink_to(real_bravo)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), case / "seg.nii.gz")

    try:
        mask_one_case(
            case,
            case,
            threshold=0.05,
            dilate_pixels=2,
            overwrite=False,
            in_place=True,
            expected_shape=volume.shape,
        )
    except RuntimeError as error:
        assert "symlinked BRAVO" in str(error)
    else:
        raise AssertionError("in-place masking unexpectedly followed a BRAVO symlink")
