"""Tests for the frozen synthetic segmentation-mask pool audit."""

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from medgen.data.loaders.seg import DEFAULT_BIN_EDGES, compute_size_bins_3d
from medgen.scripts.audit_seg_mask_pool import audit_pool


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mask_pool_audit_checks_provenance_content_and_published_paths(tmp_path: Path):
    dataset = tmp_path / "staging"
    dataset.mkdir()
    published = tmp_path / "published" / "common_seg_masks_525"
    checkpoint = tmp_path / "checkpoint_latest.pt"
    checkpoint.write_bytes(b"locked segmentation checkpoint")

    height = width = 10
    generation_depth = 4
    output_depth = 3
    atlas_path = tmp_path / "train_atlas.nii.gz"
    atlas_hwd = np.ones((height, width, generation_depth), dtype=np.float32)
    nib.save(nib.Nifti1Image(atlas_hwd, np.eye(4)), atlas_path)

    masks: list[np.ndarray] = []
    first = np.zeros((height, width, output_depth), dtype=np.float32)
    first[5, 5, 1] = 1.0
    masks.append(first)
    second = np.zeros_like(first)
    # Corner-adjacent in 3D: one lesion under the locked 26-connectivity,
    # but two components under the historical 6-connectivity default.
    second[3, 3, 0] = 1.0
    second[4, 4, 1] = 1.0
    masks.append(second)

    measured_bins: list[list[int]] = []
    for index, mask_hwd in enumerate(masks):
        sample_dir = dataset / f"{index:05d}"
        sample_dir.mkdir()
        nib.save(nib.Nifti1Image(mask_hwd, np.eye(4)), sample_dir / "seg.nii.gz")
        mask_dhw = np.transpose(mask_hwd, (2, 0, 1))
        bins = compute_size_bins_3d(
            mask_dhw,
            list(DEFAULT_BIN_EDGES),
            (1.0, 1.0, 1.0),
            7,
            connectivity=26,
        ).astype(int).tolist()
        measured_bins.append(bins)

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "git_commit": "0123456789abcdef",
        "mode": "seg_conditioned",
        "spatial_dims": 3,
        "seed": 42,
        "num_images": 2,
        "models": {
            "seg": {
                "path": str(checkpoint.resolve()),
                "sha256": _sha256(checkpoint),
            }
        },
        "sampling": {
            "strategy": "rflow",
            "ode_solver": "euler",
            "num_steps_seg": 100,
            "shift_ratio_seg": 1.0,
            "cfg_scale_seg": 1.0,
        },
        "geometry": {
            "image_size": height,
            "generation_depth": generation_depth,
            "trim_slices": generation_depth - output_depth,
            "fov_mm": 10.0,
        },
        "quality_control": {
            "brain_atlas_path": str(atlas_path.resolve()),
            "brain_atlas_provenance": {
                "path": str(atlas_path.resolve()),
                "sha256": _sha256(atlas_path),
            },
            "brain_pca_path": None,
            "seg_pca_path": None,
            "validate_size_bins": False,
            "component_connectivity": 26,
            "max_white_percentage": 0.04,
            "max_attempts_per_mask": 50,
            "brain_tolerance": 0.0,
            "brain_dilate_pixels": 0,
            "validate_brain_mask": False,
            "mask_outside_brain": False,
            "diffrs_checkpoint": None,
        },
        "samples": [
            {
                "index": index,
                "seg_noise_seed": 42 + index + 1_000_000_000,
                "seg_attempt": 0,
                "rejected_attempts": 0,
                "rejection_counts": {},
                "actual_size_bins": measured_bins[index],
            }
            for index in range(2)
        ],
    }
    (dataset / "generation_manifest.json").write_text(json.dumps(manifest))
    with (dataset / "bins.csv").open("w") as handle:
        handle.write("id,bin_0,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,total_tumors\n")
        for index, bins in enumerate(measured_bins):
            handle.write(f"{index:05d},{','.join(map(str, bins))},{sum(bins)}\n")

    report = audit_pool(
        dataset,
        atlas_path=atlas_path,
        checkpoint_path=checkpoint,
        expected_count=2,
        expected_shape=(height, width, output_depth),
        generation_depth=generation_depth,
        expected_seed=42,
        expected_steps=100,
        max_white_percentage=0.04,
        max_attempts=50,
        published_root=published,
        fov_mm=10.0,
        expected_git_commit="0123456789abcdef",
    )

    assert report["status"] == "pass"
    assert report["component_connectivity"] == 26
    assert report["dataset_root"] == str(published.resolve())
    assert report["total_rejected_draws_before_acceptance"] == 0
    assert [entry["connected_components"] for entry in report["masks"]] == [1, 1]
    assert report["masks"][1]["component_voxels"] == [2]
    assert report["masks"][0]["path"] == str(
        published.resolve() / "00000" / "seg.nii.gz"
    )
    assert len({entry["sha256"] for entry in report["masks"]}) == 2


def test_size_bin_connectivity_is_explicit_and_preserves_legacy_default():
    mask = np.zeros((2, 2, 2), dtype=np.float32)
    mask[0, 0, 0] = 1.0
    mask[1, 1, 1] = 1.0

    legacy_bins = compute_size_bins_3d(
        mask,
        list(DEFAULT_BIN_EDGES),
        (1.0, 1.0, 1.0),
        7,
    )
    pool_bins = compute_size_bins_3d(
        mask,
        list(DEFAULT_BIN_EDGES),
        (1.0, 1.0, 1.0),
        7,
        connectivity=26,
    )

    assert int(legacy_bins.sum()) == 2
    assert int(pool_bins.sum()) == 1
