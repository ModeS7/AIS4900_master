"""Fail-closed audit for a frozen generated 3D segmentation-mask pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from medgen.data.loaders.seg import DEFAULT_BIN_EDGES, compute_size_bins_3d
from medgen.metrics.brain_mask import load_brain_atlas
from medgen.scripts.generate import (
    _generated_seg_rejection_reason,
    _xyz_to_dhw,
    compute_voxel_size,
)

_SAMPLE_ID = re.compile(r"^\d{5}$")
_COMPONENT_CONNECTIVITY = 26
_COMPONENT_STRUCTURE = ndimage.generate_binary_structure(3, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_measured_bins(path: Path, expected_count: int) -> dict[int, list[int]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} bins.csv rows, found {len(rows)}")
    result: dict[int, list[int]] = {}
    for row in rows:
        sample_index = int(row["id"])
        bins = [int(row[f"bin_{index}"]) for index in range(7)]
        if int(row["total_tumors"]) != sum(bins):
            raise ValueError(f"bins.csv total does not match bins for sample {sample_index:05d}")
        if sample_index in result:
            raise ValueError(f"Duplicate bins.csv row for sample {sample_index:05d}")
        result[sample_index] = bins
    if sorted(result) != list(range(expected_count)):
        raise ValueError("bins.csv sample identifiers are not exactly 00000..N-1")
    return result


def _validate_generation_manifest(
    path: Path,
    *,
    expected_count: int,
    expected_seed: int,
    expected_steps: int,
    expected_checkpoint: Path,
    expected_atlas: Path,
    max_white_percentage: float,
    max_attempts: int,
    expected_git_commit: str | None,
    expected_shape: tuple[int, int, int],
    generation_depth: int,
    fov_mm: float,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    manifest = json.loads(path.read_text())
    expected_values = {
        "status": "complete",
        "mode": "seg_conditioned",
        "spatial_dims": 3,
        "seed": expected_seed,
        "num_images": expected_count,
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"generation_manifest.json has {key}={manifest.get(key)!r}, expected {expected!r}"
            )
    if expected_git_commit is not None and manifest.get("git_commit") != expected_git_commit:
        raise ValueError("Manifest Git commit does not match the locked source commit")

    sampling = manifest.get("sampling", {})
    sampling_expected = {
        "strategy": "rflow",
        "ode_solver": "euler",
        "num_steps_seg": expected_steps,
        "shift_ratio_seg": 1.0,
        "cfg_scale_seg": 1.0,
    }
    for key, expected in sampling_expected.items():
        if sampling.get(key) != expected:
            raise ValueError(f"Manifest sampling.{key} is not the locked value {expected!r}")

    seg_model = manifest.get("models", {}).get("seg") or {}
    if Path(seg_model.get("path", "")).resolve() != expected_checkpoint.resolve():
        raise ValueError("Manifest segmentation checkpoint does not match the locked checkpoint")
    if seg_model.get("sha256") != _sha256(expected_checkpoint):
        raise ValueError("Manifest segmentation-checkpoint hash does not match the file")
    if manifest.get("models", {}).get("image_low_t") is not None:
        raise ValueError("Mask-only manifest unexpectedly records an image model")
    if manifest.get("models", {}).get("image_high_t") is not None:
        raise ValueError("Mask-only manifest unexpectedly records a high-t image model")

    height, width, output_depth = expected_shape
    geometry_expected = {
        "image_size": height,
        "generation_depth": generation_depth,
        "trim_slices": generation_depth - output_depth,
        "fov_mm": fov_mm,
    }
    if height != width:
        raise ValueError("The generation manifest supports square in-plane masks only")
    geometry = manifest.get("geometry", {})
    for key, expected in geometry_expected.items():
        if geometry.get(key) != expected:
            raise ValueError(f"Manifest geometry.{key} is not the locked value {expected!r}")

    quality = manifest.get("quality_control", {})
    quality_expected = {
        "validate_size_bins": False,
        "component_connectivity": _COMPONENT_CONNECTIVITY,
        "max_white_percentage": max_white_percentage,
        "max_attempts_per_mask": max_attempts,
        "brain_tolerance": 0.0,
        "brain_dilate_pixels": 0,
        "brain_pca_path": None,
        "seg_pca_path": None,
        "validate_brain_mask": False,
        "mask_outside_brain": False,
        "diffrs_checkpoint": None,
    }
    for key, expected in quality_expected.items():
        if quality.get(key) != expected:
            raise ValueError(f"Manifest quality_control.{key} is not {expected!r}")
    if Path(quality.get("brain_atlas_path", "")).resolve() != expected_atlas.resolve():
        raise ValueError("Manifest atlas path does not match the locked train-only atlas")
    atlas_provenance = quality.get("brain_atlas_provenance") or {}
    if Path(atlas_provenance.get("path", "")).resolve() != expected_atlas.resolve():
        raise ValueError("Manifest atlas provenance path does not match the train-only atlas")
    if atlas_provenance.get("sha256") != _sha256(expected_atlas):
        raise ValueError("Manifest atlas hash does not match the train-only atlas")

    samples = manifest.get("samples", [])
    if len(samples) != expected_count:
        raise ValueError(f"Expected {expected_count} manifest samples, found {len(samples)}")
    by_index: dict[int, dict[str, Any]] = {}
    for sample in samples:
        index = sample.get("index")
        if not isinstance(index, int) or index in by_index:
            raise ValueError(f"Invalid or duplicate manifest sample index: {index!r}")
        if not isinstance(sample.get("seg_noise_seed"), int):
            raise ValueError(f"Sample {index:05d} has no deterministic segmentation seed")
        attempt = sample.get("seg_attempt")
        if not isinstance(attempt, int) or not 0 <= attempt < max_attempts:
            raise ValueError(f"Sample {index:05d} has invalid accepted attempt {attempt!r}")
        if sample.get("rejected_attempts") != attempt:
            raise ValueError(f"Sample {index:05d} retry provenance is inconsistent")
        rejection_counts = sample.get("rejection_counts")
        if not isinstance(rejection_counts, dict) or sum(rejection_counts.values()) != attempt:
            raise ValueError(f"Sample {index:05d} rejection counts are inconsistent")
        expected_noise_seed = expected_seed + index + 1_000_000_000 + attempt * 1_000_000
        if sample["seg_noise_seed"] != expected_noise_seed:
            raise ValueError(
                f"Sample {index:05d} has seed {sample['seg_noise_seed']}, "
                f"expected {expected_noise_seed}"
            )
        by_index[index] = sample
    if sorted(by_index) != list(range(expected_count)):
        raise ValueError("Manifest sample indices are not exactly 00000..N-1")
    seeds = [by_index[index]["seg_noise_seed"] for index in range(expected_count)]
    if len(set(seeds)) != expected_count:
        raise ValueError("Manifest segmentation seeds are not unique")
    return manifest, by_index


def audit_pool(
    dataset_root: Path,
    *,
    atlas_path: Path,
    checkpoint_path: Path,
    expected_count: int,
    expected_shape: tuple[int, int, int],
    generation_depth: int,
    expected_seed: int,
    expected_steps: int,
    max_white_percentage: float,
    max_attempts: int,
    published_root: Path | None = None,
    fov_mm: float = 240.0,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    """Audit every saved mask and return a machine-readable report."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Mask pool does not exist: {dataset_root}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not atlas_path.is_file():
        raise FileNotFoundError(f"Atlas does not exist: {atlas_path}")

    temporary_dirs = sorted(
        path.name for path in dataset_root.iterdir() if path.name.startswith(".")
    )
    if temporary_dirs:
        raise ValueError(f"Temporary entries remain in mask pool: {temporary_dirs}")
    sample_dirs = sorted(
        path for path in dataset_root.iterdir() if path.is_dir() and _SAMPLE_ID.fullmatch(path.name)
    )
    expected_names = [f"{index:05d}" for index in range(expected_count)]
    if [path.name for path in sample_dirs] != expected_names:
        raise ValueError(f"Mask directories are not exactly 00000..{expected_count - 1:05d}")

    manifest, manifest_samples = _validate_generation_manifest(
        dataset_root / "generation_manifest.json",
        expected_count=expected_count,
        expected_seed=expected_seed,
        expected_steps=expected_steps,
        expected_checkpoint=checkpoint_path,
        expected_atlas=atlas_path,
        max_white_percentage=max_white_percentage,
        max_attempts=max_attempts,
        expected_git_commit=expected_git_commit,
        expected_shape=expected_shape,
        generation_depth=generation_depth,
        fov_mm=fov_mm,
    )
    measured_bins = _load_measured_bins(dataset_root / "bins.csv", expected_count)

    atlas = load_brain_atlas(atlas_path)
    height, width, output_depth = expected_shape
    if atlas.shape != (generation_depth, height, width):
        raise ValueError(
            f"Atlas shape {atlas.shape} does not match generation shape "
            f"{(generation_depth, height, width)}"
        )
    output_atlas = atlas[:output_depth]
    voxel_spacing = _xyz_to_dhw(compute_voxel_size(height, fov_mm))

    report_root = (published_root or dataset_root).resolve()
    per_mask: list[dict[str, Any]] = []
    for index, sample_dir in enumerate(sample_dirs):
        mask_path = sample_dir / "seg.nii.gz"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask: {mask_path}")
        volume = nib.load(str(mask_path)).get_fdata().astype(np.float32)
        if volume.shape != expected_shape:
            raise ValueError(
                f"Mask {sample_dir.name} has shape {volume.shape}, expected {expected_shape}"
            )
        if not np.isfinite(volume).all():
            raise ValueError(f"Mask {sample_dir.name} contains non-finite values")
        if not np.isin(np.unique(volume), (0.0, 1.0)).all():
            raise ValueError(f"Mask {sample_dir.name} is not binary")

        mask_dhw = np.transpose(volume, (2, 0, 1))
        reason = _generated_seg_rejection_reason(
            mask_dhw,
            max_white_percentage=max_white_percentage,
            brain_atlas=output_atlas,
            brain_tolerance=0.0,
            brain_dilate_pixels=0,
        )
        if reason is not None:
            raise ValueError(f"Mask {sample_dir.name} fails frozen-pool QC: {reason}")

        binary = mask_dhw > 0.5
        labeled, component_count = ndimage.label(
            binary,
            structure=_COMPONENT_STRUCTURE,
        )
        component_voxels = np.bincount(labeled.ravel())[1:].astype(int).tolist()
        foreground_voxels = int(binary.sum())
        max_slice_fraction = float(binary.reshape(output_depth, -1).mean(axis=1).max())
        manifest_sample = manifest_samples[index]
        bins = measured_bins[index]
        if manifest_sample.get("actual_size_bins") != bins:
            raise ValueError(f"Mask {sample_dir.name} bins.csv and manifest bins disagree")
        recomputed_bins = compute_size_bins_3d(
            mask_dhw,
            list(DEFAULT_BIN_EDGES),
            voxel_spacing,
            7,
            connectivity=_COMPONENT_CONNECTIVITY,
        ).astype(int).tolist()
        if bins != recomputed_bins:
            raise ValueError(f"Mask {sample_dir.name} measured size bins are incorrect")
        per_mask.append(
            {
                "index": index,
                "path": str(report_root / sample_dir.name / "seg.nii.gz"),
                "sha256": _sha256(mask_path),
                "seg_noise_seed": manifest_sample["seg_noise_seed"],
                "seg_attempt": manifest_sample["seg_attempt"],
                "rejection_counts": manifest_sample.get("rejection_counts", {}),
                "foreground_voxels": foreground_voxels,
                "foreground_fraction": float(foreground_voxels / binary.size),
                "max_slice_foreground_fraction": max_slice_fraction,
                "connected_components": int(component_count),
                "component_voxels": component_voxels,
                "actual_size_bins": bins,
            }
        )

    hashes = [item["sha256"] for item in per_mask]
    if len(set(hashes)) != expected_count:
        raise ValueError("The frozen pool contains byte-identical duplicate mask files")
    total_rejected = sum(int(item["seg_attempt"]) for item in per_mask)
    return {
        "schema_version": 1,
        "status": "pass",
        "dataset_root": str(report_root),
        "expected_count": expected_count,
        "expected_shape_hwd": list(expected_shape),
        "component_connectivity": _COMPONENT_CONNECTIVITY,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "atlas_path": str(atlas_path.resolve()),
        "atlas_sha256": _sha256(atlas_path),
        "generation_manifest_sha256": _sha256(dataset_root / "generation_manifest.json"),
        "git_commit": manifest.get("git_commit"),
        "sampling": manifest["sampling"],
        "quality_control": manifest["quality_control"],
        "total_rejected_draws_before_acceptance": total_rejected,
        "masks": per_mask,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--output-depth", type=int, default=150)
    parser.add_argument("--generation-depth", type=int, default=160)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--max-white-per-slice", type=float, default=0.04)
    parser.add_argument("--max-attempts", type=int, required=True)
    parser.add_argument("--published-root", type=Path)
    parser.add_argument("--fov-mm", type=float, default=240.0)
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = audit_pool(
        args.dataset_root,
        atlas_path=args.atlas,
        checkpoint_path=args.checkpoint,
        expected_count=args.expected_count,
        expected_shape=(args.height, args.width, args.output_depth),
        generation_depth=args.generation_depth,
        expected_seed=args.seed,
        expected_steps=args.steps,
        max_white_percentage=args.max_white_per_slice,
        max_attempts=args.max_attempts,
        published_root=args.published_root,
        fov_mm=args.fov_mm,
        expected_git_commit=args.expected_git_commit,
    )
    _write_json_atomic(args.output, report)
    print(f"PASS: audited {args.expected_count} frozen masks; report={args.output}")


if __name__ == "__main__":
    main()
