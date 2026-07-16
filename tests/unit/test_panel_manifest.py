"""Runtime tests for the fixed generator panel's fail-closed manifests."""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def panel_manifest_module():
    path = Path(__file__).parents[2] / "IDUN" / "generate" / "panel_manifest.py"
    spec = importlib.util.spec_from_file_location("panel_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generation_manifest(input_root: Path, patient_ids: list[str], low_checkpoint: dict):
    return {
        "schema_version": 1,
        "status": "complete",
        "mode": "bravo",
        "spatial_dims": 3,
        "seed": 42,
        "num_images": 105,
        "real_seg_dir": str(input_root),
        "expected_real_cases": 105,
        "expected_real_depth": 150,
        "require_real_bravo_pairs": True,
        "validate_real_seg_masks": True,
        "models": {"image_low_t": low_checkpoint, "image_high_t": None},
        "sampling": {
            "strategy": "rflow",
            "ode_solver": "euler",
            "num_steps_bravo": 100,
            "shift_ratio_bravo": 1.0,
            "cfg_scale_bravo": 1.0,
            "handoff_t": None,
        },
        "geometry": {
            "image_size": 256,
            "generation_depth": 160,
            "trim_slices": 10,
            "fov_mm": 240.0,
        },
        "quality_control": {
            "brain_atlas_path": None,
            "brain_pca_path": None,
            "validate_brain_mask": False,
            "mask_outside_brain": False,
            "mask_outside_brain_dilate_pixels": 0,
            "diffrs_checkpoint": None,
        },
        "samples": [
            {
                "index": index,
                "patient_id": patient_id,
                "conditioning_mask": str(input_root / patient_id / "seg.nii.gz"),
                "seg_noise_seed": None,
                "image_noise_seed": 42 + index,
                "image_attempt": 0,
            }
            for index, patient_id in enumerate(patient_ids)
        ],
    }


def test_core_generation_manifest_validation_executes_and_fails_closed(
    tmp_path: Path, panel_manifest_module
):
    patient_ids = [f"Mets_{index:03d}" for index in range(105)]
    input_root = (tmp_path / "train").resolve()
    low_checkpoint = {
        "path": str((tmp_path / "run" / "checkpoint_latest.pt").resolve()),
        "sha256": "a" * 64,
    }
    manifest = _generation_manifest(input_root, patient_ids, low_checkpoint)
    path = tmp_path / "generation_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    validated = panel_manifest_module._validate_generation_manifest(
        path,
        input_root=input_root,
        input_ids=patient_ids,
        low_checkpoint=low_checkpoint,
        high_checkpoint=None,
        handoff_t=None,
    )
    assert validated["samples"][-1]["image_noise_seed"] == 146

    manifest["samples"][4]["image_noise_seed"] = 999
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Image noise seed differs at 4"):
        panel_manifest_module._validate_generation_manifest(
            path,
            input_root=input_root,
            input_ids=patient_ids,
            low_checkpoint=low_checkpoint,
            high_checkpoint=None,
            handoff_t=None,
        )


def test_metric_report_validation_requires_all_14_train_only_pools(
    tmp_path: Path, panel_manifest_module
):
    labels = [f"candidate_{index:02d}" for index in range(14)]
    datasets = {}
    for label in labels:
        dataset_path = tmp_path / label
        dataset_path.mkdir()
        manifest_path = dataset_path / "generation_manifest.json"
        manifest_path.write_text('{"status":"complete"}\n', encoding="utf-8")
        datasets[label] = {
            "source_path": str(dataset_path),
            "generation_manifest": {
                "path": str(manifest_path),
                "sha256": panel_manifest_module._sha256(manifest_path),
            },
            "total_found": 105,
            "pool_size": 105,
            "per_reference": {"train": {}},
        }
    report = {
        "config": {
            "references": {"train": 105},
            "pca_model": "none",
            "seed": 42,
            "pool_cap": 0,
            "source_git_commit": "abc123",
        },
        "datasets": datasets,
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    args = argparse.Namespace(
        report=str(path), expected_label=labels, expected_git_commit="abc123"
    )
    panel_manifest_module.validate_report(args)

    report["datasets"][labels[3]]["pool_size"] = 104
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="pool_size is not 105"):
        panel_manifest_module.validate_report(args)
