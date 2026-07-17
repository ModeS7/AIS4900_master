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


def _generation_manifest(
    input_root: Path,
    patient_ids: list[str],
    low_checkpoint: dict,
    *,
    git_commit: str = "standalone-commit",
):
    return {
        "schema_version": 1,
        "status": "complete",
        "mode": "bravo",
        "spatial_dims": 3,
        "seed": 42,
        "num_images": 105,
        "git_commit": git_commit,
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


def test_recover_handoff_validates_failed_checkout_transition_and_publishes(
    tmp_path: Path, panel_manifest_module
):
    label = "exp1_to_exp48a_t025"
    job_id = "24795269"
    observed_commit = next(iter(panel_manifest_module.RECOVERABLE_MANIFEST_WRITE_COMMITS))
    patient_ids = [f"Mets_{index:03d}" for index in range(105)]

    input_root = tmp_path / "train"
    for patient_id in patient_ids:
        patient_root = input_root / patient_id
        patient_root.mkdir(parents=True)
        (patient_root / "seg.nii.gz").write_bytes(b"seg")
        (patient_root / "bravo.nii.gz").write_bytes(b"real-bravo")

    staging_root = tmp_path / ".staging" / f"{label}.job-{job_id}"
    for index in range(105):
        sample_root = staging_root / f"{index:05d}"
        sample_root.mkdir(parents=True)
        (sample_root / "seg.nii.gz").write_bytes(b"seg")
        (sample_root / "bravo.nii.gz").write_bytes(b"generated-bravo")
    final_root = tmp_path / "panel" / label

    low_path = tmp_path / "exp48a_lowt_only_lpips_strong_20260425-160342" / "checkpoint_latest.pt"
    high_path = tmp_path / "exp1_1_1000_pixel_bravo_20260402-121556" / "checkpoint_latest.pt"
    low_path.parent.mkdir()
    high_path.parent.mkdir()
    low_path.write_bytes(b"low")
    high_path.write_bytes(b"high")
    low_checkpoint = {
        "path": str(low_path.resolve()),
        "sha256": panel_manifest_module._sha256(low_path),
    }
    high_checkpoint = {
        "path": str(high_path.resolve()),
        "sha256": panel_manifest_module._sha256(high_path),
    }
    generation_manifest = _generation_manifest(
        input_root.resolve(),
        patient_ids,
        low_checkpoint,
        git_commit=observed_commit,
    )
    generation_manifest["models"]["image_high_t"] = high_checkpoint
    generation_manifest["sampling"]["handoff_t"] = 0.25
    manifest_path = staging_root / "generation_manifest.json"
    manifest_path.write_text(
        json.dumps(generation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    original_manifest_sha256 = panel_manifest_module._sha256(manifest_path)

    failure_log = tmp_path / f"gen_exp1_to_exp48a_{job_id}.err"
    failure_log.write_text(
        "FATAL: repository commit changed after job submission: "
        f"expected {panel_manifest_module.HANDOFF_SOURCE_COMMIT}, found {observed_commit}\n",
        encoding="utf-8",
    )
    output_log = tmp_path / f"gen_exp1_to_exp48a_{job_id}.out"
    output_log.write_text(
        f"staging output:    {staging_root.resolve()}\n"
        f"Saved 105 samples to {staging_root.resolve()}\n"
        "Generation complete!\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        label=label,
        slurm_job_id=job_id,
        failure_log=str(failure_log),
        output_log=str(output_log),
        observed_manifest_write_commit=observed_commit,
        input_root=str(input_root),
        dataset_root=str(staging_root),
        final_dataset_root=str(final_root),
        low_checkpoint=str(low_path),
        high_checkpoint=str(high_path),
        handoff_t=0.25,
    )

    panel_manifest_module.recover_handoff(args)

    assert final_root.is_dir()
    assert not staging_root.exists()
    recovered = json.loads((final_root / "generation_manifest.json").read_text())
    assert recovered["git_commit"] == panel_manifest_module.HANDOFF_SOURCE_COMMIT
    recovery = recovered["provenance_recovery"]
    assert recovery["observed_manifest_write_commit"] == observed_commit
    assert recovery["original_generation_manifest_sha256"] == original_manifest_sha256
    assert recovery["slurm_job_id"] == job_id
    panel_job = json.loads((final_root / "panel_job_manifest.json").read_text())
    assert panel_job["runtime"]["git_commit"] == panel_manifest_module.HANDOFF_SOURCE_COMMIT
    assert panel_job["runtime"]["slurm_job_id"] == job_id
    assert panel_job["dataset_root"] == str(final_root.resolve())

    # Repeating the same validated request is safe and does not rewrite data.
    panel_manifest_module.recover_handoff(args)
    assert final_root.is_dir()
    assert not staging_root.exists()


def test_metric_report_validation_requires_all_14_train_only_pools(
    tmp_path: Path, panel_manifest_module
):
    labels = list(panel_manifest_module.PANEL_LABELS)
    datasets = {}
    for label in labels:
        dataset_path = tmp_path / label
        dataset_path.mkdir()
        manifest_path = dataset_path / "generation_manifest.json"
        if label == panel_manifest_module.HANDOFF_LABELS[0]:
            generation_commit = panel_manifest_module.HANDOFF_SOURCE_COMMIT
        elif label in panel_manifest_module.HANDOFF_LABELS:
            generation_commit = panel_manifest_module.POST_CHANGE_SOURCE_COMMIT
        else:
            generation_commit = panel_manifest_module.STANDALONE_SOURCE_COMMIT
        manifest_path.write_text(
            json.dumps({"status": "complete", "git_commit": generation_commit}) + "\n",
            encoding="utf-8",
        )
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
    args = argparse.Namespace(report=str(path), expected_label=labels, expected_git_commit="abc123")
    panel_manifest_module.validate_report(args)

    handoff_label = panel_manifest_module.HANDOFF_LABELS[0]
    handoff_manifest = Path(datasets[handoff_label]["generation_manifest"]["path"])
    handoff_manifest.write_text(
        json.dumps({"status": "complete", "git_commit": "wrong-commit"}) + "\n",
        encoding="utf-8",
    )
    datasets[handoff_label]["generation_manifest"]["sha256"] = panel_manifest_module._sha256(
        handoff_manifest
    )
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(
        ValueError, match="handoff generation commit is not an audited panel source"
    ):
        panel_manifest_module.validate_report(args)

    handoff_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "git_commit": panel_manifest_module.HANDOFF_SOURCE_COMMIT,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    datasets[handoff_label]["generation_manifest"]["sha256"] = panel_manifest_module._sha256(
        handoff_manifest
    )

    report["datasets"][labels[3]]["pool_size"] = 104
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="pool_size is not 105"):
        panel_manifest_module.validate_report(args)
