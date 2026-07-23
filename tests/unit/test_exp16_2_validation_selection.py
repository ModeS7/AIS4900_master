"""Focused contracts for best-checkpoint exp16_2 source selection."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import medgen.scripts.select_exp16_2_validation_source as selection

PROJECT_ROOT = Path(__file__).parents[2]
SLURM_DIR = PROJECT_ROOT / "IDUN/train/downstream/nnunet"
GPU_JOB = SLURM_DIR / "validate_best_exp16_2_synthetic_panel.slurm"
FINALIZER_JOB = SLURM_DIR / "finalize_exp16_2_validation_source.slurm"
IDENTITY = np.eye(4, dtype=np.float64)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _save_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, IDENTITY), str(path))


def _symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve())


def _summary_record(
    prediction: Path,
    reference: Path,
    *,
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, object]:
    return {
        "prediction_file": str(prediction),
        "reference_file": str(reference),
        "metrics": {"1": {"TP": tp, "FP": fp, "FN": fn}},
    }


def _verify_args(
    *,
    prediction_dir: Path,
    reference_dir: Path,
    splits: Path,
    checkpoint_best: Path,
    checkpoint_final: Path,
    output_marker: Path,
    condition: selection.Condition,
    fold: int,
    expected_folds: int,
    expected_cases: int,
    array_task_id: int,
    array_job_id: str = "123",
) -> argparse.Namespace:
    return argparse.Namespace(
        prediction_dir=prediction_dir,
        reference_dir=reference_dir,
        splits=splits,
        fold=fold,
        expected_folds=expected_folds,
        expected_cases=expected_cases,
        checkpoint_best=checkpoint_best,
        checkpoint_final=checkpoint_final,
        output_marker=output_marker,
        label=condition.label,
        experiment=condition.experiment,
        dataset_id=condition.dataset_id,
        trainer=selection.EXPECTED_TRAINER,
        plans=selection.EXPECTED_PLANS,
        configuration=selection.EXPECTED_CONFIGURATION,
        array_job_id=array_job_id,
        array_task_id=array_task_id,
        job_id=f"{array_job_id}_{array_task_id}",
        stdout=output_marker.parent / "job.out",
        stderr=output_marker.parent / "job.err",
    )


def test_split_validation_rejects_synthetic_leakage_and_ties_fail_closed(
    tmp_path: Path,
) -> None:
    splits = tmp_path / "splits_final.json"
    _write_json(
        splits,
        [
            {"train": ["BrainMetSyn_000"], "val": ["BrainMetSyn_000"]},
            {"train": [], "val": ["BrainMet_001"]},
        ],
    )

    with pytest.raises(ValueError, match="Synthetic IDs found"):
        selection._split_validation_ids(
            splits,
            0,
            expected_folds=2,
            expected_cases=1,
        )

    tied = [
        {"label": "a", "volumetric_dice": {"mean": 0.75}},
        {"label": "b", "volumetric_dice": {"mean": 0.75}},
    ]
    with pytest.raises(RuntimeError, match="Exact tie"):
        selection._select_unique_winner(tied)


def test_verify_fold_writes_atomic_best_checkpoint_marker_and_rejects_stale_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selection, "_git_state", lambda: {"commit": "test", "dirty": False})
    condition = selection.PANEL_CONDITIONS[0]
    case_ids = [f"BrainMet_{index:03d}" for index in range(21)]
    splits = tmp_path / "splits_final.json"
    _write_json(
        splits,
        [
            {"train": [], "val": case_ids},
            *({"train": [], "val": case_ids} for _ in range(4)),
        ],
    )

    shared_reference = tmp_path / "shared_reference.nii.gz"
    shared_prediction = tmp_path / "shared_prediction.nii.gz"
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[0, 0, 0] = 1
    _save_nifti(shared_reference, mask)
    _save_nifti(shared_prediction, mask)
    prediction_dir = tmp_path / "validation"
    reference_dir = tmp_path / "gt_segmentations"
    records = []
    for case_id in case_ids:
        prediction = prediction_dir / f"{case_id}.nii.gz"
        reference = reference_dir / f"{case_id}.nii.gz"
        _symlink(shared_prediction, prediction)
        _symlink(shared_reference, reference)
        records.append(_summary_record(prediction, reference, tp=1, fp=0, fn=0))
    _write_json(prediction_dir / "summary.json", {"metric_per_case": records})

    checkpoint_best = tmp_path / "checkpoint_best.pth"
    checkpoint_final = tmp_path / "checkpoint_final.pth"
    checkpoint_best.write_bytes(b"best")
    checkpoint_final.write_bytes(b"final")
    marker = tmp_path / ".best_validation_complete.json"
    args = _verify_args(
        prediction_dir=prediction_dir,
        reference_dir=reference_dir,
        splits=splits,
        checkpoint_best=checkpoint_best,
        checkpoint_final=checkpoint_final,
        output_marker=marker,
        condition=condition,
        fold=0,
        expected_folds=5,
        expected_cases=21,
        array_task_id=0,
    )

    selection._verify_fold(args)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["schema"] == selection.FOLD_MARKER_SCHEMA
    assert payload["checkpoint"]["name"] == "checkpoint_best.pth"
    assert payload["checkpoint"]["sha256"] == selection._sha256(checkpoint_best)
    assert payload["prediction"]["count"] == 21
    assert payload["split"]["validation_ids"] == sorted(case_ids)

    with pytest.raises(FileExistsError):
        selection._verify_fold(args)

    _symlink(shared_prediction, prediction_dir / "BrainMet_stale.nii.gz")
    stale_args = argparse.Namespace(
        **{**vars(args), "output_marker": tmp_path / ".stale_marker.json"}
    )
    with pytest.raises(ValueError, match="validation IDs do not match"):
        selection._verify_fold(stale_args)


def _controlled_plan() -> dict[str, object]:
    return {
        "plans_name": selection.EXPECTED_PLANS,
        "configurations": {
            selection.EXPECTED_CONFIGURATION: {
                "patch_size": [160, 192, 160],
                "batch_size": 3,
                "spacing": [1.0, 0.9375, 0.9375],
                "normalization_schemes": ["ZScoreNormalization"],
                "use_mask_for_norm": [True],
                "preprocessor_name": "DefaultPreprocessor",
                "batch_dice": False,
            }
        },
    }


def test_tiny_exact_seven_source_panel_locks_unique_validation_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all seven identities with one fold and one real validation case."""
    monkeypatch.setattr(selection, "EXPECTED_FOLDS", 1)
    monkeypatch.setattr(selection, "EXPECTED_CASES_PER_FOLD", 1)
    monkeypatch.setattr(selection, "_git_state", lambda: {"commit": "test", "dirty": False})

    real_id = "BrainMet_000"
    synthetic_ids = [f"BrainMetSyn_{index:03d}" for index in range(105)]
    raw_root = tmp_path / "nnUNet_raw"
    preprocessed_base = tmp_path / "nnUNet_preprocessed"
    validation_root = tmp_path / "job_123"
    output_dir = tmp_path / "selection" / "job_456"

    target = np.zeros((2, 2, 2), dtype=np.uint8)
    target[0, 0, 0] = 1
    empty = np.zeros_like(target)
    image = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    shared = tmp_path / "shared"
    target_file = shared / "target.nii.gz"
    empty_file = shared / "empty.nii.gz"
    image_file = shared / "image.nii.gz"
    _save_nifti(target_file, target)
    _save_nifti(empty_file, empty)
    _save_nifti(image_file, image)

    canonical = raw_root / "Dataset600_BrainMet"
    _write_json(canonical / "case_info.json", {"real_train_cases": [real_id]})
    _symlink(target_file, canonical / "labelsTr" / f"{real_id}.nii.gz")
    _symlink(image_file, canonical / "imagesTr" / f"{real_id}_0000.nii.gz")

    for source_index, condition in enumerate(selection.PANEL_CONDITIONS):
        source = raw_root / f"Dataset{condition.dataset_id}_BrainMet"
        _write_json(
            source / "case_info.json",
            {"real_train_cases": [real_id], "synthetic_cases": synthetic_ids},
        )
        _symlink(target_file, source / "labelsTr" / f"{real_id}.nii.gz")
        _symlink(image_file, source / "imagesTr" / f"{real_id}_0000.nii.gz")
        for synthetic_id in synthetic_ids:
            _symlink(target_file, source / "labelsTr" / f"{synthetic_id}.nii.gz")
            _symlink(image_file, source / "imagesTr" / f"{synthetic_id}_0000.nii.gz")

        isolated = Path(f"{preprocessed_base}_{condition.experiment}")
        preprocessed = isolated / f"Dataset{condition.dataset_id}_BrainMet"
        plan_path = preprocessed / f"{selection.EXPECTED_PLANS}.json"
        _write_json(plan_path, _controlled_plan())
        marker_text = "\n".join(
            [
                f"dataset_id={condition.dataset_id}",
                "source_dataset_id=600",
                f"target_plans={selection.EXPECTED_PLANS}",
                f"target_plan_sha256={selection._sha256(plan_path)}",
            ]
        )
        (preprocessed / ".exp16_2_d600_preprocess_complete").write_text(
            marker_text + "\n",
            encoding="utf-8",
        )
        splits = preprocessed / "splits_final.json"
        _write_json(splits, [{"train": synthetic_ids, "val": [real_id]}])
        reference_dir = preprocessed / "gt_segmentations"
        _symlink(target_file, reference_dir / f"{real_id}.nii.gz")

        task_root = validation_root / condition.experiment / "fold_0"
        prediction_dir = task_root / "validation"
        prediction_source = target_file if source_index == 0 else empty_file
        prediction = prediction_dir / f"{real_id}.nii.gz"
        _symlink(prediction_source, prediction)
        tp, fn = (1, 0) if source_index == 0 else (0, 1)
        _write_json(
            prediction_dir / "summary.json",
            {
                "metric_per_case": [
                    _summary_record(
                        prediction,
                        reference_dir / f"{real_id}.nii.gz",
                        tp=tp,
                        fp=0,
                        fn=fn,
                    )
                ]
            },
        )
        checkpoint_best = task_root / "checkpoint_best.pth"
        checkpoint_final = task_root / "checkpoint_final.pth"
        checkpoint_best.write_bytes(f"best-{source_index}".encode())
        checkpoint_final.write_bytes(f"final-{source_index}".encode())
        verify_args = _verify_args(
            prediction_dir=prediction_dir,
            reference_dir=reference_dir,
            splits=splits,
            checkpoint_best=checkpoint_best,
            checkpoint_final=checkpoint_final,
            output_marker=task_root / ".best_validation_complete.json",
            condition=condition,
            fold=0,
            expected_folds=1,
            expected_cases=1,
            array_task_id=source_index,
        )
        selection._verify_fold(verify_args)

    finalize_args = argparse.Namespace(
        condition=list(selection.PANEL_CONDITIONS),
        validation_root=validation_root,
        nnunet_raw=raw_root,
        nnunet_preprocessed=preprocessed_base,
        output_dir=output_dir,
        canonical_dataset_id=600,
        expected_folds=1,
        expected_cases_per_fold=1,
    )
    selection._finalize(finalize_args)

    result = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    assert result["winner"] == {
        "label": "original_mse",
        "experiment": selection.PANEL_CONDITIONS[0].experiment,
        "dataset_id": 650,
        "n_cases": 1,
        "mean_volumetric_dice": 1.0,
    }
    assert len(result["conditions"]) == 7
    assert result["selection_rule"]["checkpoint"] == "checkpoint_best.pth"
    assert result["selection_rule"]["official_test_used"] is False
    assert (output_dir / "validation_summary.csv").read_text().count("\n") == 8
    assert "label=original_mse" in (output_dir / "LOCKED_SOURCE.txt").read_text()
    complete = json.loads((output_dir / selection.COMPLETE_MARKER).read_text(encoding="utf-8"))
    assert complete["n_conditions"] == 7
    assert set(complete["artifacts_sha256"]) == {
        "validation_summary.csv",
        "validation_per_case.csv",
        "selection.json",
        "LOCKED_SOURCE.txt",
    }


def test_best_checkpoint_validation_slurm_is_isolated_and_exact() -> None:
    text = GPU_JOB.read_text(encoding="utf-8")
    result = subprocess.run(
        ["bash", "-n", str(GPU_JOB)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "#SBATCH --partition=GPUQ" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert '#SBATCH --constraint="a100|h100|gpu80g"' in text
    assert "#SBATCH --array=0-34" in text
    assert "readonly SOURCE_INDEX=$((TASK_ID / 5))" in text
    assert "readonly FOLD=$((TASK_ID % 5))" in text
    assert "--val \\\n    --val_best" in text
    assert 'export nnUNet_results="$STAGED_RESULTS"' in text
    assert '[[ ! -e "$TASK_ROOT" ]]' in text
    assert "$CANONICAL_FOLD_DIR/checkpoint_best.pth" in text
    assert "$CANONICAL_FOLD_DIR/checkpoint_final.pth" in text
    assert "select_exp16_2_validation_source verify-fold" in text
    assert text.index("--val_best") < text.index(" verify-fold")
    assert "labelsTs" not in text
    assert "imagesTs" not in text
    assert "sbatch " not in text

    for condition in selection.PANEL_CONDITIONS:
        assert condition.label in text
        generator = condition.experiment.removeprefix(
            "exp16_2_synthetic_105_common105_"
        ).removesuffix(f"_d{condition.dataset_id}")
        assert generator in text
        assert str(condition.dataset_id) in text


def test_validation_finalizer_is_a_separate_exact_cpu_job() -> None:
    text = FINALIZER_JOB.read_text(encoding="utf-8")
    result = subprocess.run(
        ["bash", "-n", str(FINALIZER_JOB)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "#SBATCH --partition=CPUQ" in text
    assert "#SBATCH --gres" not in text
    assert "nvidia-smi" not in text
    assert "nnUNetv2_train" not in text
    assert "VALIDATION_ARRAY_JOB_ID:?" in text
    assert "select_exp16_2_validation_source finalize" in text
    assert "--canonical-dataset-id 600" in text
    assert "--expected-folds 5" in text
    assert "--expected-cases-per-fold 21" in text
    assert "labelsTs" not in text
    assert "imagesTs" not in text
    assert "sbatch " not in text
    assert text.count("--condition ") == 7
    for condition in selection.PANEL_CONDITIONS:
        specification = f"{condition.label}|{condition.experiment}|{condition.dataset_id}"
        assert specification in text
