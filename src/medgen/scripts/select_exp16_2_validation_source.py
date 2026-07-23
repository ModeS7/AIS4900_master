"""Verify best-checkpoint fold validation and lock the exp16_2 hybrid source.

The two subcommands deliberately operate only on the 105 real training cases.
They never read the official test directories. ``verify-fold`` publishes an
atomic provenance marker after one isolated nnU-Net ``--val --val_best`` run.
``finalize`` then pools five disjoint 21-case folds for every source and ranks
the sources by the exact paper endpoint: mean patient-level 3D foreground
Dice, computed from the saved masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from medgen.metrics.paper_segmentation import sample_summary, volumetric_dice
from medgen.scripts.recompute_nnunet_metrics import (
    _load_binary,
    _mask_sha256,
    _validate_geometry,
)

FOLD_MARKER_SCHEMA = "exp16_2_best_validation_fold_v1"
SELECTION_SCHEMA = "exp16_2_validation_source_selection_v1"
COMPLETE_MARKER = ".exp16_2_validation_selection_complete.json"
EXPECTED_TRAINER = "nnUNetTrainerBrainMets"
EXPECTED_PLANS = "nnUNetResEncUNetLPlansD600"
EXPECTED_CONFIGURATION = "3d_fullres"
EXPECTED_FOLDS = 5
EXPECTED_CASES_PER_FOLD = 21


@dataclass(frozen=True)
class Condition:
    """One synthetic source participating in validation selection."""

    label: str
    experiment: str
    dataset_id: int


PANEL_CONDITIONS = (
    Condition(
        "original_mse",
        "exp16_2_synthetic_105_common105_exp1_1_1000_d650",
        650,
    ),
    Condition(
        "extended_mse",
        "exp16_2_synthetic_105_common105_exp1_1_1000plus_d651",
        651,
    ),
    Condition(
        "perceptual_continuation",
        "exp16_2_synthetic_105_common105_exp32_2_1000_d652",
        652,
    ),
    Condition(
        "strong_perceptual_continuation",
        "exp16_2_synthetic_105_common105_exp47a_d654",
        654,
    ),
    Condition(
        "weighted_huber_transition",
        "exp16_2_synthetic_105_common105_exp47c_d656",
        656,
    ),
    Condition(
        "weighted_huber_handoff",
        "exp16_2_synthetic_105_common105_exp1_to_exp48c_t025_d661",
        661,
    ),
    Condition(
        "pseudo_huber_perceptual_handoff",
        "exp16_2_synthetic_105_common105_exp1_to_exp48d_t025_d662",
        662,
    ),
)


def _parse_condition(value: str) -> Condition:
    parts = value.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"condition must be LABEL|EXPERIMENT|DATASET_ID, got {value!r}"
        )
    label, experiment, dataset_text = parts
    if not label or not experiment:
        raise argparse.ArgumentTypeError("condition label and experiment cannot be empty")
    try:
        dataset_id = int(dataset_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"dataset ID must be an integer, got {dataset_text!r}"
        ) from error
    return Condition(label=label, experiment=experiment, dataset_id=dataset_id)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(case_ids: list[str] | tuple[str, ...]) -> str:
    payload = "".join(f"{case_id}\n" for case_id in sorted(case_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_json_atomic(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        _write_json(temporary, payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_state() -> dict[str, str | bool | None]:
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _dataset_dir(root: Path, dataset_id: int) -> Path:
    matches = sorted(path for path in root.glob(f"Dataset{dataset_id:03d}_*") if path.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one Dataset{dataset_id:03d}_* under {root}, found {matches}"
        )
    return matches[0]


def _nifti_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    suffix = ".nii.gz"
    return {path.name.removesuffix(suffix): path for path in directory.glob(f"*{suffix}")}


def _modality_index(directory: Path, modality: int = 0) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    suffix = f"_{modality:04d}.nii.gz"
    return {path.name.removesuffix(suffix): path for path in directory.glob(f"*{suffix}")}


def _image_fingerprint(path: Path) -> str:
    image = nib.load(path)
    data = np.ascontiguousarray(image.get_fdata(dtype=np.float32))
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite image values in {path}")
    digest = hashlib.sha256()
    digest.update(np.asarray(data.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(image.affine, dtype=np.float64).tobytes())
    digest.update(np.asarray(image.header.get_zooms()[:3], dtype=np.float64).tobytes())
    digest.update(data.tobytes())
    return digest.hexdigest()


def _panel_index(condition: Condition) -> int:
    try:
        return PANEL_CONDITIONS.index(condition)
    except ValueError as error:
        raise ValueError(f"Condition is not in the fixed exp16_2 panel: {condition}") from error


def _require_fixed_runtime_contract(args: argparse.Namespace) -> None:
    if args.expected_folds != EXPECTED_FOLDS:
        raise ValueError(f"Expected exactly {EXPECTED_FOLDS} folds")
    expected_cases = getattr(args, "expected_cases", EXPECTED_CASES_PER_FOLD)
    expected_per_fold = getattr(
        args,
        "expected_cases_per_fold",
        EXPECTED_CASES_PER_FOLD,
    )
    if expected_cases != EXPECTED_CASES_PER_FOLD:
        raise ValueError(f"Expected exactly {EXPECTED_CASES_PER_FOLD} cases per fold")
    if expected_per_fold != EXPECTED_CASES_PER_FOLD:
        raise ValueError(f"Expected exactly {EXPECTED_CASES_PER_FOLD} cases per fold")
    canonical_dataset_id = getattr(args, "canonical_dataset_id", 600)
    if canonical_dataset_id != 600:
        raise ValueError("Dataset600 is the fixed real validation reference")


def _split_validation_ids(
    splits_path: Path,
    fold: int,
    *,
    expected_folds: int,
    expected_cases: int,
) -> list[str]:
    splits = _read_json(splits_path)
    if not isinstance(splits, list) or len(splits) != expected_folds:
        raise ValueError(
            f"{splits_path}: expected {expected_folds} folds, got "
            f"{len(splits) if isinstance(splits, list) else type(splits).__name__}"
        )
    if not 0 <= fold < expected_folds:
        raise ValueError(f"fold {fold} is outside 0--{expected_folds - 1}")
    values = splits[fold].get("val") if isinstance(splits[fold], dict) else None
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{splits_path}: fold {fold} has an invalid validation list")
    case_ids = sorted(values)
    if len(case_ids) != expected_cases or len(set(case_ids)) != expected_cases:
        raise ValueError(
            f"{splits_path}: fold {fold} must contain {expected_cases} unique cases, "
            f"got {len(case_ids)} entries and {len(set(case_ids))} unique IDs"
        )
    synthetic_ids = [case_id for case_id in case_ids if case_id.startswith("BrainMetSyn_")]
    if synthetic_ids:
        raise ValueError(f"Synthetic IDs found in real validation fold {fold}: {synthetic_ids}")
    return case_ids


def _split_training_ids(
    splits_path: Path,
    fold: int,
    *,
    expected_folds: int,
    expected_cases: int,
) -> list[str]:
    splits = _read_json(splits_path)
    if not isinstance(splits, list) or len(splits) != expected_folds:
        raise ValueError(f"{splits_path}: expected {expected_folds} folds")
    values = splits[fold].get("train") if isinstance(splits[fold], dict) else None
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{splits_path}: fold {fold} has an invalid training list")
    case_ids = sorted(values)
    if len(case_ids) != expected_cases or len(set(case_ids)) != expected_cases:
        raise ValueError(
            f"{splits_path}: fold {fold} must contain {expected_cases} unique training cases"
        )
    real_ids = [case_id for case_id in case_ids if not case_id.startswith("BrainMetSyn_")]
    if real_ids:
        raise ValueError(f"Real IDs found in synthetic-only training fold {fold}: {real_ids}")
    return case_ids


def _summary_case_records(summary_path: Path) -> dict[str, dict[str, Any]]:
    summary = _read_json(summary_path)
    records = summary.get("metric_per_case") if isinstance(summary, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"{summary_path}: missing metric_per_case list")

    parsed_records: dict[str, dict[str, Any]] = {}
    for record in records:
        try:
            prediction_file = Path(record["prediction_file"])
            reference_file = Path(record["reference_file"])
            class_metrics = record["metrics"]["1"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{summary_path}: malformed metric_per_case record") from error
        case_id = prediction_file.name.removesuffix(".nii.gz")
        if case_id in parsed_records:
            raise ValueError(f"{summary_path}: duplicate case {case_id}")
        parsed: dict[str, Any] = {
            "prediction_file": str(prediction_file.resolve()),
            "reference_file": str(reference_file.resolve()),
        }
        for key in ("TP", "FP", "FN"):
            value = class_metrics.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{summary_path}: {case_id} has invalid {key}={value!r}")
            integer = int(value)
            if integer < 0 or float(integer) != float(value):
                raise ValueError(f"{summary_path}: {case_id} has invalid {key}={value!r}")
            parsed[key] = integer
        parsed_records[case_id] = parsed
    return parsed_records


def _verify_fold(args: argparse.Namespace) -> None:
    _require_fixed_runtime_contract(args)
    condition = Condition(args.label, args.experiment, args.dataset_id)
    source_index = _panel_index(condition)
    expected_task_id = source_index * EXPECTED_FOLDS + args.fold
    if args.array_task_id != expected_task_id:
        raise ValueError(
            f"Array task {args.array_task_id} does not own {condition.label} fold "
            f"{args.fold}; expected task {expected_task_id}"
        )
    if args.trainer != EXPECTED_TRAINER:
        raise ValueError(f"Unexpected trainer: {args.trainer}")
    if args.plans != EXPECTED_PLANS:
        raise ValueError(f"Unexpected plans: {args.plans}")
    if args.configuration != EXPECTED_CONFIGURATION:
        raise ValueError(f"Unexpected configuration: {args.configuration}")
    if not str(args.array_job_id).isdigit():
        raise ValueError(f"Array job ID must be numeric: {args.array_job_id}")

    expected_ids = _split_validation_ids(
        args.splits,
        args.fold,
        expected_folds=args.expected_folds,
        expected_cases=args.expected_cases,
    )
    predictions = _nifti_index(args.prediction_dir)
    prediction_ids = sorted(predictions)
    if prediction_ids != expected_ids:
        raise ValueError(
            "Best-checkpoint validation IDs do not match the fixed fold: "
            f"missing={sorted(set(expected_ids) - set(prediction_ids))}, "
            f"extra={sorted(set(prediction_ids) - set(expected_ids))}"
        )

    summary_path = args.prediction_dir / "summary.json"
    summary_records = _summary_case_records(summary_path)
    if sorted(summary_records) != expected_ids:
        raise ValueError(f"{summary_path}: metric_per_case IDs do not match fold {args.fold}")
    for case_id in expected_ids:
        expected_prediction = str(predictions[case_id].resolve())
        expected_reference = str((args.reference_dir / f"{case_id}.nii.gz").resolve())
        record = summary_records[case_id]
        if record["prediction_file"] != expected_prediction:
            raise ValueError(
                f"{summary_path}: prediction path differs for {case_id}: "
                f"{record['prediction_file']}"
            )
        if record["reference_file"] != expected_reference:
            raise ValueError(
                f"{summary_path}: reference path differs for {case_id}: {record['reference_file']}"
            )
        if not (args.reference_dir / f"{case_id}.nii.gz").is_file():
            raise FileNotFoundError(args.reference_dir / f"{case_id}.nii.gz")
    if not args.checkpoint_best.is_file() or not args.checkpoint_final.is_file():
        raise FileNotFoundError(
            "Both checkpoint_best.pth and checkpoint_final.pth are required for --val_best"
        )

    payload = {
        "schema": FOLD_MARKER_SCHEMA,
        "created_by": "medgen.scripts.select_exp16_2_validation_source verify-fold",
        "host": platform.node(),
        "python": platform.python_version(),
        "git": _git_state(),
        "slurm": {
            "array_job_id": str(args.array_job_id),
            "array_task_id": int(args.array_task_id),
            "job_id": str(args.job_id),
            "stdout": str(args.stdout.resolve()),
            "stderr": str(args.stderr.resolve()),
        },
        "condition": {
            "label": args.label,
            "experiment": args.experiment,
            "dataset_id": args.dataset_id,
            "fold": args.fold,
            "trainer": args.trainer,
            "plans": args.plans,
            "configuration": args.configuration,
        },
        "checkpoint": {
            "name": "checkpoint_best.pth",
            "path": str(args.checkpoint_best.resolve()),
            "sha256": _sha256(args.checkpoint_best),
            "checkpoint_final_path": str(args.checkpoint_final.resolve()),
            "checkpoint_final_sha256": _sha256(args.checkpoint_final),
        },
        "split": {
            "path": str(args.splits.resolve()),
            "sha256": _sha256(args.splits),
            "validation_ids": expected_ids,
            "validation_ids_sha256": _ids_sha256(expected_ids),
        },
        "prediction": {
            "directory": str(args.prediction_dir.resolve()),
            "count": len(prediction_ids),
            "ids_sha256": _ids_sha256(prediction_ids),
            "file_sha256": {case_id: _sha256(predictions[case_id]) for case_id in prediction_ids},
            "reference_directory": str(args.reference_dir.resolve()),
            "reference_file_sha256": {
                case_id: _sha256(args.reference_dir / f"{case_id}.nii.gz")
                for case_id in prediction_ids
            },
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
        },
    }
    _write_json_atomic(args.output_marker, payload)
    print(f"Verified {args.label} fold {args.fold}: {len(expected_ids)} best-checkpoint cases")
    print(f"Marker: {args.output_marker}")


def _condition_task_root(validation_root: Path, condition: Condition, fold: int) -> Path:
    return validation_root / condition.experiment / f"fold_{fold}"


def _read_fold_marker(task_root: Path) -> dict[str, Any]:
    marker_path = task_root / ".best_validation_complete.json"
    marker = _read_json(marker_path)
    if marker.get("schema") != FOLD_MARKER_SCHEMA:
        raise ValueError(f"{marker_path}: wrong or missing marker schema")
    return marker


def _validate_marker_identity(
    marker: dict[str, Any],
    condition: Condition,
    fold: int,
    *,
    expected_cases: int,
) -> None:
    identity = marker.get("condition", {})
    expected = {
        "label": condition.label,
        "experiment": condition.experiment,
        "dataset_id": condition.dataset_id,
        "fold": fold,
        "trainer": EXPECTED_TRAINER,
        "plans": EXPECTED_PLANS,
        "configuration": EXPECTED_CONFIGURATION,
    }
    actual = {key: identity.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Fold marker identity mismatch: expected={expected}, actual={actual}")
    if marker.get("checkpoint", {}).get("name") != "checkpoint_best.pth":
        raise ValueError(f"{condition.label} fold {fold} was not validated from checkpoint_best")
    if marker.get("prediction", {}).get("count") != expected_cases:
        raise ValueError(f"{condition.label} fold {fold} has the wrong prediction count")
    expected_task_id = _panel_index(condition) * EXPECTED_FOLDS + fold
    if marker.get("slurm", {}).get("array_task_id") != expected_task_id:
        raise ValueError(f"{condition.label} fold {fold} has the wrong array task identity")


def _load_case_info(dataset_dir: Path) -> dict[str, Any]:
    payload = _read_json(dataset_dir / "case_info.json")
    real_cases = payload.get("real_train_cases")
    if not isinstance(real_cases, list) or not all(isinstance(value, str) for value in real_cases):
        raise ValueError(f"{dataset_dir}/case_info.json has invalid real_train_cases")
    synthetic_cases = payload.get("synthetic_cases", [])
    if not isinstance(synthetic_cases, list) or not all(
        isinstance(value, str) for value in synthetic_cases
    ):
        raise ValueError(f"{dataset_dir}/case_info.json has invalid synthetic_cases")
    return payload


def _load_canonical_targets(
    canonical_dataset_dir: Path,
    real_case_ids: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[float, float, float], str, str]]:
    labels = _nifti_index(canonical_dataset_dir / "labelsTr")
    missing = sorted(set(real_case_ids) - set(labels))
    if missing:
        raise ValueError(f"Canonical labelsTr is missing real cases: {missing}")

    targets = {}
    for case_id in real_case_ids:
        path = labels[case_id]
        mask, affine, spacing = _load_binary(path)
        targets[case_id] = (
            mask,
            affine,
            spacing,
            _mask_sha256(mask, affine),
            _sha256(path),
        )
    return targets


def _validate_controlled_preprocessing(
    dataset_dir: Path,
    condition: Condition,
) -> dict[str, str]:
    plan_path = dataset_dir / f"{EXPECTED_PLANS}.json"
    marker_path = dataset_dir / ".exp16_2_d600_preprocess_complete"
    plan = _read_json(plan_path)
    configuration = plan.get("configurations", {}).get(EXPECTED_CONFIGURATION, {})
    required = {
        "patch_size": [160, 192, 160],
        "batch_size": 3,
        "spacing": [1.0, 0.9375, 0.9375],
        "normalization_schemes": ["ZScoreNormalization"],
        "use_mask_for_norm": [True],
        "preprocessor_name": "DefaultPreprocessor",
        "batch_dice": False,
    }
    if plan.get("plans_name") != EXPECTED_PLANS:
        raise ValueError(f"{condition.label}: controlled plans_name differs")
    for key, expected in required.items():
        if configuration.get(key) != expected:
            raise ValueError(
                f"{condition.label}: controlled plan {key} differs: {configuration.get(key)!r}"
            )
    marker_lines = marker_path.read_text(encoding="utf-8").splitlines()
    marker = dict(line.split("=", 1) for line in marker_lines if "=" in line)
    marker_expected = {
        "dataset_id": str(condition.dataset_id),
        "source_dataset_id": "600",
        "target_plans": EXPECTED_PLANS,
    }
    for key, expected in marker_expected.items():
        if marker.get(key) != expected:
            raise ValueError(
                f"{condition.label}: preprocess marker {key}={marker.get(key)!r}, "
                f"expected {expected!r}"
            )
    if marker.get("target_plan_sha256") != _sha256(plan_path):
        raise ValueError(f"{condition.label}: controlled plan hash differs from marker")
    return {
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "preprocess_marker_path": str(marker_path.resolve()),
        "preprocess_marker_sha256": _sha256(marker_path),
    }


def _validate_partition(
    splits_path: Path,
    canonical_ids: list[str],
    synthetic_ids: list[str],
    *,
    expected_folds: int,
    expected_cases_per_fold: int,
) -> list[list[str]]:
    fold_ids = [
        _split_validation_ids(
            splits_path,
            fold,
            expected_folds=expected_folds,
            expected_cases=expected_cases_per_fold,
        )
        for fold in range(expected_folds)
    ]
    flattened = [case_id for values in fold_ids for case_id in values]
    duplicates = sorted(case_id for case_id, count in Counter(flattened).items() if count != 1)
    if duplicates:
        raise ValueError(f"Validation folds are not disjoint: {duplicates}")
    if sorted(flattened) != sorted(canonical_ids):
        raise ValueError(
            "Validation-fold union does not equal the canonical 105 real cases: "
            f"missing={sorted(set(canonical_ids) - set(flattened))}, "
            f"extra={sorted(set(flattened) - set(canonical_ids))}"
        )
    for fold in range(expected_folds):
        training_ids = _split_training_ids(
            splits_path,
            fold,
            expected_folds=expected_folds,
            expected_cases=len(synthetic_ids),
        )
        if training_ids != sorted(synthetic_ids):
            raise ValueError(
                f"{splits_path}: fold {fold} does not use the fixed common-105 "
                "synthetic training cohort"
            )
    return fold_ids


def _select_unique_winner(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("No condition summaries were supplied")
    best_mean = max(float(summary["volumetric_dice"]["mean"]) for summary in summaries)
    winners = [
        summary for summary in summaries if float(summary["volumetric_dice"]["mean"]) == best_mean
    ]
    if len(winners) != 1:
        labels = [summary["label"] for summary in winners]
        raise RuntimeError(
            "Exact tie in the prespecified validation endpoint; refusing to choose "
            f"post hoc between {labels}"
        )
    return winners[0]


def _finalize(args: argparse.Namespace) -> None:
    _require_fixed_runtime_contract(args)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if tuple(args.condition) != PANEL_CONDITIONS:
        raise ValueError("Finalization requires the exact ordered seven-source exp16_2 panel")

    canonical_dataset_dir = _dataset_dir(args.nnunet_raw, args.canonical_dataset_id)
    canonical_info = _load_case_info(canonical_dataset_dir)
    canonical_ids = sorted(canonical_info["real_train_cases"])
    expected_total = args.expected_folds * args.expected_cases_per_fold
    if len(canonical_ids) != expected_total or len(set(canonical_ids)) != expected_total:
        raise ValueError(
            f"Canonical cohort must contain {expected_total} unique real cases, "
            f"got {len(canonical_ids)}"
        )
    canonical_targets = _load_canonical_targets(canonical_dataset_dir, canonical_ids)
    canonical_images = _modality_index(canonical_dataset_dir / "imagesTr")
    missing_canonical_images = sorted(set(canonical_ids) - set(canonical_images))
    if missing_canonical_images:
        raise ValueError(f"Dataset600 imagesTr is missing real cases: {missing_canonical_images}")
    canonical_image_fingerprints = {
        case_id: _image_fingerprint(canonical_images[case_id]) for case_id in canonical_ids
    }

    reference_partition: list[list[str]] | None = None
    reference_synthetic_ids: list[str] | None = None
    reference_synthetic_masks: dict[str, str] | None = None
    per_case_rows: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []
    array_job_ids: set[str] = set()
    array_task_ids: set[int] = set()

    for condition in args.condition:
        source_dataset_dir = _dataset_dir(args.nnunet_raw, condition.dataset_id)
        source_info = _load_case_info(source_dataset_dir)
        if sorted(source_info["real_train_cases"]) != canonical_ids:
            raise ValueError(f"{condition.label}: real training cohort differs from Dataset600")
        synthetic_ids = sorted(source_info["synthetic_cases"])
        if len(synthetic_ids) != 105 or len(set(synthetic_ids)) != 105:
            raise ValueError(f"{condition.label}: expected 105 unique synthetic cases")
        if reference_synthetic_ids is None:
            reference_synthetic_ids = synthetic_ids
        elif synthetic_ids != reference_synthetic_ids:
            raise ValueError(f"{condition.label}: common-105 synthetic cohort differs")

        source_labels = _nifti_index(source_dataset_dir / "labelsTr")
        expected_source_ids = set(canonical_ids) | set(synthetic_ids)
        if set(source_labels) != expected_source_ids:
            raise ValueError(
                f"{condition.label}: labelsTr IDs differ: "
                f"missing={sorted(expected_source_ids - set(source_labels))}, "
                f"extra={sorted(set(source_labels) - expected_source_ids)}"
            )
        source_images = _modality_index(source_dataset_dir / "imagesTr")
        if set(source_images) != expected_source_ids:
            raise ValueError(
                f"{condition.label}: imagesTr IDs differ: "
                f"missing={sorted(expected_source_ids - set(source_images))}, "
                f"extra={sorted(set(source_images) - expected_source_ids)}"
            )
        for case_id in canonical_ids:
            if _image_fingerprint(source_images[case_id]) != canonical_image_fingerprints[case_id]:
                raise ValueError(f"{condition.label}: real validation BRAVO differs for {case_id}")

        current_synthetic_masks: dict[str, str] = {}
        for case_id in synthetic_ids:
            mask, affine, _ = _load_binary(source_labels[case_id])
            current_synthetic_masks[case_id] = _mask_sha256(mask, affine)
        if reference_synthetic_masks is None:
            reference_synthetic_masks = current_synthetic_masks
        elif current_synthetic_masks != reference_synthetic_masks:
            raise ValueError(f"{condition.label}: common-105 segmentation masks differ")

        isolated_root = Path(f"{args.nnunet_preprocessed}_{condition.experiment}")
        isolated_dataset_dir = _dataset_dir(isolated_root, condition.dataset_id)
        preprocessing = _validate_controlled_preprocessing(isolated_dataset_dir, condition)
        splits_path = isolated_dataset_dir / "splits_final.json"
        fold_ids = _validate_partition(
            splits_path,
            canonical_ids,
            synthetic_ids,
            expected_folds=args.expected_folds,
            expected_cases_per_fold=args.expected_cases_per_fold,
        )
        if reference_partition is None:
            reference_partition = fold_ids
        elif fold_ids != reference_partition:
            raise ValueError(f"{condition.label}: validation fold assignment differs")

        condition_rows: list[dict[str, Any]] = []
        fold_records: list[dict[str, Any]] = []
        for fold, expected_ids in enumerate(fold_ids):
            task_root = _condition_task_root(args.validation_root, condition, fold)
            marker = _read_fold_marker(task_root)
            _validate_marker_identity(
                marker,
                condition,
                fold,
                expected_cases=args.expected_cases_per_fold,
            )
            array_job_ids.add(str(marker["slurm"]["array_job_id"]))
            array_task_ids.add(int(marker["slurm"]["array_task_id"]))

            marker_split = marker["split"]
            if marker_split["validation_ids"] != expected_ids:
                raise ValueError(f"{condition.label} fold {fold}: marker split IDs differ")
            if marker_split["validation_ids_sha256"] != _ids_sha256(expected_ids):
                raise ValueError(f"{condition.label} fold {fold}: marker ID hash differs")
            if marker_split["sha256"] != _sha256(splits_path):
                raise ValueError(f"{condition.label} fold {fold}: split file changed")

            checkpoint_path = Path(marker["checkpoint"]["path"])
            if marker["checkpoint"]["sha256"] != _sha256(checkpoint_path):
                raise ValueError(f"{condition.label} fold {fold}: checkpoint_best changed")
            checkpoint_final_path = Path(marker["checkpoint"]["checkpoint_final_path"])
            if marker["checkpoint"]["checkpoint_final_sha256"] != _sha256(checkpoint_final_path):
                raise ValueError(f"{condition.label} fold {fold}: checkpoint_final changed")

            prediction_dir = Path(marker["prediction"]["directory"])
            predictions = _nifti_index(prediction_dir)
            if sorted(predictions) != expected_ids:
                raise ValueError(f"{condition.label} fold {fold}: prediction IDs changed")
            if marker["prediction"]["ids_sha256"] != _ids_sha256(sorted(predictions)):
                raise ValueError(f"{condition.label} fold {fold}: prediction ID hash differs")
            current_prediction_hashes = {
                case_id: _sha256(predictions[case_id]) for case_id in expected_ids
            }
            if marker["prediction"].get("file_sha256") != current_prediction_hashes:
                raise ValueError(f"{condition.label} fold {fold}: prediction file changed")
            summary_path = Path(marker["prediction"]["summary_path"])
            if marker["prediction"]["summary_sha256"] != _sha256(summary_path):
                raise ValueError(f"{condition.label} fold {fold}: summary.json changed")
            native_records = _summary_case_records(summary_path)
            if sorted(native_records) != expected_ids:
                raise ValueError(f"{condition.label} fold {fold}: summary IDs changed")
            expected_reference_dir = isolated_dataset_dir / "gt_segmentations"
            if marker["prediction"].get("reference_directory") != str(
                expected_reference_dir.resolve()
            ):
                raise ValueError(f"{condition.label} fold {fold}: reference directory differs")
            current_reference_hashes = {
                case_id: _sha256(expected_reference_dir / f"{case_id}.nii.gz")
                for case_id in expected_ids
            }
            if marker["prediction"].get("reference_file_sha256") != current_reference_hashes:
                raise ValueError(f"{condition.label} fold {fold}: reference file changed")
            for case_id in expected_ids:
                if native_records[case_id]["prediction_file"] != str(
                    predictions[case_id].resolve()
                ):
                    raise ValueError(
                        f"{condition.label} fold {fold}: summary prediction path differs"
                    )
                if native_records[case_id]["reference_file"] != str(
                    (expected_reference_dir / f"{case_id}.nii.gz").resolve()
                ):
                    raise ValueError(
                        f"{condition.label} fold {fold}: summary reference path differs"
                    )

            fold_dice: list[float] = []
            for case_id in expected_ids:
                canonical_mask, canonical_affine, canonical_spacing, target_mask_sha, target_sha = (
                    canonical_targets[case_id]
                )
                source_mask, source_affine, source_spacing = _load_binary(source_labels[case_id])
                _validate_geometry(
                    case_id,
                    source_mask,
                    canonical_mask,
                    source_affine,
                    canonical_affine,
                    source_spacing,
                    canonical_spacing,
                )
                if not np.array_equal(source_mask, canonical_mask):
                    raise ValueError(f"{condition.label}: target mask differs for {case_id}")

                reference_mask, reference_affine, reference_spacing = _load_binary(
                    expected_reference_dir / f"{case_id}.nii.gz"
                )
                _validate_geometry(
                    case_id,
                    reference_mask,
                    canonical_mask,
                    reference_affine,
                    canonical_affine,
                    reference_spacing,
                    canonical_spacing,
                )
                if not np.array_equal(reference_mask, canonical_mask):
                    raise ValueError(
                        f"{condition.label}: preprocessed validation target differs for {case_id}"
                    )

                prediction, prediction_affine, prediction_spacing = _load_binary(
                    predictions[case_id]
                )
                _validate_geometry(
                    case_id,
                    prediction,
                    canonical_mask,
                    prediction_affine,
                    canonical_affine,
                    prediction_spacing,
                    canonical_spacing,
                )
                dice = volumetric_dice(prediction, canonical_mask)
                actual_counts = {
                    "TP": int(np.logical_and(prediction, canonical_mask).sum()),
                    "FP": int(np.logical_and(prediction, np.logical_not(canonical_mask)).sum()),
                    "FN": int(np.logical_and(np.logical_not(prediction), canonical_mask).sum()),
                }
                recorded_counts = {
                    key: int(native_records[case_id][key]) for key in ("TP", "FP", "FN")
                }
                if actual_counts != recorded_counts:
                    raise ValueError(
                        f"{condition.label} {case_id}: saved-mask counts "
                        f"{actual_counts} differ from nnU-Net counts {recorded_counts}"
                    )
                row = {
                    "condition": condition.label,
                    "experiment": condition.experiment,
                    "dataset_id": condition.dataset_id,
                    "fold": fold,
                    "case_id": case_id,
                    "volumetric_dice": dice,
                    "prediction_path": str(predictions[case_id].resolve()),
                    "prediction_sha256": _sha256(predictions[case_id]),
                    "target_path": str(
                        (canonical_dataset_dir / "labelsTr" / f"{case_id}.nii.gz").resolve()
                    ),
                    "target_sha256": target_sha,
                    "target_mask_sha256": target_mask_sha,
                    "checkpoint_best_sha256": marker["checkpoint"]["sha256"],
                }
                per_case_rows.append(row)
                condition_rows.append(row)
                fold_dice.append(dice)

            fold_records.append(
                {
                    "fold": fold,
                    "case_ids": expected_ids,
                    "volumetric_dice": sample_summary(fold_dice),
                    "marker": marker,
                }
            )

        values = [float(row["volumetric_dice"]) for row in condition_rows]
        if len(values) != expected_total:
            raise RuntimeError(
                f"{condition.label}: expected {expected_total} cases, got {len(values)}"
            )
        condition_summaries.append(
            {
                "label": condition.label,
                "experiment": condition.experiment,
                "dataset_id": condition.dataset_id,
                "n_cases": len(values),
                "volumetric_dice": sample_summary(values),
                "controlled_preprocessing": preprocessing,
                "folds": fold_records,
            }
        )

    if len(array_job_ids) != 1:
        raise ValueError(f"Fold markers came from multiple array jobs: {sorted(array_job_ids)}")
    expected_task_ids = set(range(len(PANEL_CONDITIONS) * EXPECTED_FOLDS))
    if array_task_ids != expected_task_ids:
        raise ValueError(
            "Fold markers do not contain the exact 35 array tasks: "
            f"missing={sorted(expected_task_ids - array_task_ids)}, "
            f"extra={sorted(array_task_ids - expected_task_ids)}"
        )
    array_job_id = next(iter(array_job_ids))
    if args.validation_root.name != f"job_{array_job_id}":
        raise ValueError(
            f"Validation root {args.validation_root} does not belong to array {array_job_id}"
        )
    winner = _select_unique_winner(condition_summaries)

    output_parent = args.output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = output_parent / f".{args.output_dir.name}.tmp_{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir()
    try:
        summary_path = temporary / "validation_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = [
                "condition",
                "experiment",
                "dataset_id",
                "n_cases",
                "selected",
                "dice_mean",
                "dice_std",
                "dice_median",
                "dice_q1",
                "dice_q3",
                "dice_min",
                "dice_max",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for result in condition_summaries:
                stats = result["volumetric_dice"]
                writer.writerow(
                    {
                        "condition": result["label"],
                        "experiment": result["experiment"],
                        "dataset_id": result["dataset_id"],
                        "n_cases": result["n_cases"],
                        "selected": result["label"] == winner["label"],
                        "dice_mean": stats["mean"],
                        "dice_std": stats["std"],
                        "dice_median": stats["median"],
                        "dice_q1": stats["q1"],
                        "dice_q3": stats["q3"],
                        "dice_min": stats["min"],
                        "dice_max": stats["max"],
                    }
                )

        per_case_path = temporary / "validation_per_case.csv"
        with per_case_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_case_rows[0]))
            writer.writeheader()
            writer.writerows(per_case_rows)

        selection = {
            "schema": SELECTION_SCHEMA,
            "created_by": "medgen.scripts.select_exp16_2_validation_source finalize",
            "command": sys.argv,
            "host": platform.node(),
            "python": platform.python_version(),
            "nibabel": nib.__version__,
            "numpy": np.__version__,
            "git": _git_state(),
            "validation_array_job_id": array_job_id,
            "selection_rule": {
                "endpoint": "mean patient-level complete-volume foreground Dice",
                "cohort": "105 real training cases, each predicted once out of fold",
                "checkpoint": "checkpoint_best.pth",
                "empty_empty": 1.0,
                "ranking": "highest full-precision arithmetic mean",
                "exact_tie": "fail closed; no post-hoc tie break",
                "official_test_used": False,
            },
            "conditions": condition_summaries,
            "winner": {
                "label": winner["label"],
                "experiment": winner["experiment"],
                "dataset_id": winner["dataset_id"],
                "n_cases": winner["n_cases"],
                "mean_volumetric_dice": winner["volumetric_dice"]["mean"],
            },
        }
        _write_json(temporary / "selection.json", selection)
        (temporary / "LOCKED_SOURCE.txt").write_text(
            "\n".join(
                [
                    f"label={winner['label']}",
                    f"experiment={winner['experiment']}",
                    f"dataset_id={winner['dataset_id']}",
                    f"n_cases={winner['n_cases']}",
                    f"mean_patient_volumetric_dice={winner['volumetric_dice']['mean']}",
                    f"validation_array_job_id={array_job_id}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        artifact_paths = {
            "validation_summary.csv": summary_path,
            "validation_per_case.csv": per_case_path,
            "selection.json": temporary / "selection.json",
            "LOCKED_SOURCE.txt": temporary / "LOCKED_SOURCE.txt",
        }
        _write_json(
            temporary / COMPLETE_MARKER,
            {
                "schema": SELECTION_SCHEMA,
                "winner": winner["label"],
                "n_conditions": len(condition_summaries),
                "n_cases_per_condition": expected_total,
                "validation_array_job_id": array_job_id,
                "artifacts_sha256": {name: _sha256(path) for name, path in artifact_paths.items()},
            },
        )
        os.replace(temporary, args.output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    print("=== Locked exp16_2 hybrid source ===")
    print(f"Condition: {winner['label']}")
    print(f"Experiment: {winner['experiment']}")
    print(f"Dataset: {winner['dataset_id']}")
    print(f"Validation Dice: {winner['volumetric_dice']['mean']:.10f}")
    print(f"Output: {args.output_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-fold")
    verify.add_argument("--prediction-dir", type=Path, required=True)
    verify.add_argument("--reference-dir", type=Path, required=True)
    verify.add_argument("--splits", type=Path, required=True)
    verify.add_argument("--fold", type=int, required=True)
    verify.add_argument("--expected-folds", type=int, default=5)
    verify.add_argument("--expected-cases", type=int, default=21)
    verify.add_argument("--checkpoint-best", type=Path, required=True)
    verify.add_argument("--checkpoint-final", type=Path, required=True)
    verify.add_argument("--output-marker", type=Path, required=True)
    verify.add_argument("--label", required=True)
    verify.add_argument("--experiment", required=True)
    verify.add_argument("--dataset-id", type=int, required=True)
    verify.add_argument("--trainer", required=True)
    verify.add_argument("--plans", required=True)
    verify.add_argument("--configuration", required=True)
    verify.add_argument("--array-job-id", required=True)
    verify.add_argument("--array-task-id", type=int, required=True)
    verify.add_argument("--job-id", required=True)
    verify.add_argument("--stdout", type=Path, required=True)
    verify.add_argument("--stderr", type=Path, required=True)
    verify.set_defaults(function=_verify_fold)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--condition", action="append", type=_parse_condition, required=True)
    finalize.add_argument("--validation-root", type=Path, required=True)
    finalize.add_argument("--nnunet-raw", type=Path, required=True)
    finalize.add_argument("--nnunet-preprocessed", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--canonical-dataset-id", type=int, default=600)
    finalize.add_argument("--expected-folds", type=int, default=5)
    finalize.add_argument("--expected-cases-per-fold", type=int, default=21)
    finalize.set_defaults(function=_finalize)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if getattr(args, "expected_folds", 0) < 1:
        raise ValueError("expected fold count must be positive")
    expected_cases = getattr(args, "expected_cases", None)
    if expected_cases is not None and expected_cases < 1:
        raise ValueError("expected case count must be positive")
    expected_per_fold = getattr(args, "expected_cases_per_fold", None)
    if expected_per_fold is not None and expected_per_fold < 1:
        raise ValueError("expected cases per fold must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
