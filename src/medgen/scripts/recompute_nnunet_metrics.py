"""Recompute paper segmentation metrics from saved nnU-Net predictions only.

This command never loads a checkpoint and never invokes nnU-Net inference.  It
strictly pairs existing prediction NIfTIs with converted ``labelsTs`` files,
then writes full-precision, auditable metrics for the MIA paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import scipy
from scipy import stats

from medgen.metrics.paper_segmentation import (
    LEGACY_SIZE_BINS_MM,
    brats_mets_lesionwise_dice,
    hd95_mm,
    matched_component_metrics,
    sample_summary,
    slicewise_dice,
    volumetric_dice,
    volumetric_iou,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Condition:
    """One already-evaluated paper condition."""

    label: str
    experiment: str
    dataset_id: int
    provenance_stdout: Path


def _parse_condition(value: str) -> Condition:
    parts = value.split("|")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"condition must be LABEL|EXPERIMENT|DATASET_ID|PROVENANCE_STDOUT, got {value!r}"
        )
    label, experiment, dataset_text, provenance_text = parts
    if not label or not experiment:
        raise argparse.ArgumentTypeError("condition label and experiment cannot be empty")
    if not provenance_text:
        raise argparse.ArgumentTypeError("condition provenance log cannot be empty")
    try:
        dataset_id = int(dataset_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"dataset ID must be an integer, got {dataset_text!r}"
        ) from error
    return Condition(
        label=label,
        experiment=experiment,
        dataset_id=dataset_id,
        provenance_stdout=Path(provenance_text),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        type=_parse_condition,
        required=True,
        help=(
            "repeatable LABEL|EXPERIMENT|DATASET_ID|PROVENANCE_STDOUT specification; "
            "the matching .err log is checked automatically"
        ),
    )
    parser.add_argument("--nnunet-results", type=Path, required=True)
    parser.add_argument("--nnunet-raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", required=True, help="condition label for paired tests")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="condition label to compare with --baseline; repeat for one test family",
    )
    parser.add_argument("--expected-cases", type=int, default=51)
    parser.add_argument("--min-component-voxels", type=int, default=5)
    parser.add_argument("--detection-threshold", type=float, default=0.1)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask_sha256(mask: np.ndarray, affine: np.ndarray) -> str:
    """Hash binary content and geometry independently of gzip metadata."""
    digest = hashlib.sha256()
    digest.update(np.asarray(mask.shape, dtype=np.int64).tobytes())
    digest.update(np.asarray(affine, dtype=np.float64).tobytes())
    digest.update(np.packbits(np.asarray(mask, dtype=np.uint8).ravel()).tobytes())
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_state() -> dict[str, Any]:
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _dataset_labels_dir(nnunet_raw: Path, dataset_id: int) -> Path:
    matches = sorted(nnunet_raw.glob(f"Dataset{dataset_id:03d}_*/labelsTs"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one labelsTs for Dataset{dataset_id:03d}, found {matches}"
        )
    return matches[0]


def _prediction_dir(nnunet_results: Path, experiment: str) -> Path:
    return nnunet_results / experiment / f"eval_{experiment}" / "predictions"


def _validate_provenance(
    condition: Condition,
    prediction_dir: Path,
    *,
    expected_cases: int,
) -> dict[str, Any]:
    """Validate and fingerprint the canonical five-fold inference/evaluation logs."""
    stdout_path = condition.provenance_stdout
    stderr_path = stdout_path.with_suffix(".err")
    if stdout_path.suffix != ".out":
        raise ValueError(
            f"{condition.label}: provenance stdout must end in .out, got {stdout_path}"
        )
    if not stdout_path.is_file() or not stderr_path.is_file():
        raise FileNotFoundError(
            f"{condition.label}: missing provenance pair: {stdout_path}, {stderr_path}"
        )

    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    fold_marker = "Folds: (0, 1, 2, 3, 4)"
    required_stdout = {
        "five_fold_marker": fold_marker,
        "prediction_directory": f"Output: {prediction_dir}",
        "inference_completion": "Inference complete:",
    }
    missing_stdout = [name for name, marker in required_stdout.items() if marker not in stdout]
    if missing_stdout:
        raise ValueError(
            f"{condition.label}: provenance stdout lacks {missing_stdout}: {stdout_path}"
        )
    evaluation_marker = f"Evaluating {expected_cases} cases"
    if evaluation_marker not in stderr:
        raise ValueError(
            f"{condition.label}: provenance stderr lacks {evaluation_marker!r}: {stderr_path}"
        )

    return {
        "stdout_path": str(stdout_path.resolve()),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_path": str(stderr_path.resolve()),
        "stderr_sha256": _sha256(stderr_path),
        "folds": [0, 1, 2, 3, 4],
        "documented_prediction_dir": str(prediction_dir),
        "documented_evaluation_cases": expected_cases,
        "scope": (
            "documents how the saved pool was produced; current mask content is "
            "independently fingerprinted by this recomputation"
        ),
    }


def _nifti_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    return {path.name.removesuffix(".nii.gz"): path for path in directory.glob("*.nii.gz")}


def _strict_pairs(
    prediction_dir: Path,
    labels_dir: Path,
    *,
    expected_cases: int,
) -> list[tuple[str, Path, Path]]:
    predictions = _nifti_files(prediction_dir)
    targets = _nifti_files(labels_dir)
    if len(predictions) != expected_cases or len(targets) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} NIfTIs in each directory; "
            f"found predictions={len(predictions)}, targets={len(targets)}"
        )
    if predictions.keys() != targets.keys():
        missing_predictions = sorted(targets.keys() - predictions.keys())
        missing_targets = sorted(predictions.keys() - targets.keys())
        raise ValueError(
            "Prediction/target case sets differ: "
            f"missing_predictions={missing_predictions}, missing_targets={missing_targets}"
        )
    return [(case_id, predictions[case_id], targets[case_id]) for case_id in sorted(predictions)]


def _load_binary(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI at {path}, got shape {image.shape}")
    data = np.asanyarray(image.dataobj)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite mask values in {path}")
    unique = np.unique(data)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"Mask is not binary at {path}: values={unique.tolist()}")
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(image.affine))
    return data.astype(bool, copy=False), np.asarray(image.affine), spacing


def _validate_geometry(
    case_id: str,
    prediction: np.ndarray,
    target: np.ndarray,
    prediction_affine: np.ndarray,
    target_affine: np.ndarray,
    prediction_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"{case_id}: shape mismatch prediction={prediction.shape}, target={target.shape}"
        )
    if not np.allclose(prediction_affine, target_affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"{case_id}: prediction and target affines differ")
    if not np.allclose(prediction_spacing, target_spacing, rtol=1e-5, atol=1e-6):
        raise ValueError(
            f"{case_id}: spacing mismatch prediction={prediction_spacing}, target={target_spacing}"
        )

    # scipy's sampled Euclidean distance transform supports orthogonal voxel
    # axes with arbitrary spacing. Rotations/reflections are harmless, but a
    # sheared grid would require distances computed from the full affine.
    linear = np.asarray(target_affine[:3, :3], dtype=np.float64)
    axis_lengths = np.linalg.norm(linear, axis=0)
    if not np.all(np.isfinite(linear)) or np.any(axis_lengths <= 0):
        raise ValueError(f"{case_id}: invalid affine basis for physical distances")
    directions = linear / axis_lengths
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError(
            f"{case_id}: sheared/non-orthogonal affine is unsupported for physical HD95"
        )


def _case_metrics(
    case_id: str,
    prediction_path: Path,
    target_path: Path,
    *,
    min_component_voxels: int,
    detection_threshold: float,
) -> dict[str, Any]:
    prediction, prediction_affine, prediction_spacing = _load_binary(prediction_path)
    target, target_affine, target_spacing = _load_binary(target_path)
    _validate_geometry(
        case_id,
        prediction,
        target,
        prediction_affine,
        target_affine,
        prediction_spacing,
        target_spacing,
    )

    true_positive_voxels = int(np.logical_and(prediction, target).sum())
    false_positive_voxels = int(np.logical_and(prediction, ~target).sum())
    false_negative_voxels = int(np.logical_and(~prediction, target).sum())
    predicted_positive_voxels = true_positive_voxels + false_positive_voxels
    target_positive_voxels = true_positive_voxels + false_negative_voxels
    lesion = matched_component_metrics(
        prediction,
        target,
        target_spacing,
        min_voxels=min_component_voxels,
        detection_threshold=detection_threshold,
    )
    lesion.pop("parameters")

    return {
        "case_id": case_id,
        "prediction_path": str(prediction_path),
        "target_path": str(target_path),
        "prediction_sha256": _sha256(prediction_path),
        "target_sha256": _sha256(target_path),
        "prediction_mask_sha256": _mask_sha256(prediction, prediction_affine),
        "target_mask_sha256": _mask_sha256(target, target_affine),
        "shape": list(prediction.shape),
        "spacing_mm": list(target_spacing),
        "affine": target_affine.tolist(),
        "voxel_counts": {
            "tp": true_positive_voxels,
            "fp": false_positive_voxels,
            "fn": false_negative_voxels,
        },
        "volumetric_dice": volumetric_dice(prediction, target),
        "volumetric_iou": volumetric_iou(prediction, target),
        "brats_mets_lesionwise_dice": brats_mets_lesionwise_dice(prediction, target),
        "voxel_precision": (
            true_positive_voxels / predicted_positive_voxels if predicted_positive_voxels else None
        ),
        "voxel_recall": (
            true_positive_voxels / target_positive_voxels if target_positive_voxels else None
        ),
        "sagittal_slicewise_dice": slicewise_dice(prediction, target, axis=0),
        "coronal_slicewise_dice": slicewise_dice(prediction, target, axis=1),
        "axial_slicewise_dice": slicewise_dice(prediction, target, axis=2),
        "sagittal_foreground_slicewise_dice": slicewise_dice(
            prediction,
            target,
            axis=0,
            include_empty_slices=False,
        ),
        "coronal_foreground_slicewise_dice": slicewise_dice(
            prediction,
            target,
            axis=1,
            include_empty_slices=False,
        ),
        "axial_foreground_slicewise_dice": slicewise_dice(
            prediction,
            target,
            axis=2,
            include_empty_slices=False,
        ),
        "hd95": hd95_mm(prediction, target, target_spacing, target_affine),
        "lesion": lesion,
    }


def _wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return [float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))]


def _defined_summary(values: list[float | None]) -> dict[str, float | int | None]:
    return sample_summary([value for value in values if value is not None])


def _summarize_condition(
    cases: dict[str, dict[str, Any]],
    *,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    case_values = list(cases.values())
    voxel_tp = sum(case["voxel_counts"]["tp"] for case in case_values)
    voxel_fp = sum(case["voxel_counts"]["fp"] for case in case_values)
    voxel_fn = sum(case["voxel_counts"]["fn"] for case in case_values)
    precision = voxel_tp / (voxel_tp + voxel_fp) if voxel_tp + voxel_fp else None
    recall = voxel_tp / (voxel_tp + voxel_fn) if voxel_tp + voxel_fn else None
    micro_dice_denominator = 2 * voxel_tp + voxel_fp + voxel_fn
    micro_iou_denominator = voxel_tp + voxel_fp + voxel_fn
    micro_dice = 2 * voxel_tp / micro_dice_denominator if micro_dice_denominator else 1.0
    micro_iou = voxel_tp / micro_iou_denominator if micro_iou_denominator else 1.0

    hd95_statuses = Counter(case["hd95"]["status"] for case in case_values)
    conditional = [
        case["hd95"]["conditional_mm"]
        for case in case_values
        if case["hd95"]["conditional_mm"] is not None
    ]
    nonempty_conditional = [
        case["hd95"]["conditional_mm"]
        for case in case_values
        if case["hd95"]["status"] == "nonempty_pair"
    ]
    failure_aware = [case["hd95"]["failure_aware_mm"] for case in case_values]

    count_keys = (
        "n_gt",
        "n_pred_retained",
        "tp",
        "fn",
        "fp",
        "ignored_gt_below_min_voxels",
        "ignored_pred_below_min_voxels",
        "unmatched_pred_touching_excluded_gt",
    )
    lesion_counts = {
        key: sum(case["lesion"]["counts"][key] for case in case_values) for key in count_keys
    }
    lesion_tp = lesion_counts["tp"]
    lesion_fn = lesion_counts["fn"]
    lesion_fp = lesion_counts["fp"]
    n_gt = lesion_counts["n_gt"]
    n_evaluated_predictions = lesion_tp + lesion_fp
    lesion_sensitivity = lesion_tp / n_gt if n_gt else None
    lesion_precision = lesion_tp / n_evaluated_predictions if n_evaluated_predictions else None
    lesion_f1_denominator = 2 * lesion_tp + lesion_fp + lesion_fn
    lesion_f1 = 2 * lesion_tp / lesion_f1_denominator if lesion_f1_denominator else 1.0

    gt_records = [record for case in case_values for record in case["lesion"]["gt_lesions"]]
    fp_records = [
        record for case in case_values for record in case["lesion"]["false_positive_lesions"]
    ]
    matched_scores = [float(record["matched_dice"]) for record in gt_records if record["detected"]]
    all_gt_scores = [float(record["matched_dice"]) for record in gt_records]
    penalized_denominator = n_gt + lesion_fp
    penalized_dice = sum(matched_scores) / penalized_denominator if penalized_denominator else 1.0

    size_results = {}
    for size_name in LEGACY_SIZE_BINS_MM:
        records = [record for record in gt_records if record["size_category"] == size_name]
        detected = [record for record in records if record["detected"]]
        tp_size = len(detected)
        total_size = len(records)
        size_results[size_name] = {
            "n_gt": total_size,
            "tp": tp_size,
            "fn": total_size - tp_size,
            "sensitivity": tp_size / total_size if total_size else None,
            "sensitivity_wilson_95ci": _wilson_interval(tp_size, total_size),
            "matched_dice": sample_summary([float(record["matched_dice"]) for record in detected]),
            "all_gt_dice": sample_summary([float(record["matched_dice"]) for record in records]),
            "fp_by_predicted_size": sum(
                record["size_category"] == size_name for record in fp_records
            ),
        }

    patient_fp = [case["lesion"]["counts"]["fp"] for case in case_values]
    patient_sensitivity = [
        case["lesion"]["sensitivity"]
        for case in case_values
        if case["lesion"]["sensitivity"] is not None
    ]
    patient_precision = [
        case["lesion"]["precision"]
        for case in case_values
        if case["lesion"]["precision"] is not None
    ]
    patient_f1 = [case["lesion"]["f1"] for case in case_values]
    patient_penalized_dice = [case["lesion"]["penalized_lesion_dice"] for case in case_values]
    volumetric_dice_values = np.asarray(
        [case["volumetric_dice"] for case in case_values],
        dtype=np.float64,
    )
    brats_mets_dice_values = np.asarray(
        [case["brats_mets_lesionwise_dice"] for case in case_values],
        dtype=np.float64,
    )

    return {
        "n_cases": len(case_values),
        "volumetric_dice": {
            **sample_summary(volumetric_dice_values),
            "bootstrap_mean_95ci": _bootstrap_mean_ci(
                volumetric_dice_values,
                draws=bootstrap_draws,
                seed=seed,
            ),
        },
        "volumetric_iou": sample_summary([case["volumetric_iou"] for case in case_values]),
        "brats_mets_lesionwise_dice": {
            **sample_summary(brats_mets_dice_values),
            "bootstrap_mean_95ci": _bootstrap_mean_ci(
                brats_mets_dice_values,
                draws=bootstrap_draws,
                seed=seed,
            ),
        },
        "sagittal_slicewise_dice": sample_summary(
            [case["sagittal_slicewise_dice"] for case in case_values]
        ),
        "coronal_slicewise_dice": sample_summary(
            [case["coronal_slicewise_dice"] for case in case_values]
        ),
        "axial_slicewise_dice": sample_summary(
            [case["axial_slicewise_dice"] for case in case_values]
        ),
        "sagittal_foreground_slicewise_dice": _defined_summary(
            [case["sagittal_foreground_slicewise_dice"] for case in case_values]
        ),
        "coronal_foreground_slicewise_dice": _defined_summary(
            [case["coronal_foreground_slicewise_dice"] for case in case_values]
        ),
        "axial_foreground_slicewise_dice": _defined_summary(
            [case["axial_foreground_slicewise_dice"] for case in case_values]
        ),
        "patient_macro": {
            "voxel_precision": _defined_summary([case["voxel_precision"] for case in case_values]),
            "voxel_recall": _defined_summary([case["voxel_recall"] for case in case_values]),
        },
        "voxel_micro": {
            "tp": voxel_tp,
            "fp": voxel_fp,
            "fn": voxel_fn,
            "dice": micro_dice,
            "iou": micro_iou,
            "precision": precision,
            "recall": recall,
        },
        "hd95_mm": {
            "status_counts": dict(sorted(hd95_statuses.items())),
            "conditional_valid_cases": sample_summary(conditional),
            "conditional_nonempty_pairs": sample_summary(nonempty_conditional),
            "failure_aware_all_cases": sample_summary(failure_aware),
            "empty_policy": (
                "both-empty=0; exactly-one-empty=image physical diagonal; "
                "conditional value is null for exactly-one-empty"
            ),
        },
        "matched_lesions": {
            "counts": lesion_counts,
            "sensitivity": lesion_sensitivity,
            "sensitivity_wilson_95ci": _wilson_interval(lesion_tp, n_gt),
            "precision": lesion_precision,
            "f1": lesion_f1,
            "false_positives_per_patient": {
                "total": lesion_fp,
                "mean": lesion_fp / len(case_values) if case_values else None,
                "distribution": sample_summary(patient_fp),
            },
            "matched_dice": sample_summary(matched_scores),
            "all_gt_dice": sample_summary(all_gt_scores),
            "penalized_lesion_dice": float(penalized_dice),
            "per_patient_penalized_lesion_dice": sample_summary(patient_penalized_dice),
            "per_patient_sensitivity": sample_summary(patient_sensitivity),
            "per_patient_precision": sample_summary(patient_precision),
            "per_patient_f1": sample_summary(patient_f1),
            "by_gt_size": size_results,
        },
    }


def _evaluate_condition(
    condition: Condition,
    *,
    prediction_dir: Path,
    labels_dir: Path,
    provenance: dict[str, Any],
    pairs: list[tuple[str, Path, Path]],
    min_component_voxels: int,
    detection_threshold: float,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    logger.info(
        "[%s] prediction-only evaluation: %d cases\n  predictions=%s\n  targets=%s",
        condition.label,
        len(pairs),
        prediction_dir,
        labels_dir,
    )

    cases = {}
    for index, (case_id, prediction_path, target_path) in enumerate(pairs, start=1):
        cases[case_id] = _case_metrics(
            case_id,
            prediction_path,
            target_path,
            min_component_voxels=min_component_voxels,
            detection_threshold=detection_threshold,
        )
        if index == 1 or index % 10 == 0 or index == len(pairs):
            logger.info("[%s] %d/%d cases", condition.label, index, len(pairs))

    return {
        "condition": {
            "label": condition.label,
            "experiment": condition.experiment,
            "dataset_id": condition.dataset_id,
            "prediction_dir": str(prediction_dir),
            "labels_dir": str(labels_dir),
            "prediction_provenance": provenance,
        },
        "method": {
            "source": "saved prediction masks only; no model loading or inference",
            "component_connectivity": 26,
            "minimum_component_voxels": min_component_voxels,
            "detection_rule": f"unsmoothed full-component Dice > {detection_threshold}",
            "matching": "one-to-one maximum-cardinality then maximum-Dice",
            "dice_views": {
                "patient_volumetric": "complete 3D foreground mask; empty/empty=1",
                "pooled_voxel": "all patient voxels pooled before Dice calculation",
                "brats_mets_lesionwise": (
                    "26-connected lesions; missed GT and prediction-only lesions contribute 0; "
                    "any voxel overlap detects a GT lesion for this metric"
                ),
                "slicewise_all": (
                    "sagittal, coronal, and axial; correctly predicted empty slices=1"
                ),
                "slicewise_foreground": (
                    "sagittal, coronal, and axial; slices with both masks empty omitted"
                ),
                "matched_lesion": "detected one-to-one component pairs only",
                "all_gt_lesion": "all retained GT components; missed lesions=0",
                "fp_penalized_lesion": (
                    "matched Dice sum divided by retained GT plus unmatched predictions"
                ),
            },
            "primary_mean_bootstrap": {
                "draws": bootstrap_draws,
                "seed": seed,
                "interval": "pointwise 2.5th and 97.5th percentiles",
            },
            "size_measure": (
                "2D Feret diameter of the largest 4-connected region on the "
                "maximum-area axial slice"
            ),
            "size_bins_mm": {
                "tiny": [0.0, 10.0],
                "small": [10.0, 20.0],
                "medium": [20.0, 30.0],
                "large": [30.0, None],
            },
            "hd95": {
                "estimator": (
                    "symmetric maximum of the two directed 95th-percentile surface distances"
                ),
                "surface": (
                    "surface voxels from erosion with a full 3x3x3 (26-neighbour) footprint"
                ),
                "geometry": ("native NIfTI voxel spacing on an affine-validated orthogonal grid"),
                "empty_policy": (
                    "both-empty=0 mm; exactly-one-empty conditional=null and "
                    "failure-aware=image physical diagonal"
                ),
            },
        },
        "summary": _summarize_condition(
            cases,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
        ),
        "per_case": cases,
    }


def _rank_biserial(differences: np.ndarray) -> tuple[float, int]:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0, 0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative), int(nonzero.size)


def _bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(draws, differences.size))
    means = differences[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return [float(lower), float(upper)]


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=np.float64)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[int(index)])
        running_max = max(running_max, candidate)
        adjusted[int(index)] = running_max
    return [float(value) for value in adjusted]


def _paired_statistics(
    results: dict[str, dict[str, Any]],
    *,
    baseline_label: str,
    compare_labels: list[str],
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    baseline_cases = results[baseline_label]["per_case"]
    baseline_ids = set(baseline_cases)
    comparisons = []
    for comparison_index, label_name in enumerate(compare_labels):
        condition_cases = results[label_name]["per_case"]
        if set(condition_cases) != baseline_ids:
            raise ValueError(f"Cannot pair {label_name} with {baseline_label}: case IDs differ")
        unequal_targets = [
            case_id
            for case_id in baseline_ids
            if condition_cases[case_id]["target_mask_sha256"]
            != baseline_cases[case_id]["target_mask_sha256"]
        ]
        if unequal_targets:
            raise ValueError(
                f"Cannot pair {label_name} with {baseline_label}: "
                f"target masks differ for {sorted(unequal_targets)}"
            )
        case_ids = sorted(baseline_ids)
        baseline = np.asarray([baseline_cases[case_id]["volumetric_dice"] for case_id in case_ids])
        condition = np.asarray(
            [condition_cases[case_id]["volumetric_dice"] for case_id in case_ids]
        )
        differences = condition - baseline
        nonzero_count = int(np.count_nonzero(differences))
        if nonzero_count == 0:
            statistic, p_value = 0.0, 1.0
        else:
            test = stats.wilcoxon(
                condition,
                baseline,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            statistic, p_value = float(test.statistic), float(test.pvalue)
        rank_biserial, effective_n = _rank_biserial(differences)
        difference_sd = float(differences.std(ddof=1))
        bootstrap_ci = _bootstrap_mean_ci(
            differences,
            draws=bootstrap_draws,
            seed=seed + comparison_index,
        )
        comparisons.append(
            {
                "condition": label_name,
                "n_patients": len(case_ids),
                "n_zero_differences": len(case_ids) - nonzero_count,
                "n_effective": effective_n,
                "baseline_mean": float(baseline.mean()),
                "condition_mean": float(condition.mean()),
                "mean_difference": float(differences.mean()),
                "median_difference": float(np.median(differences)),
                "percent_of_baseline": float(condition.mean() / baseline.mean() * 100),
                "percent_change": float(differences.mean() / baseline.mean() * 100),
                "bootstrap_mean_difference_95ci": bootstrap_ci,
                "bootstrap_ci_lo": bootstrap_ci[0],
                "bootstrap_ci_hi": bootstrap_ci[1],
                "wilcoxon_w": statistic,
                "p_raw": p_value,
                "cohen_dz": (
                    float(differences.mean() / difference_sd) if difference_sd > 0 else 0.0
                ),
                "rank_biserial": rank_biserial,
            }
        )

    raw_p = [comparison["p_raw"] for comparison in comparisons]
    holm = _holm_adjust(raw_p)
    family_size = len(comparisons)
    for comparison, holm_value in zip(comparisons, holm, strict=True):
        comparison["p_holm"] = holm_value
        comparison["p_bonferroni"] = min(1.0, comparison["p_raw"] * family_size)

    return {
        "metric": "per-patient complete-volume foreground Dice",
        "baseline": baseline_label,
        "family_size": family_size,
        "alpha": 0.05,
        "bonferroni_alpha": 0.05 / family_size if family_size else None,
        "wilcoxon": {
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "method": "auto",
        },
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": seed,
        "comparisons": comparisons,
    }


def _validate_cohorts(
    results: dict[str, dict[str, Any]],
    *,
    baseline_label: str,
) -> dict[str, str]:
    """Verify identical GT content whenever a condition shares baseline case IDs."""
    baseline_cases = results[baseline_label]["per_case"]
    baseline_ids = set(baseline_cases)
    statuses = {}
    for label_name, result in results.items():
        cases = result["per_case"]
        if set(cases) != baseline_ids:
            statuses[label_name] = "different_case_set"
            continue
        unequal = [
            case_id
            for case_id in baseline_ids
            if cases[case_id]["target_mask_sha256"] != baseline_cases[case_id]["target_mask_sha256"]
        ]
        if unequal:
            raise ValueError(
                f"{label_name} shares baseline case IDs but target masks differ: {sorted(unequal)}"
            )
        statuses[label_name] = "identical_case_ids_and_target_masks"
    return statuses


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label_name, result in results.items():
        summary = result["summary"]
        row: dict[str, Any] = {
            "condition": label_name,
            "experiment": result["condition"]["experiment"],
            "dataset_id": result["condition"]["dataset_id"],
            "n_cases": summary["n_cases"],
            "dice_mean": summary["volumetric_dice"]["mean"],
            "dice_std": summary["volumetric_dice"]["std"],
            "dice_ci_lo": summary["volumetric_dice"]["bootstrap_mean_95ci"][0],
            "dice_ci_hi": summary["volumetric_dice"]["bootstrap_mean_95ci"][1],
            "iou_mean": summary["volumetric_iou"]["mean"],
            "iou_std": summary["volumetric_iou"]["std"],
            "brats_mets_dice_mean": summary["brats_mets_lesionwise_dice"]["mean"],
            "brats_mets_dice_std": summary["brats_mets_lesionwise_dice"]["std"],
            "brats_mets_dice_ci_lo": summary["brats_mets_lesionwise_dice"]["bootstrap_mean_95ci"][
                0
            ],
            "brats_mets_dice_ci_hi": summary["brats_mets_lesionwise_dice"]["bootstrap_mean_95ci"][
                1
            ],
            "slicewise_dice_mean": summary["sagittal_slicewise_dice"]["mean"],
            "slicewise_dice_std": summary["sagittal_slicewise_dice"]["std"],
            "sagittal_slicewise_dice_mean": summary["sagittal_slicewise_dice"]["mean"],
            "sagittal_slicewise_dice_std": summary["sagittal_slicewise_dice"]["std"],
            "coronal_slicewise_dice_mean": summary["coronal_slicewise_dice"]["mean"],
            "coronal_slicewise_dice_std": summary["coronal_slicewise_dice"]["std"],
            "axial_slicewise_dice_mean": summary["axial_slicewise_dice"]["mean"],
            "axial_slicewise_dice_std": summary["axial_slicewise_dice"]["std"],
            "sagittal_foreground_slicewise_dice_mean": summary[
                "sagittal_foreground_slicewise_dice"
            ]["mean"],
            "coronal_foreground_slicewise_dice_mean": summary["coronal_foreground_slicewise_dice"][
                "mean"
            ],
            "axial_foreground_slicewise_dice_mean": summary["axial_foreground_slicewise_dice"][
                "mean"
            ],
            "voxel_dice_micro": summary["voxel_micro"]["dice"],
            "voxel_iou_micro": summary["voxel_micro"]["iou"],
            "voxel_precision": summary["voxel_micro"]["precision"],
            "voxel_recall": summary["voxel_micro"]["recall"],
            "patient_precision_mean": summary["patient_macro"]["voxel_precision"]["mean"],
            "patient_recall_mean": summary["patient_macro"]["voxel_recall"]["mean"],
            "hd95_conditional_mean_mm": summary["hd95_mm"]["conditional_valid_cases"]["mean"],
            "hd95_conditional_std_mm": summary["hd95_mm"]["conditional_valid_cases"]["std"],
            "hd95_conditional_n": summary["hd95_mm"]["conditional_valid_cases"]["n"],
            "hd95_failure_aware_mean_mm": summary["hd95_mm"]["failure_aware_all_cases"]["mean"],
            "hd95_failure_aware_std_mm": summary["hd95_mm"]["failure_aware_all_cases"]["std"],
            "hd95_empty_prediction_n": summary["hd95_mm"]["status_counts"].get(
                "empty_prediction", 0
            ),
            "hd95_empty_target_n": summary["hd95_mm"]["status_counts"].get("empty_target", 0),
            "lesion_tp": summary["matched_lesions"]["counts"]["tp"],
            "lesion_fn": summary["matched_lesions"]["counts"]["fn"],
            "lesion_fp": summary["matched_lesions"]["counts"]["fp"],
            "lesion_sensitivity": summary["matched_lesions"]["sensitivity"],
            "lesion_precision": summary["matched_lesions"]["precision"],
            "lesion_f1": summary["matched_lesions"]["f1"],
            "fp_per_patient": summary["matched_lesions"]["false_positives_per_patient"]["mean"],
            "matched_lesion_dice_mean": summary["matched_lesions"]["matched_dice"]["mean"],
            "matched_lesion_dice_std": summary["matched_lesions"]["matched_dice"]["std"],
            "all_gt_lesion_dice_mean": summary["matched_lesions"]["all_gt_dice"]["mean"],
            "all_gt_lesion_dice_std": summary["matched_lesions"]["all_gt_dice"]["std"],
            "penalized_lesion_dice": summary["matched_lesions"]["penalized_lesion_dice"],
            "patient_penalized_lesion_dice_mean": summary["matched_lesions"][
                "per_patient_penalized_lesion_dice"
            ]["mean"],
            "patient_penalized_lesion_dice_std": summary["matched_lesions"][
                "per_patient_penalized_lesion_dice"
            ]["std"],
        }
        for size_name in LEGACY_SIZE_BINS_MM:
            size = summary["matched_lesions"]["by_gt_size"][size_name]
            row[f"{size_name}_n_gt"] = size["n_gt"]
            row[f"{size_name}_sensitivity"] = size["sensitivity"]
            interval = size["sensitivity_wilson_95ci"]
            row[f"{size_name}_sensitivity_ci_lo"] = interval[0] if interval else None
            row[f"{size_name}_sensitivity_ci_hi"] = interval[1] if interval else None
            row[f"{size_name}_all_gt_dice_mean"] = size["all_gt_dice"]["mean"]
            row[f"{size_name}_all_gt_dice_std"] = size["all_gt_dice"]["std"]
            row[f"{size_name}_matched_dice_mean"] = size["matched_dice"]["mean"]
            row[f"{size_name}_matched_dice_std"] = size["matched_dice"]["std"]
            row[f"{size_name}_matched_dice_n"] = size["matched_dice"]["n"]
            row[f"{size_name}_fp"] = size["fp_by_predicted_size"]
        rows.append(row)
    return rows


def _per_case_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label_name, result in results.items():
        for case_id, case in result["per_case"].items():
            counts = case["lesion"]["counts"]
            rows.append(
                {
                    "condition": label_name,
                    "case_id": case_id,
                    "volumetric_dice": case["volumetric_dice"],
                    "volumetric_iou": case["volumetric_iou"],
                    "brats_mets_lesionwise_dice": case["brats_mets_lesionwise_dice"],
                    "voxel_precision": case["voxel_precision"],
                    "voxel_recall": case["voxel_recall"],
                    "sagittal_slicewise_dice": case["sagittal_slicewise_dice"],
                    "coronal_slicewise_dice": case["coronal_slicewise_dice"],
                    "axial_slicewise_dice": case["axial_slicewise_dice"],
                    "sagittal_foreground_slicewise_dice": case[
                        "sagittal_foreground_slicewise_dice"
                    ],
                    "coronal_foreground_slicewise_dice": case["coronal_foreground_slicewise_dice"],
                    "axial_foreground_slicewise_dice": case["axial_foreground_slicewise_dice"],
                    "voxel_tp": case["voxel_counts"]["tp"],
                    "voxel_fp": case["voxel_counts"]["fp"],
                    "voxel_fn": case["voxel_counts"]["fn"],
                    "hd95_status": case["hd95"]["status"],
                    "hd95_conditional_mm": case["hd95"]["conditional_mm"],
                    "hd95_failure_aware_mm": case["hd95"]["failure_aware_mm"],
                    "lesion_n_gt": counts["n_gt"],
                    "lesion_tp": counts["tp"],
                    "lesion_fn": counts["fn"],
                    "lesion_fp": counts["fp"],
                    "lesion_sensitivity": case["lesion"]["sensitivity"],
                    "lesion_precision": case["lesion"]["precision"],
                    "lesion_f1": case["lesion"]["f1"],
                    "penalized_lesion_dice": case["lesion"]["penalized_lesion_dice"],
                }
            )
    return rows


def _per_lesion_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for label_name, result in results.items():
        for case_id, case in result["per_case"].items():
            for record in case["lesion"]["gt_lesions"]:
                rows.append(
                    {
                        "condition": label_name,
                        "case_id": case_id,
                        "record_type": "ground_truth",
                        **record,
                    }
                )
            for record in case["lesion"]["false_positive_lesions"]:
                rows.append(
                    {
                        "condition": label_name,
                        "case_id": case_id,
                        "record_type": "false_positive_prediction",
                        **record,
                    }
                )
    return rows


def _print_headline(results: dict[str, dict[str, Any]]) -> None:
    logger.info("=== Prediction-only paper metric panel ===")
    logger.info(
        "%-18s %8s %8s %8s %8s %10s %8s %8s %8s",
        "condition",
        "Dice",
        "BraTS",
        "Prec",
        "Recall",
        "HD95mm",
        "LesSens",
        "LesF1",
        "FP/pat",
    )
    for label_name, result in results.items():
        summary = result["summary"]
        logger.info(
            "%-18s %8.4f %8.4f %8.4f %8.4f %10.3f %8.4f %8.4f %8.3f",
            label_name,
            summary["volumetric_dice"]["mean"],
            summary["brats_mets_lesionwise_dice"]["mean"],
            summary["voxel_micro"]["precision"],
            summary["voxel_micro"]["recall"],
            summary["hd95_mm"]["failure_aware_all_cases"]["mean"],
            summary["matched_lesions"]["sensitivity"],
            summary["matched_lesions"]["f1"],
            summary["matched_lesions"]["false_positives_per_patient"]["mean"],
        )


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.expected_cases < 1:
        raise ValueError("--expected-cases must be positive")
    if args.bootstrap_draws < 1:
        raise ValueError("--bootstrap-draws must be positive")

    labels = [condition.label for condition in args.condition]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Condition labels must be unique: {labels}")
    if args.baseline not in labels:
        raise ValueError(f"Unknown baseline {args.baseline!r}")
    if len(args.compare) != len(set(args.compare)):
        raise ValueError(f"Comparison labels must be unique: {args.compare}")
    unknown_comparisons = sorted(set(args.compare) - set(labels))
    if unknown_comparisons:
        raise ValueError(f"Unknown comparison labels: {unknown_comparisons}")

    # Resolve and validate every pool before spending time on surface distances.
    resolved: dict[
        str,
        tuple[Path, Path, list[tuple[str, Path, Path]], dict[str, Any]],
    ] = {}
    for condition in args.condition:
        prediction_dir = _prediction_dir(args.nnunet_results, condition.experiment)
        labels_dir = _dataset_labels_dir(args.nnunet_raw, condition.dataset_id)
        provenance = _validate_provenance(
            condition,
            prediction_dir,
            expected_cases=args.expected_cases,
        )
        pairs = _strict_pairs(
            prediction_dir,
            labels_dir,
            expected_cases=args.expected_cases,
        )
        resolved[condition.label] = (prediction_dir, labels_dir, pairs, provenance)
    logger.info(
        "Preflight passed for all %d prediction pools and provenance pairs",
        len(resolved),
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_by": "medgen.scripts.recompute_nnunet_metrics",
        "command": sys.argv,
        "host": platform.node(),
        "python": platform.python_version(),
        "packages": {
            "nibabel": nib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_image": _package_version("scikit-image"),
        },
        "git": _git_state(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "parameters": {
            "expected_cases": args.expected_cases,
            "min_component_voxels": args.min_component_voxels,
            "detection_threshold_strictly_greater_than": args.detection_threshold,
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
        },
        "conditions": [
            {
                "label": condition.label,
                "experiment": condition.experiment,
                "dataset_id": condition.dataset_id,
                "prediction_provenance": resolved[condition.label][3],
            }
            for condition in args.condition
        ],
    }
    _write_json(args.output_dir / "manifest.json", metadata)

    results: dict[str, dict[str, Any]] = {}
    for condition_index, condition in enumerate(args.condition):
        prediction_dir, labels_dir, pairs, provenance = resolved[condition.label]
        result = _evaluate_condition(
            condition,
            prediction_dir=prediction_dir,
            labels_dir=labels_dir,
            provenance=provenance,
            pairs=pairs,
            min_component_voxels=args.min_component_voxels,
            detection_threshold=args.detection_threshold,
            bootstrap_draws=args.bootstrap_draws,
            seed=args.seed + condition_index,
        )
        results[condition.label] = result
        _write_json(args.output_dir / "conditions" / f"{condition.label}.json", result)

    cohort_validation = _validate_cohorts(results, baseline_label=args.baseline)
    paired = _paired_statistics(
        results,
        baseline_label=args.baseline,
        compare_labels=args.compare,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    aggregate = {
        "metadata": metadata,
        "cohort_validation": cohort_validation,
        "conditions": results,
        "paired_statistics": paired,
    }
    _write_json(args.output_dir / "summary.json", aggregate)
    _write_json(args.output_dir / "paired_statistics.json", paired)

    summary_rows = _summary_rows(results)
    summary_fields = list(summary_rows[0])
    _write_csv(args.output_dir / "summary.csv", summary_rows, summary_fields)

    per_case_rows = _per_case_rows(results)
    _write_csv(args.output_dir / "per_case.csv", per_case_rows, list(per_case_rows[0]))

    per_lesion_rows = _per_lesion_rows(results)
    lesion_fields = sorted({key for row in per_lesion_rows for key in row})
    _write_csv(args.output_dir / "per_lesion.csv", per_lesion_rows, lesion_fields)

    comparison_rows = paired["comparisons"]
    _write_csv(
        args.output_dir / "paired_statistics.csv",
        comparison_rows,
        list(comparison_rows[0]) if comparison_rows else ["condition"],
    )
    _print_headline(results)
    logger.info("Wrote new non-overwriting prediction-only results to %s", args.output_dir)


if __name__ == "__main__":
    main()
