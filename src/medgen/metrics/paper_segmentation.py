"""Prediction-only segmentation metrics used by the MIA paper audit.

This module deliberately does not reuse :mod:`tracker_seg`: that historical
tracker masks predictions to each ground-truth component and applies ``+1``
smoothing, so its output is not a conventional lesion Dice or detection
metric.  The functions below operate on complete connected components and use
one-to-one matching.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)
LEGACY_SIZE_BINS_MM: dict[str, tuple[float, float]] = {
    "tiny": (0.0, 10.0),
    "small": (10.0, 20.0),
    "medium": (20.0, 30.0),
    "large": (30.0, float("inf")),
}


def sample_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return a strict-JSON-safe descriptive summary using sample SD."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1) if array.size > 1 else 0.0),
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def volumetric_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    """Complete-volume foreground Dice; empty/empty is scored as one."""
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    denominator = int(pred.sum()) + int(gt.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(pred, gt).sum()) / denominator


def volumetric_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    """Complete-volume foreground IoU; empty/empty is scored as one."""
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum()) / union


def brats_mets_lesionwise_dice(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    min_overlap_voxels: int = 1,
) -> float:
    """BraTS-Mets lesion-wise Dice with missed and false-positive lesions scored zero.

    Ground-truth and predicted lesions use 26-connectivity.  Each ground-truth
    lesion contributes its local bounding-box Dice when it has the required
    voxel overlap and contributes zero otherwise.  Each predicted component
    that touches no ground-truth lesion contributes one additional zero.  The
    patient score is the mean of those contributions.  This intentionally
    remains separate from the study-defined one-to-one detection analysis,
    which uses a full-component Dice threshold and minimum component size.
    """
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(f"Expected equal 3D masks, got {pred.shape} and {gt.shape}")
    if min_overlap_voxels < 1:
        raise ValueError("min_overlap_voxels must be positive")

    gt_labels, n_gt = label(gt, structure=CONNECTIVITY_26)
    pred_labels, n_pred = label(pred, structure=CONNECTIVITY_26)
    if n_gt == 0 and n_pred == 0:
        return 1.0

    contributions: list[float] = []
    prediction_touched = np.zeros(n_pred + 1, dtype=bool)
    for gt_id in range(1, n_gt + 1):
        gt_component = gt_labels == gt_id
        if int(np.logical_and(pred, gt_component).sum()) < min_overlap_voxels:
            contributions.append(0.0)
            continue

        coordinates = np.where(gt_component)
        bounding_box = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)
        local_prediction = pred[bounding_box]
        local_target = gt_component[bounding_box]
        denominator = int(local_prediction.sum()) + int(local_target.sum())
        contributions.append(
            2.0 * float(np.logical_and(local_prediction, local_target).sum()) / denominator
        )
        for pred_id in np.unique(pred_labels[gt_component]):
            if pred_id != 0:
                prediction_touched[int(pred_id)] = True

    contributions.extend(0.0 for pred_id in range(1, n_pred + 1) if not prediction_touched[pred_id])
    return float(np.mean(contributions))


def slicewise_dice(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    axis: int = 0,
    include_empty_slices: bool = True,
) -> float | None:
    """Mean slice-wise Dice on one array axis.

    With ``include_empty_slices=True`` correctly predicted empty slices score
    one, matching the paper's sagittal endpoint.  With it disabled, slices
    where both masks are empty are omitted.  The latter is undefined when all
    slices are empty and therefore returns ``None``.
    """
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: prediction={pred.shape}, target={gt.shape}")
    if not 0 <= axis < pred.ndim:
        raise ValueError(f"Invalid slice axis {axis} for {pred.ndim}D input")

    reduce_axes = tuple(index for index in range(pred.ndim) if index != axis)
    intersection = np.logical_and(pred, gt).sum(axis=reduce_axes, dtype=np.float64)
    denominator = pred.sum(axis=reduce_axes, dtype=np.float64) + gt.sum(
        axis=reduce_axes, dtype=np.float64
    )
    nonempty = denominator > 0
    if not include_empty_slices:
        if not nonempty.any():
            return None
        return float((2.0 * intersection[nonempty] / denominator[nonempty]).mean())

    values = np.ones_like(denominator, dtype=np.float64)
    np.divide(2.0 * intersection, denominator, out=values, where=nonempty)
    return float(values.mean())


def image_diagonal_mm(shape: Sequence[int], affine: np.ndarray) -> float:
    """Maximum centre-to-centre distance between affine-transformed image corners."""
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D shape, got {tuple(shape)}")
    affine_array = np.asarray(affine, dtype=np.float64)
    if affine_array.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 affine, got {affine_array.shape}")

    corners = np.asarray(
        [
            (x, y, z, 1.0)
            for x in (0, int(shape[0]) - 1)
            for y in (0, int(shape[1]) - 1)
            for z in (0, int(shape[2]) - 1)
        ],
        dtype=np.float64,
    )
    world = (affine_array @ corners.T).T[:, :3]
    deltas = world[:, None, :] - world[None, :, :]
    return float(np.sqrt(np.square(deltas).sum(axis=-1)).max())


def hd95_mm(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float],
    affine: np.ndarray,
) -> dict[str, float | str | None]:
    """Compute symmetric physical HD95 with explicit empty-mask handling.

    For two non-empty masks, the result is the maximum of the two directed
    95th-percentile surface distances.  Both-empty masks score 0 mm.  When
    exactly one mask is empty, conditional HD95 is undefined and therefore
    ``None``; a separate failure-aware value uses the physical image diagonal.
    Both values are exported so the paper can state its policy explicitly.
    """
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: prediction={pred.shape}, target={gt.shape}")
    if pred.ndim != 3 or spacing_array.shape != (3,):
        raise ValueError("HD95 expects 3D masks and three spacing values")
    if not np.all(np.isfinite(spacing_array)) or np.any(spacing_array <= 0):
        raise ValueError(f"Invalid voxel spacing: {spacing_array.tolist()}")

    pred_nonempty = bool(pred.any())
    gt_nonempty = bool(gt.any())
    diagonal = image_diagonal_mm(pred.shape, affine)
    if not pred_nonempty and not gt_nonempty:
        return {
            "status": "both_empty",
            "conditional_mm": 0.0,
            "failure_aware_mm": 0.0,
            "empty_case_penalty_mm": diagonal,
        }
    if not pred_nonempty:
        return {
            "status": "empty_prediction",
            "conditional_mm": None,
            "failure_aware_mm": diagonal,
            "empty_case_penalty_mm": diagonal,
        }
    if not gt_nonempty:
        return {
            "status": "empty_target",
            "conditional_mm": None,
            "failure_aware_mm": diagonal,
            "empty_case_penalty_mm": diagonal,
        }

    footprint = np.ones((3, 3, 3), dtype=bool)
    pred_surface = np.logical_xor(
        pred,
        binary_erosion(pred, structure=footprint, border_value=0),
    )
    gt_surface = np.logical_xor(
        gt,
        binary_erosion(gt, structure=footprint, border_value=0),
    )

    distance_to_gt = distance_transform_edt(~gt_surface, sampling=spacing_array)
    pred_to_gt = distance_to_gt[pred_surface]
    del distance_to_gt
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing_array)
    gt_to_pred = distance_to_pred[gt_surface]
    del distance_to_pred

    value = max(
        float(np.percentile(pred_to_gt, 95)),
        float(np.percentile(gt_to_pred, 95)),
    )
    return {
        "status": "nonempty_pair",
        "conditional_mm": value,
        "failure_aware_mm": value,
        "empty_case_penalty_mm": diagonal,
    }


def classify_legacy_size(diameter_mm: float) -> str:
    """Classify the paper's study-defined 0--10--20--30 mm size strata."""
    for name, (lower, upper) in LEGACY_SIZE_BINS_MM.items():
        if lower <= diameter_mm < upper:
            return name
    raise ValueError(f"Invalid lesion diameter: {diameter_mm}")


def axial_feret_diameter_mm(component: np.ndarray, spacing: Sequence[float]) -> float:
    """Legacy maximum-area axial-slice Feret diameter in physical millimetres.

    Raw converted NIfTI arrays are ordered ``(X, Y, Z)``.  The axial plane is
    therefore ``X,Y`` and slices are selected along ``Z``.  The paper datasets
    have isotropic in-plane spacing (0.9375 mm); a mismatch is rejected rather
    than silently applying an incorrect scalar conversion.
    """
    mask = np.asarray(component, dtype=bool)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if mask.ndim != 3 or spacing_array.shape != (3,):
        raise ValueError("Feret diameter expects a 3D mask and three spacing values")
    if not mask.any():
        raise ValueError("Cannot measure an empty component")
    if not math.isclose(spacing_array[0], spacing_array[1], rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(
            "Legacy axial Feret requires isotropic in-plane spacing; "
            f"got {spacing_array[:2].tolist()}"
        )

    slice_index = int(np.argmax(mask.sum(axis=(0, 1))))
    axial = mask[:, :, slice_index]
    # Preserve the paper's existing size-bin implementation: the 3D lesion is
    # 26-connected, while scipy's default 4-connectivity is used again within
    # its maximum-area 2D slice before selecting the largest region.
    labeled, _ = label(axial)
    regions = regionprops(labeled)
    if not regions:
        raise RuntimeError("No 2D region found in a non-empty 3D component")
    region = max(regions, key=lambda item: item.area)
    return float(region.feret_diameter_max * spacing_array[0])


def _components(
    mask: np.ndarray,
    *,
    min_voxels: int,
) -> tuple[np.ndarray, list[int], list[int], dict[int, int]]:
    labeled, count = label(mask, structure=CONNECTIVITY_26)
    voxel_counts = np.bincount(labeled.ravel(), minlength=count + 1)
    retained = [index for index in range(1, count + 1) if voxel_counts[index] >= min_voxels]
    ignored = [index for index in range(1, count + 1) if voxel_counts[index] < min_voxels]
    sizes = {index: int(voxel_counts[index]) for index in retained}
    return labeled, retained, ignored, sizes


def _dice_matrix(
    gt_labels: np.ndarray,
    gt_ids: Sequence[int],
    gt_sizes: dict[int, int],
    pred_labels: np.ndarray,
    pred_ids: Sequence[int],
    pred_sizes: dict[int, int],
) -> np.ndarray:
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    pred_column = {component_id: column for column, component_id in enumerate(pred_ids)}
    for row, gt_id in enumerate(gt_ids):
        overlaps = np.bincount(pred_labels[gt_labels == gt_id].ravel())
        for pred_id in np.flatnonzero(overlaps):
            column = pred_column.get(int(pred_id))
            if column is None:
                continue
            intersection = int(overlaps[pred_id])
            matrix[row, column] = 2.0 * intersection / (gt_sizes[gt_id] + pred_sizes[int(pred_id)])
    return matrix


def _one_to_one_matches(
    dice_matrix: np.ndarray,
    *,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, then maximum-Dice, strict-threshold matching."""
    if dice_matrix.size == 0:
        return []
    eligible = dice_matrix > threshold
    if not eligible.any():
        return []

    # One additional valid edge must outweigh every possible Dice-sum change,
    # so assignment first maximizes the number of detections and then Dice.
    bonus = float(min(dice_matrix.shape) + 1)
    weights = np.where(eligible, bonus + dice_matrix, 0.0)
    rows, columns = linear_sum_assignment(-weights)
    return [
        (int(row), int(column), float(dice_matrix[row, column]))
        for row, column in zip(rows, columns, strict=True)
        if eligible[row, column]
    ]


def matched_component_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float],
    *,
    min_voxels: int = 5,
    detection_threshold: float = 0.1,
) -> dict[str, object]:
    """Study-defined one-to-one lesion detection and full-component Dice.

    Components use 26-connectivity.  GT and prediction components below
    ``min_voxels`` are outside the analysis.  Eligible matches must have
    unsmoothed full-component Dice strictly greater than ``detection_threshold``.
    Every retained unmatched prediction is a false positive, including one that
    touches only a discarded sub-threshold GT component.  The latter contact is
    recorded so this edge case remains auditable without suppressing false
    positives.
    """
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(f"Expected equal 3D masks, got {pred.shape} and {gt.shape}")
    if min_voxels < 1:
        raise ValueError("min_voxels must be positive")
    if not 0 <= detection_threshold < 1:
        raise ValueError("detection_threshold must be in [0, 1)")

    gt_labels, gt_ids, ignored_gt_ids, gt_sizes = _components(gt, min_voxels=min_voxels)
    pred_labels, pred_ids, ignored_pred_ids, pred_sizes = _components(
        pred,
        min_voxels=min_voxels,
    )
    matrix = _dice_matrix(gt_labels, gt_ids, gt_sizes, pred_labels, pred_ids, pred_sizes)
    assignments = _one_to_one_matches(matrix, threshold=detection_threshold)

    matched_by_gt = {row: (column, score) for row, column, score in assignments}
    matched_pred_columns = {column for _, column, _ in assignments}
    ignored_gt_mask = np.isin(gt_labels, ignored_gt_ids) if ignored_gt_ids else None

    gt_records: list[dict[str, object]] = []
    for row, gt_id in enumerate(gt_ids):
        component = gt_labels == gt_id
        diameter = axial_feret_diameter_mm(component, spacing)
        match = matched_by_gt.get(row)
        matched_column = match[0] if match is not None else None
        matched_score = match[1] if match is not None else 0.0
        best_score = float(matrix[row].max()) if matrix.shape[1] else 0.0
        gt_records.append(
            {
                "gt_component_id": int(gt_id),
                "volume_voxels": gt_sizes[gt_id],
                "diameter_mm": diameter,
                "size_category": classify_legacy_size(diameter),
                "detected": match is not None,
                "matched_pred_component_id": (
                    int(pred_ids[matched_column]) if matched_column is not None else None
                ),
                "matched_dice": float(matched_score),
                "best_candidate_dice": best_score,
            }
        )

    fp_records: list[dict[str, object]] = []
    unmatched_pred_touching_excluded_gt = 0
    for column, pred_id in enumerate(pred_ids):
        if column in matched_pred_columns:
            continue
        pred_component = pred_labels == pred_id
        overlaps_ignored_gt = (
            bool(np.logical_and(pred_component, ignored_gt_mask).any())
            if ignored_gt_mask is not None
            else False
        )
        unmatched_pred_touching_excluded_gt += int(overlaps_ignored_gt)
        diameter = axial_feret_diameter_mm(pred_component, spacing)
        fp_records.append(
            {
                "pred_component_id": int(pred_id),
                "volume_voxels": pred_sizes[pred_id],
                "diameter_mm": diameter,
                "size_category": classify_legacy_size(diameter),
                "touches_excluded_gt_component": overlaps_ignored_gt,
            }
        )

    tp = len(assignments)
    fn = len(gt_ids) - tp
    fp = len(fp_records)
    evaluated_predictions = tp + fp
    sensitivity = tp / len(gt_ids) if gt_ids else None
    precision = tp / evaluated_predictions if evaluated_predictions else None
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else 1.0
    matched_scores = [score for _, _, score in assignments]
    all_gt_scores = [float(record["matched_dice"]) for record in gt_records]
    penalized_denominator = len(gt_ids) + fp
    penalized_dice = (
        float(sum(matched_scores) / penalized_denominator) if penalized_denominator else 1.0
    )

    return {
        "parameters": {
            "connectivity": 26,
            "min_component_voxels": min_voxels,
            "detection_rule": f"full_component_dice > {detection_threshold}",
            "matching": "one_to_one_max_cardinality_then_dice",
            "diameter": "2D Feret on maximum-area axial slice",
            "size_bins_mm": LEGACY_SIZE_BINS_MM,
        },
        "counts": {
            "n_gt": len(gt_ids),
            "n_pred_retained": len(pred_ids),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "ignored_gt_below_min_voxels": len(ignored_gt_ids),
            "ignored_pred_below_min_voxels": len(ignored_pred_ids),
            "unmatched_pred_touching_excluded_gt": unmatched_pred_touching_excluded_gt,
        },
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": float(f1),
        "penalized_lesion_dice": penalized_dice,
        "matched_dice": sample_summary(matched_scores),
        "all_gt_dice": sample_summary(all_gt_scores),
        "gt_lesions": gt_records,
        "false_positive_lesions": fp_records,
    }
