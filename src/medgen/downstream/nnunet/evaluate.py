"""Post-training evaluation of nnU-Net predictions using MedGen metrics.

After nnU-Net training and inference, this module:
1. Loads nnU-Net predictions and ground truth NIfTIs
2. Computes global metrics (precision, recall, HD95) via GlobalSegMetrics
3. Computes per-tumor-size Dice/IoU via SegRegionalMetricsTracker
4. Saves results as JSON

Reuses existing metrics infrastructure:
    - medgen.metrics.seg_metrics.GlobalSegMetrics
    - medgen.metrics.regional.tracker_seg.SegRegionalMetricsTracker
"""
import json
import logging
import os

import nibabel as nib
import numpy as np
import torch

from medgen.metrics.regional.tracker_seg import SegRegionalMetricsTracker
from medgen.metrics.seg_metrics import GlobalSegMetrics

logger = logging.getLogger(__name__)

# Default voxel spacing for brainmetshare-3 (D, H, W) in mm
DEFAULT_VOXEL_SPACING_3D = (1.0, 0.9375, 0.9375)


def _load_nifti_binary(path: str) -> np.ndarray:
    """Load a NIfTI file and binarize to bool array."""
    data = nib.load(path).get_fdata()
    return (data > 0.5).astype(bool)


def _find_prediction_pairs(
    pred_dir: str,
    gt_dir: str,
) -> list[tuple[str, str, str]]:
    """Find matching prediction / ground-truth pairs.

    Args:
        pred_dir: Directory containing prediction NIfTIs.
        gt_dir: Directory containing ground truth NIfTIs (labelsTs/).

    Returns:
        List of (case_id, pred_path, gt_path) tuples.
    """
    pairs = []
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith('.nii.gz'):
            continue
        case_id = fname.replace('.nii.gz', '')
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            logger.warning(f"No ground truth for {case_id}, skipping")
            continue
        pairs.append((case_id, os.path.join(pred_dir, fname), gt_path))
    return pairs


def _dice_from_counts(intersection: float, union: float) -> float:
    """Dice = 2·∩/(|A|+|B|) with the TN convention: empty/empty → 1.0.

    Empty/empty means both pred and gt have zero foreground voxels (true
    negative for the whole region). Scoring 0.0 in that case (the old
    bug at evaluate.py:138) penalises perfect tumour-free predictions.
    """
    if union <= 0:
        return 1.0
    return 2.0 * float(intersection) / float(union)


def _lesionwise_dice_bratsmets(
    pred: np.ndarray,
    gt: np.ndarray,
    min_overlap_voxels: int = 1,
) -> float:
    """BraTS-Mets 2023/2024 lesion-wise Dice.

    Algorithm (matches the BraTS-Mets challenge evaluation):
      1. 3D connected components in GT (26-connectivity) → GT lesions.
      2. 3D connected components in pred (26-connectivity) → predicted lesions.
      3. Per GT lesion: compute voxel Dice between that lesion and the union of
         predicted voxels overlapping its bounding box.
         - If at least `min_overlap_voxels` predicted voxels lie inside the
           GT lesion: it's a detected lesion, contributes its voxel Dice.
         - Otherwise: missed (false negative), contributes Dice = 0.
      4. Per predicted lesion with NO overlap with any GT lesion (spurious
         lesion = FP): contributes Dice = 0.
      5. Patient lesion-wise Dice = mean of all (TP Dices + 0's for FN's + 0's
         for FP-only predicted lesions). Denominator = n_GT + n_FP_only_pred.

    This penalises spurious predictions exactly the way volumetric Dice does
    NOT — a single tiny FP blob counts the same as a missed lesion. For sparse
    multi-lesion brain mets this is the most informative segmentation metric,
    and it's the official BraTS-Mets ranking metric (arXiv:2306.00838 §3.2).

    Reference: Moawad et al. 2023, BraTS-Mets challenge report.
    """
    from scipy.ndimage import label as _ccl
    structure = np.ones((3, 3, 3), dtype=np.uint8)  # 26-connectivity in 3D
    gt_lab, n_gt = _ccl(gt, structure=structure)
    pred_lab, n_pred = _ccl(pred, structure=structure)
    if n_gt == 0 and n_pred == 0:
        return 1.0  # empty/empty -> perfect (matches the TN convention)
    contributions: list[float] = []
    # Track which predicted lesions intersect at least one GT lesion.
    pred_lesion_hit = np.zeros(n_pred + 1, dtype=bool)
    for gid in range(1, n_gt + 1):
        gt_mask = (gt_lab == gid)
        overlap = pred & gt_mask
        n_overlap = int(overlap.sum())
        if n_overlap >= min_overlap_voxels:
            # Per-lesion voxel Dice: include all predicted voxels that touch
            # this lesion (within its bounding box), so neighbouring FPs don't
            # contaminate a different lesion's score.
            coords = np.where(gt_mask)
            slc = tuple(slice(c.min(), c.max() + 1) for c in coords)
            p_local = pred[slc] & (pred_lab[slc] != 0)
            g_local = gt_mask[slc]
            inter = int((p_local & g_local).sum())
            denom = int(p_local.sum()) + int(g_local.sum())
            contributions.append(2.0 * inter / max(denom, 1))
            # Mark all predicted lesions that touched this GT as "hit"
            for pid in np.unique(pred_lab[gt_mask]):
                if pid != 0:
                    pred_lesion_hit[pid] = True
        else:
            contributions.append(0.0)  # missed lesion (FN)
    # Add one Dice=0 per predicted lesion that didn't touch any GT (FP-only)
    for pid in range(1, n_pred + 1):
        if not pred_lesion_hit[pid]:
            contributions.append(0.0)
    if not contributions:
        return float('nan')
    return float(np.mean(contributions))


def _slicewise_dice_yi2023(pred: np.ndarray, gt: np.ndarray,
                           axis: int = 0) -> float:
    """Ottesen 2023 slice-wise Dice with empty-empty slices scored as 1.0.

    Matches §2.5 of Ottesen et al. 2023 (PMC9889663):
    "Dice = (2·TP)/(2·TP + FP + FN)... Segmentation performance was
    evaluated using a slice-wise dice similarity coefficient. All correctly
    predicted zero-slices were given a perfect dice score of 1."

    The slice axis must be the acquisition plane. BrainMetShare BRAVO is
    acquired SAGITTAL (Table 1 of the paper). In nibabel's RAS+ layout for
    this dataset the array shape is [X=LR, Y=AP, Z=SI]; sagittal slices are
    therefore stacked along axis 0 (LR). Using the last axis (Z=SI = axial)
    gives ~0.73 instead of ~0.85 because there are fewer, thicker slabs and
    therefore fewer empty-empty 1.0 contributions to the mean. The model is
    the same; only the metric axis differs.
    """
    n_slices = pred.shape[axis]
    if n_slices == 0:
        return float('nan')
    # Vectorized: reduce all axes except `axis` to get per-slice voxel sums.
    reduce_axes = tuple(i for i in range(pred.ndim) if i != axis)
    inter = np.logical_and(pred, gt).sum(axis=reduce_axes).astype(np.float64)
    pred_sum = pred.sum(axis=reduce_axes).astype(np.float64)
    gt_sum = gt.sum(axis=reduce_axes).astype(np.float64)
    union = pred_sum + gt_sum
    # Empty/empty -> 1.0; else 2*TP / (|P| + |G|).
    dices = np.where(union > 0, 2.0 * inter / np.maximum(union, 1e-12), 1.0)
    return float(dices.mean())


def evaluate_predictions(
    pred_dir: str,
    gt_dir: str,
    output_path: str | None = None,
    tensorboard_dir: str | None = None,
    voxel_spacing: tuple[float, ...] = DEFAULT_VOXEL_SPACING_3D,
    image_size: int = 256,
    fov_mm: float = 240.0,
    spatial_dims: int = 3,
) -> dict:
    """Evaluate nnU-Net predictions against ground truth.

    Args:
        pred_dir: Directory with prediction NIfTIs (from nnUNetPredictor).
        gt_dir: Directory with ground truth NIfTIs (labelsTs/).
        output_path: Where to save results JSON (optional).
        tensorboard_dir: TensorBoard log directory to write test metrics to.
            If provided, logs test/ scalars to the training TensorBoard.
        voxel_spacing: Voxel spacing in mm (D, H, W).
        image_size: Image size in pixels (H=W).
        fov_mm: Field of view in mm.
        spatial_dims: 2 or 3.

    Returns:
        Dict with 'global_metrics', 'regional_metrics', 'detection_metrics',
        'dice_variants', 'per_case', and 'num_cases'. Per-patient Dice
        aggregates are 'dice_mean/std' (volumetric foreground — PRIMARY),
        'dice_yi2023_slicewise_mean/std' (Ottesen 2023 sagittal slice-wise),
        and 'dice_lesionwise_bratsmets_mean/std' (BraTS-Mets lesion-wise).
    """
    pairs = _find_prediction_pairs(pred_dir, gt_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No prediction-GT pairs found. pred_dir={pred_dir}, gt_dir={gt_dir}"
        )

    logger.info(f"Evaluating {len(pairs)} cases")

    # Initialize metrics
    global_metrics = GlobalSegMetrics(compute_hd95=True, device=torch.device('cpu'))
    regional_tracker = SegRegionalMetricsTracker(
        image_size=image_size,
        fov_mm=fov_mm,
        spatial_dims=spatial_dims,
        voxel_spacing=voxel_spacing,
    )
    global_metrics.reset()
    regional_tracker.reset()

    per_case = {}

    for case_id, pred_path, gt_path in pairs:
        pred_np = _load_nifti_binary(pred_path)
        gt_np = _load_nifti_binary(gt_path)

        if pred_np.shape != gt_np.shape:
            logger.warning(
                f"Shape mismatch for {case_id}: pred={pred_np.shape}, gt={gt_np.shape}"
            )
            continue

        # Convert to tensors [1, 1, ...] (batch=1, channel=1)
        pred_t = torch.from_numpy(pred_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy(gt_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        # For 3D NIfTI data: shape is [H, W, D] -> need [1, 1, D, H, W]
        if spatial_dims == 3 and pred_t.ndim == 5:
            # NIfTI convention: [H, W, D] -> permute to [D, H, W]
            pred_t = pred_t.permute(0, 1, 4, 2, 3)
            gt_t = gt_t.permute(0, 1, 4, 2, 3)

        # Update global metrics (already binary, no sigmoid needed)
        global_metrics.update(pred_t, gt_t, apply_sigmoid=False)

        # Update regional metrics
        regional_tracker.update(pred_t, gt_t, apply_sigmoid=False)

        # ── Per-case Dice variants ─────────────────────────────────
        # (1) Per-patient volumetric foreground Dice — PRIMARY.
        intersection = float((pred_np & gt_np).sum())
        union = float(pred_np.sum()) + float(gt_np.sum())
        dice = _dice_from_counts(intersection, union)

        # (2) BraTS-Mets lesion-wise Dice (FPs count as Dice=0).
        dice_lesion = _lesionwise_dice_bratsmets(pred_np, gt_np)

        # (3) Ottesen 2023 sagittal slice-wise Dice with empty-slice = 1.0.
        dice_yi = _slicewise_dice_yi2023(pred_np, gt_np)

        per_case[case_id] = {
            'dice': float(dice),
            'dice_lesionwise_bratsmets': float(dice_lesion),
            'dice_yi2023_slicewise': float(dice_yi),
        }

        logger.info(
            f"  {case_id}: Dice={dice:.4f}  "
            f"Lesion={dice_lesion:.4f}  Yi2023={dice_yi:.4f}"
        )

    # Compute final metrics
    global_results = global_metrics.compute()
    regional_results = regional_tracker.compute()
    detection_results = regional_tracker.get_detection_summary()

    # Compute per-tumor Dice std (across-tumor variability) from raw records.
    # The tracker only stores sums internally, so std isn't in compute()'s
    # output. We add it here so the report can show `dice ± std` per bin.
    per_tumor_records = regional_tracker.get_per_tumor_records()
    if per_tumor_records:
        all_dices = np.array([r['dice'] for r in per_tumor_records], dtype=np.float64)
        regional_results['dice_std'] = (
            float(all_dices.std(ddof=1)) if all_dices.size > 1 else 0.0
        )
        for size_name in ('tiny', 'small', 'medium', 'large'):
            size_dices = np.array(
                [r['dice'] for r in per_tumor_records if r['size_cat'] == size_name],
                dtype=np.float64,
            )
            if size_dices.size > 1:
                regional_results[f'dice_{size_name}_std'] = float(size_dices.std(ddof=1))
            elif size_dices.size == 1:
                regional_results[f'dice_{size_name}_std'] = 0.0
            else:
                regional_results[f'dice_{size_name}_std'] = float('nan')

    # Aggregate per-patient Dice variants across cases (mean ± std).
    # 'dice' is the TN-fixed volumetric per-patient Dice (was the buggy
    # field at evaluate.py:138; same name kept so existing tables stay valid
    # but become correct on empty-empty cases).
    dice_summary = {}
    for key in (
        'dice',
        'dice_lesionwise_bratsmets',
        'dice_yi2023_slicewise',
    ):
        values = np.asarray(
            [c[key] for c in per_case.values() if not np.isnan(c[key])],
            dtype=np.float64,
        )
        if values.size == 0:
            dice_summary[f'{key}_mean'] = float('nan')
            dice_summary[f'{key}_std'] = float('nan')
            dice_summary[f'{key}_n'] = 0
        else:
            dice_summary[f'{key}_mean'] = float(values.mean())
            dice_summary[f'{key}_std'] = float(values.std(ddof=1) if values.size > 1 else 0.0)
            dice_summary[f'{key}_n'] = int(values.size)

    results = {
        'global_metrics': global_results,
        'regional_metrics': regional_results,
        'detection_metrics': detection_results,
        'dice_variants': dice_summary,
        'per_case': per_case,
        'num_cases': len(per_case),
    }

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    # Log to TensorBoard
    if tensorboard_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_scalar('test/precision', global_results['precision'], 0)
        writer.add_scalar('test/recall', global_results['recall'], 0)
        if 'hd95' in global_results:
            writer.add_scalar('test/hd95', global_results['hd95'], 0)
        writer.add_scalar('test/dice', regional_results.get('dice', 0), 0)
        writer.add_scalar('test/iou', regional_results.get('iou', 0), 0)
        for size in ('tiny', 'small', 'medium', 'large'):
            d = regional_results.get(f'dice_{size}', float('nan'))
            if not np.isnan(d):
                writer.add_scalar(f'test/dice_{size}', d, 0)
        # Per-lesion detection rates (overall and by size).
        writer.add_scalar(
            'test/detection_rate',
            detection_results.get('detection_rate', 0.0),
            0,
        )
        writer.add_scalar(
            'test/false_positives',
            detection_results.get('false_positives', 0.0),
            0,
        )
        for size in ('tiny', 'small', 'medium', 'large'):
            rate = detection_results.get(f'detection_rate_{size}')
            if rate is not None:
                writer.add_scalar(f'test/detection_rate_{size}', rate, 0)
        # Per-patient Dice variants (mean across cases).
        for key in (
            'dice',
            'dice_lesionwise_bratsmets',
            'dice_yi2023_slicewise',
        ):
            v = dice_summary[f'{key}_mean']
            if not np.isnan(v):
                writer.add_scalar(f'test/{key}_per_patient_mean', v, 0)
        writer.close()
        logger.info(f"Test metrics logged to TensorBoard: {tensorboard_dir}")

    # Log summary via logger (not print) so output respects handlers + redirect contexts.
    logger.info(f"=== Evaluation Results ({len(per_case)} cases) ===")
    hd95_suffix = (
        f", HD95={global_results['hd95']:.2f}mm" if 'hd95' in global_results else ""
    )
    logger.info(
        f"Global: precision={global_results['precision']:.4f}, "
        f"recall={global_results['recall']:.4f}{hd95_suffix}"
    )
    overall_d = regional_results.get('dice', 0.0)
    overall_d_std = regional_results.get('dice_std', float('nan'))
    logger.info(
        f"Regional: overall_dice={overall_d:.4f} ± {overall_d_std:.4f}, "
        f"overall_iou={regional_results.get('iou', 0):.4f}"
    )
    for size in ('tiny', 'small', 'medium', 'large'):
        d = regional_results.get(f'dice_{size}', float('nan'))
        d_std = regional_results.get(f'dice_{size}_std', float('nan'))
        n = regional_results.get(f'n_tumors_{size}', 0)
        logger.info(
            f"  {size}: dice={d:.4f} ± {d_std:.4f}  (n={n})"
        )

    logger.info(
        f"Per-lesion detection rate (Dice > "
        f"{regional_tracker.detection_threshold:.2f} criterion):"
    )
    overall_det = detection_results.get('detection_rate', 0.0)
    fp_total = detection_results.get('false_positives', 0)
    logger.info(
        f"  overall: {overall_det*100:.1f}%   FP total: {int(fp_total)}"
    )
    for size in ('tiny', 'small', 'medium', 'large'):
        rate = detection_results.get(f'detection_rate_{size}')
        n = regional_results.get(f'n_tumors_{size}', 0)
        fp = detection_results.get(f'fp_{size}', 0)
        if rate is not None and n > 0:
            logger.info(
                f"  {size}: detection={rate*100:>5.1f}%  "
                f"(n={n}, FPs={int(fp)})"
            )

    logger.info("Dice variants (per-patient mean ± std across cases):")
    logger.info(
        f"  PRIMARY  volumetric (per-volume foreground):  "
        f"{dice_summary['dice_mean']:.4f} ± {dice_summary['dice_std']:.4f}  "
        f"(nnU-Net / BrainMetShare / most brain-mets literature)"
    )
    logger.info(
        f"  lesion-wise (BraTS-Mets, FPs penalised):      "
        f"{dice_summary['dice_lesionwise_bratsmets_mean']:.4f} ± "
        f"{dice_summary['dice_lesionwise_bratsmets_std']:.4f}  "
        f"(BraTS-Mets 2023/2024 ranking metric)"
    )
    logger.info(
        f"  Ottesen 2023 slice-wise (sagittal axis):      "
        f"{dice_summary['dice_yi2023_slicewise_mean']:.4f} ± "
        f"{dice_summary['dice_yi2023_slicewise_std']:.4f}  "
        f"(literature: 0.85 ± 0.13 for nnU-Net, PMC9889663)"
    )

    return results
