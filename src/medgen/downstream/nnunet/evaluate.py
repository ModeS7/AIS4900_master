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


def _load_brain_mask(image_path: str) -> np.ndarray:
    """Derive a binary brain mask from a skull-stripped channel-0 NIfTI.

    BrainMetShare ships skull-stripped (BET applied upstream by the Grøvik
    et al. release pipeline), so `image > 0` recovers the exact mask Grøvik
    2020 used for their voxel-Dice gating. See
    memory/project_brainmetshare_skull_stripped.md.
    """
    data = nib.load(image_path).get_fdata()
    return (data > 0).astype(bool)


def _find_prediction_pairs(
    pred_dir: str,
    gt_dir: str,
    images_dir: str | None = None,
) -> list[tuple[str, str, str, str | None]]:
    """Find matching prediction / ground-truth / channel-0-image triples.

    Args:
        pred_dir: Directory containing prediction NIfTIs.
        gt_dir: Directory containing ground truth NIfTIs (labelsTs/).
        images_dir: Optional nnU-Net imagesTs directory. When given, the
            channel-0 image (`<case>_0000.nii.gz`) is paired alongside so the
            Grøvik-style brain-mask gating can be applied. If missing or the
            file isn't found for a case, the image path is returned as None
            and Grøvik-Dice falls back to a `pred ∪ gt`-derived mask.

    Returns:
        List of (case_id, pred_path, gt_path, image_path_or_None) tuples.
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
        image_path: str | None = None
        if images_dir is not None:
            candidate = os.path.join(images_dir, f"{case_id}_0000.nii.gz")
            if os.path.exists(candidate):
                image_path = candidate
            else:
                logger.warning(
                    f"No channel-0 image for {case_id} at {candidate}; "
                    "Grøvik-Dice will fall back to pred∪gt mask for this case."
                )
        pairs.append((case_id, os.path.join(pred_dir, fname), gt_path, image_path))
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


def _grovik_dice(pred: np.ndarray, gt: np.ndarray,
                 brain_mask: np.ndarray | None) -> float:
    """Grøvik 2020 per-patient voxel-Dice, restricted to inside the brain mask.

    Matches §Statistical Analysis of Grøvik et al. 2020 (arXiv:1903.07988):
    "Only voxels within the brain mask were considered when calculating AUC"
    and the Dice/F1 score is computed on the same brain-gated voxel set at
    the optimal probability threshold. Returns one float per patient; mean ±
    std across patients gives the comparable summary (paper: 0.79 ± 0.12).
    """
    if brain_mask is None:
        # No channel-0 image available → use the pred∪gt envelope as a
        # conservative proxy. Doesn't bias the metric (any voxel that could
        # contribute to Dice is included) and stays bounded.
        brain_mask = pred | gt
    p = pred & brain_mask
    g = gt & brain_mask
    intersection = float(np.logical_and(p, g).sum())
    union = float(p.sum()) + float(g.sum())
    return _dice_from_counts(intersection, union)


def _slicewise_dice_yi2023(pred: np.ndarray, gt: np.ndarray) -> float:
    """Ottesen 2023 slice-wise Dice with empty-empty slices scored as 1.0.

    Matches §2.5 of Ottesen et al. 2023 (PMC9889663):
    "Dice = (2·TP)/(2·TP + FP + FN)... Segmentation performance was
    evaluated using a slice-wise dice similarity coefficient. All correctly
    predicted zero-slices were given a perfect dice score of 1."

    Iterates the last array axis (axis 2 for nibabel's `[H, W, D]` layout =
    axial slice direction for BrainMetShare BRAVO). Returns the mean over
    slices for ONE patient; mean ± std across patients gives the comparable
    summary (paper nnU-Net column: 0.85 ± 0.13).
    """
    n_slices = pred.shape[-1]
    if n_slices == 0:
        return float('nan')
    dices = np.empty(n_slices, dtype=np.float64)
    for d in range(n_slices):
        p_slice = pred[..., d]
        g_slice = gt[..., d]
        intersection = float(np.logical_and(p_slice, g_slice).sum())
        union = float(p_slice.sum()) + float(g_slice.sum())
        dices[d] = _dice_from_counts(intersection, union)
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
    images_dir: str | None = None,
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
        images_dir: Optional nnU-Net imagesTs directory. When given, the
            channel-0 image is used to derive a per-patient brain mask
            (`image > 0`, matching the BrainMetShare skull-stripped release)
            for Grøvik-Dice gating. See
            memory/project_brainmetshare_skull_stripped.md.

    Returns:
        Dict with 'global_metrics', 'regional_metrics', 'per_case', plus
        aggregate fields 'dice_per_patient_mean/std',
        'dice_grovik_mean/std', and 'dice_yi2023_slicewise_mean/std'.
    """
    pairs = _find_prediction_pairs(pred_dir, gt_dir, images_dir=images_dir)
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

    for case_id, pred_path, gt_path, image_path in pairs:
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
        # (1) Per-patient volumetric Dice (TN-fixed; was the buggy 0/1 above).
        intersection = float((pred_np & gt_np).sum())
        union = float(pred_np.sum()) + float(gt_np.sum())
        dice = _dice_from_counts(intersection, union)

        # (2) Grøvik 2020 voxel-Dice over brain-masked region.
        brain_mask = _load_brain_mask(image_path) if image_path is not None else None
        if brain_mask is not None and brain_mask.shape != pred_np.shape:
            logger.warning(
                f"Brain-mask shape {brain_mask.shape} ≠ pred shape "
                f"{pred_np.shape} for {case_id}; falling back to pred∪gt mask."
            )
            brain_mask = None
        dice_grovik = _grovik_dice(pred_np, gt_np, brain_mask)

        # (3) Ottesen 2023 slice-wise Dice with empty-slice = 1.0.
        dice_yi = _slicewise_dice_yi2023(pred_np, gt_np)

        per_case[case_id] = {
            'dice': float(dice),
            'dice_grovik': float(dice_grovik),
            'dice_yi2023_slicewise': float(dice_yi),
        }

        logger.info(
            f"  {case_id}: Dice={dice:.4f}  "
            f"Grøvik={dice_grovik:.4f}  Yi2023={dice_yi:.4f}"
        )

    # Compute final metrics
    global_results = global_metrics.compute()
    regional_results = regional_tracker.compute()

    # Aggregate per-patient Dice variants across cases (mean ± std).
    # 'dice' is the TN-fixed volumetric per-patient Dice (was the buggy
    # field at evaluate.py:138; same name kept so existing tables stay valid
    # but become correct on empty-empty cases).
    dice_summary = {}
    for key in ('dice', 'dice_grovik', 'dice_yi2023_slicewise'):
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
        # Per-patient Dice variants (mean across cases).
        for key in ('dice', 'dice_grovik', 'dice_yi2023_slicewise'):
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
    logger.info(
        f"Regional: overall_dice={regional_results.get('dice', 0):.4f}, "
        f"overall_iou={regional_results.get('iou', 0):.4f}"
    )
    for size in ('tiny', 'small', 'medium', 'large'):
        d = regional_results.get(f'dice_{size}', float('nan'))
        logger.info(f"  {size}: dice={d:.4f}")

    logger.info("Dice variants (per-patient mean ± std across cases):")
    logger.info(
        f"  volumetric (TN-fixed):       "
        f"{dice_summary['dice_mean']:.4f} ± {dice_summary['dice_std']:.4f}  "
        f"(within-thesis primary)"
    )
    logger.info(
        f"  Grøvik 2020 brain-gated:     "
        f"{dice_summary['dice_grovik_mean']:.4f} ± "
        f"{dice_summary['dice_grovik_std']:.4f}  "
        f"(literature: 0.79 ± 0.12, arXiv:1903.07988)"
    )
    logger.info(
        f"  Ottesen 2023 slice-wise:     "
        f"{dice_summary['dice_yi2023_slicewise_mean']:.4f} ± "
        f"{dice_summary['dice_yi2023_slicewise_std']:.4f}  "
        f"(literature: 0.85 ± 0.13 for nnU-Net, PMC9889663)"
    )

    return results
