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
    """Find matching prediction-ground truth NIfTI pairs.

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
        if os.path.exists(gt_path):
            pairs.append((case_id, os.path.join(pred_dir, fname), gt_path))
        else:
            logger.warning(f"No ground truth for {case_id}, skipping")
    return pairs


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
        Dict with 'global_metrics', 'regional_metrics', 'per_case'.
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

        # Per-case Dice
        intersection = (pred_np & gt_np).sum()
        union = pred_np.sum() + gt_np.sum()
        dice = (2.0 * intersection) / max(union, 1)
        per_case[case_id] = {'dice': float(dice)}

        logger.info(f"  {case_id}: Dice={dice:.4f}")

    # Compute final metrics
    global_results = global_metrics.compute()
    regional_results = regional_tracker.compute()

    results = {
        'global_metrics': global_results,
        'regional_metrics': regional_results,
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

    return results
