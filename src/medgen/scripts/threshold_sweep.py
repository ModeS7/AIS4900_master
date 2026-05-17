"""Post-hoc threshold sweep on nnU-Net softmax probabilities.

Background
----------
nnU-Net binarises predictions via argmax over softmax channels, equivalent
to thresholding the foreground probability at 0.5. For class-imbalanced
tasks like brain-metastasis segmentation (foreground is 0.001-0.06% of
voxels), the model often outputs sharply-peaked-but-cautious foreground
probabilities, and lowering the threshold below 0.5 trades a small
increase in FP for a larger increase in TP, lifting volumetric Dice.

This script does NOT retrain; it only re-binarises already-computed
softmax outputs at different thresholds and recomputes volumetric Dice.

Two modes
---------

1. ``--mode sweep``: sweep thresholds on a single dataset, print Dice
   per threshold. Useful as an upper-bound check ("how much could
   threshold tuning possibly help?") but selecting the optimum on the
   test set itself is biased. Quote the per-threshold curve, NOT a
   single "best" number, for any thesis-grade claim.

2. ``--mode tune-eval``: pass ``--tune-probs-dir <val>`` and
   ``--tune-gt-dir <val_gt>``, then ``--eval-probs-dir <test>`` and
   ``--eval-gt-dir <test_gt>``. The script (a) picks the threshold that
   maximises mean per-volume Dice on the tune set, (b) applies it to
   the eval set, (c) reports Dice on the eval set. This is the
   publishable workflow matching Ottesen 2025's "per-fold threshold
   tuning on validation" — supply each fold's val set as the tune set,
   or aggregate all 5 folds' vals.

Input format
------------
Each .npz file produced by nnUNetv2 with ``save_probabilities=True``
contains a ``probabilities`` array of shape ``[n_classes, ...]``. For
binary segmentation n_classes=2 (background=0, foreground=1). Ground
truth NIfTI files (``labelsTs/<case>.nii.gz``) are loaded and binarised
with ``> 0.5``.

The script asserts that probability arrays match GT spatial shape; if
they don't, it errors out loudly so the user knows the saved probs are
in preprocessed (resampled) space and need post-processing first.

Usage
-----

    # Mode 1: sweep on test set (upper bound)
    python -m medgen.scripts.threshold_sweep \\
        --mode sweep \\
        --probs-dir runs/.../eval_exp3/predictions \\
        --gt-dir   .../labelsTs \\
        --output   runs/.../eval_exp3/threshold_sweep_test.json

    # Mode 2: tune on val, eval on test (publishable)
    python -m medgen.scripts.threshold_sweep \\
        --mode tune-eval \\
        --tune-probs-dir .../fold_X/validation \\
        --tune-gt-dir    .../labelsTr  \\
        --eval-probs-dir runs/.../eval_exp3/predictions \\
        --eval-gt-dir    .../labelsTs \\
        --output         runs/.../eval_exp3/threshold_sweep_val_tuned.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: tuple[float, ...] = (
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
)
FG_CHANNEL = 1  # nnU-Net binary seg: channel 0 = bg, channel 1 = fg


def _load_probs(npz_path: str) -> np.ndarray:
    """Load the foreground-channel probability map from an nnU-Net .npz.

    Returns shape [D, H, W] float32 array of foreground probabilities.
    """
    data = np.load(npz_path)
    if 'probabilities' not in data.files:
        raise KeyError(
            f"{npz_path} has no 'probabilities' key. Found: {data.files}"
        )
    probs = data['probabilities']
    if probs.ndim != 4 or probs.shape[0] < 2:
        raise ValueError(
            f"{npz_path}: expected probabilities of shape [n_classes>=2, D, H, W], "
            f"got {probs.shape}"
        )
    return probs[FG_CHANNEL].astype(np.float32)


def _load_gt(gt_path: str) -> np.ndarray:
    """Load a GT NIfTI as a boolean foreground mask, axes ordered to [D, H, W]."""
    arr = nib.load(gt_path).get_fdata()
    # nibabel returns [H, W, D] for our brain-mets NIfTIs (RAS+). Match the
    # convention used by `evaluate.py` which permutes to [D, H, W] for
    # downstream metrics; here we do the same so probs and GT align.
    if arr.ndim != 3:
        raise ValueError(f"{gt_path}: expected 3D NIfTI, got shape {arr.shape}")
    arr = np.transpose(arr, (2, 0, 1))  # [H, W, D] -> [D, H, W]
    return arr > 0.5


def _find_case_pairs(probs_dir: str, gt_dir: str) -> list[tuple[str, str, str]]:
    """Find (case_id, npz_path, gt_path) triples for every probs file with a GT."""
    pairs: list[tuple[str, str, str]] = []
    for fname in sorted(os.listdir(probs_dir)):
        if not fname.endswith('.npz'):
            continue
        case_id = fname[:-len('.npz')]
        gt_path = os.path.join(gt_dir, f'{case_id}.nii.gz')
        if not os.path.isfile(gt_path):
            logger.warning(f"No GT for {case_id}, skipping")
            continue
        pairs.append((case_id, os.path.join(probs_dir, fname), gt_path))
    if not pairs:
        raise FileNotFoundError(
            f"No probs/GT pairs found. probs_dir={probs_dir}, gt_dir={gt_dir}"
        )
    return pairs


def _volumetric_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-volume Dice with TN convention (empty/empty = 1.0)."""
    inter = float(np.logical_and(pred, gt).sum())
    union = float(pred.sum()) + float(gt.sum())
    if union <= 0:
        return 1.0
    return 2.0 * inter / union


def sweep_thresholds(
    pairs: list[tuple[str, str, str]],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> dict:
    """For each threshold, compute mean ± std volumetric Dice over all cases.

    Returns
    -------
    dict with keys:
        'thresholds':    list[float]                          (the sweep grid)
        'dice_per_thr':  list[float]  (mean over cases per threshold)
        'std_per_thr':   list[float]  (std over cases per threshold)
        'per_case':      dict[case_id, dict[thr, dice]]       (raw values)
        'best_threshold': float       (argmax of dice_per_thr)
        'best_dice':      float
        'baseline_dice_at_0.5': float (current nnU-Net default)
    """
    thresholds = sorted(set(thresholds))
    per_case: dict[str, dict[str, float]] = {}
    dice_matrix = np.zeros((len(pairs), len(thresholds)), dtype=np.float64)

    for i, (case_id, npz_path, gt_path) in enumerate(pairs):
        gt = _load_gt(gt_path)
        probs = _load_probs(npz_path)
        if probs.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for {case_id}: probs={probs.shape} vs gt={gt.shape}. "
                f"The .npz is likely in preprocessed (resampled) space — re-run "
                f"inference with save_probabilities and ensure nnU-Net writes the "
                f"original-space probabilities."
            )
        case_dices: dict[str, float] = {}
        for j, thr in enumerate(thresholds):
            pred = probs > thr
            d = _volumetric_dice(pred, gt)
            dice_matrix[i, j] = d
            case_dices[f'{thr:.2f}'] = d
        per_case[case_id] = case_dices
        logger.info(f"  {case_id}: swept {len(thresholds)} thresholds")

    dice_mean = dice_matrix.mean(axis=0)
    dice_std = (
        dice_matrix.std(axis=0, ddof=1)
        if dice_matrix.shape[0] > 1
        else np.zeros_like(dice_mean)
    )
    best_idx = int(np.argmax(dice_mean))
    baseline_idx = thresholds.index(0.50) if 0.50 in thresholds else None

    return {
        'thresholds': list(thresholds),
        'dice_per_thr': dice_mean.tolist(),
        'std_per_thr': dice_std.tolist(),
        'per_case': per_case,
        'best_threshold': float(thresholds[best_idx]),
        'best_dice': float(dice_mean[best_idx]),
        'best_dice_std': float(dice_std[best_idx]),
        'baseline_dice_at_0.5': (
            float(dice_mean[baseline_idx]) if baseline_idx is not None else float('nan')
        ),
        'n_cases': len(pairs),
    }


def evaluate_at_threshold(
    pairs: list[tuple[str, str, str]],
    threshold: float,
) -> dict:
    """Apply a single threshold and report mean ± std volumetric Dice."""
    dices: list[float] = []
    per_case: dict[str, float] = {}
    for case_id, npz_path, gt_path in pairs:
        gt = _load_gt(gt_path)
        probs = _load_probs(npz_path)
        if probs.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for {case_id}: probs={probs.shape} vs gt={gt.shape}"
            )
        d = _volumetric_dice(probs > threshold, gt)
        dices.append(d)
        per_case[case_id] = d
    arr = np.asarray(dices, dtype=np.float64)
    return {
        'threshold': float(threshold),
        'dice_mean': float(arr.mean()),
        'dice_std': float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        'n_cases': int(arr.size),
        'per_case': per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['sweep', 'tune-eval'])
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=list(DEFAULT_THRESHOLDS),
                        help='Threshold grid (default: 0.10..0.80 step 0.05)')
    parser.add_argument('--output', required=True,
                        help='Path to write the result JSON.')

    # Mode 1
    parser.add_argument('--probs-dir', help='[sweep] Directory of .npz probability files')
    parser.add_argument('--gt-dir', help='[sweep] Directory of GT NIfTIs')

    # Mode 2
    parser.add_argument('--tune-probs-dir',
                        help='[tune-eval] Probability dir to tune threshold on (e.g. val)')
    parser.add_argument('--tune-gt-dir', help='[tune-eval] GT dir matching tune-probs-dir')
    parser.add_argument('--eval-probs-dir',
                        help='[tune-eval] Probability dir to apply chosen threshold (e.g. test)')
    parser.add_argument('--eval-gt-dir', help='[tune-eval] GT dir matching eval-probs-dir')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if args.mode == 'sweep':
        if not (args.probs_dir and args.gt_dir):
            parser.error('--mode sweep requires --probs-dir and --gt-dir')
        pairs = _find_case_pairs(args.probs_dir, args.gt_dir)
        logger.info(f"Sweep over {len(pairs)} cases x {len(args.thresholds)} thresholds")
        result = sweep_thresholds(pairs, args.thresholds)
        # Console summary
        logger.info("=== Threshold sweep result ===")
        for thr, d, s in zip(
            result['thresholds'],
            result['dice_per_thr'],
            result['std_per_thr'],
            strict=True,
        ):
            marker = '  *' if thr == result['best_threshold'] else '   '
            logger.info(f"{marker} t={thr:.2f}: dice={d:.4f} ± {s:.4f}")
        logger.info(
            f"BASELINE (t=0.50): {result['baseline_dice_at_0.5']:.4f}"
        )
        logger.info(
            f"BEST     (t={result['best_threshold']:.2f}): {result['best_dice']:.4f} "
            f"± {result['best_dice_std']:.4f}"
        )
        delta = result['best_dice'] - result['baseline_dice_at_0.5']
        logger.info(f"GAIN over baseline: +{delta:.4f}")
        out = result
    else:
        required = (args.tune_probs_dir, args.tune_gt_dir,
                    args.eval_probs_dir, args.eval_gt_dir)
        if not all(required):
            parser.error('--mode tune-eval requires --tune-probs-dir, --tune-gt-dir, '
                         '--eval-probs-dir, --eval-gt-dir')
        tune_pairs = _find_case_pairs(args.tune_probs_dir, args.tune_gt_dir)
        eval_pairs = _find_case_pairs(args.eval_probs_dir, args.eval_gt_dir)
        logger.info(f"Tune on {len(tune_pairs)} cases, eval on {len(eval_pairs)} cases")
        tune_result = sweep_thresholds(tune_pairs, args.thresholds)
        best_t = tune_result['best_threshold']
        logger.info(f"Optimal threshold on tune set: t={best_t:.2f} "
                    f"(tune dice={tune_result['best_dice']:.4f})")
        eval_result = evaluate_at_threshold(eval_pairs, best_t)
        # Also report eval Dice at 0.5 for the gain estimate
        eval_baseline = evaluate_at_threshold(eval_pairs, 0.5)
        logger.info("=== Tune-Eval result ===")
        logger.info(
            f"BASELINE eval (t=0.50): {eval_baseline['dice_mean']:.4f} "
            f"± {eval_baseline['dice_std']:.4f}"
        )
        logger.info(
            f"TUNED    eval (t={best_t:.2f}): {eval_result['dice_mean']:.4f} "
            f"± {eval_result['dice_std']:.4f}"
        )
        gain = eval_result['dice_mean'] - eval_baseline['dice_mean']
        logger.info(f"GAIN over baseline: +{gain:.4f}")
        out = {
            'tune_sweep': tune_result,
            'chosen_threshold': best_t,
            'eval_at_baseline_0.5': eval_baseline,
            'eval_at_tuned': eval_result,
            'gain_over_baseline': float(gain),
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info(f"Results written to {args.output}")


if __name__ == '__main__':
    main()
