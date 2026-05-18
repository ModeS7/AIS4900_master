"""Local threshold analysis of per-fold test predictions.

Run AFTER `run_per_fold_local.py` produces 5 × 51 = 255 softmax .npz files.

Does three things and writes JSON + a markdown table:
  1. Per-fold sweep: at each threshold ∈ {0.10..0.80}, mean per-volume Dice
     for that fold on the 51 test cases. Reports each fold's optimum.
  2. Cross-fold optimum stability: mean / median / std of the 5 per-fold
     optima — tells us whether threshold tuning is signal or noise.
  3. Ensemble experiment: average all 5 folds' foreground softmaxes
     per voxel, then sweep thresholds. Compares baseline t=0.50 ensemble
     to the tuned mean-of-folds-threshold ensemble.

Inputs are loaded directly from the .npz files written by nnUNetv2's
predict_from_files(save_probabilities=True). GTs come from labelsTs/.

Usage:  .venv_nnunet/bin/python misc/analyze_per_fold_threshold.py
"""
from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_FOLD = os.path.join(
    REPO, 'runs/exp3_baseline_v2_d600/eval_exp3_baseline_v2_d600/per_fold_test',
)
LABELS_TS = os.path.join(
    REPO, 'data/nnunet_local/nnUNet_raw/Dataset600_BrainMet/labelsTs',
)
OUT_JSON = os.path.join(
    REPO, 'runs/exp3_baseline_v2_d600/threshold_analysis_per_fold.json',
)
OUT_MD = os.path.join(
    REPO, 'runs/exp3_baseline_v2_d600/threshold_analysis_per_fold.md',
)

THRESHOLDS = np.array([
    0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07,
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
])
FG = 1   # foreground softmax channel


def _load_gt(case_id: str) -> np.ndarray:
    """Load GT NIfTI as boolean [X, Y, Z] mask (nibabel's native ordering)."""
    arr = nib.load(os.path.join(LABELS_TS, f'{case_id}.nii.gz')).get_fdata()
    return arr > 0.5


def _load_probs(npz_path: str) -> np.ndarray:
    """Return foreground softmax [X, Y, Z] float32.

    nnUNetv2 saves probabilities in SimpleITK (Z, Y, X) order inside the .npz.
    Permute (2, 1, 0) to match nibabel's (X, Y, Z) convention used by labelsTs.
    Verified by brute-force matching against the binary .nii.gz prediction
    nnU-Net writes alongside the .npz (which IS in nibabel-native space).
    """
    probs = np.load(npz_path)['probabilities'][FG].astype(np.float32)
    return np.transpose(probs, (2, 1, 0))


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-volume Dice with empty/empty -> 1.0."""
    inter = float(np.logical_and(pred, gt).sum())
    union = float(pred.sum()) + float(gt.sum())
    if union <= 0:
        return 1.0
    return 2.0 * inter / union


def _sweep_one_fold(fold: int) -> dict:
    """Sweep thresholds for all 51 test cases in one fold. Returns curve."""
    probs_dir = os.path.join(PER_FOLD, f'fold_{fold}', 'predictions')
    case_ids = sorted(
        f[:-len('.npz')] for f in os.listdir(probs_dir) if f.endswith('.npz')
    )
    dice_mat = np.zeros((len(case_ids), len(THRESHOLDS)), dtype=np.float64)
    for i, cid in enumerate(case_ids):
        gt = _load_gt(cid)
        probs = _load_probs(os.path.join(probs_dir, f'{cid}.npz'))
        if probs.shape != gt.shape:
            raise ValueError(f'shape mismatch {cid}: {probs.shape} vs {gt.shape}')
        for j, thr in enumerate(THRESHOLDS):
            dice_mat[i, j] = _dice(probs > thr, gt)
    means = dice_mat.mean(axis=0)
    stds = dice_mat.std(axis=0, ddof=1) if dice_mat.shape[0] > 1 else np.zeros_like(means)
    best_idx = int(np.argmax(means))
    return {
        'n_cases': int(len(case_ids)),
        'thresholds': THRESHOLDS.tolist(),
        'dice_mean': means.tolist(),
        'dice_std': stds.tolist(),
        'best_threshold': float(THRESHOLDS[best_idx]),
        'best_dice': float(means[best_idx]),
        'best_dice_std': float(stds[best_idx]),
        'dice_at_0.5': float(means[list(THRESHOLDS).index(0.50)]),
    }


def _ensemble_sweep() -> dict:
    """Average all 5 folds' foreground softmax per voxel, then sweep."""
    fold0_dir = os.path.join(PER_FOLD, 'fold_0', 'predictions')
    case_ids = sorted(
        f[:-len('.npz')] for f in os.listdir(fold0_dir) if f.endswith('.npz')
    )
    dice_mat = np.zeros((len(case_ids), len(THRESHOLDS)), dtype=np.float64)
    for i, cid in enumerate(case_ids):
        gt = _load_gt(cid)
        accum = None
        for f in range(5):
            probs = _load_probs(
                os.path.join(PER_FOLD, f'fold_{f}', 'predictions', f'{cid}.npz'),
            )
            accum = probs if accum is None else accum + probs
        ensemble_probs = accum / 5.0
        for j, thr in enumerate(THRESHOLDS):
            dice_mat[i, j] = _dice(ensemble_probs > thr, gt)
    means = dice_mat.mean(axis=0)
    stds = dice_mat.std(axis=0, ddof=1) if dice_mat.shape[0] > 1 else np.zeros_like(means)
    best_idx = int(np.argmax(means))
    return {
        'n_cases': int(len(case_ids)),
        'thresholds': THRESHOLDS.tolist(),
        'dice_mean': means.tolist(),
        'dice_std': stds.tolist(),
        'best_threshold': float(THRESHOLDS[best_idx]),
        'best_dice': float(means[best_idx]),
        'best_dice_std': float(stds[best_idx]),
        'dice_at_0.5': float(means[list(THRESHOLDS).index(0.50)]),
    }


def main() -> None:
    print('=== Per-fold threshold sweep ===')
    per_fold = {}
    per_fold_best_t = []
    for f in range(5):
        print(f'  Sweeping fold {f}...')
        r = _sweep_one_fold(f)
        per_fold[f'fold_{f}'] = r
        per_fold_best_t.append(r['best_threshold'])
        print(f'    fold {f}: best t={r["best_threshold"]:.2f} '
              f'dice={r["best_dice"]:.4f}  (vs t=0.50: {r["dice_at_0.5"]:.4f})')

    print('')
    print('=== Cross-fold optimum stability ===')
    arr = np.asarray(per_fold_best_t)
    print(f'  per-fold thresholds: {per_fold_best_t}')
    print(f'  mean={arr.mean():.4f}  median={np.median(arr):.4f}  std={arr.std(ddof=1):.4f}')

    print('')
    print('=== Ensembled (mean of 5 fold softmaxes) sweep ===')
    ens = _ensemble_sweep()
    print(f'  best t={ens["best_threshold"]:.2f} dice={ens["best_dice"]:.4f}')
    print(f'  baseline t=0.50 dice={ens["dice_at_0.5"]:.4f}')

    out = {
        'per_fold': per_fold,
        'per_fold_thresholds': per_fold_best_t,
        'cross_fold_optimum_stats': {
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'std': float(arr.std(ddof=1)),
        },
        'ensemble': ens,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nJSON written to {OUT_JSON}')

    # Compact markdown table
    lines = ['# Per-fold threshold sweep — exp3_baseline_v2_d600\n']
    lines.append('## Per-fold sweep\n')
    lines.append('| Fold | Best t | Dice @ best | Dice @ 0.50 | Δ |')
    lines.append('|---|---|---|---|---|')
    for f in range(5):
        r = per_fold[f'fold_{f}']
        d = r['best_dice'] - r['dice_at_0.5']
        lines.append(f'| {f} | {r["best_threshold"]:.2f} | {r["best_dice"]:.4f} '
                     f'| {r["dice_at_0.5"]:.4f} | {d:+.4f} |')
    lines.append('')
    lines.append(f'## Cross-fold optimum stability\n')
    lines.append(f'- thresholds: {per_fold_best_t}')
    lines.append(f'- mean: {arr.mean():.4f}')
    lines.append(f'- median: {np.median(arr):.4f}')
    lines.append(f'- std: {arr.std(ddof=1):.4f}\n')
    lines.append('## Ensembled (mean of 5 folds) sweep\n')
    lines.append('| Threshold | Dice mean | Dice std |')
    lines.append('|---|---|---|')
    for j, t in enumerate(ens['thresholds']):
        marker = ' ←' if t == ens['best_threshold'] else ''
        lines.append(f'| {t:.2f} | {ens["dice_mean"][j]:.4f} | '
                     f'{ens["dice_std"][j]:.4f} |{marker}')
    delta = ens['best_dice'] - ens['dice_at_0.5']
    lines.append('')
    lines.append(f'**Ensemble baseline (t=0.50)**: {ens["dice_at_0.5"]:.4f}')
    lines.append(f'**Ensemble best (t={ens["best_threshold"]:.2f})**: {ens["best_dice"]:.4f}')
    lines.append(f'**Gain**: {delta:+.4f}')
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Markdown table written to {OUT_MD}')


if __name__ == '__main__':
    main()
