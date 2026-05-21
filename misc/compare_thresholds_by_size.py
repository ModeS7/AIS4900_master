"""Size-stratified Dice + detection-rate comparison at two thresholds.

Compares t=0.50 (nnU-Net default) vs t=0.005 (our swept optimum) on the
51-case test set, broken down by lesion size bucket (tiny / small / medium /
large per RANO-BM thresholds at the 1.0×0.9375×0.9375 mm BrainMetShare voxel).

Reuses the per-fold softmax .npz files produced by run_per_fold_local.py.
The ensemble probability is the per-voxel mean across the 5 folds, exactly
matching what nnU-Net's standard ensemble inference outputs.

Output: prints a comparison table and writes JSON. No medgen import (the
existing tracker class lives behind medgen.__init__ which loads the diffusion
pipeline — too heavy for the nnunet venv). The size-binning logic mirrors
src/medgen/metrics/regional/tracker_seg.py and src/medgen/scripts/
classify_synth_by_lesion_size.py.
"""
from __future__ import annotations

import json
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import label as scipy_label

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_FOLD = os.path.join(
    REPO, 'runs/exp3_baseline_v2_d600/eval_exp3_baseline_v2_d600/per_fold_test',
)
LABELS_TS = os.path.join(
    REPO, 'data/nnunet_local/nnUNet_raw/Dataset600_BrainMet/labelsTs',
)
OUT_JSON = os.path.join(
    REPO, 'runs/exp3_baseline_v2_d600/threshold_size_comparison.json',
)

# Voxel spacing: BrainMetShare-3 = (X=0.9375, Y=0.9375, Z=1.0) mm (nibabel XYZ).
VOXEL_SPACING_MM = (0.9375, 0.9375, 1.0)

# RANO-BM tumor-size thresholds in mm (Feret diameter). Mirrors
# medgen.metrics.constants.TUMOR_SIZE_THRESHOLDS_MM.
SIZE_BUCKETS = [
    ('tiny',   0.0,  10.0),
    ('small',  10.0, 20.0),
    ('medium', 20.0, 30.0),
    ('large',  30.0, np.inf),
]

# Matching criterion for "detected": at least one predicted voxel overlapping
# the GT lesion. Mirrors what evaluate.py's lesion-wise Dice uses for FN/TP
# discrimination. (The tracker class uses a `detection_threshold` Dice > 0.1
# criterion; we use min-overlap-voxels=1 to align with BraTS-Mets convention.)
MIN_OVERLAP_VOXELS = 1

# 26-connectivity 3D
STRUCTURE_3D = np.ones((3, 3, 3), dtype=np.uint8)


def _feret_diameter_3d(lesion_mask_xyz: np.ndarray, spacing_xyz: tuple[float, ...]) -> float:
    """Centroid-distance approximation of Feret diameter in mm.

    For lesions with <500 voxels (typical here) the O(N²) is trivial. For very
    large lesions this becomes slow but still completes in seconds.
    """
    coords = np.argwhere(lesion_mask_xyz).astype(np.float64)
    if coords.shape[0] == 0:
        return 0.0
    if coords.shape[0] == 1:
        return float(np.linalg.norm(np.asarray(spacing_xyz)))
    coords *= np.asarray(spacing_xyz)[np.newaxis, :]
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return float(np.sqrt((diffs * diffs).sum(axis=2).max()))


def _bucket_of(diameter_mm: float) -> str:
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= diameter_mm < hi:
            return name
    return 'large'


def _load_ensemble_probs(case_id: str) -> np.ndarray:
    """Mean of 5 folds' foreground softmax, in nibabel (X, Y, Z) order."""
    accum = None
    for f in range(5):
        npz = os.path.join(PER_FOLD, f'fold_{f}', 'predictions', f'{case_id}.npz')
        p = np.load(npz)['probabilities'][1].astype(np.float32)
        # Permute SimpleITK ZYX → nibabel XYZ
        p = np.transpose(p, (2, 1, 0))
        accum = p if accum is None else accum + p
    return accum / 5.0


def _load_gt(case_id: str) -> np.ndarray:
    return nib.load(os.path.join(LABELS_TS, f'{case_id}.nii.gz')).get_fdata() > 0.5


def _per_case_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute per-GT-lesion match info + per-FP-only-predicted-lesion count.

    Returns a dict where:
        gt_lesions: list of dicts with keys {bucket, diameter, dice, detected}
        fp_lesions_per_bucket: dict[bucket, count] of predicted lesions that
                               had zero overlap with any GT (false-positive blobs).
                               Bucket inferred from the predicted lesion's diameter.
    """
    gt_lab, n_gt = scipy_label(gt, structure=STRUCTURE_3D)
    pred_lab, n_pred = scipy_label(pred, structure=STRUCTURE_3D)

    gt_lesions = []
    pred_hit = np.zeros(n_pred + 1, dtype=bool)

    for gid in range(1, n_gt + 1):
        g_mask = (gt_lab == gid)
        d_mm = _feret_diameter_3d(g_mask, VOXEL_SPACING_MM)
        bucket = _bucket_of(d_mm)
        overlap = pred & g_mask
        n_overlap = int(overlap.sum())
        detected = n_overlap >= MIN_OVERLAP_VOXELS

        if detected:
            # Per-lesion voxel Dice within bbox (matches BraTS-Mets convention)
            coords = np.where(g_mask)
            slc = tuple(slice(c.min(), c.max() + 1) for c in coords)
            p_local = pred[slc] & (pred_lab[slc] != 0)
            g_local = g_mask[slc]
            inter = int((p_local & g_local).sum())
            denom = int(p_local.sum()) + int(g_local.sum())
            dice = 2.0 * inter / max(denom, 1)
            for pid in np.unique(pred_lab[g_mask]):
                if pid != 0:
                    pred_hit[pid] = True
        else:
            dice = 0.0

        gt_lesions.append({
            'bucket': bucket, 'diameter_mm': d_mm,
            'dice': dice, 'detected': detected,
        })

    fp_by_bucket: dict[str, int] = {b: 0 for b, *_ in SIZE_BUCKETS}
    for pid in range(1, n_pred + 1):
        if pred_hit[pid]:
            continue
        # Spurious predicted lesion — find its diameter, bucket it
        d_mm = _feret_diameter_3d(pred_lab == pid, VOXEL_SPACING_MM)
        fp_by_bucket[_bucket_of(d_mm)] += 1

    return {'gt_lesions': gt_lesions, 'fp_by_bucket': fp_by_bucket}


def _aggregate(per_case_stats: list[dict], thr_label: str) -> dict:
    """Compile per-bucket Dice, detection rate, FP counts across the 51 cases.

    Mirrors what SegRegionalMetricsTracker.compute() + get_detection_summary()
    do in the regular eval pipeline, but for one threshold's predictions.
    """
    by_bucket: dict[str, dict[str, list]] = {
        b: {'dices': [], 'detected': [], 'fp': 0}
        for b, *_ in SIZE_BUCKETS
    }
    for cs in per_case_stats:
        for g in cs['gt_lesions']:
            by_bucket[g['bucket']]['dices'].append(g['dice'])
            by_bucket[g['bucket']]['detected'].append(g['detected'])
        for b, n in cs['fp_by_bucket'].items():
            by_bucket[b]['fp'] += n

    out = {'threshold_label': thr_label, 'per_bucket': {}}
    all_dices: list[float] = []
    all_detected: list[bool] = []
    total_fp = 0
    for b, *_ in SIZE_BUCKETS:
        dices = by_bucket[b]['dices']
        det = by_bucket[b]['detected']
        fp = by_bucket[b]['fp']
        out['per_bucket'][b] = {
            'n_tumors': len(dices),
            'dice_mean': float(np.mean(dices)) if dices else float('nan'),
            'dice_std': float(np.std(dices, ddof=1)) if len(dices) > 1 else 0.0,
            'detection_rate': float(np.mean(det)) if det else float('nan'),
            'n_detected': int(sum(det)),
            'fp_count': fp,
        }
        all_dices.extend(dices)
        all_detected.extend(det)
        total_fp += fp

    out['overall'] = {
        'n_tumors': len(all_dices),
        'dice_mean': float(np.mean(all_dices)) if all_dices else float('nan'),
        'dice_std': float(np.std(all_dices, ddof=1)) if len(all_dices) > 1 else 0.0,
        'detection_rate': float(np.mean(all_detected)) if all_detected else float('nan'),
        'n_detected': int(sum(all_detected)),
        'fp_count': total_fp,
    }
    return out


def main() -> None:
    case_ids = sorted(
        f[:-len('.npz')]
        for f in os.listdir(os.path.join(PER_FOLD, 'fold_0', 'predictions'))
        if f.endswith('.npz')
    )
    print(f'Evaluating {len(case_ids)} cases at t=0.50 and t=0.005 (ensemble)')

    stats_050: list[dict] = []
    stats_005: list[dict] = []

    for i, cid in enumerate(case_ids):
        probs = _load_ensemble_probs(cid)
        gt = _load_gt(cid)
        if probs.shape != gt.shape:
            raise RuntimeError(f'shape mismatch {cid}: {probs.shape} vs {gt.shape}')
        stats_050.append(_per_case_stats(probs > 0.50, gt))
        stats_005.append(_per_case_stats(probs > 0.005, gt))
        if (i + 1) % 10 == 0 or i + 1 == len(case_ids):
            print(f'  {i+1}/{len(case_ids)} done')

    res_050 = _aggregate(stats_050, 't=0.50')
    res_005 = _aggregate(stats_005, 't=0.005')

    # Side-by-side console table
    print('\n' + '=' * 92)
    print(f'{"Bucket":<10}{"n_GT":>6}'
          f' {"Dice@0.50":>12} {"Det@0.50":>10} {"FP@0.50":>8}'
          f' {"Dice@0.005":>13} {"Det@0.005":>11} {"FP@0.005":>9}')
    print('-' * 92)
    for b, *_ in SIZE_BUCKETS:
        a = res_050['per_bucket'][b]
        c = res_005['per_bucket'][b]
        d50 = f"{a['dice_mean']:.3f}±{a['dice_std']:.3f}"
        d05 = f"{c['dice_mean']:.3f}±{c['dice_std']:.3f}"
        print(f'{b:<10}{a["n_tumors"]:>6}'
              f' {d50:>12} {a["detection_rate"]*100:>9.1f}% {a["fp_count"]:>8}'
              f' {d05:>13} {c["detection_rate"]*100:>10.1f}% {c["fp_count"]:>9}')
    a = res_050['overall']
    c = res_005['overall']
    d50 = f"{a['dice_mean']:.3f}±{a['dice_std']:.3f}"
    d05 = f"{c['dice_mean']:.3f}±{c['dice_std']:.3f}"
    print('-' * 92)
    print(f'{"OVERALL":<10}{a["n_tumors"]:>6}'
          f' {d50:>12} {a["detection_rate"]*100:>9.1f}% {a["fp_count"]:>8}'
          f' {d05:>13} {c["detection_rate"]*100:>10.1f}% {c["fp_count"]:>9}')
    print('=' * 92)

    out = {'t_050': res_050, 't_005': res_005}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nJSON written to {OUT_JSON}')


if __name__ == '__main__':
    main()
