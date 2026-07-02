#!/usr/bin/env python3
"""Objectively choose the PCA brain-shape filter's resolution, k, and threshold.

Turns the PCA shape filter into a validated binary detector and selects its
hyperparameters by discrimination performance on held-out labelled data — no
visual/threshold-by-eye tuning.

Protocol:
    - Fit PCA on real ``--fit-splits`` (default: train).
    - GOOD = real ``--good-splits`` (default: test1), unseen by the PCA.
    - BAD  = broken brains the filter must reject:
        (a) real broken generations (``--bad-gen-root`` / ``--bad-gen-dirs``), and
        (b) synthetic corruptions of the GOOD brains (``--corruptions``):
            invert / shift / scramble / rot90 / flip_lr.
    - For each (resolution, k): fit PCA on train, score GOOD+BAD by reconstruction
      error, and report ROC-AUC of error separating GOOD from BAD (higher error =>
      more broken). Pick (resolution, k) with the best AUC (or the simplest config
      within noise — e.g. full-rank @ 64^3 if it ties).
    - Threshold: the error accepting ``--accept-percentile`` (default 95) of GOOD;
      report the resulting broken-rejection rate per BAD type.

AUC ~1 => the filter cleanly separates real from broken; AUC ~0.5 => that
corruption is (near-)in-distribution (expected for flip_lr — brains are ~symmetric).

Usage:
    python -m medgen.scripts.validate_pca_filter \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --bad-gen-root ~/MedicalDataSets/generated \
        --bad-gen-dirs exp23_bravo_real_seg_test1_nofilter \
        --output runs/eval/pca_filter_validation.json
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

from medgen.scripts.compute_brain_pca import load_masks_for_splits, reconstruction_errors

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Fixed seed so shift/scramble corruptions are reproducible run-to-run.
_RNG = np.random.default_rng(0)


def corrupt(mask3d: np.ndarray, kind: str) -> np.ndarray:
    """Apply a synthetic structural corruption to a 3D binary mask [D, H, W]."""
    d, h, w = mask3d.shape
    if kind == 'invert':                      # upside-down anatomy (flip D and H)
        return mask3d[::-1, ::-1, :].copy()
    if kind == 'shift':                       # brain off-centre (~25% roll on D,H)
        return np.roll(mask3d, shift=(d // 4, h // 4), axis=(0, 1)).copy()
    if kind == 'rot90':                       # wrong orientation (90 deg in-plane)
        return np.rot90(mask3d, k=1, axes=(1, 2)).copy()
    if kind == 'flip_lr':                     # near-symmetric -> deliberately weak/hard case
        return mask3d[:, :, ::-1].copy()
    if kind == 'scramble':                    # shuffle 2x2x2 octants -> destroys global structure
        out = np.empty_like(mask3d)
        dm, hm, wm = d // 2, h // 2, w // 2
        octs = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
        perm = _RNG.permutation(len(octs))
        for src_i, dst_i in enumerate(perm):
            sa, sb, sc = octs[src_i]
            da, db, dc = octs[dst_i]
            out[da*dm:da*dm+dm, db*hm:db*hm+hm, dc*wm:dc*wm+wm] = \
                mask3d[sa*dm:sa*dm+dm, sb*hm:sb*hm+hm, sc*wm:sc*wm+wm]
        return out
    raise ValueError(f"Unknown corruption {kind!r}")


def auc_higher_is_bad(good_err: np.ndarray, bad_err: np.ndarray) -> float:
    """ROC-AUC treating reconstruction error as a 'broken' score (higher => BAD)."""
    good_err = np.asarray(good_err)[None, :]
    bad_err = np.asarray(bad_err)[:, None]
    gt = (bad_err > good_err).sum()
    eq = (bad_err == good_err).sum()
    return float((gt + 0.5 * eq) / (bad_err.size * good_err.size))


def main():
    p = argparse.ArgumentParser(description="Validate + tune the PCA brain-shape filter")
    p.add_argument('--data-root', required=True, help='brainmetshare-3 root (train/, test1/, ...)')
    p.add_argument('--fit-splits', nargs='+', default=['train'])
    p.add_argument('--good-splits', nargs='+', default=['test1'])
    p.add_argument('--bad-gen-root', default=None, help='Root containing broken-generation dirs')
    p.add_argument('--bad-gen-dirs', nargs='+', default=[], help='Broken-generation subdir(s) under --bad-gen-root')
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--depth', type=int, default=160)
    p.add_argument('--brain-threshold', type=float, default=0.05)
    p.add_argument('--pca-configs', nargs='+', default=['40:64', '80:128'],
                   help="Resolutions as 'depth:size' (default coarse 40x64x64 and fine 80x128x128)")
    p.add_argument('--k-values', nargs='+', default=['30', '60', '100', 'full'])
    p.add_argument('--corruptions', nargs='+',
                   default=['invert', 'shift', 'scramble', 'rot90', 'flip_lr'])
    p.add_argument('--accept-percentile', type=float, default=95.0,
                   help='Threshold accepts this %% of GOOD (held-out real) brains (default 95)')
    p.add_argument('--output', default='runs/eval/pca_filter_validation.json')
    args = p.parse_args()

    data_root = Path(args.data_root)
    rows = []

    for cfg in args.pca_configs:
        pd_, ps_ = (int(x) for x in cfg.split(':'))
        target = (pd_, ps_, ps_)
        logger.info(f"\n===== Resolution {ps_}x{ps_}x{pd_} =====")

        fit = load_masks_for_splits(data_root, args.fit_splits, args.depth, args.image_size, args.brain_threshold, target)
        good = load_masks_for_splits(data_root, args.good_splits, args.depth, args.image_size, args.brain_threshold, target)

        # Real broken generations (BAD)
        bad_sets: dict[str, np.ndarray] = {}
        if args.bad_gen_root and args.bad_gen_dirs:
            gen = load_masks_for_splits(Path(args.bad_gen_root), args.bad_gen_dirs,
                                        args.depth, args.image_size, args.brain_threshold, target)
            bad_sets['gen_real'] = gen

        # Synthetic corruptions of the GOOD brains (BAD)
        good3d = good.reshape(-1, *target)
        for kind in args.corruptions:
            bad_sets[kind] = np.stack([corrupt(m, kind).reshape(-1) for m in good3d]).astype(np.float32)

        # Fit PCA on train
        mean = fit.mean(axis=0)
        _U, S, Vt = np.linalg.svd(fit - mean, full_matrices=False)
        max_k = fit.shape[0] - 1
        comps = Vt[:max_k]
        cum_evr = np.cumsum(S ** 2) / (S ** 2).sum()

        for kraw in args.k_values:
            k = max_k if str(kraw).lower() == 'full' else min(int(kraw), max_k)
            eg = reconstruction_errors(good, mean, comps, k)
            per_bad_err = {name: reconstruction_errors(b, mean, comps, k) for name, b in bad_sets.items()}
            aucs = {name: auc_higher_is_bad(eg, be) for name, be in per_bad_err.items()}
            all_bad = np.concatenate(list(per_bad_err.values()))
            aucs['ALL'] = auc_higher_is_bad(eg, all_bad)
            thr = float(np.percentile(eg, args.accept_percentile))
            reject = {name: float((be > thr).mean()) for name, be in per_bad_err.items()}
            rows.append({
                'resolution': f"{ps_}x{ps_}x{pd_}", 'k': k, 'cum_evr': float(cum_evr[k - 1]),
                'auc': aucs, f'threshold_p{args.accept_percentile:g}': thr,
                'reject_rate': reject, 'good_err_mean': float(eg.mean()),
            })
            logger.info(f"  k={k:>4} cumEVR={cum_evr[k-1]:.3f}  AUC_ALL={aucs['ALL']:.3f}  "
                        + " ".join(f"{n}={aucs[n]:.2f}" for n in aucs if n != 'ALL'))

    best = max(rows, key=lambda r: r['auc']['ALL'])
    logger.info(f"\n>>> BEST by AUC_ALL: resolution={best['resolution']} k={best['k']} "
                f"(cumEVR={best['cum_evr']:.3f}) AUC_ALL={best['auc']['ALL']:.3f}")
    logger.info(f"    threshold (accept {args.accept_percentile:g}% real) rejects: {best['reject_rate']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'rows': rows, 'best': best}, indent=2))
    logger.info(f"Saved report to {out}")


if __name__ == '__main__':
    main()
