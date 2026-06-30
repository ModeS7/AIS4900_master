#!/usr/bin/env python3
"""Perception-Distortion proof of the MSE mean-blur failure mode.

Quantifies the MSE-averaging failure on real BrainMetShare volumes by placing
each model on the perception-distortion plane (Blau & Michaeli, CVPR 2018):
an MSE-trained baseline is *best* on full-reference fidelity (PSNR/SSIM) yet
*worst* on distributional realism (FID), while perceptual fine-tunes trade
fidelity for realism. The failure is perceptual (averaging onto a non-valid
conditional mean), not loss of high-frequency energy -- so we measure it
perceptually (FID), as the theorem requires (a distortion metric cannot see it).

For each model and low-t timestep (where fine detail is committed during
generation), this computes, on the same real test volumes:
  1. DISTORTION (full-reference, paired): PSNR + SSIM(x_hat0, x0), brain-masked.
       Single-step denoise: x_hat0 = x_t + t*v_pred  (predict_single_step).
  2. PERCEPTION (no-reference, distributional): FID/KID/CMMD/FID-RadImageNet of
       the denoised set {x_hat0} vs the real test features (same pipeline as the
       generation FID in find_optimal_steps).
  3. AVERAGING signature: pixel-std of x_hat0 across n_noise seeds. ~0 means the
       model returns the conditional mean rather than a sample -- the mechanism.

Run once per single-model checkpoint into a shared output root, then assemble
the plane across the per-model JSON files.

Usage:
    python -m medgen.scripts.eval_perception_distortion \
        --checkpoint runs/diffusion_3d/bravo/exp1_1_1000_.../checkpoint_latest.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --output-dir eval_perception_distortion/baseline \
        --t-values 0.05 0.1 0.2 --num-volumes 25 --n-noise 8
"""
import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from medgen.diffusion import RFlowStrategy, load_diffusion_model
from medgen.scripts.diagnose_mean_blur import (
    discover_test_patients,
    load_volume,
    predict_single_step,
)
from medgen.scripts.eval_ode_solvers import (
    compute_all_metrics,
    discover_splits,
    get_or_cache_reference_features,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def masked_psnr(pred: np.ndarray, real: np.ndarray, mask: np.ndarray) -> float:
    """PSNR over brain-masked voxels, data range [0, 1]."""
    m = mask > 0.5
    if m.sum() == 0:
        return float('nan')
    mse = float(np.mean((pred[m] - real[m]) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def volume_ssim(pred: torch.Tensor, real: torch.Tensor) -> float:
    """3D SSIM via MONAI (best-effort). pred/real: [1,1,D,H,W] in [0,1]."""
    try:
        from monai.metrics import SSIMMetric
        metric = SSIMMetric(spatial_dims=3, data_range=1.0)
        val = metric(pred, real)
        return float(val.mean().item())
    except Exception as e:  # SSIM is a secondary distortion axis; PSNR is primary
        logger.warning(f"SSIM unavailable ({type(e).__name__}: {e}); reporting NaN")
        return float('nan')


def main() -> None:
    p = argparse.ArgumentParser(
        description="Perception-Distortion proof of MSE mean-blur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", required=True, help="Single-model checkpoint")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--t-values", nargs="+", type=float, default=[0.05, 0.1, 0.2],
                   help="Normalized low-t values to denoise at (default: 0.05 0.1 0.2)")
    p.add_argument("--num-volumes", type=int, default=25,
                   help="Real test volumes for distortion + FID (default: 25)")
    p.add_argument("--n-noise", type=int, default=8,
                   help="Noise seeds per volume for the pred_std averaging signature (default: 8)")
    p.add_argument("--pred-std-volumes", type=int, default=5,
                   help="How many volumes to use for the pred_std signature (default: 5)")
    p.add_argument("--depth", type=int, default=160)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--trim-slices", type=int, default=10)
    p.add_argument("--ref-split", default="test")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model + strategy (pixel-space RFlow bravo, conditioned on seg) ──
    logger.info(f"Loading model: {args.checkpoint}")
    model = load_diffusion_model(
        args.checkpoint, device=device, compile_model=False, spatial_dims=3,
    )
    model.eval()
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000, image_size=args.image_size,
        depth_size=args.depth, spatial_dims=3,
    )
    T = strategy.scheduler.num_train_timesteps

    # ── Real test volumes (bravo + seg) ──
    data_root = Path(args.data_root)
    patients = discover_test_patients(data_root, args.num_volumes)
    logger.info(f"Loaded {len(patients)} real test patients")
    cached = []
    for pid, bravo_path, seg_path in patients:
        clean = load_volume(bravo_path, args.depth)
        seg = (load_volume(seg_path, args.depth) > 0.5).astype(np.float32)
        cached.append((pid, clean, seg))

    # ── Reference features (real test split) for distributional FID ──
    logger.info(f"Preparing reference features ({args.ref_split} split)...")
    splits = discover_splits(data_root, modality="bravo")
    if args.ref_split not in splits:
        raise ValueError(
            f"Reference split '{args.ref_split}' not found. Available: {list(splits.keys())}"
        )
    cache_dir = out / "reference_features"
    ref_features = get_or_cache_reference_features(
        {args.ref_split: splits[args.ref_split]}, cache_dir, device,
        args.depth, args.trim_slices, args.image_size,
        modality="bravo", build_all=False,
    )
    eval_ref = {args.ref_split: ref_features[args.ref_split]}

    n_pred_std = min(args.pred_std_volumes, len(cached))
    records = []

    for t_norm in args.t_values:
        logger.info("=" * 60)
        logger.info(f"t = {t_norm:.3f}")
        denoised, psnrs, ssims, pred_stds = [], [], [], []

        for vol_idx, (pid, clean_np, seg_np) in enumerate(cached):
            clean = torch.from_numpy(clean_np)[None, None].to(device)
            seg = torch.from_numpy(seg_np)[None, None].to(device)
            mask_np = (clean_np > 0.02).astype(np.float32)

            # Distortion + perception: one denoise per volume (base seed).
            torch.manual_seed(args.seed + vol_idx)
            noise = torch.randn_like(clean)
            x0 = predict_single_step(model, strategy, clean, seg, noise, t_norm, T, device)
            x0_np = x0.squeeze().detach().cpu().numpy().astype(np.float32)
            denoised.append(x0_np)
            psnrs.append(masked_psnr(x0_np, clean_np, mask_np))
            ssims.append(volume_ssim(x0, clean))

            # Averaging signature: pred_std across n_noise seeds (subset of volumes).
            if vol_idx < n_pred_std:
                t_int = max(1, min(T - 1, round(t_norm * T)))
                preds = []
                for i in range(args.n_noise):
                    torch.manual_seed(args.seed + 10000 * vol_idx + 100 * t_int + i)
                    n_i = torch.randn_like(clean)
                    preds.append(
                        predict_single_step(model, strategy, clean, seg, n_i, t_norm, T, device).cpu()
                    )
                pred_stds.append(float(torch.stack(preds, 0).std(0).mean().item()))
                del preds

            del clean, seg, noise, x0
            torch.cuda.empty_cache()

        # Distributional perception (FID/KID/CMMD/FID-RadImageNet) of the denoised set.
        split_metrics = compute_all_metrics(denoised, eval_ref, device, args.trim_slices)
        m = asdict(split_metrics[args.ref_split])

        rec = {
            "t": t_norm,
            "num_volumes": len(denoised),
            # distortion (full-reference) -- MSE baseline should be BEST here
            "psnr_mean": float(np.nanmean(psnrs)),
            "psnr_std": float(np.nanstd(psnrs)),
            "ssim_mean": float(np.nanmean(ssims)),
            # perception (distributional) -- MSE baseline should be WORST here
            "fid": m["fid"],
            "fid_radimagenet": m["fid_radimagenet"],
            "kid": m["kid_mean"],
            "cmmd": m["cmmd"],
            # mechanism -- averaging / posterior-mean collapse signature
            "pred_std_mean": float(np.mean(pred_stds)) if pred_stds else None,
        }
        records.append(rec)
        logger.info(
            f"  PSNR={rec['psnr_mean']:.2f} SSIM={rec['ssim_mean']:.4f} | "
            f"FID={rec['fid']:.3f} FID_RIN={rec['fid_radimagenet']:.4f} "
            f"KID={rec['kid']:.5f} CMMD={rec['cmmd']:.4f} | "
            f"pred_std={rec['pred_std_mean']}"
        )

    result = {
        "checkpoint": args.checkpoint,
        "data_root": str(data_root),
        "num_volumes": args.num_volumes,
        "n_noise": args.n_noise,
        "ref_split": args.ref_split,
        "t_values": args.t_values,
        "records": records,
    }
    with open(out / "perception_distortion.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Saved {out / 'perception_distortion.json'}")
    logger.info("Distortion = PSNR/SSIM (full-ref). Perception = FID (distributional).")
    logger.info("PROOF: an MSE baseline should land at best-PSNR + worst-FID;")
    logger.info("perceptual fine-tunes trade PSNR for FID -- the perception-distortion tradeoff.")


if __name__ == "__main__":
    main()
