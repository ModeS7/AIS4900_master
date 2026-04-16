#!/usr/bin/env python3
"""Measure velocity prediction quality across the noise schedule.

For a trained RFlow model, this measures how well the model predicts velocity
at each candidate timestep t. Used to empirically identify the boundary where
prediction quality breaks down — informs the σ_boundary for two-expert
sampling (eDiff-I-style ensemble).

Methodology (mirrors Karras et al. EDM 2022, Fig. 3):
- For a fine grid of t values, take held-out real volumes
- Add noise at level t (deterministic seed for reproducibility)
- Predict velocity with the trained model
- Compute multiple error metrics:
    * Velocity MSE       — direct training objective
    * Predicted-clean L1 — pixel error after reconstruction
    * LPIPS (slice-wise) — perceptual error
    * High-freq energy ratio — spectral fidelity at fine scales
- Plot all metrics; identify the inflection point.

Usage:
    python -m medgen.scripts.measure_velocity_breakdown \
        --bravo-model runs/checkpoint_latest.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --output-dir runs/velocity_breakdown \
        --num-volumes 10 --depth 160
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from medgen.diffusion import RFlowStrategy, load_diffusion_model

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# Default timestep grid: dense at low t (where breakdown expected), coarser elsewhere
DEFAULT_T_GRID = [
    0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20,
    0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
]


def load_volume(path: Path, depth: int) -> np.ndarray:
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    vmax = vol.max()
    if vmax > 0:
        vol /= vmax
    vol = np.transpose(vol, (2, 0, 1))
    if vol.shape[0] < depth:
        vol = np.pad(vol, ((0, depth - vol.shape[0]), (0, 0), (0, 0)))
    elif vol.shape[0] > depth:
        vol = vol[:depth]
    return vol


def discover_test_patients(
    data_root: Path, max_n: int,
) -> list[tuple[str, Path, Path]]:
    """Find patients with bravo + seg in test split."""
    test_dir = data_root / 'test'
    if not test_dir.exists():
        test_dir = data_root / 'test_new'
    patients = []
    for d in sorted(test_dir.iterdir()):
        if not d.is_dir():
            continue
        bravo = d / "bravo.nii.gz"
        seg = d / "seg.nii.gz"
        if bravo.exists() and seg.exists():
            patients.append((d.name, bravo, seg))
            if len(patients) >= max_n:
                break
    return patients


def compute_high_freq_ratio(volume: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of FFT energy above radial frequency 0.25 (above Nyquist/4)."""
    if mask is not None:
        volume = volume * mask
    fft = np.fft.fftn(volume)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    d, h, w = volume.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)

    high_mask = radius > 0.25
    total_energy = power.sum()
    if total_energy <= 0:
        return 0.0
    return float(power[high_mask].sum() / total_energy)


def slicewise_lpips(
    pred: torch.Tensor, target: torch.Tensor, lpips_fn, n_slices: int = 8,
) -> float:
    """LPIPS averaged over n_slices central axial slices.

    LPIPS is 2D, ImageNet-pretrained — applied to slices, RGB-replicated.
    """
    # pred, target: [1, 1, D, H, W]
    D = pred.shape[2]
    # Sample equally-spaced slices around the brain (skip top/bottom 25%)
    slice_indices = np.linspace(D // 4, 3 * D // 4, n_slices, dtype=int)
    losses = []
    for s in slice_indices:
        p2d = pred[:, :, s].repeat(1, 3, 1, 1)  # [1, 3, H, W]
        t2d = target[:, :, s].repeat(1, 3, 1, 1)
        # LPIPS expects values in [-1, 1]
        p2d = p2d.float() * 2 - 1
        t2d = t2d.float() * 2 - 1
        with torch.no_grad():
            losses.append(lpips_fn(p2d, t2d).item())
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure velocity prediction quality across noise schedule")
    parser.add_argument("--bravo-model", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True,
                        help="Dataset root (brainmetshare-3)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-volumes", type=int, default=10,
                        help="Held-out test volumes to average over (default: 10)")
    parser.add_argument("--depth", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-grid", nargs='+', type=float, default=DEFAULT_T_GRID,
                        help="Timestep grid in [0, 1] (normalized)")
    parser.add_argument("--n-slices-lpips", type=int, default=8,
                        help="Number of axial slices for LPIPS (default: 8)")
    parser.add_argument("--skip-lpips", action='store_true',
                        help="Skip LPIPS computation (faster, no extra deps)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')

    # ── Load model ──
    logger.info(f"Loading bravo model from {args.bravo_model}")
    model = load_diffusion_model(
        args.bravo_model, device=device, compile_model=False, spatial_dims=3,
    )
    model.eval()

    strategy = RFlowStrategy()
    strategy.setup_scheduler(num_timesteps=1000, image_size=256, depth_size=args.depth, spatial_dims=3)
    T = strategy.scheduler.num_train_timesteps

    # ── Load LPIPS (optional) ──
    lpips_fn = None
    if not args.skip_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex').to(device)
            lpips_fn.eval()
            logger.info("LPIPS (AlexNet) loaded")
        except ImportError:
            logger.warning("lpips package not installed; skipping LPIPS metric")
            lpips_fn = None

    # ── Load test volumes ──
    data_root = Path(args.data_root)
    patients = discover_test_patients(data_root, args.num_volumes)
    logger.info(f"Found {len(patients)} test patients")
    if not patients:
        raise RuntimeError(f"No test patients found in {data_root}")

    # Pre-load + cache volumes (small set, fits easily)
    cached = []
    for pid, bp, sp in patients:
        clean = load_volume(bp, args.depth)
        seg = (load_volume(sp, args.depth) > 0.5).astype(np.float32)
        cached.append((pid, clean, seg))
    logger.info(f"Loaded {len(cached)} volumes")

    # ── Sweep timesteps ──
    results: list[dict] = []
    for t_norm in args.t_grid:
        t_int = round(t_norm * T)
        t_int = max(1, min(T - 1, t_int))  # avoid extremes the scheduler may not handle
        per_vol_metrics = {
            'v_mse': [], 'x0_l1': [], 'x0_lpips': [],
            'hf_ratio_pred': [], 'hf_ratio_real': [], 'hf_ratio_diff': [],
        }
        logger.info(f"Measuring t={t_norm:.3f} (t_int={t_int}) — {len(cached)} volumes...")

        for vol_idx, (_pid, clean_np, seg_np) in enumerate(cached):
            # Deterministic noise per (vol, t) for reproducibility
            torch.manual_seed(args.seed + vol_idx * 1000 + t_int)
            clean = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0).to(device)
            seg = torch.from_numpy(seg_np).unsqueeze(0).unsqueeze(0).to(device)
            noise = torch.randn_like(clean)
            t_tensor = torch.tensor([t_int], device=device, dtype=torch.long)

            # x_t via scheduler (consistent with training)
            x_t = strategy.scheduler.add_noise(clean, noise, t_tensor)

            # Model input: [x_t, seg]
            model_input = torch.cat([x_t, seg], dim=1)

            with torch.no_grad():
                with autocast('cuda', dtype=torch.bfloat16):
                    v_pred = model(model_input, t_tensor)
                v_pred = v_pred.float()

                # True velocity target: v = x_0 - noise
                v_true = strategy.compute_target(clean, noise)
                # Predicted clean: x_0 = x_t + (t/T) * v_pred
                x0_pred = strategy.compute_predicted_clean(x_t, v_pred, t_tensor)

            # Velocity MSE
            v_mse = F.mse_loss(v_pred, v_true).item()

            # Predicted-clean L1
            x0_l1 = F.l1_loss(x0_pred.clamp(0, 1), clean).item()

            # LPIPS (slice-wise) — only if available
            x0_lpips = float('nan')
            if lpips_fn is not None:
                x0_lpips = slicewise_lpips(
                    x0_pred.clamp(0, 1), clean, lpips_fn,
                    n_slices=args.n_slices_lpips,
                )

            # High-freq energy ratio
            x0_pred_np = x0_pred.clamp(0, 1).squeeze().cpu().numpy()
            mask = (clean_np > 0.02).astype(np.float32)
            hf_pred = compute_high_freq_ratio(x0_pred_np, mask)
            hf_real = compute_high_freq_ratio(clean_np, mask)
            hf_diff = hf_real - hf_pred  # positive = pred is missing high-freq

            per_vol_metrics['v_mse'].append(v_mse)
            per_vol_metrics['x0_l1'].append(x0_l1)
            per_vol_metrics['x0_lpips'].append(x0_lpips)
            per_vol_metrics['hf_ratio_pred'].append(hf_pred)
            per_vol_metrics['hf_ratio_real'].append(hf_real)
            per_vol_metrics['hf_ratio_diff'].append(hf_diff)

            del clean, seg, noise, x_t, model_input, v_pred, v_true, x0_pred
        torch.cuda.empty_cache()

        agg = {'t_norm': t_norm, 't_int': t_int}
        for key, vals in per_vol_metrics.items():
            agg[f'{key}_mean'] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))
        results.append(agg)
        logger.info(
            f"  t={t_norm:.3f}: v_mse={agg['v_mse_mean']:.4f}  "
            f"x0_l1={agg['x0_l1_mean']:.4f}  "
            f"lpips={agg['x0_lpips_mean']:.4f}  "
            f"hf_diff={agg['hf_ratio_diff_mean']:.5f}"
        )

    # ── Save raw data ──
    with open(output_dir / 'velocity_breakdown.json', 'w') as f:
        json.dump({
            'checkpoint': args.bravo_model,
            'num_volumes': len(cached),
            't_grid': args.t_grid,
            'results': results,
        }, f, indent=2)

    # ── Plot ──
    t_vals = np.array([r['t_norm'] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    def add_breakdown_lines(ax):
        # Mark theoretical SNR=16 (t=0.20) and SNR=4 (t=0.33) thresholds
        ax.axvline(x=0.20, color='red', linestyle='--', alpha=0.5,
                   label='SNR=16 (t=0.20)')
        ax.axvline(x=0.33, color='orange', linestyle='--', alpha=0.5,
                   label='SNR=4 (t=0.33)')

    # Velocity MSE
    ax = axes[0, 0]
    means = np.array([r['v_mse_mean'] for r in results])
    stds = np.array([r['v_mse_std'] for r in results])
    ax.errorbar(t_vals, means, yerr=stds, marker='o', linewidth=2, capsize=3, color='steelblue')
    ax.set_xlabel('t (normalized noise level)')
    ax.set_ylabel('Velocity MSE')
    ax.set_title('Velocity prediction MSE (training objective)\nFlat-ish = model is "competent" by its own standard')
    add_breakdown_lines(ax)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # x0 L1
    ax = axes[0, 1]
    means = np.array([r['x0_l1_mean'] for r in results])
    stds = np.array([r['x0_l1_std'] for r in results])
    ax.errorbar(t_vals, means, yerr=stds, marker='o', linewidth=2, capsize=3, color='green')
    ax.set_xlabel('t (normalized noise level)')
    ax.set_ylabel('Predicted-clean L1 error')
    ax.set_title('||x₀_pred - x₀||_1 — pixel reconstruction error')
    add_breakdown_lines(ax)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # LPIPS
    ax = axes[1, 0]
    if lpips_fn is not None:
        means = np.array([r['x0_lpips_mean'] for r in results])
        stds = np.array([r['x0_lpips_std'] for r in results])
        ax.errorbar(t_vals, means, yerr=stds, marker='o', linewidth=2, capsize=3, color='purple')
        ax.set_ylabel('LPIPS (slice-wise mean)')
        ax.set_title('Perceptual error of predicted clean image\nSharp climb = perceptual breakdown')
    else:
        ax.text(0.5, 0.5, 'LPIPS skipped\n(install lpips to enable)',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('LPIPS (skipped)')
    ax.set_xlabel('t (normalized noise level)')
    add_breakdown_lines(ax)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # High-freq ratio difference
    ax = axes[1, 1]
    means = np.array([r['hf_ratio_diff_mean'] for r in results])
    stds = np.array([r['hf_ratio_diff_std'] for r in results])
    ax.errorbar(t_vals, means, yerr=stds, marker='o', linewidth=2, capsize=3, color='darkorange',
                label='real_hf − pred_hf')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.4)
    ax.set_xlabel('t (normalized noise level)')
    ax.set_ylabel('High-freq energy deficit (real − pred)')
    ax.set_title('Spectral fidelity: how much high-frequency content is missing\nPositive = predicted is too smooth')
    add_breakdown_lines(ax)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.suptitle(
        'Velocity prediction quality vs noise level — '
        f'exp1_1_1000 over {len(cached)} held-out volumes',
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(str(output_dir / 'velocity_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Summary ──
    logger.info("=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    logger.info(f"Output: {output_dir / 'velocity_breakdown.png'}")
    logger.info(f"Data:   {output_dir / 'velocity_breakdown.json'}")

    # Identify likely breakdown point: where x0_l1 starts increasing sharply
    x0_l1_means = np.array([r['x0_l1_mean'] for r in results])
    # Compute log-log slope at each point (relative to next)
    if len(t_vals) > 2:
        log_t = np.log(t_vals)
        log_l1 = np.log(np.maximum(x0_l1_means, 1e-8))
        slopes = np.diff(log_l1) / np.diff(log_t)
        # Find where slope is most positive (steepest climb)
        idx_max_slope = int(np.argmax(slopes))
        breakdown_t = float(t_vals[idx_max_slope])
        logger.info(
            f"Steepest L1 climb between t={t_vals[idx_max_slope]:.3f} "
            f"and t={t_vals[idx_max_slope + 1]:.3f} "
            f"(log-log slope = {slopes[idx_max_slope]:.2f})"
        )
        logger.info(f"Suggested boundary: t ≈ {breakdown_t:.3f} (review the plot)")


if __name__ == '__main__':
    main()
