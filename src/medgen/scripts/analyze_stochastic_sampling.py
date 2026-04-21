#!/usr/bin/env python3
"""Stochastic-Euler ablation for RFlow sampling.

Tests whether *inference-time* noise injection alone can break the
posterior-mean collapse we diagnosed in exp1_1_1000 (MSE → deterministic
Euler → mean-blur). Single hyperparameter: `noise_scale` (σ). Nothing else
changes — same model, same seeds, same step count. The model is NOT
retrained.

Stochastic Euler step:

    x_{t-Δt} = x_t + Δt · v_θ(x_t, t) + σ · √|Δt_norm| · η,   η ~ N(0, I)

where Δt_norm = (t - next_t) / num_train_timesteps (in [0, 1]). σ = 0
recovers deterministic Euler (production behaviour). The √|Δt| scaling
is the standard SDE convention — keeps the total injected variance
independent of step count.

Grid: σ ∈ {0.0, 0.01, 0.03, 0.05, 0.1} — bracketing "no effect" through
"clearly disruptive". N=5 seeds per scale, paired seeds (same noise start
across scales → cleanly isolates the σ effect).

Metrics:
  - per-band energy ratio to real (very_low .. very_high) — spectral test
  - Frangi mean inside brain — vessel-like structure
  - cross-seed std (per-voxel std over the 5 paired seeds) — diversity
  - HF energy ratio (bands high + very_high combined) — blur proxy

Outputs:
  Figures: band bars, Frangi bar, cross-seed-std bar, mid-slice grid
  NIfTI:   each generated volume under <out>/vols/sigma_XXX/seed_XXX.nii.gz
  JSON:    per-config metrics + raw per-seed values

Usage:
    python -m medgen.scripts.analyze_stochastic_sampling \\
        --checkpoint /path/to/exp1_1_1000_checkpoint_latest.pt \\
        --data-root /path/to/brainmetshare-3 \\
        --real-dir /path/to/brainmetshare-3/test1 \\
        --output-dir runs/eval/stochastic_sampling \\
        --num-seeds 5 --num-steps 32 \\
        --sigmas 0.0 0.01 0.03 0.05 0.1
"""
import argparse
import gc
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from skimage.filters import frangi
from torch.amp import autocast

from medgen.diffusion import RFlowStrategy, load_diffusion_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


BRAIN_THRESHOLD = 0.05
FRANGI_SIGMAS = (0.5, 1.0, 1.5, 2.0, 3.0)

BANDS: dict[str, tuple[float, float]] = {
    'very_low':  (0.00, 0.05),
    'low':       (0.05, 0.10),
    'low_mid':   (0.10, 0.20),
    'mid':       (0.20, 0.30),
    'high':      (0.30, 0.40),
    'very_high': (0.40, 0.50),
}
DEFAULT_NUM_STEPS = 32
DEFAULT_SIGMAS = [0.0, 0.01, 0.03, 0.05, 0.1]


# ────────────────────────────────────────────────────────────────
# I/O helpers (consistent with analyze_hybrid_generation.py)
# ────────────────────────────────────────────────────────────────
def load_volume(path: Path, depth: int) -> np.ndarray:
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    vol = np.transpose(vol, (2, 0, 1))
    d = vol.shape[0]
    if d < depth:
        vol = np.pad(vol, ((0, depth - d), (0, 0), (0, 0)))
    elif d > depth:
        vol = vol[:depth]
    return vol


def load_seg(path: Path, depth: int) -> np.ndarray:
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vol = np.transpose(vol, (2, 0, 1))
    d = vol.shape[0]
    if d < depth:
        vol = np.pad(vol, ((0, depth - d), (0, 0), (0, 0)))
    elif d > depth:
        vol = vol[:depth]
    return (vol > 0.5).astype(np.float32)


def discover_patients(data_root: Path, split: str, num: int
                      ) -> list[tuple[str, Path, Path]]:
    d = data_root / split
    if not d.exists():
        d = data_root / 'test'
    out = []
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        b = sub / 'bravo.nii.gz'
        s = sub / 'seg.nii.gz'
        if b.exists() and s.exists():
            out.append((sub.name, b, s))
            if len(out) >= num:
                break
    return out


def save_nifti(vol: np.ndarray, path: Path) -> None:
    """Save [D, H, W] as [H, W, D] NIfTI (matching brainmetshare convention)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.transpose(vol, (1, 2, 0)).astype(np.float32)
    nib.save(nib.Nifti1Image(out, affine=np.eye(4)), str(path))


# ────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────
def band_energies(vol: np.ndarray) -> dict[str, float]:
    fft = np.fft.fftshift(np.fft.fftn(vol))
    power = np.abs(fft) ** 2
    d, h, w = vol.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)
    return {
        name: float(power[(radius >= lo) & (radius < hi)].sum())
        for name, (lo, hi) in BANDS.items()
    }


def brain_mask_simple(vol: np.ndarray, thresh: float = BRAIN_THRESHOLD) -> np.ndarray:
    m = (vol > thresh).astype(np.uint8)
    filled = np.zeros_like(m)
    for i in range(m.shape[0]):
        filled[i] = binary_fill_holes(m[i])
    return filled.astype(bool)


def vessel_mean(vol: np.ndarray) -> float:
    mask = brain_mask_simple(vol)
    if not mask.any():
        return 0.0
    v = frangi(vol, sigmas=FRANGI_SIGMAS, alpha=0.5, beta=0.5,
               gamma=None, black_ridges=False).astype(np.float32)
    return float(v[mask].mean())


def compute_metrics(vol: np.ndarray, real_be: dict[str, float]) -> dict:
    be = band_energies(vol)
    ratios = {f'ratio_{name}': (be[name] / real_be[name] if real_be[name] > 0 else 0.0)
              for name in BANDS}
    # HF proxy: (high + very_high) / real_total_same_bands
    hf_e = be['high'] + be['very_high']
    hf_real = real_be['high'] + real_be['very_high']
    ratios['ratio_hf_combined'] = float(hf_e / hf_real) if hf_real > 0 else 0.0
    ratios['vessel_mean'] = vessel_mean(vol)
    return ratios


def real_band_energies_from_volumes(volumes: list[np.ndarray]) -> dict[str, float]:
    per_vol = [band_energies(v) for v in volumes]
    return {name: float(np.mean([p[name] for p in per_vol])) for name in BANDS}


# ────────────────────────────────────────────────────────────────
# Stochastic Euler generation
# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def stochastic_generate(
    model,
    strategy: RFlowStrategy,
    seg: torch.Tensor,
    num_steps: int,
    T: int,
    sigma: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Euler generation with optional per-step Gaussian noise injection.

    sigma = 0 → deterministic (matches production).
    sigma > 0 → x_{t-Δt} += sigma · √|Δt_norm| · η each step except the final.
    The final step runs deterministic (we want a clean output, not a noisy one).
    """
    torch.manual_seed(seed)
    d, h, w = seg.shape[2], seg.shape[3], seg.shape[4]
    x_t = torch.randn(1, 1, d, h, w, device=device)

    steps = torch.linspace(T, 0.0, num_steps + 1, device=device)
    all_t = steps[:-1]
    all_next = steps[1:]

    strategy.scheduler.set_timesteps(
        num_inference_steps=num_steps, device=device,
        input_img_size_numel=d * h * w,
    )

    for i, (t, next_t) in enumerate(zip(all_t, all_next)):
        t_batch = t.unsqueeze(0).to(device)
        model_input = torch.cat([x_t, seg], dim=1)
        with autocast('cuda', dtype=torch.bfloat16):
            velocity = model(model_input, t_batch)
        x_t, _ = strategy.scheduler.step(velocity.float(), t, x_t, next_t)

        is_last = (i == len(all_t) - 1)
        if sigma > 0.0 and not is_last:
            dt_norm = abs(float((t - next_t).item()) / T)
            noise_amp = sigma * math.sqrt(dt_norm)
            x_t = x_t + noise_amp * torch.randn_like(x_t)

    return x_t.clamp(0, 1).squeeze().cpu().numpy()


# ────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────
def plot_band_bars(results: dict, output_base: Path) -> None:
    sigmas = sorted(results.keys())
    band_names = list(BANDS.keys())
    means = np.zeros((len(sigmas), len(band_names)))
    stds = np.zeros((len(sigmas), len(band_names)))
    for i, s in enumerate(sigmas):
        for j, bn in enumerate(band_names):
            key = f'ratio_{bn}'
            vals = [r[key] for r in results[s]['per_seed']]
            means[i, j] = float(np.mean(vals))
            stds[i, j] = float(np.std(vals))

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(band_names))
    width = 0.9 / max(len(sigmas), 1)
    cmap = plt.get_cmap('viridis')
    for i, s in enumerate(sigmas):
        color = cmap(i / max(len(sigmas) - 1, 1))
        ax.bar(x + (i - len(sigmas) / 2 + 0.5) * width, means[i],
               yerr=stds[i], capsize=2, width=width,
               label=f'σ={s:.3f}', color=color, alpha=0.85)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.6,
               label='real (ratio=1)')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.set_ylabel('per-band energy ratio to real')
    ax.set_title('Stochastic sampling — per-band energy by noise scale (mean ± std)')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=9, ncol=2)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_scalar_bar(results: dict, metric: str, ylabel: str,
                    output_base: Path, ref_line: float | None = None,
                    ref_label: str | None = None) -> None:
    sigmas = sorted(results.keys())
    means = [float(np.mean([r[metric] for r in results[s]['per_seed']])) for s in sigmas]
    stds = [float(np.std([r[metric] for r in results[s]['per_seed']])) for s in sigmas]
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(len(sigmas) - 1, 1)) for i in range(len(sigmas))]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(sigmas))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.9,
           edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'σ={s:.3f}' for s in sigmas])
    ax.set_ylabel(ylabel)
    ax.set_title(f'Stochastic sampling — {metric}')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    if ref_line is not None:
        ax.axhline(ref_line, color='black', linestyle='--', linewidth=0.8,
                   alpha=0.6, label=ref_label or f'ref = {ref_line:.4f}')
        ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_axial_grid(samples: dict[float, list[np.ndarray]], output_base: Path) -> None:
    """Grid: rows = σ, cols = seeds. Shows how one seed evolves across σ.

    (Because seeds are paired, column k is the same initial noise across rows.)
    """
    sigmas = sorted(samples.keys())
    n_seeds = len(next(iter(samples.values())))
    fig, axes = plt.subplots(len(sigmas), n_seeds,
                             figsize=(2.4 * n_seeds, 2.4 * len(sigmas)),
                             squeeze=False)
    for r, s in enumerate(sigmas):
        for c, vol in enumerate(samples[s]):
            zmid = vol.shape[0] // 2
            axes[r, c].imshow(vol[zmid], cmap='gray', vmin=0, vmax=1)
            axes[r, c].axis('off')
            if c == 0:
                axes[r, c].set_ylabel(f'σ={s:.3f}', fontsize=9)
            if r == 0:
                axes[r, c].set_title(f'seed {c}', fontsize=9)
    plt.suptitle('Axial mid-slice by noise scale × seed (paired seeds across rows)',
                 fontsize=10)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=180, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Stochastic-Euler ablation for RFlow")
    parser.add_argument('--checkpoint', required=True,
                        help='Path to exp1_1_1000 checkpoint (or any RFlow bravo model)')
    parser.add_argument('--data-root', required=True,
                        help='Root of brainmetshare-3 (for seg conditioning)')
    parser.add_argument('--real-dir', required=True,
                        help='Directory of real volumes for reference band energies')
    parser.add_argument('--output-dir', default='runs/eval/stochastic_sampling')
    parser.add_argument('--seg-split', default='test1')
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument('--sigmas', type=float, nargs='+', default=DEFAULT_SIGMAS)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--seed-base', type=int, default=42)
    parser.add_argument('--save-volumes', action='store_true',
                        help='Save every generated volume as NIfTI (eats disk; off by default)')
    parser.add_argument('--thesis-dir',
                        default='/home/mode/NTNU/AIS4900_doc/AIS4900-master-thesis/Images/stochastic_sampling')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Real band-energy reference
    real_root = Path(args.real_dir)
    logger.info(f"Computing real band energies from {real_root}")
    real_files = sorted(real_root.glob("*/bravo.nii.gz"))[:args.num_seeds * 2]
    if not real_files:
        raise SystemExit(f"No real volumes under {real_root}")
    real_vols = [load_volume(f, args.depth) for f in real_files]
    real_be = real_band_energies_from_volumes(real_vols)
    logger.info(f"  real bands: {real_be}")
    real_vessel_mean = float(np.mean([vessel_mean(v) for v in real_vols]))
    logger.info(f"  real vessel mean: {real_vessel_mean:.5f}")

    # Model + strategy
    logger.info(f"Loading model: {args.checkpoint}")
    model = load_diffusion_model(args.checkpoint, device=device,
                                 in_channels=2, out_channels=1,
                                 spatial_dims=3).eval()
    strategy = RFlowStrategy()
    T = strategy.scheduler.num_train_timesteps

    # Paired seg conditioning — same seg across σ values for every seed
    patients = discover_patients(Path(args.data_root), args.seg_split, args.num_seeds)
    if len(patients) < args.num_seeds:
        raise SystemExit(f"Only {len(patients)} patients found, need {args.num_seeds}")
    seg_tensors: list[torch.Tensor] = []
    for name, _, sp in patients:
        s_np = load_seg(sp, args.depth)
        seg_tensors.append(torch.from_numpy(s_np).unsqueeze(0).unsqueeze(0).to(device))
        logger.info(f"  seed seg: {name}")

    # Generation loop — outer: σ, inner: seed (so σ=0 runs first as baseline)
    results: dict[float, dict] = {}
    samples_for_grid: dict[float, list[np.ndarray]] = {}

    for sigma in args.sigmas:
        logger.info(f"σ = {sigma}")
        per_seed_metrics = []
        seed_vols: list[np.ndarray] = []
        for i, seg in enumerate(seg_tensors):
            seed = args.seed_base + i
            vol = stochastic_generate(model, strategy, seg, args.num_steps,
                                       T, sigma, seed, device)
            seed_vols.append(vol)
            m = compute_metrics(vol, real_be)
            per_seed_metrics.append(m)
            logger.info(f"  seed {i}: hf={m['ratio_hf_combined']:.3f}  "
                        f"vessel={m['vessel_mean']:.4f}")
            if args.save_volumes:
                save_nifti(vol, output_dir / 'vols' / f'sigma_{sigma:.3f}'
                           / f'seed_{i:03d}.nii.gz')

        # Cross-seed std (per-voxel) — then take its mean
        stack = np.stack(seed_vols, axis=0)
        cross_seed_std_map = stack.std(axis=0)
        cross_seed_std_mean = float(cross_seed_std_map.mean())

        results[sigma] = {
            'per_seed': per_seed_metrics,
            'cross_seed_std': cross_seed_std_mean,
        }
        samples_for_grid[sigma] = seed_vols
        logger.info(f"  σ={sigma}: cross_seed_std = {cross_seed_std_mean:.5f}")

        gc.collect()
        torch.cuda.empty_cache()

    # Plots
    logger.info("Plotting")
    plot_band_bars(results, output_dir / 'band_ratios')
    plot_scalar_bar(results, 'ratio_hf_combined',
                    'HF energy ratio (high + very_high bands)',
                    output_dir / 'hf_ratio', ref_line=1.0, ref_label='real = 1.0')
    plot_scalar_bar(results, 'vessel_mean', 'Mean Frangi (inside brain)',
                    output_dir / 'vessel_mean',
                    ref_line=real_vessel_mean,
                    ref_label=f'real = {real_vessel_mean:.4f}')

    # Single scalar across σ: cross-seed std
    sigmas_sorted = sorted(results.keys())
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    std_vals = [results[s]['cross_seed_std'] for s in sigmas_sorted]
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(len(sigmas_sorted) - 1, 1)) for i in range(len(sigmas_sorted))]
    ax.bar([f'σ={s:.3f}' for s in sigmas_sorted], std_vals, color=colors,
           alpha=0.9, edgecolor='black', linewidth=0.4)
    ax.set_ylabel('Mean cross-seed voxel std')
    ax.set_title('Cross-seed diversity (higher = more variation across seeds)')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_dir / f'cross_seed_std.{ext}', dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("  saved cross_seed_std.png + .pdf")

    plot_axial_grid(samples_for_grid, output_dir / 'axial_grid')

    # Summary print + JSON
    print()
    print("=" * 100)
    print("Stochastic sampling summary")
    print("=" * 100)
    header = f"{'σ':>8}{'hf_ratio (mean±std)':>24}{'vessel_mean (mean±std)':>28}{'cross_seed_std':>20}"
    print(header)
    print("-" * len(header))
    for s in sigmas_sorted:
        hf_vals = [r['ratio_hf_combined'] for r in results[s]['per_seed']]
        ves_vals = [r['vessel_mean'] for r in results[s]['per_seed']]
        print(
            f"{s:>8.3f}"
            f"{float(np.mean(hf_vals)):>14.4f}±{float(np.std(hf_vals)):.4f}"
            f"{float(np.mean(ves_vals)):>18.4f}±{float(np.std(ves_vals)):.4f}"
            f"{results[s]['cross_seed_std']:>20.5f}"
        )
    print(f"  real vessel_mean = {real_vessel_mean:.4f}")
    print("=" * 100)

    with open(output_dir / 'stochastic_results.json', 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'data_root': args.data_root,
            'real_dir': args.real_dir,
            'num_seeds': args.num_seeds,
            'num_steps': args.num_steps,
            'sigmas': sorted(args.sigmas),
            'real_band_energies': real_be,
            'real_vessel_mean': real_vessel_mean,
            'results': {
                f'{s:.4f}': results[s] for s in sigmas_sorted
            },
        }, f, indent=2)
    logger.info(f"Saved: {output_dir / 'stochastic_results.json'}")

    if args.thesis_dir:
        thesis_dir = Path(args.thesis_dir)
        thesis_dir.mkdir(parents=True, exist_ok=True)
        for name in ('band_ratios', 'hf_ratio', 'vessel_mean', 'cross_seed_std',
                     'axial_grid'):
            for ext in ('png', 'pdf'):
                src = output_dir / f'{name}.{ext}'
                if src.exists():
                    (thesis_dir / f'{name}.{ext}').write_bytes(src.read_bytes())
        logger.info(f"Copied figures to thesis dir: {thesis_dir}")


if __name__ == '__main__':
    main()
