#!/usr/bin/env python3
"""Per-t hybrid generation ablation (Phase 1 #5).

At each Euler step during generation, choose which checkpoint's velocity to use
based on the current t. Tests whether splitting the trajectory into phases and
using different models for each phase beats using any single model throughout.

Design: two checkpoints (typically baseline + one fine-tune) and a set of
t-boundaries. For each boundary, generate with "high-t model" used when
`t > boundary` and "low-t model" used when `t <= boundary`. Compare eight
configurations:

  - `baseline`:   baseline throughout (reference)
  - `ft`:         fine-tune throughout (reference)
  - `split_XX`:   baseline at high t, fine-tune at low t, boundary XX ∈ {0.3, 0.5, 0.7}
  - `rev_XX`:     fine-tune at high t, baseline at low t, boundary XX ∈ {0.3, 0.5, 0.7}

For each config × seed, compute:
  - Band-energy ratios to real (very_low / low / mid / high / very_high)
  - Brain coherence (largest connected component / total mask)
  - Mean Frangi vesselness inside brain

Plots per-band ratio per config (bar chart) + axial-slice comparison grid.

Interpretation heuristics:
  - If `split_0.5` beats both `baseline` and `ft` → two-model ensemble is best
  - If `ft` beats all splits → single fine-tune everywhere is best (no hand-off)
  - If `baseline` beats all splits → fine-tune failed (shouldn't happen)
  - If `rev_XX` beats `split_XX` → fine-tune was trained for the WRONG t-range

Usage:
    python -m medgen.scripts.analyze_hybrid_generation \\
        --baseline /path/to/baseline.pt \\
        --fine-tune /path/to/ft.pt \\
        --ft-label exp37_3 \\
        --data-root /path/to/brainmetshare-3 \\
        --real-dir /path/to/brainmetshare-3/test1 \\
        --output-dir runs/eval/hybrid_generation
"""
import argparse
import gc
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as cc_label
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

DEFAULT_BOUNDARIES = [0.3, 0.5, 0.7]
DEFAULT_NUM_STEPS = 32


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


# ────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────
def radial_band_ratios(vol: np.ndarray, real_band_energies: dict[str, float]
                       ) -> dict[str, float]:
    """Per-band energy ratio to real."""
    fft = np.fft.fftshift(np.fft.fftn(vol))
    power = np.abs(fft) ** 2
    d, h, w = vol.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)

    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (radius >= lo) & (radius < hi)
        e = power[mask].sum()
        real_e = real_band_energies[name]
        out[name] = float(e / real_e) if real_e > 0 else 0.0
    return out


def real_band_energies_from_volumes(volumes: list[np.ndarray]) -> dict[str, float]:
    """Compute mean per-band energy across a list of real volumes."""
    per_vol = []
    for vol in volumes:
        fft = np.fft.fftshift(np.fft.fftn(vol))
        power = np.abs(fft) ** 2
        d, h, w = vol.shape
        cd, ch, cw = d // 2, h // 2, w // 2
        dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
        radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)
        per_vol.append({
            name: float(power[(radius >= lo) & (radius < hi)].sum())
            for name, (lo, hi) in BANDS.items()
        })
    # Average across volumes
    return {
        name: float(np.mean([pv[name] for pv in per_vol]))
        for name in BANDS
    }


def brain_coherence(vol: np.ndarray) -> float:
    mask = (vol > BRAIN_THRESHOLD).astype(np.uint8)
    filled = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        filled[i] = binary_fill_holes(mask[i])
    total = filled.sum()
    if total == 0:
        return 0.0
    labels, _ = cc_label(filled)
    if labels.max() == 0:
        return 0.0
    largest = np.bincount(labels.flatten())[1:].max()
    return float(largest / total)


def vessel_mean(vol: np.ndarray) -> float:
    mask = vol > BRAIN_THRESHOLD
    if not mask.any():
        return 0.0
    v = frangi(vol, sigmas=FRANGI_SIGMAS, alpha=0.5, beta=0.5,
               gamma=None, black_ridges=False).astype(np.float32)
    return float(v[mask].mean())


def compute_metrics(vol: np.ndarray, real_be: dict[str, float]) -> dict:
    out = radial_band_ratios(vol, real_be)
    out['brain_coherence'] = brain_coherence(vol)
    out['vessel_mean'] = vessel_mean(vol)
    return out


# ────────────────────────────────────────────────────────────────
# Hybrid generation
# ────────────────────────────────────────────────────────────────
def hybrid_generate(
    model_hi,                 # velocity source for t > boundary
    model_lo,                 # velocity source for t <= boundary
    strategy,
    seg: torch.Tensor,
    num_steps: int,
    T: int,
    boundary_norm: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Euler generation with model chosen per-step by t."""
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

    boundary_abs = boundary_norm * T

    for t, next_t in zip(all_t, all_next):
        t_batch = t.unsqueeze(0).to(device)
        model_input = torch.cat([x_t, seg], dim=1)
        use_model = model_hi if t.item() > boundary_abs else model_lo

        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            velocity = use_model(model_input, t_batch)
        x_t, _ = strategy.scheduler.step(velocity.float(), t, x_t, next_t)

    return x_t.clamp(0, 1).squeeze().cpu().numpy()


# ────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────
def plot_band_bars(results: dict, real_be: dict, output_base: Path) -> None:
    configs = list(results.keys())
    band_names = list(BANDS.keys())
    means = np.zeros((len(configs), len(band_names)))
    stds = np.zeros((len(configs), len(band_names)))
    for i, cfg in enumerate(configs):
        for j, bn in enumerate(band_names):
            vals = [s[bn] for s in results[cfg]['per_seed']]
            means[i, j] = float(np.mean(vals))
            stds[i, j] = float(np.std(vals))

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(band_names))
    width = 0.9 / max(len(configs), 1)
    cmap = plt.get_cmap('tab10')
    for i, cfg in enumerate(configs):
        color = cmap(i % 10)
        ax.bar(x + (i - len(configs) / 2 + 0.5) * width, means[i],
               yerr=stds[i], capsize=2, width=width, label=cfg,
               color=color, alpha=0.85)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.6,
               label='real (ratio=1)')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.set_ylabel('per-band energy ratio to real')
    ax.set_title('Hybrid generation — per-band energy by config (mean ± std)')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_metric_bars(results: dict, metric: str, ylabel: str,
                     output_base: Path) -> None:
    configs = list(results.keys())
    means = [float(np.mean([s[metric] for s in results[cfg]['per_seed']])) for cfg in configs]
    stds = [float(np.std([s[metric] for s in results[cfg]['per_seed']])) for cfg in configs]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(configs))
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(configs))]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
           edgecolor='black', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Hybrid generation — {metric}')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_axial_grid(
    samples: dict[str, np.ndarray],
    output_base: Path,
) -> None:
    """Show one axial slice per config side by side."""
    names = list(samples.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3), squeeze=False)
    slice_idx = next(iter(samples.values())).shape[0] // 2
    for i, name in enumerate(names):
        vol = samples[name]
        axes[0, i].imshow(vol[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(name, fontsize=9)
        axes[0, i].axis('off')
    plt.suptitle('Axial slice — one seed per config', fontsize=10)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.5 — hybrid generation ablation")
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--fine-tune', required=True)
    parser.add_argument('--ft-label', default='ft',
                        help='Short label for the fine-tune (used in plots)')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--real-dir', required=True,
                        help='Directory of reference real volumes (for band normalization)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument('--boundaries', nargs='+', type=float, default=DEFAULT_BOUNDARIES)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--seg-split', default='test1')
    parser.add_argument('--seed-base', type=int, default=42)
    parser.add_argument('--no-vessel', action='store_true',
                        help='Skip Frangi vesselness (Frangi is slow)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')

    # Load data (seg conditioning + real reference)
    data_root = Path(args.data_root)
    patients = discover_patients(data_root, args.seg_split, args.num_seeds)
    if len(patients) < args.num_seeds:
        raise SystemExit(f"Need {args.num_seeds} patients, found {len(patients)}")
    logger.info(f"Using {len(patients)} patients from {args.seg_split}")
    cached_seg: list[tuple[str, torch.Tensor]] = []
    for pid, _, sp in patients:
        s_np = load_seg(sp, args.depth)
        cached_seg.append((pid, torch.from_numpy(s_np).unsqueeze(0).unsqueeze(0).to(device)))

    # Real reference volumes → per-band energy normalization
    logger.info(f"Computing real reference band energies from {args.real_dir}")
    real_root = Path(args.real_dir)
    real_files = sorted(real_root.glob('*/bravo.nii.gz'))[:args.num_seeds]
    real_vols = [load_volume(fp, args.depth) for fp in real_files]
    real_be = real_band_energies_from_volumes(real_vols)
    logger.info("  real band energies: "
                + ", ".join(f"{n}={v:.3e}" for n, v in real_be.items()))

    # Setup strategy
    strategy = RFlowStrategy()
    strategy.setup_scheduler(num_timesteps=1000, image_size=256,
                             depth_size=args.depth, spatial_dims=3)
    T = strategy.scheduler.num_train_timesteps

    # Load both checkpoints simultaneously
    logger.info(f"Loading baseline: {args.baseline}")
    baseline = load_diffusion_model(args.baseline, device=device,
                                    compile_model=False, spatial_dims=3)
    baseline.eval()

    logger.info(f"Loading fine-tune ({args.ft_label}): {args.fine_tune}")
    ft = load_diffusion_model(args.fine_tune, device=device,
                              compile_model=False, spatial_dims=3)
    ft.eval()

    # Build list of configs
    configs: list[tuple[str, object, object, float]] = []
    configs.append(('baseline', baseline, baseline, 0.0))   # never switches → all baseline
    configs.append((args.ft_label, ft, ft, 0.0))            # never switches → all ft
    for b in args.boundaries:
        # baseline at high t, ft at low t
        configs.append((f'split_{b:.2f}', baseline, ft, b))
        # ft at high t, baseline at low t
        configs.append((f'rev_{b:.2f}',    ft, baseline, b))

    # Run all configs × seeds
    results: dict[str, dict] = {}
    example_samples: dict[str, np.ndarray] = {}  # first seed per config

    for cfg_name, model_hi, model_lo, boundary in configs:
        logger.info(f"--- Config: {cfg_name} ---")
        per_seed_stats: list[dict] = []
        for i, (pid, seg) in enumerate(cached_seg):
            vol_np = hybrid_generate(
                model_hi, model_lo, strategy, seg,
                num_steps=args.num_steps, T=T,
                boundary_norm=boundary,
                seed=args.seed_base + i,
                device=device,
            )
            if args.no_vessel:
                metrics = radial_band_ratios(vol_np, real_be)
                metrics['brain_coherence'] = brain_coherence(vol_np)
                metrics['vessel_mean'] = 0.0
            else:
                metrics = compute_metrics(vol_np, real_be)
            metrics['patient'] = pid
            per_seed_stats.append(metrics)
            if i == 0:
                example_samples[cfg_name] = vol_np
            logger.info(f"  [{cfg_name}] seed {i}: "
                        + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                                    if k in ('mid', 'high', 'very_high', 'vessel_mean')))
            torch.cuda.empty_cache()

        results[cfg_name] = {'per_seed': per_seed_stats,
                             'boundary': boundary}

    # Plots
    logger.info("Plotting")
    plot_band_bars(results, real_be, output_dir / 'hybrid_band_ratios')
    plot_metric_bars(results, 'brain_coherence',
                     'Brain coherence (largest CC / total)',
                     output_dir / 'hybrid_brain_coherence')
    if not args.no_vessel:
        plot_metric_bars(results, 'vessel_mean',
                         'Mean Frangi vesselness (inside brain)',
                         output_dir / 'hybrid_vessel_mean')
    plot_axial_grid(example_samples, output_dir / 'hybrid_axial_grid')

    # JSON dump
    with open(output_dir / 'hybrid_generation_results.json', 'w') as f:
        json.dump({
            'baseline': args.baseline,
            'fine_tune': args.fine_tune,
            'ft_label': args.ft_label,
            'boundaries': args.boundaries,
            'num_steps': args.num_steps,
            'num_seeds': args.num_seeds,
            'seg_split': args.seg_split,
            'real_band_energies': real_be,
            'configs': {
                name: {
                    'boundary': r['boundary'],
                    'per_seed': r['per_seed'],
                    'mean_band_ratios': {
                        bn: float(np.mean([s[bn] for s in r['per_seed']]))
                        for bn in BANDS
                    },
                    'mean_brain_coherence': float(np.mean(
                        [s['brain_coherence'] for s in r['per_seed']])),
                    'mean_vessel': float(np.mean(
                        [s['vessel_mean'] for s in r['per_seed']])),
                } for name, r in results.items()
            },
        }, f, indent=2)
    logger.info(f"Saved: {output_dir / 'hybrid_generation_results.json'}")

    # Terminal summary — highlight the vessel band (the gap we care about)
    print()
    print("=" * 78)
    print("Hybrid generation summary — ratio to real at key bands")
    print("=" * 78)
    print(f"{'config':<16} {'mid':>10} {'high':>10} {'very_high':>12} "
          f"{'coherence':>12} {'vessel':>12}")
    print("-" * 78)
    for name in results:
        r = results[name]
        mid = float(np.mean([s['mid'] for s in r['per_seed']]))
        hi = float(np.mean([s['high'] for s in r['per_seed']]))
        vhi = float(np.mean([s['very_high'] for s in r['per_seed']]))
        coh = float(np.mean([s['brain_coherence'] for s in r['per_seed']]))
        ves = float(np.mean([s['vessel_mean'] for s in r['per_seed']]))
        print(f"{name:<16} {mid:>10.3f} {hi:>10.3f} {vhi:>12.3f} "
              f"{coh:>12.4f} {ves:>12.5f}")
    print("=" * 78)

    del baseline, ft
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
