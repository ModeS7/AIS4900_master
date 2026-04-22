#!/usr/bin/env python3
"""Generation trajectory emergence analysis (Phase 1 #3).

Starting from pure noise, run Euler integration to t=0 while tracking at each
step what features have emerged. For each step, project to x̂_0 = x + t*v_pred
and compute:

  - Brain mask fraction (fraction of volume above brain threshold)
  - Brain coherence (largest connected component / total mask)
  - HF energy ratio (above Nyquist/4)
  - Mid-band energy fraction (0.20-0.30 of Nyquist)
  - Very-high-band energy fraction (0.40-0.50 — vessel scale)
  - Frangi vesselness mean (inside brain)

Plots each metric vs t. Shows exactly when during generation each feature
gets committed. Compares across multiple seeds for robustness.

Inputs (one at a time, SLURM wrapper for multi-model):
  - A diffusion checkpoint
  - A seg mask directory (uses real test1 seg as conditioning)

Usage:
    python -m medgen.scripts.analyze_generation_trajectory \\
        --bravo-model /path/to/checkpoint.pt \\
        --data-root /path/to/brainmetshare-3 \\
        --output-dir runs/eval/trajectory_emergence_<model>_<timestamp> \\
        --num-seeds 3 --num-steps 50
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
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as cc_label
from skimage.filters import frangi
from torch.amp import autocast

from medgen.diffusion import RFlowStrategy, load_diffusion_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


BRAIN_THRESHOLD = 0.05
VESSEL_THRESHOLD = 0.05
FRANGI_SIGMAS = (0.5, 1.0, 1.5, 2.0, 3.0)

# t values (normalized) at which to save axial-slice snapshots for the montage.
SNAPSHOT_T = [0.95, 0.80, 0.60, 0.40, 0.20, 0.10, 0.02]


def load_seg(path: Path, depth: int) -> np.ndarray:
    """Load seg.nii.gz as binary mask [D, H, W]."""
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vol = np.transpose(vol, (2, 0, 1))
    d = vol.shape[0]
    if d < depth:
        vol = np.pad(vol, ((0, depth - d), (0, 0), (0, 0)))
    elif d > depth:
        vol = vol[:depth]
    return (vol > 0.5).astype(np.float32)


# ────────────────────────────────────────────────────────────────
# Emergence metrics — compute on predicted x̂_0 (clamped to [0, 1])
# ────────────────────────────────────────────────────────────────
def brain_mask_fraction(vol: np.ndarray) -> float:
    return float((vol > BRAIN_THRESHOLD).mean())


def brain_coherence(vol: np.ndarray) -> float:
    """Largest connected component / total mask (1.0 = one blob, lower = fragmented)."""
    mask = (vol > BRAIN_THRESHOLD).astype(np.uint8)
    # Fill holes slice-wise (cheap 2D)
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


def radial_band_energy(vol: np.ndarray, lo: float, hi: float) -> float:
    """Fraction of FFT energy in radial band [lo, hi] of Nyquist (0.5)."""
    fft = np.fft.fftshift(np.fft.fftn(vol))
    power = np.abs(fft) ** 2
    d, h, w = vol.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)
    band = (radius >= lo) & (radius < hi)
    total = power.sum()
    if total <= 0:
        return 0.0
    return float(power[band].sum() / total)


def hf_energy_ratio(vol: np.ndarray) -> float:
    """Fraction of FFT energy above Nyquist/4 (matches diagnose_mean_blur.py)."""
    return radial_band_energy(vol, 0.25, 0.5)


def vessel_score(vol: np.ndarray) -> float:
    """Mean Frangi vesselness inside brain mask."""
    mask = vol > BRAIN_THRESHOLD
    if not mask.any():
        return 0.0
    # Frangi is expensive — downsample slightly for speed during trajectory tracking
    v = frangi(vol, sigmas=FRANGI_SIGMAS, alpha=0.5, beta=0.5,
               gamma=None, black_ridges=False).astype(np.float32)
    return float(v[mask].mean())


def compute_emergence_metrics(vol_np: np.ndarray, include_vessel: bool) -> dict:
    """Compute all emergence metrics for a clamped [D, H, W] prediction."""
    out = {
        'brain_frac': brain_mask_fraction(vol_np),
        'brain_coherence': brain_coherence(vol_np),
        'hf_energy': hf_energy_ratio(vol_np),
        'band_mid': radial_band_energy(vol_np, 0.20, 0.30),
        'band_very_high': radial_band_energy(vol_np, 0.40, 0.50),
    }
    # Frangi is slow (~5s per 256³ volume) — skip during most steps
    out['vessel_score'] = vessel_score(vol_np) if include_vessel else float('nan')
    return out


# ────────────────────────────────────────────────────────────────
# Generation loop — saves x̂_0 projections per step
# ────────────────────────────────────────────────────────────────
def generate_trajectory(
    model, strategy, seg: torch.Tensor,
    num_steps: int, T: int, seed: int,
    snapshot_ts_norm: list[float],
    vessel_every_n_steps: int,
    device: torch.device,
) -> tuple[list[dict], list[np.ndarray], list[float]]:
    """Run one full generation trajectory, computing metrics at each step.

    Returns:
      per_step_metrics: list of metric dicts (one per step)
      snapshots: list of (D, H, W) x̂_0 numpy arrays at snapshot_ts_norm
      snapshot_t_actual: t values (normalized) at which snapshots were taken
    """
    torch.manual_seed(seed)
    # Init pure noise
    d, h, w = seg.shape[2], seg.shape[3], seg.shape[4]
    x_t = torch.randn(1, 1, d, h, w, device=device)

    # Uniform timestep sequence from T down to 0
    steps = torch.linspace(T, 0.0, num_steps + 1, device=device)
    all_t = steps[:-1]
    all_next = steps[1:]

    strategy.scheduler.set_timesteps(
        num_inference_steps=num_steps, device=device,
        input_img_size_numel=d * h * w,
    )

    per_step: list[dict] = []
    snapshots: list[np.ndarray] = []
    snapshot_ts: list[float] = []

    # Build list of (step_idx, snapshot_t) to save at
    wanted_snapshot_ts = [t * T for t in snapshot_ts_norm]  # convert normalized -> absolute
    snapshot_idx_set: set[int] = set()
    for wt in wanted_snapshot_ts:
        idx = int(torch.argmin(torch.abs(all_t - wt)).item())
        snapshot_idx_set.add(idx)

    for step_idx, (t, next_t) in enumerate(zip(all_t, all_next)):
        t_batch = t.unsqueeze(0).to(device)
        model_input = torch.cat([x_t, seg], dim=1)
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            velocity = model(model_input, t_batch)
        velocity = velocity.float()

        # Compute predicted clean x̂_0 for metrics
        x0_pred = strategy.compute_predicted_clean(x_t, velocity, t_batch).clamp(0, 1)
        x0_np = x0_pred.squeeze().cpu().numpy()

        include_vessel = (step_idx % vessel_every_n_steps == 0) or (step_idx == num_steps - 1)
        m = compute_emergence_metrics(x0_np, include_vessel=include_vessel)
        m['step_idx'] = step_idx
        m['t_abs'] = float(t.item())
        m['t_norm'] = float(t.item() / T)
        per_step.append(m)

        if step_idx in snapshot_idx_set:
            snapshots.append(x0_np.copy())
            snapshot_ts.append(m['t_norm'])

        # Euler step to next x_t
        x_t, _ = strategy.scheduler.step(velocity, t, x_t, next_t)

    return per_step, snapshots, snapshot_ts


# ────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────
METRIC_INFO = [
    ('brain_frac', 'Brain mask fraction', 'teal'),
    ('brain_coherence', 'Brain coherence (largest CC / total)', 'darkgreen'),
    ('hf_energy', 'HF energy ratio (>Nyquist/4)', 'steelblue'),
    ('band_mid', 'Mid-band energy fraction (0.20-0.30)', 'darkorange'),
    ('band_very_high', 'Very-high-band energy (0.40-0.50 — vessels)', 'firebrick'),
    ('vessel_score', 'Frangi vesselness mean (inside brain)', 'purple'),
]


def plot_emergence_curves(all_trajectories: list[list[dict]], output_base: Path) -> None:
    """Multi-panel plot — one panel per metric, one line per seed + mean."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()

    for ax, (mkey, mtitle, color) in zip(axes, METRIC_INFO):
        # Gather per-seed curves, align on t_norm
        t_vals = np.array([m['t_norm'] for m in all_trajectories[0]])
        seed_curves = []
        for traj in all_trajectories:
            vals = np.array([m[mkey] for m in traj], dtype=np.float64)
            seed_curves.append(vals)
        seed_curves = np.array(seed_curves)  # [n_seeds, n_steps]

        # For vessel_score (computed sparsely), interpolate NaNs
        if mkey == 'vessel_score':
            for i in range(seed_curves.shape[0]):
                row = seed_curves[i]
                valid = ~np.isnan(row)
                if valid.any():
                    seed_curves[i] = np.interp(
                        np.arange(len(row)),
                        np.where(valid)[0], row[valid],
                    )

        # Plot per-seed (thin) + mean (thick)
        for row in seed_curves:
            ax.plot(t_vals, row, color=color, alpha=0.25, linewidth=0.8)
        mean_curve = np.nanmean(seed_curves, axis=0)
        ax.plot(t_vals, mean_curve, color=color, linewidth=2.0, label='mean')

        ax.set_xlabel('t (normalized) — generation goes right→left')
        ax.set_ylabel(mkey)
        ax.set_title(mtitle, fontsize=10)
        ax.invert_xaxis()  # So generation is left-to-right visually
        ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=8)

    plt.suptitle('Generation trajectory — feature emergence per t (mean + per-seed)',
                 fontsize=11)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_snapshot_montage(
    all_snapshots: list[list[np.ndarray]],
    all_ts: list[list[float]],
    output_base: Path,
) -> None:
    """Axial-slice montage. Rows = seeds, cols = snapshot t values."""
    n_seeds = len(all_snapshots)
    n_t = max(len(ts) for ts in all_ts) if all_ts else 0
    if n_t == 0 or n_seeds == 0:
        return
    fig, axes = plt.subplots(n_seeds, n_t, figsize=(2.4 * n_t, 2.4 * n_seeds),
                             squeeze=False)

    for i, (snaps, ts) in enumerate(zip(all_snapshots, all_ts)):
        for j, (snap, t_norm) in enumerate(zip(snaps, ts)):
            ax = axes[i, j]
            slice_idx = snap.shape[0] // 2
            ax.imshow(snap[slice_idx], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if i == 0:
                ax.set_title(f't={t_norm:.2f}', fontsize=9)
            if j == 0:
                ax.text(-0.15, 0.5, f'seed {i}', transform=ax.transAxes,
                        fontsize=9, rotation=90, va='center')

    plt.suptitle('x̂_0 projection at snapshot t — generation progresses left→right',
                 fontsize=10)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def discover_seg_files(data_root: Path, split: str, num_files: int) -> list[Path]:
    """Find `num_files` seg.nii.gz files under the given split."""
    d = data_root / split
    if not d.exists():
        d = data_root / 'test'
    segs = sorted(d.glob('*/seg.nii.gz'))
    return segs[:num_files]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.3 — trajectory emergence")
    parser.add_argument('--bravo-model', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-seeds', type=int, default=3)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--seg-split', default='test1')
    parser.add_argument('--seed-base', type=int, default=42)
    parser.add_argument('--vessel-every-n-steps', type=int, default=5,
                        help='Frangi is slow — compute only every N steps (interpolated later)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')

    logger.info(f"Loading bravo model from {args.bravo_model}")
    model = load_diffusion_model(
        args.bravo_model, device=device, compile_model=False, spatial_dims=3,
    )
    model.eval()
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000, image_size=256, depth_size=args.depth, spatial_dims=3,
    )
    T = strategy.scheduler.num_train_timesteps

    data_root = Path(args.data_root)
    segs = discover_seg_files(data_root, args.seg_split, args.num_seeds)
    if len(segs) < args.num_seeds:
        raise SystemExit(f"Need {args.num_seeds} seg files, found {len(segs)}")
    logger.info(f"Using {len(segs)} seg masks from {args.seg_split}")

    # Resume support: if emergence_metrics.json already exists in output_dir,
    # load the seeds already processed and skip them on this run.
    results_path = output_dir / 'emergence_metrics.json'
    all_trajectories: list[list[dict]] = []
    all_snapshots: list[list[np.ndarray]] = []
    all_snap_ts: list[list[float]] = []
    resume_completed = 0
    if results_path.exists():
        try:
            with open(results_path) as f:
                prior = json.load(f)
            if prior.get('num_steps') == args.num_steps \
                    and prior.get('seg_split') == args.seg_split \
                    and prior.get('checkpoint') == args.bravo_model:
                prior_trajs = prior.get('seed_trajectories', [])
                all_trajectories = prior_trajs
                resume_completed = len(prior_trajs)
                # Snapshots are not persisted in JSON; resumed seeds will miss
                # their snapshot slices — montage shows only newly-computed ones.
                all_snapshots = [[] for _ in range(resume_completed)]
                all_snap_ts = [[] for _ in range(resume_completed)]
                logger.info(
                    f"RESUMING: found existing {results_path} with {resume_completed} "
                    f"seeds already completed. Skipping to seed {resume_completed}."
                )
            else:
                logger.warning(
                    f"Existing {results_path} has different args "
                    f"(num_steps/seg_split/checkpoint), starting fresh."
                )
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Could not parse existing {results_path} ({e}); starting fresh.")

    for i, seg_path in enumerate(segs):
        if i < resume_completed:
            continue
        logger.info(f"=== Seed {i} (seg={seg_path.parent.name}) ===")
        seg_np = load_seg(seg_path, args.depth)
        seg = torch.from_numpy(seg_np).unsqueeze(0).unsqueeze(0).to(device)

        per_step, snaps, snap_ts = generate_trajectory(
            model, strategy, seg,
            num_steps=args.num_steps,
            T=T,
            seed=args.seed_base + i,
            snapshot_ts_norm=SNAPSHOT_T,
            vessel_every_n_steps=args.vessel_every_n_steps,
            device=device,
        )
        all_trajectories.append(per_step)
        all_snapshots.append(snaps)
        all_snap_ts.append(snap_ts)

        # Cleanup GPU between seeds
        torch.cuda.empty_cache()

        # Incremental save — survives SLURM timeout / OOM / cancellation.
        # Overwrites previous partial file; new file contains all seeds so far.
        with open(results_path, 'w') as f:
            json.dump({
                'checkpoint': args.bravo_model,
                'num_seeds': args.num_seeds,
                'num_steps': args.num_steps,
                'snapshot_ts_norm': SNAPSHOT_T,
                'seg_split': args.seg_split,
                'vessel_every_n_steps': args.vessel_every_n_steps,
                'seed_trajectories': all_trajectories,
            }, f, indent=2)
        logger.info(f"  Saved progress: {len(all_trajectories)}/{len(segs)} seeds → {results_path}")

    # Plots (JSON was already saved incrementally per-seed above).
    # Snapshot montage uses only this-run snapshots; seeds resumed from a prior
    # run will show as gaps — this is a recoverable cosmetic artifact.
    logger.info("Plotting emergence curves")
    plot_emergence_curves(all_trajectories, output_dir / 'emergence_curves')
    plot_snapshot_montage(all_snapshots, all_snap_ts, output_dir / 'snapshot_montage')

    # Simple summary: at what t does each metric reach 90% of its final value?
    print()
    print("=" * 76)
    print("Emergence timing — per seed, t at which metric reaches 90% of final")
    print("=" * 76)
    for mkey, _mtitle, _ in METRIC_INFO:
        tvals = []
        for traj in all_trajectories:
            vals = np.array([m[mkey] for m in traj], dtype=np.float64)
            # Handle NaN in vessel_score
            valid = ~np.isnan(vals)
            if not valid.any():
                continue
            final = vals[valid][-1]
            target = 0.9 * final
            ts = np.array([m['t_norm'] for m in traj])
            # Find first t (going right→left, so from large t to small) where |val| >= target
            idx = np.where(valid & (vals >= target))[0]
            if len(idx) > 0:
                tvals.append(ts[idx[0]])
        if tvals:
            print(f"  {mkey:<20} reaches 90% at t_norm = {np.mean(tvals):.3f} "
                  f"(std {np.std(tvals):.3f})")
    print("=" * 76)


if __name__ == '__main__':
    main()
