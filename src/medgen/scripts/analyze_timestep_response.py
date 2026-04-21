#!/usr/bin/env python3
"""Timestep-response diagnostic.

Sweeps a fine grid of t values over [0, 1]; at each t, adds calibrated RFlow
noise to N real volumes and measures velocity MSE. Used to verify:

  (a) The model's MSE vs t curve is **smooth** — no spikes or
      discontinuities that would indicate a pathology in the timestep
      embedding (e.g. sinusoidal frequencies aliasing at a specific t,
      or a training gap in the t distribution).
  (b) Per-volume curves follow the same **shape** — no single volume
      drives the mean; if one volume shows a spike at t* but others
      don't, that's data-specific not model-specific.
  (c) The loss landscape over t matches RFlow expectations. For velocity
      prediction under x_t = (1-t)·x_0 + t·ε with target v = x_0 - ε:
      MSE should be smooth and U-shaped (or similar continuous shape)
      because the model has the most information about v near t=0.5.

Fixed noise per volume: for each real x_0 we draw ε once and reuse it
across ALL t values. Eliminates noise variance from the plot so any
jaggedness in the curve comes from the model's response, not sampling.

Outputs:
  Figure 1: mean±std MSE vs t (primary smooth-curve diagnostic)
  Figure 2: per-volume curves overlaid (shape consistency check)
  Figure 3: d(MSE)/dt (finite-difference derivative — spike detector)
  Figure 4: MSE distribution at representative t values (histogram)
  JSON:     raw t-array and MSE matrix [n_t, n_vol]

Usage:
    python -m medgen.scripts.analyze_timestep_response \\
        --checkpoint /path/to/exp1_1_1000_checkpoint_latest.pt \\
        --data-root /path/to/brainmetshare-3 \\
        --seg-split test1 \\
        --output-dir runs/eval/timestep_response \\
        --num-volumes 10 --num-t 1000
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
from torch.amp import autocast

from medgen.diffusion import RFlowStrategy, load_diffusion_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


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


@torch.no_grad()
def sweep_timestep_mse(
    model,
    real_volumes: list[torch.Tensor],
    seg_tensors: list[torch.Tensor],
    t_values: np.ndarray,
    T: int,
    device: torch.device,
    log_every: int = 50,
) -> np.ndarray:
    """For each (t, volume) pair, return velocity-MSE.

    Output: [n_t, n_vol] matrix.

    Protocol:
      - Each volume draws ε once (reused across all t for that volume)
      - x_t = (1-t) x_0 + t ε
      - v_target = x_0 - ε  (constant per-volume across t)
      - v_pred = model([x_t, seg], t*T)
      - mse = mean((v_pred - v_target)^2)
    """
    n_t = len(t_values)
    n_vol = len(real_volumes)
    mse = np.zeros((n_t, n_vol), dtype=np.float64)

    # Fix ε per volume so any curve jaggedness is the model, not noise re-sampling
    torch.manual_seed(42)
    eps_list = [torch.randn_like(x0) for x0 in real_volumes]

    v_targets = [x0 - eps for x0, eps in zip(real_volumes, eps_list)]

    for i, t_norm in enumerate(t_values):
        t_model = torch.tensor([float(t_norm) * T], device=device, dtype=torch.float32)
        for j, (x0, eps, seg, vtgt) in enumerate(
            zip(real_volumes, eps_list, seg_tensors, v_targets)
        ):
            x_t = (1.0 - float(t_norm)) * x0 + float(t_norm) * eps
            model_input = torch.cat([x_t, seg], dim=1)

            with autocast('cuda', dtype=torch.bfloat16):
                v_pred = model(model_input, t_model)

            diff = v_pred.float() - vtgt
            mse[i, j] = float((diff * diff).mean().item())

        if (i + 1) % log_every == 0 or i == 0:
            logger.info(f"  t={float(t_norm):.4f} ({i + 1}/{n_t})  "
                        f"mean MSE across {n_vol} volumes: {mse[i].mean():.4f}")

    return mse


# ────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────
def plot_mean_curve(
    t: np.ndarray,
    mse: np.ndarray,  # [n_t, n_vol]
    output_base: Path,
) -> None:
    mean = mse.mean(axis=1)
    std = mse.std(axis=1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(t, mean, color='tab:blue', linewidth=1.5, label=f'mean (n={mse.shape[1]} vols)')
    ax.fill_between(t, mean - std, mean + std, color='tab:blue', alpha=0.25, label='±1 std')
    ax.set_xlabel('t (normalized, 0=clean  1=noise)')
    ax.set_ylabel('velocity MSE')
    ax.set_title('Velocity MSE vs t — smoothness diagnostic')
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='best')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_per_volume_overlay(
    t: np.ndarray,
    mse: np.ndarray,
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_vol = mse.shape[1]
    cmap = plt.get_cmap('tab10')
    for j in range(n_vol):
        ax.plot(t, mse[:, j], color=cmap(j % 10), linewidth=0.8, alpha=0.75,
                label=f'vol {j}')
    ax.set_xlabel('t')
    ax.set_ylabel('velocity MSE')
    ax.set_title('Per-volume MSE vs t — shape consistency check')
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='best', ncol=min(n_vol, 5), fontsize=8)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_derivative(
    t: np.ndarray,
    mse: np.ndarray,
    output_base: Path,
) -> None:
    """Finite-difference derivative of mean MSE — spike detector.

    A smooth curve has a continuous derivative. A spike in MSE → sharp
    positive-then-negative pair in the derivative (easy to spot visually).
    """
    mean = mse.mean(axis=1)
    dmean = np.gradient(mean, t)
    abs_d = np.abs(dmean)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(t, dmean, color='tab:red', linewidth=0.9)
    axes[0].axhline(0, color='black', linewidth=0.4, alpha=0.6)
    axes[0].set_ylabel('d(MSE)/dt')
    axes[0].set_title('Derivative of mean MSE — discontinuities manifest as sharp spikes')
    axes[0].grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
    axes[0].set_axisbelow(True)

    # Mark top-5 |derivative| peaks (candidate anomalies)
    top_k = 5
    peak_idx = np.argsort(abs_d)[-top_k:]
    for pi in peak_idx:
        axes[0].axvline(t[pi], color='orange', linestyle='--', linewidth=0.7, alpha=0.7)
        axes[0].text(t[pi], dmean[pi], f't={t[pi]:.3f}', fontsize=7,
                     color='orange', ha='left', va='bottom')

    axes[1].plot(t, abs_d, color='tab:purple', linewidth=0.9)
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('|d(MSE)/dt|')
    axes[1].set_title(f'|derivative| — top {top_k} peaks marked (candidate anomalies)')
    axes[1].grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_histograms(
    t: np.ndarray,
    mse: np.ndarray,
    output_base: Path,
    t_samples: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> None:
    """MSE distribution across volumes at a few representative t values."""
    fig, axes = plt.subplots(1, len(t_samples), figsize=(3.2 * len(t_samples), 4),
                             sharey=True, squeeze=False)
    for ax, ts in zip(axes[0], t_samples):
        idx = int(np.argmin(np.abs(t - ts)))
        vals = mse[idx]
        ax.hist(vals, bins=15, color='tab:blue', alpha=0.75,
                edgecolor='black', linewidth=0.4)
        ax.axvline(vals.mean(), color='red', linestyle='--', linewidth=1,
                   label=f'mean={vals.mean():.3f}')
        ax.set_title(f't = {t[idx]:.3f}', fontsize=10)
        ax.set_xlabel('MSE')
        ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=8)
    axes[0, 0].set_ylabel('n volumes')
    plt.suptitle('MSE distribution across volumes at selected t', fontsize=11)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Timestep-response diagnostic")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', default='runs/eval/timestep_response')
    parser.add_argument('--seg-split', default='test1')
    parser.add_argument('--num-volumes', type=int, default=10)
    parser.add_argument('--num-t', type=int, default=1000,
                        help='Number of t values in [t_min, t_max]')
    parser.add_argument('--t-min', type=float, default=0.001,
                        help='Avoid exactly 0 (trivial) and exactly 1 (no information)')
    parser.add_argument('--t-max', type=float, default=0.999)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--thesis-dir',
                        default='/home/mode/NTNU/AIS4900_doc/AIS4900-master-thesis/Images/timestep_response')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    logger.info(f"Loading model: {args.checkpoint}")
    model = load_diffusion_model(args.checkpoint, device=device,
                                 in_channels=2, out_channels=1,
                                 spatial_dims=3).eval()
    strategy = RFlowStrategy()
    strategy.setup_scheduler(num_timesteps=1000, image_size=256,
                             depth_size=args.depth, spatial_dims=3)
    T = strategy.scheduler.num_train_timesteps
    logger.info(f"  num_train_timesteps = {T}")

    patients = discover_patients(Path(args.data_root), args.seg_split,
                                 args.num_volumes)
    if len(patients) < args.num_volumes:
        raise SystemExit(f"Only {len(patients)} patients found, need {args.num_volumes}")

    logger.info("Loading volumes and seg masks")
    real_volumes: list[torch.Tensor] = []
    seg_tensors: list[torch.Tensor] = []
    for name, bp, sp in patients:
        x0 = load_volume(bp, args.depth)
        seg = load_seg(sp, args.depth)
        real_volumes.append(torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).to(device))
        seg_tensors.append(torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).to(device))
        logger.info(f"  {name}")

    t_values = np.linspace(args.t_min, args.t_max, args.num_t)
    logger.info(f"Sweeping {args.num_t} t values in [{args.t_min}, {args.t_max}]")

    mse = sweep_timestep_mse(
        model, real_volumes, seg_tensors,
        t_values=t_values, T=T, device=device, log_every=max(1, args.num_t // 20),
    )

    # Plots
    logger.info("Plotting")
    plot_mean_curve(t_values, mse, output_dir / 'mse_mean')
    plot_per_volume_overlay(t_values, mse, output_dir / 'mse_per_volume')
    plot_derivative(t_values, mse, output_dir / 'mse_derivative')
    plot_histograms(t_values, mse, output_dir / 'mse_histograms')

    # Summary numbers
    dmean = np.gradient(mse.mean(axis=1), t_values)
    abs_d = np.abs(dmean)
    peak_idx = np.argsort(abs_d)[-10:][::-1]
    print()
    print("=" * 80)
    print("Timestep response summary")
    print("=" * 80)
    print(f"  t range:        [{t_values.min():.4f}, {t_values.max():.4f}]")
    print(f"  MSE mean range: [{mse.mean(axis=1).min():.4f}, {mse.mean(axis=1).max():.4f}]")
    print(f"  MSE std range:  [{mse.std(axis=1).min():.4f}, {mse.std(axis=1).max():.4f}]")
    print("  Top 10 |derivative| peaks (candidate anomalies):")
    for pi in peak_idx:
        print(f"    t = {t_values[pi]:.4f}  "
              f"MSE = {mse[pi].mean():.4f}  "
              f"|dMSE/dt| = {abs_d[pi]:.3f}")
    print("=" * 80)

    with open(output_dir / 'timestep_response_results.json', 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'data_root': args.data_root,
            'seg_split': args.seg_split,
            'num_volumes': args.num_volumes,
            'num_t': args.num_t,
            't_min': args.t_min,
            't_max': args.t_max,
            't_values': t_values.tolist(),
            'mse_matrix': mse.tolist(),
            'mse_mean': mse.mean(axis=1).tolist(),
            'mse_std': mse.std(axis=1).tolist(),
            'top_abs_derivative_peaks': [
                {'t': float(t_values[pi]), 'mse_mean': float(mse[pi].mean()),
                 'abs_dmse_dt': float(abs_d[pi])}
                for pi in peak_idx
            ],
        }, f, indent=2)
    logger.info(f"Saved: {output_dir / 'timestep_response_results.json'}")

    if args.thesis_dir:
        try:
            thesis_dir = Path(args.thesis_dir)
            thesis_dir.mkdir(parents=True, exist_ok=True)
            for name in ('mse_mean', 'mse_per_volume', 'mse_derivative', 'mse_histograms'):
                for ext in ('png', 'pdf'):
                    src = output_dir / f'{name}.{ext}'
                    if src.exists():
                        (thesis_dir / f'{name}.{ext}').write_bytes(src.read_bytes())
            logger.info(f"Copied figures to thesis dir: {thesis_dir}")
        except OSError as e:
            logger.warning(f"Could not copy to thesis dir {args.thesis_dir}: {e}")


if __name__ == '__main__':
    main()
