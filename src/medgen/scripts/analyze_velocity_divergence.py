#!/usr/bin/env python3
"""Per-t velocity divergence map (Phase 1 #4).

For each fine-tuned checkpoint, measure how much its velocity field diverges
from the baseline (exp1_1_1000) at each t. This reveals *where* each fine-tune
actually moved the model — which may or may not match its intended t-schedule.

For fixed real x_0 volumes and a fixed noise tensor per seed:
  1. Build x_t = add_noise(x_0, noise, t) for a grid of t values.
  2. Run the baseline model: v_base = baseline(x_t, t).
  3. Run each fine-tune: v_ft = fine_tune(x_t, t).
  4. Compute:
       - rel_l2(t, ft) = ||v_ft - v_base||_2 / (||v_base||_2 + eps)
       - cosine(t, ft) = <v_ft, v_base> / (||v_ft|| ||v_base||)
  5. Plot divergence vs t per fine-tune. High divergence = fine-tune changed
     velocity there; low = fine-tune kept baseline behavior.

Interprets against intended t-schedules:
  - exp32 family expected: high divergence at low t (aux schedule active there)
  - exp37 family expected: high divergence at mid t (aux schedule active there)
  - exp37_3 expected: highest divergence among all (heavier aux weights)

Usage:
    python -m medgen.scripts.analyze_velocity_divergence \\
        --baseline /path/to/exp1_1_1000/checkpoint_latest.pt \\
        --fine-tunes label1=/path/to/ft1.pt label2=/path/to/ft2.pt \\
        --data-root /path/to/brainmetshare-3 \\
        --output-dir runs/eval/velocity_divergence
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
from torch.amp import autocast

from medgen.diffusion import RFlowStrategy, load_diffusion_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# T-grid (normalized). More dense at the ends where schedules switch.
DEFAULT_T_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.95]

# Consistent palette with other Phase-1 scripts.
MODEL_COLORS: dict[str, str] = {
    'exp1_1_1000':  'gray',
    'exp32_1_1000': 'tab:blue',
    'exp32_2_1000': 'tab:cyan',
    'exp32_3_1000': 'tab:purple',
    'exp37_1':      'tab:orange',
    'exp37_2':      'tab:red',
    'exp37_3':      'tab:brown',
}


def load_volume(path: Path, depth: int) -> np.ndarray:
    """Load NIfTI -> [D, H, W] in [0, 1]."""
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
                      ) -> list[tuple[Path, Path]]:
    d = data_root / split
    if not d.exists():
        d = data_root / 'test'
    patients = []
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        b = sub / 'bravo.nii.gz'
        s = sub / 'seg.nii.gz'
        if b.exists() and s.exists():
            patients.append((b, s))
            if len(patients) >= num:
                break
    return patients


def compute_velocity(model, strategy, x_t: torch.Tensor, seg: torch.Tensor,
                     t_int: int, device: torch.device) -> torch.Tensor:
    """Single forward pass: returns velocity prediction as float32 CPU tensor."""
    t_tensor = torch.tensor([t_int], device=device, dtype=torch.long)
    model_input = torch.cat([x_t, seg], dim=1)
    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        v = model(model_input, t_tensor)
    return v.float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.4 — velocity divergence map")
    parser.add_argument('--baseline', required=True,
                        help='Baseline checkpoint path (typically exp1_1_1000)')
    parser.add_argument('--fine-tunes', nargs='+', required=True,
                        help='Fine-tune checkpoints in label=path format')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-volumes', type=int, default=5)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--seg-split', default='test1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t-grid', nargs='+', type=float, default=DEFAULT_T_GRID)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')

    # Parse fine-tune args
    ft_specs: list[tuple[str, str]] = []
    for spec in args.fine_tunes:
        if '=' not in spec:
            raise SystemExit(f"--fine-tunes expects label=path, got {spec!r}")
        label, path = spec.split('=', 1)
        ft_specs.append((label, path))

    # Load data volumes (cache on CPU)
    data_root = Path(args.data_root)
    patients = discover_patients(data_root, args.seg_split, args.num_volumes)
    if len(patients) < args.num_volumes:
        raise SystemExit(f"Only found {len(patients)} patients, need {args.num_volumes}")
    logger.info(f"Using {len(patients)} patients from {args.seg_split}")

    cached: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for bp, sp in patients:
        x = torch.from_numpy(load_volume(bp, args.depth)).unsqueeze(0).unsqueeze(0)
        s = torch.from_numpy(load_seg(sp, args.depth)).unsqueeze(0).unsqueeze(0)
        cached.append((bp.parent.name, x, s))

    # Setup strategy (same for all models)
    strategy = RFlowStrategy()
    strategy.setup_scheduler(num_timesteps=1000, image_size=256,
                             depth_size=args.depth, spatial_dims=3)
    T = strategy.scheduler.num_train_timesteps

    # Precompute noisy inputs per (volume, t) — fixed noise per seed
    # This ensures baseline + all fine-tunes see identical inputs.
    logger.info("Precomputing x_t tensors on CPU (keeps baseline/ft inputs identical)")
    noisy_inputs: dict[tuple[int, float], tuple[torch.Tensor, torch.Tensor]] = {}
    for vol_idx, (_pid, x, s) in enumerate(cached):
        for t_norm in args.t_grid:
            torch.manual_seed(args.seed + vol_idx * 10000 + int(t_norm * 1000))
            noise = torch.randn_like(x)
            t_int = max(1, min(T - 1, round(t_norm * T)))
            t_tensor = torch.tensor([t_int], dtype=torch.long)
            # Add noise in CPU (same operation on all models)
            x_t = strategy.scheduler.add_noise(x, noise, t_tensor)
            noisy_inputs[(vol_idx, t_norm)] = (x_t, s)

    # -------- Baseline velocities --------
    logger.info(f"Loading baseline: {args.baseline}")
    baseline = load_diffusion_model(args.baseline, device=device,
                                    compile_model=False, spatial_dims=3)
    baseline.eval()

    baseline_vels: dict[tuple[int, float], torch.Tensor] = {}
    for vol_idx, _ in enumerate(cached):
        for t_norm in args.t_grid:
            x_t, s = noisy_inputs[(vol_idx, t_norm)]
            x_t_gpu = x_t.to(device)
            s_gpu = s.to(device)
            t_int = max(1, min(T - 1, round(t_norm * T)))
            v = compute_velocity(baseline, strategy, x_t_gpu, s_gpu, t_int, device)
            baseline_vels[(vol_idx, t_norm)] = v  # CPU float32
            del x_t_gpu, s_gpu
        torch.cuda.empty_cache()
        logger.info(f"  baseline vol {vol_idx + 1}/{len(cached)}")

    # Free baseline from GPU before loading fine-tunes
    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    # -------- Fine-tune velocities + divergence --------
    eps = 1e-8
    results: dict[str, dict] = {}
    for label, path in ft_specs:
        logger.info(f"Loading fine-tune '{label}': {path}")
        ft = load_diffusion_model(path, device=device, compile_model=False, spatial_dims=3)
        ft.eval()

        per_vol_t: dict[tuple[int, float], dict] = {}
        for vol_idx, _ in enumerate(cached):
            for t_norm in args.t_grid:
                x_t, s = noisy_inputs[(vol_idx, t_norm)]
                x_t_gpu = x_t.to(device)
                s_gpu = s.to(device)
                t_int = max(1, min(T - 1, round(t_norm * T)))
                v_ft = compute_velocity(ft, strategy, x_t_gpu, s_gpu, t_int, device)
                v_base = baseline_vels[(vol_idx, t_norm)]

                diff = (v_ft - v_base).flatten()
                vb = v_base.flatten()
                vf = v_ft.flatten()

                rel_l2 = (diff.norm() / (vb.norm() + eps)).item()
                cos = torch.nn.functional.cosine_similarity(
                    vf.unsqueeze(0), vb.unsqueeze(0), dim=1
                ).item()

                per_vol_t[(vol_idx, t_norm)] = {
                    'rel_l2': rel_l2,
                    'cos': cos,
                    'v_base_norm': float(vb.norm().item()),
                    'v_ft_norm': float(vf.norm().item()),
                }

                del x_t_gpu, s_gpu, v_ft
            torch.cuda.empty_cache()
            logger.info(f"  [{label}] vol {vol_idx + 1}/{len(cached)}")

        # Aggregate per t (mean, std across volumes)
        agg = {}
        for t_norm in args.t_grid:
            rows = [per_vol_t[(v, t_norm)] for v in range(len(cached))]
            agg[t_norm] = {
                'rel_l2_mean': float(np.mean([r['rel_l2'] for r in rows])),
                'rel_l2_std':  float(np.std([r['rel_l2'] for r in rows])),
                'cos_mean':    float(np.mean([r['cos'] for r in rows])),
                'cos_std':     float(np.std([r['cos'] for r in rows])),
            }
        results[label] = {'per_vol_t': per_vol_t, 'agg': agg}

        del ft
        gc.collect()
        torch.cuda.empty_cache()

    # -------- Plots --------
    t_arr = np.array(args.t_grid)

    # Figure 1 — relative L2 divergence vs t
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, d in results.items():
        means = np.array([d['agg'][t]['rel_l2_mean'] for t in args.t_grid])
        stds  = np.array([d['agg'][t]['rel_l2_std'] for t in args.t_grid])
        color = MODEL_COLORS.get(label)
        ax.errorbar(t_arr, means, yerr=stds, marker='o', linewidth=1.8,
                    capsize=3, label=label, color=color, alpha=0.85)
    ax.set_xlabel('t (normalized noise level) — low t = near clean')
    ax.set_ylabel('||v_finetune - v_baseline||₂ / ||v_baseline||₂')
    ax.set_title('Velocity divergence vs baseline, per t')
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=9, ncol=2)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_dir / f'velocity_divergence_rel_l2.{ext}',
                    dpi=180, bbox_inches='tight')
    plt.close()
    logger.info("  saved velocity_divergence_rel_l2.png + .pdf")

    # Figure 2 — cosine similarity vs t (higher = more similar to baseline)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, d in results.items():
        means = np.array([d['agg'][t]['cos_mean'] for t in args.t_grid])
        stds  = np.array([d['agg'][t]['cos_std'] for t in args.t_grid])
        color = MODEL_COLORS.get(label)
        ax.errorbar(t_arr, means, yerr=stds, marker='o', linewidth=1.8,
                    capsize=3, label=label, color=color, alpha=0.85)
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5,
               label='identical to baseline')
    ax.set_xlabel('t (normalized noise level)')
    ax.set_ylabel('cosine similarity to baseline velocity')
    ax.set_title('Velocity cosine similarity vs baseline, per t\n(1.0 = same direction; lower = more different)')
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=9, ncol=2)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_dir / f'velocity_cosine.{ext}',
                    dpi=180, bbox_inches='tight')
    plt.close()
    logger.info("  saved velocity_cosine.png + .pdf")

    # Figure 3 — per-model heatmap of rel_l2 × volume (shows consistency across data)
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(ft_specs)), 5.5))
    heatmap = np.zeros((len(ft_specs), len(args.t_grid)))
    for i, (label, _) in enumerate(ft_specs):
        for j, t_norm in enumerate(args.t_grid):
            heatmap[i, j] = results[label]['agg'][t_norm]['rel_l2_mean']
    im = ax.imshow(heatmap, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(np.arange(len(args.t_grid)))
    ax.set_xticklabels([f'{t:.2f}' for t in args.t_grid], rotation=0)
    ax.set_yticks(np.arange(len(ft_specs)))
    ax.set_yticklabels([label for label, _ in ft_specs])
    ax.set_xlabel('t (normalized)')
    ax.set_ylabel('fine-tune')
    ax.set_title('Mean relative L2 divergence — hot regions = where each fine-tune changed')
    plt.colorbar(im, ax=ax, label='rel_l2')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_dir / f'velocity_divergence_heatmap.{ext}',
                    dpi=180, bbox_inches='tight')
    plt.close()
    logger.info("  saved velocity_divergence_heatmap.png + .pdf")

    # -------- JSON + summary --------
    summary = {
        'baseline': args.baseline,
        'num_volumes': len(cached),
        'seed': args.seed,
        't_grid': args.t_grid,
        'models': {
            label: {
                'checkpoint': path,
                'agg': {str(t): v for t, v in results[label]['agg'].items()},
            } for label, path in ft_specs
        },
    }
    with open(output_dir / 'velocity_divergence_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved: {output_dir / 'velocity_divergence_results.json'}")

    # Text table — max divergence per model, t at which max occurs
    print()
    print("=" * 76)
    print("Velocity divergence summary")
    print("=" * 76)
    print(f"{'model':<16} {'max_rel_l2':>12} {'t at max':>10} {'min_cosine':>12} {'t at min':>10}")
    print("-" * 76)
    for label, _ in ft_specs:
        agg = results[label]['agg']
        rel_l2_vals = [(t, agg[t]['rel_l2_mean']) for t in args.t_grid]
        cos_vals = [(t, agg[t]['cos_mean']) for t in args.t_grid]
        t_max, max_l2 = max(rel_l2_vals, key=lambda x: x[1])
        t_min_cos, min_cos = min(cos_vals, key=lambda x: x[1])
        print(f"{label:<16} {max_l2:>12.4f} {t_max:>10.3f} {min_cos:>12.4f} {t_min_cos:>10.3f}")
    print("=" * 76)


if __name__ == '__main__':
    main()
