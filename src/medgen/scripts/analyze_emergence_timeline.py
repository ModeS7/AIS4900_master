#!/usr/bin/env python3
"""Feature emergence timeline — post-process trajectory emergence JSONs.

For each model's `emergence_metrics.json` (produced by
analyze_generation_trajectory.py), compute per-metric emergence time:
the largest t (earliest in generation, since t goes 1→0) at which the
metric first falls inside ±tolerance of its near-final value and stays
there for the remainder of the trajectory.

  structure is built here ────▶ texture here ────▶ fine detail here

Produces:
  Figure 1: Horizontal timeline — one row per metric, one bar per model
            spanning [emergence_t, 0.0]. The "locked-in" region.
  Figure 2: Normalized metric curves (metric / final_value) overlaid
            per model — visually obvious when each metric saturates.
  Figure 3: Emergence-t heatmap — rows=metric, cols=model.
  JSON:     per-metric per-model emergence_t, final_value, tolerance.

Run on one or more trajectory_emergence dirs:

    python -m medgen.scripts.analyze_emergence_timeline \\
        --input-dirs runs/eval/trajectory_emergence_exp32_1_1000_* \\
                      runs/eval/trajectory_emergence_exp1_1_1000_* \\
        --output-dir runs/eval/emergence_timeline
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Metrics to include in the timeline. Excludes `brain_coherence` because it
# equals 1.0 throughout (the threshold-connected-component metric degenerates
# at pure noise — every positive-noise voxel is part of one big blob, so the
# metric artifactually reports "coherent" at t=1).
METRICS = [
    'brain_frac',       # when the brain shape is established
    'band_mid',         # when mid-frequency detail locks in
    'hf_energy',        # overall HF energy
    'band_very_high',   # when the finest frequencies emerge
    'vessel_score',     # when vessel-like tubes stabilize
]

METRIC_LABELS = {
    'brain_frac':      'brain fraction',
    'band_mid':        'mid-band energy',
    'hf_energy':       'HF energy (all bands ≥ 0.3)',
    'band_very_high':  'very-high band energy',
    'vessel_score':    'Frangi (vessel-like)',
}


MODEL_COLORS: dict[str, str] = {
    'exp1_1_1000':  'gray',
    'exp32_1_1000': 'tab:blue',
    'exp32_2_1000': 'tab:cyan',
    'exp32_3_1000': 'tab:purple',
    'exp37_1':      'tab:orange',
    'exp37_2':      'tab:red',
    'exp37_3':      'tab:brown',
}


def infer_model_label(input_dir: Path) -> str:
    """Extract model label from directory name.

    Handles both:
      trajectory_emergence_exp1_1_1000_20260421-022058
      trajectory_emergence_exp37_3_20260421-041315
    """
    name = input_dir.name
    if name.startswith('trajectory_emergence_'):
        remainder = name[len('trajectory_emergence_'):]
        # Strip the timestamp suffix (last two underscore-separated tokens, YYYYMMDD-HHMMSS)
        parts = remainder.split('_')
        if len(parts) > 1 and '-' in parts[-1] and parts[-1].replace('-', '').isdigit():
            parts = parts[:-1]
        return '_'.join(parts)
    return name


def load_trajectory(input_dir: Path) -> dict:
    """Load emergence_metrics.json and aggregate metrics over seeds.

    Returns:
        {
          't': np.ndarray[n_steps],  # shared across seeds
          'metrics': {metric: np.ndarray[n_steps]},  # mean over seeds, NaN-safe
        }
    """
    path = input_dir / 'emergence_metrics.json'
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        d = json.load(f)

    seeds = d['seed_trajectories']  # list[seed] of list[step_dict]
    if not seeds:
        raise ValueError(f"No seeds in {path}")

    # Assume all seeds share the same t grid (they do — generated identically)
    n_steps = len(seeds[0])
    t = np.array([seeds[0][i]['t_norm'] for i in range(n_steps)])

    metrics_mean: dict[str, np.ndarray] = {}
    for m in METRICS:
        stacked = np.full((len(seeds), n_steps), np.nan, dtype=np.float64)
        for si, seed_traj in enumerate(seeds):
            for ti, step in enumerate(seed_traj):
                v = step.get(m)
                if v is not None and np.isfinite(v):
                    stacked[si, ti] = v
        metrics_mean[m] = np.nanmean(stacked, axis=0)

    return {'t': t, 'metrics': metrics_mean, 'meta': {
        'n_seeds': len(seeds),
        'n_steps': n_steps,
    }}


def compute_emergence_t(
    t: np.ndarray,
    values: np.ndarray,
    tolerance: float = 0.10,
    final_window: int = 1,
) -> tuple[float, float]:
    """Find the largest t where the metric enters ±tolerance band around
    its final value and stays there for the rest of the trajectory.

    Walks from t=0 (end of trajectory) backwards; flags the first index
    at which the metric leaves the band. Returns t just *after* (i.e.
    smaller t than) that flagged point.

    Defaults: tolerance=10%, final_window=1 (use the last value itself
    as "final"). With a larger final_window you average end-of-trajectory
    values, which can push the last sample itself *outside* a tight band
    for still-growing metrics — hence window=1 is the cleanest choice.

    Args:
        t: [n_steps], decreasing from ≈1 to ≈0
        values: [n_steps], same orientation
        tolerance: relative tolerance around final value (±tolerance · |final|)
        final_window: number of end steps to average for "final value"

    Returns:
        (emergence_t, final_value). emergence_t is NaN if the metric
        never stabilizes (e.g. all-NaN values).
    """
    # Filter out NaN (e.g. vessel_score is computed only every N steps)
    valid = np.isfinite(values)
    if valid.sum() < final_window + 1:
        return float('nan'), float('nan')

    t_v = t[valid]
    v_v = values[valid]
    # Sort by decreasing t (trajectory order)
    order = np.argsort(-t_v)
    t_v = t_v[order]
    v_v = v_v[order]

    final_value = float(np.mean(v_v[-final_window:]))
    if abs(final_value) < 1e-12:
        # Avoid divide-by-zero on degenerate metrics
        return float('nan'), final_value
    tol_abs = tolerance * abs(final_value)
    in_band = np.abs(v_v - final_value) <= tol_abs

    # Walk from end (t→0) backwards, find last contiguous in-band region
    emergence_idx = len(v_v)  # default: never
    for i in range(len(v_v) - 1, -1, -1):
        if in_band[i]:
            emergence_idx = i
        else:
            break
    if emergence_idx >= len(v_v):
        return float('nan'), final_value
    return float(t_v[emergence_idx]), final_value


# ────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────
def plot_timeline(
    emergence: dict[str, dict[str, float]],
    output_base: Path,
) -> None:
    """Horizontal bars: one row per metric, one bar per model showing
    [emergence_t, 0] — the "locked-in" region of the trajectory.
    """
    models = list(emergence.keys())
    n_models = len(models)
    metric_order = list(METRICS)

    fig, ax = plt.subplots(figsize=(12, 0.55 * len(metric_order) * n_models + 2.5))
    y = 0
    yticks = []
    yticklabels = []

    for m in metric_order:
        for mi, model in enumerate(models):
            info = emergence[model].get(m, {})
            t_em = info.get('emergence_t', float('nan'))
            if np.isnan(t_em):
                y += 1
                continue
            color = MODEL_COLORS.get(model, f'C{mi}')
            # Bar from [0, t_em], representing the locked-in span from t_em down to 0
            ax.barh(y, t_em, left=0, color=color, alpha=0.85,
                    edgecolor='black', linewidth=0.4,
                    label=model if m == metric_order[0] else None)
            ax.text(t_em + 0.01, y, f't={t_em:.3f}', fontsize=7, va='center')
            yticks.append(y)
            yticklabels.append(f'{METRIC_LABELS[m]}  ({model})')
            y += 1
        y += 0.5  # gap between metric groups

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('t (0 = clean end, 1 = noise start)')
    ax.set_title('Feature emergence timeline: bar = "metric within ±5% of final from this t onward"')
    ax.grid(True, axis='x', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    # Legend only shows once per model
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='lower right', fontsize=8)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_normalized_curves(
    trajectories: dict[str, dict],
    output_base: Path,
) -> None:
    """Per-metric subplot: metric / final_value vs t, one line per model.

    Makes saturation points visually obvious (they flatten at 1.0).
    """
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(11, 2.4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, m in zip(axes, METRICS):
        for model, traj in trajectories.items():
            t = traj['t']
            v = traj['metrics'][m]
            valid = np.isfinite(v)
            if not valid.any():
                continue
            final_vals = v[valid][-3:]  # last 3 valid values
            final = float(np.mean(final_vals)) if final_vals.size else 1.0
            if abs(final) < 1e-12:
                continue
            norm = v / final
            color = MODEL_COLORS.get(model, 'gray')
            ax.plot(t[valid], norm[valid], color=color, linewidth=1.3,
                    alpha=0.9, label=model)

        ax.axhline(1.0, color='black', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.axhspan(0.95, 1.05, color='black', alpha=0.08)  # ±5% tolerance band
        ax.set_ylabel(f'{METRIC_LABELS[m]}\n(fraction of final)')
        ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
        ax.set_axisbelow(True)
        ax.invert_xaxis()  # t decreases left→right = time progression

    axes[-1].set_xlabel('t (left = noise, right = clean)')
    axes[0].legend(loc='upper left', fontsize=8, ncol=2)
    plt.suptitle('Per-metric normalized curves (value / final)', fontsize=11)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_heatmap(
    emergence: dict[str, dict[str, float]],
    output_base: Path,
) -> None:
    """Rows=metric, cols=model — colored by emergence_t."""
    models = list(emergence.keys())
    metrics = list(METRICS)
    grid = np.full((len(metrics), len(models)), np.nan)
    for j, model in enumerate(models):
        for i, m in enumerate(metrics):
            info = emergence[model].get(m, {})
            grid[i, j] = info.get('emergence_t', np.nan)

    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * len(models), 0.6 * len(metrics) + 1.5))
    im = ax.imshow(grid, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([METRIC_LABELS[m] for m in metrics], fontsize=9)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=9)

    for i in range(len(metrics)):
        for j in range(len(models)):
            v = grid[i, j]
            txt = 'n/a' if np.isnan(v) else f'{v:.3f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='white' if np.isnan(v) or v < 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='emergence t (earlier = larger)')
    ax.set_title('Emergence t per metric × model (brighter = later in trajectory)')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def print_table(emergence: dict[str, dict[str, float]]) -> None:
    """Human-readable summary table."""
    models = list(emergence.keys())
    print()
    print("=" * (22 + 16 * len(models)))
    print("Feature emergence times (larger t = earlier lock-in)")
    print("=" * (22 + 16 * len(models)))
    header = f"{'metric':<22}" + "".join(f"{m:>16}" for m in models)
    print(header)
    print("-" * len(header))
    for m in METRICS:
        row = f"{METRIC_LABELS[m]:<22}"
        for model in models:
            info = emergence[model].get(m, {})
            te = info.get('emergence_t', float('nan'))
            if np.isnan(te):
                row += f"{'n/a':>16}"
            else:
                row += f"{te:>16.3f}"
        print(row)
    print("=" * (22 + 16 * len(models)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature emergence timeline across one or more models",
    )
    parser.add_argument('--input-dirs', nargs='+', required=True,
                        help='trajectory_emergence_* directories')
    parser.add_argument('--output-dir', default='runs/eval/emergence_timeline')
    parser.add_argument('--tolerance', type=float, default=0.10,
                        help='Relative ±tolerance around final value (default 0.10)')
    parser.add_argument('--final-window', type=int, default=1,
                        help='Number of end steps averaged for "final value" (default 1)')
    parser.add_argument('--thesis-dir',
                        default='/home/mode/NTNU/AIS4900_doc/AIS4900-master-thesis/Images/emergence_timeline')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories: dict[str, dict] = {}
    emergence: dict[str, dict[str, dict[str, float]]] = {}

    for raw in args.input_dirs:
        d = Path(raw)
        if not d.is_dir():
            logger.warning(f"Not a directory, skipping: {d}")
            continue
        try:
            traj = load_trajectory(d)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load {d}: {e}")
            continue
        label = infer_model_label(d)
        trajectories[label] = traj
        logger.info(f"loaded {label}: {traj['meta']['n_seeds']} seeds × "
                    f"{traj['meta']['n_steps']} steps")

        em: dict[str, dict[str, float]] = {}
        for m in METRICS:
            te, final = compute_emergence_t(
                traj['t'], traj['metrics'][m],
                tolerance=args.tolerance, final_window=args.final_window,
            )
            em[m] = {'emergence_t': te, 'final_value': final}
        emergence[label] = em

    if not trajectories:
        raise SystemExit("No valid trajectory directories found.")

    logger.info("Plotting")
    plot_timeline(emergence, output_dir / 'timeline_bars')
    plot_normalized_curves(trajectories, output_dir / 'normalized_curves')
    plot_heatmap(emergence, output_dir / 'emergence_heatmap')

    print_table(emergence)

    with open(output_dir / 'emergence_timeline.json', 'w') as f:
        json.dump({
            'input_dirs': [str(Path(d)) for d in args.input_dirs],
            'tolerance': args.tolerance,
            'final_window': args.final_window,
            'metrics': METRICS,
            'emergence': {
                model: {m: em[m] for m in METRICS}
                for model, em in emergence.items()
            },
        }, f, indent=2)
    logger.info(f"Saved: {output_dir / 'emergence_timeline.json'}")

    if args.thesis_dir:
        try:
            thesis_dir = Path(args.thesis_dir)
            thesis_dir.mkdir(parents=True, exist_ok=True)
            for name in ('timeline_bars', 'normalized_curves', 'emergence_heatmap'):
                for ext in ('png', 'pdf'):
                    src = output_dir / f'{name}.{ext}'
                    if src.exists():
                        (thesis_dir / f'{name}.{ext}').write_bytes(src.read_bytes())
            logger.info(f"Copied figures to thesis dir: {thesis_dir}")
        except OSError as e:
            logger.warning(f"Could not copy to thesis dir {args.thesis_dir}: {e}")


if __name__ == '__main__':
    main()
