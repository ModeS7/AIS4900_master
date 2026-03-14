#!/usr/bin/env python3
"""Extract generation metric trajectories from TensorBoard runs for 3D bravo experiments."""

import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("/home/mode/NTNU/AIS4900_master/runs_tb/diffusion_3d/bravo")

METRICS = [
    "Generation/KID_mean_val",
    "Generation/CMMD_val",
    "Generation/extended_KID_mean_val",
    "Generation/extended_CMMD_val",
    "Generation_3d/KID_mean_val",
    "Generation_3d/CMMD_val",
    "Generation_Diversity/extended_LPIPS",
    "Generation_Diversity/extended_MSSSIM",
    "test_best_generation/FID_bravo",
]

MIN_POINTS = 50
ROLLING_WINDOW = 20
LAST_N = 100


def classify_trend(pct_change: float) -> str:
    if pct_change < -10:
        return "strong improve"
    elif pct_change < -3:
        return "improving"
    elif pct_change <= 3:
        return "flat"
    elif pct_change <= 10:
        return "worsening"
    else:
        return "degrading"


def extract_metrics(logdir: str) -> dict[str, list[tuple[int, float]]]:
    """Extract scalar metrics from a TensorBoard logdir."""
    ea = EventAccumulator(logdir)
    ea.Reload()
    available_tags = ea.Tags().get("scalars", [])

    results = {}
    for metric in METRICS:
        if metric in available_tags:
            events = ea.Scalars(metric)
            # Store as (step, value) pairs
            results[metric] = [(e.step, e.value) for e in events]
    return results


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # Use 'valid' mode then pad front with NaN
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, smoothed])


def compute_slope_pct(smoothed: np.ndarray, last_n: int) -> tuple[float, float]:
    """Compute linear regression slope over last_n smoothed values.

    Returns (slope_pct_per_100, mean_last_n).
    slope_pct_per_100 = percentage change per 100 epochs relative to mean.
    """
    tail = smoothed[~np.isnan(smoothed)]
    if len(tail) < 10:
        return float("nan"), float("nan")
    tail = tail[-last_n:]
    x = np.arange(len(tail))
    # Linear regression
    coeffs = np.polyfit(x, tail, 1)
    slope = coeffs[0]  # units per epoch
    mean_val = np.mean(tail)
    if abs(mean_val) < 1e-12:
        return float("nan"), mean_val
    # % change over 100 epochs relative to mean
    # slope has units of [metric_units / epoch_index]
    # slope * 100 = absolute change over 100 data points
    # / mean_val * 100 = convert to percentage
    pct = (slope * 100) / abs(mean_val) * 100
    return pct, mean_val


def main():
    exp_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(exp_dirs)} experiment directories\n")

    # Collect all results
    all_results = []  # list of (exp_name, metric_name, n_points, mean_last100, slope_pct, trend)

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        # TFEvents are in tensorboard/ subdirectory
        tb_dir = exp_dir / "tensorboard"
        logdir = str(tb_dir) if tb_dir.is_dir() else str(exp_dir)

        try:
            metrics = extract_metrics(logdir)
        except Exception as e:
            print(f"  WARN: {exp_name}: {e}")
            continue

        if not metrics:
            continue

        for metric_name, data in metrics.items():
            n_points = len(data)
            if n_points < MIN_POINTS:
                continue

            steps, values = zip(*data)
            values = np.array(values, dtype=float)

            smoothed = smooth(values, ROLLING_WINDOW)
            slope_pct, mean_val = compute_slope_pct(smoothed, LAST_N)
            trend = classify_trend(slope_pct) if not np.isnan(slope_pct) else "N/A"

            all_results.append((exp_name, metric_name, n_points, mean_val, slope_pct, trend, int(steps[-1])))

    if not all_results:
        print("No metrics with sufficient data points found.")
        return

    # Print detailed per-metric tables
    metric_groups = defaultdict(list)
    for row in all_results:
        metric_groups[row[1]].append(row)

    for metric_name in METRICS:
        rows = metric_groups.get(metric_name, [])
        if not rows:
            continue

        # Sort by slope (ascending = most improving first)
        rows.sort(key=lambda r: r[4] if not np.isnan(r[4]) else 999)

        print(f"\n{'=' * 120}")
        print(f"  {metric_name}  ({len(rows)} experiments with >{MIN_POINTS} points)")
        print(f"{'=' * 120}")
        print(f"  {'Experiment':<60} {'Points':>6} {'Last Step':>9} {'Mean(last100)':>14} {'Slope%/100ep':>13} {'Trend':>15}")
        print(f"  {'-'*60} {'-'*6} {'-'*9} {'-'*14} {'-'*13} {'-'*15}")

        for exp_name, _, n_points, mean_val, slope_pct, trend, last_step in rows:
            slope_str = f"{slope_pct:+.2f}%" if not np.isnan(slope_pct) else "N/A"
            mean_str = f"{mean_val:.6f}" if not np.isnan(mean_val) else "N/A"
            print(f"  {exp_name:<60} {n_points:>6} {last_step:>9} {mean_str:>14} {slope_str:>13} {trend:>15}")

    # Summary: best experiment per metric
    print(f"\n\n{'=' * 120}")
    print(f"  BEST EXPERIMENT PER METRIC (lowest mean over last 100 epochs)")
    print(f"{'=' * 120}")
    print(f"  {'Metric':<40} {'Best Experiment':<50} {'Mean':>14} {'Trend':>15}")
    print(f"  {'-'*40} {'-'*50} {'-'*14} {'-'*15}")

    for metric_name in METRICS:
        rows = metric_groups.get(metric_name, [])
        if not rows:
            continue
        # For FID/KID/CMMD lower is better; for Diversity higher is better
        if "Diversity" in metric_name:
            best = max(rows, key=lambda r: r[3] if not np.isnan(r[3]) else -999)
        else:
            best = min(rows, key=lambda r: r[3] if not np.isnan(r[3]) else 999)

        exp_name, _, _, mean_val, _, trend, _ = best
        mean_str = f"{mean_val:.6f}" if not np.isnan(mean_val) else "N/A"
        print(f"  {metric_name:<40} {exp_name:<50} {mean_str:>14} {trend:>15}")

    # Cross-metric summary for KID: sort all experiments by KID slope
    print(f"\n\n{'=' * 120}")
    print(f"  ALL EXPERIMENTS SORTED BY Generation/KID_mean_val TRAJECTORY SLOPE")
    print(f"{'=' * 120}")

    kid_rows = metric_groups.get("Generation/KID_mean_val", [])
    if kid_rows:
        kid_rows.sort(key=lambda r: r[4] if not np.isnan(r[4]) else 999)
        print(f"  {'Experiment':<60} {'KID Mean':>10} {'KID Slope':>12} {'KID Trend':>15}")
        print(f"  {'-'*60} {'-'*10} {'-'*12} {'-'*15}")
        for exp_name, _, _, mean_val, slope_pct, trend, _ in kid_rows:
            slope_str = f"{slope_pct:+.2f}%" if not np.isnan(slope_pct) else "N/A"
            mean_str = f"{mean_val:.6f}" if not np.isnan(mean_val) else "N/A"
            print(f"  {exp_name:<60} {mean_str:>10} {slope_str:>12} {trend:>15}")
    else:
        print("  No experiments with Generation/KID_mean_val data.")


if __name__ == "__main__":
    main()
