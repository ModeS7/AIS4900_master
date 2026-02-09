"""
Bin Adherence Evaluation for 3D Seg-Conditioned Models.

Evaluates whether models learned proper size bin conditioning by generating
segmentation masks under controlled conditions and measuring how well the
output matches the requested bin distribution.

Supports two model types:
- FiLM (exp2_1): Size bins embedded into timestep embedding via SizeBinModelWrapper
- Input (exp2b_1): Size bins as 7 spatial maps concatenated with noise (8 input channels)

Usage:
    python misc/eval_bin_adherence.py \
        --checkpoint runs/diffusion_3d/.../checkpoint_best.pt \
        --model_type film \
        --output_dir results/bin_adherence/exp2_1

    python misc/eval_bin_adherence.py \
        --checkpoint runs/diffusion_3d/.../checkpoint_best.pt \
        --model_type input \
        --output_dir results/bin_adherence/exp2b_1
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from medgen.core import setup_cuda_optimizations
from medgen.data import make_binary
from medgen.data.loaders.datasets import (
    DEFAULT_BIN_EDGES,
    compute_size_bins_3d,
    create_size_bin_maps,
)
from medgen.diffusion import RFlowStrategy, load_diffusion_model
from medgen.scripts.generate import compute_voxel_size, generate_batch, get_noise_shape

setup_cuda_optimizations()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
)
logger = logging.getLogger(__name__)

# ============================================================================
# Test Conditions
# ============================================================================

TEST_CONDITIONS: list[tuple[str, list[int]]] = [
    # Single tumor per bin (7 conditions)
    ("single_bin0", [1, 0, 0, 0, 0, 0, 0]),
    ("single_bin1", [0, 1, 0, 0, 0, 0, 0]),
    ("single_bin2", [0, 0, 1, 0, 0, 0, 0]),
    ("single_bin3", [0, 0, 0, 1, 0, 0, 0]),
    ("single_bin4", [0, 0, 0, 0, 1, 0, 0]),
    ("single_bin5", [0, 0, 0, 0, 0, 1, 0]),
    ("single_bin6", [0, 0, 0, 0, 0, 0, 1]),
    # Multiple tumors in same bin (2 conditions)
    ("multi_bin2_x2", [0, 0, 2, 0, 0, 0, 0]),
    ("multi_bin6_x2", [0, 0, 0, 0, 0, 0, 2]),
    # Clinical combinations (3 conditions)
    ("combo_small_large", [0, 0, 1, 0, 0, 0, 1]),
    ("combo_mid_pair", [0, 1, 0, 1, 0, 0, 0]),
    ("combo_three_mid", [0, 0, 1, 1, 1, 0, 0]),
    # Empty (1 condition)
    ("empty", [0, 0, 0, 0, 0, 0, 0]),
]


# ============================================================================
# Metrics
# ============================================================================

def compute_sample_metrics(
    requested: np.ndarray,
    actual: np.ndarray,
) -> dict[str, float]:
    """Compute adherence metrics for a single sample.

    Args:
        requested: Requested bin counts [7].
        actual: Actual bin counts from generated seg [7].

    Returns:
        Dict of metric_name -> value.
    """
    exact_match = float(np.array_equal(requested, actual))
    total_count_match = float(requested.sum() == actual.sum())
    presence_match = float((requested.sum() > 0) == (actual.sum() > 0))
    per_bin_match = (requested == actual).astype(float)
    per_bin_accuracy = float(per_bin_match.mean())
    mae = float(np.abs(requested - actual).mean())

    return {
        "exact_match": exact_match,
        "total_count_match": total_count_match,
        "presence_match": presence_match,
        "per_bin_accuracy": per_bin_accuracy,
        "mae": mae,
        # Individual bin matches for detailed analysis
        **{f"bin{i}_match": float(per_bin_match[i]) for i in range(7)},
    }


# ============================================================================
# Generation
# ============================================================================

def generate_single_sample(
    model: torch.nn.Module,
    strategy: RFlowStrategy,
    bins: list[int],
    model_type: str,
    num_steps: int,
    image_size: int,
    depth: int,
    device: torch.device,
    cfg_scale: float,
    seed: int,
) -> np.ndarray:
    """Generate one segmentation mask with given bin conditioning.

    Args:
        model: Loaded diffusion model.
        strategy: RFlow strategy with scheduler set up.
        bins: Requested size bin counts [7].
        model_type: "film" or "input".
        num_steps: Number of denoising steps.
        image_size: H/W resolution.
        depth: Volume depth.
        device: Torch device.
        cfg_scale: Classifier-free guidance scale.
        seed: Random seed for this sample.

    Returns:
        Binary segmentation mask as numpy array [D, H, W].
    """
    torch.manual_seed(seed)

    noise = torch.randn(
        get_noise_shape(1, 1, 3, image_size, depth), device=device
    )

    if model_type == "film":
        size_bins = torch.tensor([bins], dtype=torch.long, device=device)
        seg = generate_batch(
            model, strategy, noise, num_steps, device,
            size_bins=size_bins, cfg_scale=cfg_scale,
        )
    elif model_type == "input":
        bin_maps = create_size_bin_maps(
            torch.tensor(bins, dtype=torch.long, device=device),
            (depth, image_size, image_size),
            normalize=True, max_count=10,
        ).unsqueeze(0).to(device)  # [1, 7, D, H, W]
        seg = generate_batch(
            model, strategy, noise, num_steps, device,
            bin_maps=bin_maps, cfg_scale=cfg_scale,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Post-process: normalize + binarize (matches generate.py)
    seg_np = seg[0, 0].cpu().numpy()  # [D, H, W]
    seg_np = (seg_np - seg_np.min()) / (seg_np.max() - seg_np.min() + 1e-8)
    seg_binary = make_binary(seg_np, threshold=0.5)

    return seg_binary


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def run_evaluation(args: argparse.Namespace) -> None:
    """Run the full bin adherence evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_scales = [float(s) for s in args.cfg_scales.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save eval config
    git_hash = _get_git_hash()
    eval_config = {
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "cfg_scales": cfg_scales,
        "num_repeats": args.num_repeats,
        "num_steps": args.num_steps,
        "image_size": args.image_size,
        "depth": args.depth,
        "seed": args.seed,
        "num_conditions": len(TEST_CONDITIONS),
        "total_samples": len(TEST_CONDITIONS) * len(cfg_scales) * args.num_repeats,
        "git_hash": git_hash,
        "timestamp": datetime.now().isoformat(),
        "bin_edges": DEFAULT_BIN_EDGES,
    }
    with open(output_dir / "eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)

    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"CFG scales: {cfg_scales}")
    logger.info(f"Conditions: {len(TEST_CONDITIONS)}, Repeats: {args.num_repeats}")
    logger.info(f"Total samples: {eval_config['total_samples']}")

    # Load model
    if args.model_type == "film":
        in_channels, out_channels = 1, 1
    else:
        in_channels, out_channels = 8, 1

    logger.info("Loading model...")
    model = load_diffusion_model(
        args.checkpoint, device=device,
        in_channels=in_channels, out_channels=out_channels,
        compile_model=True, spatial_dims=3,
    )

    # Set up strategy
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000,
        image_size=args.image_size,
        depth_size=args.depth,
        spatial_dims=3,
    )

    # Voxel spacing for bin computation
    voxel_spacing = compute_voxel_size(args.image_size)

    # Collect all results
    all_results: list[dict] = []
    sample_idx = 0
    start_time = time.time()

    for cond_name, bins in TEST_CONDITIONS:
        requested = np.array(bins)
        for cfg_scale in cfg_scales:
            for repeat in range(args.num_repeats):
                seed = args.seed + sample_idx

                seg_binary = generate_single_sample(
                    model, strategy, bins, args.model_type,
                    args.num_steps, args.image_size, args.depth,
                    device, cfg_scale, seed,
                )

                # Compute actual bins from generated mask
                actual = compute_size_bins_3d(
                    seg_binary, DEFAULT_BIN_EDGES, voxel_spacing, 7,
                )

                # Compute metrics
                metrics = compute_sample_metrics(requested, actual)

                result = {
                    "condition": cond_name,
                    "cfg_scale": cfg_scale,
                    "repeat": repeat,
                    "seed": seed,
                    **{f"req_bin{i}": int(bins[i]) for i in range(7)},
                    **{f"act_bin{i}": int(actual[i]) for i in range(7)},
                    **metrics,
                }
                all_results.append(result)

                sample_idx += 1
                if sample_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = sample_idx / elapsed
                    remaining = (eval_config["total_samples"] - sample_idx) / rate
                    logger.info(
                        f"Progress: {sample_idx}/{eval_config['total_samples']} "
                        f"({rate:.1f} samples/min, ~{remaining:.0f}s remaining)"
                    )

                torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    logger.info(f"Generation complete: {sample_idx} samples in {elapsed:.0f}s")

    # Write per-sample CSV
    _write_per_sample_csv(all_results, output_dir / "results_per_sample.csv")

    # Write summary CSVs
    _write_summary_by_cfg(all_results, cfg_scales, output_dir / "summary_by_cfg.csv")
    _write_summary_by_condition(all_results, output_dir / "summary_by_condition.csv")

    # Print summary table
    _print_summary(all_results, cfg_scales, args.model_type)


# ============================================================================
# Output Writers
# ============================================================================

def _write_per_sample_csv(results: list[dict], path: Path) -> None:
    """Write per-sample results CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Saved per-sample results: {path}")


def _write_summary_by_cfg(
    results: list[dict],
    cfg_scales: list[float],
    path: Path,
) -> None:
    """Write summary aggregated by CFG scale."""
    rows = []
    for cfg in cfg_scales:
        subset = [r for r in results if r["cfg_scale"] == cfg]
        rows.append(_aggregate_metrics(subset, cfg_label=str(cfg)))

    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved summary by CFG: {path}")


def _write_summary_by_condition(results: list[dict], path: Path) -> None:
    """Write summary aggregated by condition name."""
    conditions = list(dict.fromkeys(r["condition"] for r in results))
    rows = []
    for cond in conditions:
        subset = [r for r in results if r["condition"] == cond]
        rows.append(_aggregate_metrics(subset, condition_label=cond))

    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved summary by condition: {path}")


def _aggregate_metrics(
    subset: list[dict],
    cfg_label: str | None = None,
    condition_label: str | None = None,
) -> dict[str, str | float]:
    """Aggregate metrics over a subset of results."""
    n = len(subset)
    if n == 0:
        return {}

    row: dict[str, str | float] = {}
    if cfg_label is not None:
        row["cfg_scale"] = cfg_label
    if condition_label is not None:
        row["condition"] = condition_label
    row["n_samples"] = n

    for metric in ["exact_match", "total_count_match", "presence_match", "per_bin_accuracy"]:
        values = [r[metric] for r in subset]
        row[metric] = round(sum(values) / n, 4)

    mae_values = [r["mae"] for r in subset]
    row["mae_mean"] = round(sum(mae_values) / n, 4)
    row["mae_std"] = round(float(np.std(mae_values)), 4)

    return row


def _print_summary(
    results: list[dict],
    cfg_scales: list[float],
    model_type: str,
) -> None:
    """Print formatted summary table to stdout."""
    print(f"\n{'=' * 70}")
    print(f"  Bin Adherence: {model_type.upper()}")
    print(f"{'=' * 70}\n")

    # By CFG Scale
    print("By CFG Scale:")
    header = f"{'CFG':>6} | {'Exact':>8} | {'Presence':>9} | {'Tot.Count':>10} | {'Per-Bin':>8} | {'MAE':>6}"
    print(header)
    print("-" * len(header))

    for cfg in cfg_scales:
        subset = [r for r in results if r["cfg_scale"] == cfg]
        n = len(subset)
        if n == 0:
            continue
        exact = sum(r["exact_match"] for r in subset) / n * 100
        presence = sum(r["presence_match"] for r in subset) / n * 100
        total_count = sum(r["total_count_match"] for r in subset) / n * 100
        per_bin = sum(r["per_bin_accuracy"] for r in subset) / n * 100
        mae = sum(r["mae"] for r in subset) / n
        print(f"{cfg:>6.1f} | {exact:>7.1f}% | {presence:>8.1f}% | {total_count:>9.1f}% | {per_bin:>7.1f}% | {mae:>6.2f}")

    # By Condition
    print(f"\nBy Condition:")
    conditions = list(dict.fromkeys(r["condition"] for r in results))
    header = f"{'Condition':>20} | {'Exact':>8} | {'Presence':>9} | {'Tot.Count':>10} | {'Per-Bin':>8} | {'MAE':>6}"
    print(header)
    print("-" * len(header))

    for cond in conditions:
        subset = [r for r in results if r["condition"] == cond]
        n = len(subset)
        if n == 0:
            continue
        exact = sum(r["exact_match"] for r in subset) / n * 100
        presence = sum(r["presence_match"] for r in subset) / n * 100
        total_count = sum(r["total_count_match"] for r in subset) / n * 100
        per_bin = sum(r["per_bin_accuracy"] for r in subset) / n * 100
        mae = sum(r["mae"] for r in subset) / n
        print(f"{cond:>20} | {exact:>7.1f}% | {presence:>8.1f}% | {total_count:>9.1f}% | {per_bin:>7.1f}% | {mae:>6.2f}")

    print()


# ============================================================================
# Utilities
# ============================================================================

def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate bin adherence of 3D seg-conditioned diffusion models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FiLM model (exp2_1)
  python misc/eval_bin_adherence.py \\
      --checkpoint runs/diffusion_3d/.../checkpoint_best.pt \\
      --model_type film \\
      --output_dir results/bin_adherence/exp2_1

  # Input conditioning model (exp2b_1)
  python misc/eval_bin_adherence.py \\
      --checkpoint runs/diffusion_3d/.../checkpoint_best.pt \\
      --model_type input \\
      --output_dir results/bin_adherence/exp2b_1
        """,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["film", "input"],
        help="Model conditioning type: 'film' (SizeBinWrapper) or 'input' (bin maps concat)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--cfg_scales", type=str, default="1.0,2.0,3.0,5.0",
        help="Comma-separated CFG scales to evaluate (default: '1.0,2.0,3.0,5.0')",
    )
    parser.add_argument(
        "--num_repeats", type=int, default=5,
        help="Number of repeats per condition (default: 5)",
    )
    parser.add_argument(
        "--num_steps", type=int, default=50,
        help="Number of denoising steps (default: 50)",
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Image height/width in pixels (default: 256)",
    )
    parser.add_argument(
        "--depth", type=int, default=160,
        help="Volume depth in slices (default: 160)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    logger.info("=" * 60)
    logger.info("Bin Adherence Evaluation")
    logger.info("=" * 60)
    run_evaluation(args)
    logger.info("Done!")


if __name__ == "__main__":
    main()
