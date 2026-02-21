"""
Bin Adherence Evaluation for 3D Seg-Conditioned Models.

Evaluates whether models learned proper size bin conditioning by loading
real segmentation volumes from the dataset, extracting their bin vectors,
and testing if the model can generate volumes matching those real distributions.

Supports two model types:
- FiLM (exp2_1, exp2c_1): Size bins embedded via SizeBinModelWrapper
- Input (exp2b_1): Size bins as 7 spatial maps concatenated with noise

Usage:
    python misc/eval_bin_adherence.py \
        --checkpoint runs/diffusion_3d/.../checkpoint_best.pt \
        --model_type film \
        --data_dir /path/to/brainmetshare-3 \
        --output_dir results/bin_adherence/exp2_1
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from medgen.core import setup_cuda_optimizations
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
# Extract Real Conditions from Dataset
# ============================================================================

def extract_conditions_from_data(
    data_dir: str,
    split: str,
    bin_edges: list[float],
    num_bins: int,
) -> list[tuple[str, list[int]]]:
    """Extract bin vectors from real 3D segmentation volumes.

    Loads each patient's seg.nii.gz, computes the bin vector using the
    NIfTI header voxel spacing, and returns unique bin vectors as conditions.

    Args:
        data_dir: Root dataset directory (e.g. /path/to/brainmetshare-3).
        split: Dataset split to use ('train', 'val', 'test').
        bin_edges: Size bin edges in mm.
        num_bins: Number of bins (including overflow).

    Returns:
        List of (condition_name, bin_vector) tuples.
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise NotADirectoryError(f"Split directory not found: {split_dir}")

    patients = sorted(
        p for p in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, p))
    )

    logger.info(f"Scanning {len(patients)} patients in {split_dir}...")

    # Collect bin vectors from all positive volumes
    all_vectors: list[tuple[int, ...]] = []

    for patient in patients:
        seg_path = os.path.join(split_dir, patient, "seg.nii.gz")
        if not os.path.exists(seg_path):
            continue

        nii_img = nib.load(seg_path)
        data = nii_img.get_fdata()

        # Skip empty volumes (no tumors)
        if data.sum() == 0:
            continue

        # Use voxel spacing from NIfTI header (ground truth physical spacing)
        zooms = nii_img.header.get_zooms()[:3]
        voxel_spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

        bins = compute_size_bins_3d(data, bin_edges, voxel_spacing, num_bins)
        all_vectors.append(tuple(bins.tolist()))

    logger.info(f"Found {len(all_vectors)} positive volumes")

    # Count unique vectors and sort by frequency (most common first)
    vector_counts = Counter(all_vectors)
    unique_vectors = vector_counts.most_common()

    logger.info(f"Unique bin vectors: {len(unique_vectors)}")
    for vec, count in unique_vectors:
        logger.info(f"  {list(vec)} x{count}")

    # Build conditions: each unique vector is a test condition
    conditions: list[tuple[str, list[int]]] = []
    for i, (vec, count) in enumerate(unique_vectors):
        bins_str = "_".join(str(v) for v in vec)
        name = f"real_{i:02d}_{bins_str}_x{count}"
        conditions.append((name, list(vec)))

    # Always include empty condition (model should handle unconditional)
    empty = [0] * num_bins
    if tuple(empty) not in vector_counts:
        conditions.append(("empty", empty))

    return conditions


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
        **{f"bin{i}_match": float(per_bin_match[i]) for i in range(len(requested))},
    }


# ============================================================================
# Generation
# ============================================================================

def generate_samples_batched(
    model: torch.nn.Module,
    strategy: RFlowStrategy,
    bins: list[int],
    model_type: str,
    num_steps: int,
    image_size: int,
    depth: int,
    device: torch.device,
    cfg_scale: float,
    seeds: list[int],
    batch_size: int,
) -> list[np.ndarray]:
    """Generate multiple segmentation masks in batches.

    All samples share the same conditioning (bins + cfg_scale),
    only the noise seed differs. Batching gives ~Nx speedup.

    Args:
        model: Loaded diffusion model.
        strategy: RFlow strategy with scheduler set up.
        bins: Requested size bin counts [num_bins].
        model_type: "film" or "input".
        num_steps: Number of denoising steps.
        image_size: H/W resolution.
        depth: Volume depth.
        device: Torch device.
        cfg_scale: Classifier-free guidance scale.
        seeds: List of seeds, one per sample.
        batch_size: Max samples per forward pass.

    Returns:
        List of binary segmentation masks as numpy arrays [D, H, W].
    """
    all_volumes: list[np.ndarray] = []

    for chunk_start in range(0, len(seeds), batch_size):
        chunk_seeds = seeds[chunk_start:chunk_start + batch_size]
        B = len(chunk_seeds)

        # Generate noise with per-sample reproducibility
        noise_list = []
        for seed in chunk_seeds:
            torch.manual_seed(seed)
            noise_list.append(
                torch.randn(get_noise_shape(1, 1, 3, image_size, depth), device=device)
            )
        noise = torch.cat(noise_list, dim=0)  # [B, 1, D, H, W]

        if model_type == "film":
            size_bins = torch.tensor([bins] * B, dtype=torch.long, device=device)
            seg = generate_batch(
                model, strategy, noise, num_steps, device,
                size_bins=size_bins, cfg_scale=cfg_scale,
            )
        elif model_type == "input":
            single_map = create_size_bin_maps(
                torch.tensor(bins, dtype=torch.long, device=device),
                (depth, image_size, image_size),
                normalize=True, max_count=10,
            )  # [7, D, H, W]
            bin_maps = single_map.unsqueeze(0).expand(B, -1, -1, -1, -1).to(device)
            seg = generate_batch(
                model, strategy, noise, num_steps, device,
                bin_maps=bin_maps, cfg_scale=cfg_scale,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Post-process each volume: clamp + binarize (matches generation_sampling.py)
        for j in range(B):
            vol = torch.clamp(seg[j, 0].float(), 0, 1)
            all_volumes.append((vol > 0.5).float().cpu().numpy())

    return all_volumes


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def run_evaluation(args: argparse.Namespace) -> None:
    """Run the full bin adherence evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_scales = [float(s) for s in args.cfg_scales.split(",")]
    num_bins = len(DEFAULT_BIN_EDGES)  # 7 bins (6 bounded + overflow)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract real conditions from dataset
    conditions = extract_conditions_from_data(
        args.data_dir, args.split, DEFAULT_BIN_EDGES, num_bins,
    )
    logger.info(f"Testing {len(conditions)} conditions from real data")

    # Save eval config
    git_hash = _get_git_hash()
    eval_config = {
        "checkpoint": args.checkpoint,
        "model_type": args.model_type,
        "data_dir": args.data_dir,
        "split": args.split,
        "cfg_scales": cfg_scales,
        "num_repeats": args.num_repeats,
        "num_steps": args.num_steps,
        "image_size": args.image_size,
        "depth": args.depth,
        "seed": args.seed,
        "num_conditions": len(conditions),
        "total_samples": len(conditions) * len(cfg_scales) * args.num_repeats,
        "conditions": [(name, bins) for name, bins in conditions],
        "git_hash": git_hash,
        "timestamp": datetime.now().isoformat(),
        "bin_edges": DEFAULT_BIN_EDGES,
    }
    with open(output_dir / "eval_config.json", "w") as f:
        json.dump(eval_config, f, indent=2)

    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"CFG scales: {cfg_scales}")
    logger.info(f"Conditions: {len(conditions)}, Repeats: {args.num_repeats}")
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

    # Voxel spacing for evaluating generated volumes
    voxel_spacing = compute_voxel_size(args.image_size)

    # Collect all results
    all_results: list[dict] = []
    sample_idx = 0
    total_samples = eval_config["total_samples"]
    start_time = time.time()

    for cond_name, bins in conditions:
        requested = np.array(bins)
        for cfg_scale in cfg_scales:
            # Generate all repeats in one batched call
            seeds = [args.seed + sample_idx + r for r in range(args.num_repeats)]
            volumes = generate_samples_batched(
                model, strategy, bins, args.model_type,
                args.num_steps, args.image_size, args.depth,
                device, cfg_scale, seeds, args.batch_size,
            )

            for repeat, seg_binary in enumerate(volumes):
                # Compute actual bins from generated mask
                actual = compute_size_bins_3d(
                    seg_binary, DEFAULT_BIN_EDGES, voxel_spacing, num_bins,
                )

                metrics = compute_sample_metrics(requested, actual)

                result = {
                    "condition": cond_name,
                    "cfg_scale": cfg_scale,
                    "repeat": repeat,
                    "seed": seeds[repeat],
                    **{f"req_bin{i}": int(bins[i]) for i in range(num_bins)},
                    **{f"act_bin{i}": int(actual[i]) for i in range(num_bins)},
                    **metrics,
                }
                all_results.append(result)

            sample_idx += args.num_repeats
            elapsed = time.time() - start_time
            rate = sample_idx / elapsed * 60  # samples per minute
            remaining = (total_samples - sample_idx) / max(rate, 0.01) * 60
            logger.info(
                f"Progress: {sample_idx}/{total_samples} "
                f"({rate:.1f} samples/min, ~{remaining:.0f}s remaining) "
                f"[{cond_name}, cfg={cfg_scale}]"
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
    header = f"{'Condition':>40} | {'Exact':>8} | {'Presence':>9} | {'Tot.Count':>10} | {'Per-Bin':>8} | {'MAE':>6}"
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
        print(f"{cond:>40} | {exact:>7.1f}% | {presence:>8.1f}% | {total_count:>9.1f}% | {per_bin:>7.1f}% | {mae:>6.2f}")

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
      --data_dir /path/to/brainmetshare-3 \\
      --output_dir results/bin_adherence/exp2_1

  # Input conditioning model (exp2b_1)
  python misc/eval_bin_adherence.py \\
      --checkpoint runs/diffusion_3d/.../checkpoint_best.pt \\
      --model_type input \\
      --data_dir /path/to/brainmetshare-3 \\
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
        "--data_dir", type=str, required=True,
        help="Root dataset directory containing train/val splits with seg.nii.gz volumes",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to extract conditions from (default: 'train')",
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
        "--batch_size", type=int, default=4,
        help="Volumes per forward pass (default: 4, limited by GPU memory)",
    )
    parser.add_argument(
        "--num_steps", type=int, default=37,
        help="Number of denoising steps (default: 37)",
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
    logger.info("Bin Adherence Evaluation (real data conditions)")
    logger.info("=" * 60)
    run_evaluation(args)
    logger.info("Done!")


if __name__ == "__main__":
    main()
