"""Quick diagnostic: does size bin conditioning change the output at all?

Generates volumes with real bins vs zero bins (same noise seed) and compares.
If the model ignores conditioning, outputs will be nearly identical.

Auto-detects model type (FiLM vs Input) from checkpoint.
"""

import argparse
import torch
import numpy as np

from medgen.core import setup_cuda_optimizations
from medgen.data.loaders.datasets import (
    DEFAULT_BIN_EDGES,
    compute_size_bins_3d,
    create_size_bin_maps,
)
from medgen.data.utils import binarize_seg
from medgen.diffusion import RFlowStrategy
from medgen.diffusion.loading import load_diffusion_model_with_metadata
from medgen.scripts.generate import compute_voxel_size, generate_batch, get_noise_shape

setup_cuda_optimizations()

NUM_BINS = 7


def main():
    parser = argparse.ArgumentParser(description="Debug: does conditioning change output?")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", default="model", help="Label for printout")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=160)
    parser.add_argument("--num_steps", type=str, default="40,100,400",
                        help="Comma-separated step counts to test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model — auto-detect everything from checkpoint
    print(f"\n{'='*70}")
    print(f"  {args.label}")
    print(f"  {args.checkpoint}")
    print(f"{'='*70}")
    result = load_diffusion_model_with_metadata(
        args.checkpoint, device=device,
        compile_model=True, spatial_dims=3,
    )
    model = result.model
    wrapper_type = result.wrapper_type

    if wrapper_type == 'size_bin':
        model_type = 'film'
    elif wrapper_type == 'raw':
        model_type = 'input'
    else:
        raise ValueError(f"Unexpected wrapper type: {wrapper_type}")

    print(f"Detected: wrapper={wrapper_type}, model_type={model_type}")

    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000,
        image_size=args.image_size,
        depth_size=args.depth,
        spatial_dims=3,
    )

    voxel_spacing = compute_voxel_size(args.image_size)
    step_counts = [int(s) for s in args.num_steps.split(",")]

    # Test conditions — from easy to hard
    conditions = {
        "zeros":        [0, 0, 0, 0, 0, 0, 0],
        "1_small":      [0, 1, 0, 0, 0, 0, 0],
        "3_medium":     [0, 0, 3, 0, 0, 0, 0],
        "many_mixed":   [3, 4, 2, 0, 0, 0, 0],
        "heavy":        [7, 10, 3, 1, 0, 0, 0],
    }

    # Same noise for all conditions — crucial for fair comparison
    torch.manual_seed(args.seed)
    noise = torch.randn(
        get_noise_shape(1, 1, 3, args.image_size, args.depth), device=device
    )

    for num_steps in step_counts:
        print(f"\n{'='*70}")
        print(f"  Steps: {num_steps}")
        print(f"{'='*70}")
        print(f"\n{'Condition':<20} {'Tumor voxels':>14} {'Num tumors':>12} {'Actual bins'}")
        print("-" * 80)

        results = {}
        for name, bins in conditions.items():
            if model_type == 'film':
                size_bins = torch.tensor([bins], dtype=torch.long, device=device)
                seg = generate_batch(
                    model, strategy, noise.clone(), num_steps, device,
                    size_bins=size_bins, cfg_scale=1.0,
                )
            else:  # input
                bin_map = create_size_bin_maps(
                    torch.tensor(bins, dtype=torch.long, device=device),
                    (args.depth, args.image_size, args.image_size),
                    normalize=True, max_count=10,
                )
                bin_maps = bin_map.unsqueeze(0).to(device)
                seg = generate_batch(
                    model, strategy, noise.clone(), num_steps, device,
                    bin_maps=bin_maps, cfg_scale=1.0,
                )

            seg_bin = binarize_seg(seg[0, 0]).cpu().numpy()
            tumor_voxels = int(seg_bin.sum())
            actual = compute_size_bins_3d(seg_bin, DEFAULT_BIN_EDGES, voxel_spacing, NUM_BINS)

            results[name] = {"seg": seg_bin, "voxels": tumor_voxels, "actual_bins": actual.tolist()}
            print(f"{name:<20} {tumor_voxels:>14,} {int(actual.sum()):>12} {actual.tolist()}")

        # Pairwise comparison
        print(f"\n{'Pair':<45} {'Diff voxels':>12} {'Dice':>8}")
        print("-" * 70)
        names = list(results.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = results[names[i]]["seg"]
                b = results[names[j]]["seg"]
                diff = int(np.abs(a - b).sum())
                dice = 2 * ((a > 0) & (b > 0)).sum() / (a.sum() + b.sum() + 1e-8)
                pair = f"{names[i]} vs {names[j]}"
                print(f"{pair:<45} {diff:>12,} {dice:>8.4f}")


if __name__ == "__main__":
    main()
