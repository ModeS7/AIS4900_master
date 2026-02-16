#!/usr/bin/env python3
"""Evaluate DiffRS (Diffusion Rejection Sampling) against baseline generation.

Generates 3D BRAVO volumes with baseline Euler sampler and optionally DiffRS,
using identical noise and real segmentation masks as conditioning. Computes
FID, KID, CMMD against reference data (per-split and combined) to measure
whether DiffRS improves generation quality.

Experimental design:
  - Baseline: Standard Euler sampler at multiple step counts [10, 25, 50]
  - DiffRS: Same step counts with rejection sampling (higher NFE, hopefully better quality)
  - All configs use the SAME noise tensors and SAME real seg masks
  - NFE counted per config for fair comparison

Usage:
    # Baseline only (no DiffRS head needed)
    python -m medgen.scripts.eval_diffrs \\
        --bravo-model runs/checkpoint_bravo.pt \\
        --data-root ~/data/brainmetshare-3 \\
        --num-volumes 25 --output-dir results/eval_diffrs

    # With DiffRS comparison
    python -m medgen.scripts.eval_diffrs \\
        --bravo-model runs/checkpoint_bravo.pt \\
        --diffrs-head runs/diffrs_head.pt \\
        --data-root ~/data/brainmetshare-3 \\
        --num-volumes 25 --output-dir results/eval_diffrs

    # Quick test
    python -m medgen.scripts.eval_diffrs \\
        --bravo-model runs/checkpoint_bravo.pt \\
        --data-root ~/data/brainmetshare-3 \\
        --num-volumes 5 --output-dir results/eval_diffrs_quick --quick

    # Resume interrupted run
    python -m medgen.scripts.eval_diffrs \\
        --bravo-model runs/checkpoint_bravo.pt \\
        --diffrs-head runs/diffrs_head.pt \\
        --data-root ~/data/brainmetshare-3 \\
        --num-volumes 25 --output-dir results/eval_diffrs --resume

    # Smoke test (no checkpoint needed)
    python -m medgen.scripts.eval_diffrs --smoke-test
"""
import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from medgen.scripts.eval_ode_solvers import (
    compute_all_metrics,
    discover_splits,
    generate_noise_tensors,
    get_or_cache_reference_features,
    load_conditioning,
    save_conditioning,
    save_volumes,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

BASELINE_STEPS = [10, 25, 50]
DIFFRS_STEPS = [10, 25, 50]
QUICK_STEPS = [10, 25]


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    """A single evaluation configuration."""
    name: str          # e.g., "baseline_025", "diffrs_025"
    method: str        # "baseline" or "diffrs"
    steps: int         # Number of Euler steps

    # DiffRS hyperparameters (only used when method="diffrs")
    rej_percentile: float = 0.75
    backsteps: int = 1
    max_iter: int = 999999
    iter_warmup: int = 10

    @property
    def dir_name(self) -> str:
        return self.name

    @property
    def label(self) -> str:
        if self.method == "diffrs":
            return f"DiffRS/{self.steps} (γ={self.rej_percentile})"
        return f"Euler/{self.steps}"


@dataclass
class SplitMetrics:
    """Metrics computed against one reference split."""
    fid: float
    kid_mean: float
    kid_std: float
    cmmd: float


@dataclass
class EvalResult:
    """Full result for one evaluation configuration."""
    name: str
    method: str
    steps: int
    nfe_total: int
    nfe_per_volume: float
    wall_time_s: float
    time_per_volume_s: float
    num_volumes: int
    metrics: dict[str, dict[str, float]]   # split -> {fid, kid_mean, kid_std, cmmd}


# ═══════════════════════════════════════════════════════════════════════════════
# NFE counting
# ═══════════════════════════════════════════════════════════════════════════════

class NFECounter(nn.Module):
    """Wraps a model to count forward passes (= function evaluations)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self._wrapped = model
        self.nfe = 0

    def forward(self, *args, **kwargs):
        self.nfe += 1
        return self._wrapped(*args, **kwargs)

    def reset(self):
        self.nfe = 0

    def __getattr__(self, name: str):
        if name in ('_wrapped', 'nfe', 'training', '_parameters', '_buffers',
                     '_modules', '_backward_hooks', '_forward_hooks',
                     '_forward_pre_hooks', '_state_dict_hooks',
                     '_load_state_dict_pre_hooks'):
            return super().__getattr__(name)
        return getattr(self._wrapped, name)


# ═══════════════════════════════════════════════════════════════════════════════
# Build evaluation configs
# ═══════════════════════════════════════════════════════════════════════════════

def build_eval_configs(
    has_diffrs: bool,
    quick: bool = False,
    rej_percentile: float = 0.75,
    backsteps: int = 1,
    max_iter: int = 999999,
    iter_warmup: int = 10,
) -> list[EvalConfig]:
    """Build evaluation configurations."""
    steps_list = QUICK_STEPS if quick else BASELINE_STEPS
    configs = []

    # Baseline configs
    for steps in steps_list:
        configs.append(EvalConfig(
            name=f"baseline_{steps:03d}",
            method="baseline",
            steps=steps,
        ))

    # DiffRS configs (same step counts)
    if has_diffrs:
        diffrs_steps = QUICK_STEPS if quick else DIFFRS_STEPS
        for steps in diffrs_steps:
            configs.append(EvalConfig(
                name=f"diffrs_{steps:03d}",
                method="diffrs",
                steps=steps,
                rej_percentile=rej_percentile,
                backsteps=backsteps,
                max_iter=max_iter,
                iter_warmup=iter_warmup,
            ))

    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Volume generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_volumes(
    model: nn.Module,
    strategy,
    noise_list: list[torch.Tensor],
    cond_list: list[tuple[str, torch.Tensor]],
    eval_cfg: EvalConfig,
    device: torch.device,
    diffrs_discriminator=None,
) -> tuple[list[np.ndarray], int, float]:
    """Generate BRAVO volumes for one evaluation configuration.

    Args:
        model: BRAVO model (will be wrapped with NFECounter).
        strategy: RFlowStrategy instance.
        noise_list: Pre-generated noise tensors.
        cond_list: (patient_id, seg_tensor) pairs.
        eval_cfg: Evaluation configuration.
        device: CUDA device.
        diffrs_discriminator: DiffRSDiscriminator (required for DiffRS configs).

    Returns:
        (volumes_list, total_nfe, wall_time_seconds)
    """
    counter = NFECounter(model)
    counter.reset()

    # Build DiffRS kwargs if needed
    gen_kwargs = {}
    if eval_cfg.method == "diffrs":
        if diffrs_discriminator is None:
            raise ValueError("DiffRS config requires a discriminator")
        gen_kwargs['diffrs_discriminator'] = diffrs_discriminator
        gen_kwargs['diffrs_config'] = {
            'rej_percentile': eval_cfg.rej_percentile,
            'backsteps': eval_cfg.backsteps,
            'max_iter': eval_cfg.max_iter,
            'iter_warmup': eval_cfg.iter_warmup,
        }

    volumes = []
    start_time = time.time()

    for i, (noise, (_patient_id, seg_tensor)) in enumerate(zip(noise_list, cond_list)):
        seg_on_device = seg_tensor.to(device)
        model_input = torch.cat([noise, seg_on_device], dim=1)  # [1, 2, D, H, W]

        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            with torch.no_grad():
                result = strategy.generate(
                    counter, model_input, eval_cfg.steps, device,
                    **gen_kwargs,
                )

        vol_np = torch.clamp(result[0, 0], 0, 1).cpu().float().numpy()  # [D, H, W]
        volumes.append(vol_np)

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - start_time
            logger.info(
                "    %d/%d volumes (%.0fs, NFE=%d)",
                i + 1, len(noise_list), elapsed, counter.nfe,
            )

    wall_time = time.time() - start_time
    return volumes, counter.nfe, wall_time


# ═══════════════════════════════════════════════════════════════════════════════
# Results output
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_table(results: list[EvalResult], primary_split: str = 'all') -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 120)
    print(f"DiffRS Evaluation Results (vs '{primary_split}' reference)")
    print("=" * 120)
    print(f"{'Config':<24} {'Method':<10} {'Steps':>5} {'NFE/vol':>8} {'Time/vol':>9} "
          f"{'FID':>10} {'KID':>14} {'CMMD':>10}")
    print("-" * 120)

    # Group by step count for easy comparison
    by_steps: dict[int, list[EvalResult]] = {}
    for r in results:
        by_steps.setdefault(r.steps, []).append(r)

    for steps in sorted(by_steps.keys()):
        group = by_steps[steps]
        for r in sorted(group, key=lambda x: x.method):
            m = r.metrics.get(primary_split, {})
            if not m:
                continue
            kid_str = f"{m['kid_mean']:.6f}±{m['kid_std']:.4f}"
            print(f"{r.name:<24} {r.method:<10} {r.steps:>5} {r.nfe_per_volume:>8.0f} "
                  f"{r.time_per_volume_s:>8.1f}s "
                  f"{m['fid']:>10.2f} {kid_str:>14} {m['cmmd']:>10.6f}")

        # Show improvement if both baseline and DiffRS present
        baseline = [r for r in group if r.method == "baseline"]
        diffrs = [r for r in group if r.method == "diffrs"]
        if baseline and diffrs:
            bm = baseline[0].metrics.get(primary_split, {})
            dm = diffrs[0].metrics.get(primary_split, {})
            if bm and dm:
                fid_delta = dm['fid'] - bm['fid']
                kid_delta = dm['kid_mean'] - bm['kid_mean']
                cmmd_delta = dm['cmmd'] - bm['cmmd']
                nfe_ratio = diffrs[0].nfe_per_volume / max(baseline[0].nfe_per_volume, 1)
                print(f"  {'Δ (DiffRS - baseline)':<34} {'':>5} {f'{nfe_ratio:.1f}x':>8} "
                      f"{'':>9} "
                      f"{fid_delta:>+10.2f} {kid_delta:>+14.6f} {cmmd_delta:>+10.6f}")
        print()

    print("=" * 120)


def save_results_csv(results: list[EvalResult], path: Path) -> None:
    """Save results as CSV for thesis tables/plots."""
    all_splits = set()
    for r in results:
        all_splits.update(r.metrics.keys())
    all_splits = sorted(all_splits)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['name', 'method', 'steps', 'nfe_per_volume',
                  'wall_time_s', 'time_per_volume_s', 'num_volumes']
        for split in all_splits:
            header.extend([f'fid_{split}', f'kid_mean_{split}',
                          f'kid_std_{split}', f'cmmd_{split}'])
        writer.writerow(header)

        for r in sorted(results, key=lambda x: (x.method, x.steps)):
            row = [r.name, r.method, r.steps, f"{r.nfe_per_volume:.1f}",
                   f"{r.wall_time_s:.1f}", f"{r.time_per_volume_s:.1f}", r.num_volumes]
            for split in all_splits:
                m = r.metrics.get(split, {})
                row.extend([
                    f"{m.get('fid', ''):.4f}" if m else '',
                    f"{m.get('kid_mean', ''):.6f}" if m else '',
                    f"{m.get('kid_std', ''):.6f}" if m else '',
                    f"{m.get('cmmd', ''):.6f}" if m else '',
                ])
            writer.writerow(row)

    logger.info("CSV saved to %s", path)


def save_results_json(results: list[EvalResult], path: Path) -> None:
    """Save full structured results as JSON."""
    data = [asdict(r) for r in results]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_results_json(path: Path) -> list[EvalResult]:
    """Load results from JSON for resume support."""
    with open(path) as f:
        data = json.load(f)
    return [EvalResult(**entry) for entry in data]


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _run_smoke_test() -> None:
    """Fast end-to-end pipeline verification with tiny dummy model."""
    import tempfile

    from monai.networks.nets import DiffusionModelUNet

    from medgen.diffusion import RFlowStrategy
    from medgen.diffusion.diffrs import DiffRSDiscriminator, DiffRSHead

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(tempfile.mkdtemp(prefix="eval_diffrs_smoke_"))

    image_size = 16
    depth = 8
    trim_slices = 2
    num_volumes = 2
    seed = 42
    voxel_size = (1.0, 1.0, 1.0)

    logger.info("=== SMOKE TEST (output: %s) ===", output_dir)

    # Create tiny model (in_channels=2 for noise+seg conditioning)
    model = DiffusionModelUNet(
        spatial_dims=3, in_channels=2, out_channels=1,
        channels=[8, 16], attention_levels=[False, False],
        num_res_blocks=1, num_head_channels=8, norm_num_groups=8,
    ).to(device)

    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000, image_size=image_size,
        depth_size=depth, spatial_dims=3,
    )

    # Create tiny DiffRS head
    from medgen.diffusion.diffrs import get_bottleneck_channels
    bottleneck_ch = get_bottleneck_channels(model)
    head = DiffRSHead(in_channels=bottleneck_ch, spatial_dims=3).to(device)
    discriminator = DiffRSDiscriminator(model, head, device)

    # Fake conditioning
    cond_list = [
        (f"smoke_{i:03d}", torch.randint(0, 2, (1, 1, depth, image_size, image_size)).float())
        for i in range(num_volumes)
    ]

    # Fake reference features
    effective_slices = depth - trim_slices
    ref_features = {
        'fake_split': {
            'resnet': torch.randn(num_volumes * effective_slices, 2048),
            'clip': torch.randn(num_volumes * effective_slices, 512),
        },
    }

    noise_list = generate_noise_tensors(num_volumes, depth, image_size, device, seed)

    # Test configs: baseline and diffrs
    configs = [
        EvalConfig("baseline_003", "baseline", 3),
        EvalConfig("diffrs_003", "diffrs", 3, iter_warmup=2),
    ]

    all_results: list[EvalResult] = []
    for eval_cfg in configs:
        logger.info("[%s] %s", eval_cfg.name, eval_cfg.label)

        volumes, total_nfe, wall_time = generate_volumes(
            model, strategy, noise_list, cond_list, eval_cfg, device,
            diffrs_discriminator=discriminator if eval_cfg.method == "diffrs" else None,
        )

        split_metrics = compute_all_metrics(volumes, ref_features, device, trim_slices)

        result = EvalResult(
            name=eval_cfg.name, method=eval_cfg.method, steps=eval_cfg.steps,
            nfe_total=total_nfe, nfe_per_volume=total_nfe / num_volumes,
            wall_time_s=wall_time, time_per_volume_s=wall_time / num_volumes,
            num_volumes=num_volumes,
            metrics={s: asdict(m) for s, m in split_metrics.items()},
        )
        all_results.append(result)

    print_results_table(all_results, primary_split='fake_split')
    save_results_json(all_results, output_dir / "results.json")
    save_results_csv(all_results, output_dir / "results.csv")

    # Verify outputs
    loaded = load_results_json(output_dir / "results.json")
    assert len(loaded) == len(configs)

    logger.info("=== SMOKE TEST PASSED (%s) ===", output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DiffRS against baseline generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--bravo-model', default=None,
                        help='Path to trained bravo model checkpoint')
    parser.add_argument('--diffrs-head', default=None,
                        help='Path to trained DiffRS head checkpoint (enables DiffRS evaluation)')
    parser.add_argument('--data-root', default=None,
                        help='Root of dataset (e.g., /path/to/brainmetshare-3)')
    parser.add_argument('--output-dir', default='results/eval_diffrs',
                        help='Output directory')
    parser.add_argument('--num-volumes', type=int, default=25,
                        help='Volumes per configuration (default: 25)')
    parser.add_argument('--cond-split', default='val',
                        help='Split for conditioning seg masks (default: val)')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Image H/W (auto-detected from checkpoint)')
    parser.add_argument('--depth', type=int, default=None,
                        help='Generation depth (auto-detected from checkpoint)')
    parser.add_argument('--trim-slices', type=int, default=10,
                        help='Slices to trim from end (default: 10)')
    parser.add_argument('--fov-mm', type=float, default=240.0,
                        help='Field of view in mm (default: 240.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer step counts')
    parser.add_argument('--resume', action='store_true',
                        help='Resume: skip configs with existing results')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Smoke test with tiny dummy model')
    # DiffRS hyperparameters
    parser.add_argument('--rej-percentile', type=float, default=0.75,
                        help='DiffRS rejection percentile gamma (default: 0.75)')
    parser.add_argument('--backsteps', type=int, default=1,
                        help='DiffRS backsteps on rejection (default: 1)')
    parser.add_argument('--max-iter', type=int, default=999999,
                        help='DiffRS max NFE per sample (default: 999999)')
    parser.add_argument('--iter-warmup', type=int, default=10,
                        help='DiffRS warmup iterations (default: 10)')
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()
        return

    if not args.bravo_model:
        parser.error("--bravo-model is required (unless --smoke-test)")
    if not args.data_root:
        parser.error("--data-root is required (unless --smoke-test)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = output_dir / "results.json"
    results_csv_path = output_dir / "results.csv"

    # ── Auto-detect dimensions from checkpoint ───────────────────────────
    logger.info("Loading checkpoint metadata...")
    ckpt = torch.load(args.bravo_model, map_location='cpu', weights_only=False)
    ckpt_cfg = ckpt.get('config', {})
    if hasattr(ckpt_cfg, 'model'):
        model_cfg = ckpt_cfg.model
        image_size = args.image_size or getattr(model_cfg, 'image_size', 256)
        depth = args.depth or getattr(model_cfg, 'depth_size', 160)
    else:
        image_size = args.image_size or 256
        depth = args.depth or 160
    del ckpt

    voxel_size = (args.fov_mm / image_size, args.fov_mm / image_size, 1.0)

    # ── Build evaluation configs ─────────────────────────────────────────
    has_diffrs = args.diffrs_head is not None
    eval_configs = build_eval_configs(
        has_diffrs=has_diffrs,
        quick=args.quick,
        rej_percentile=args.rej_percentile,
        backsteps=args.backsteps,
        max_iter=args.max_iter,
        iter_warmup=args.iter_warmup,
    )

    # ── Save experiment config ───────────────────────────────────────────
    experiment_config = {
        'bravo_model': str(Path(args.bravo_model).resolve()),
        'diffrs_head': str(Path(args.diffrs_head).resolve()) if args.diffrs_head else None,
        'data_root': str(Path(args.data_root).resolve()),
        'cond_split': args.cond_split,
        'num_volumes': args.num_volumes,
        'image_size': image_size,
        'depth': depth,
        'trim_slices': args.trim_slices,
        'seed': args.seed,
        'has_diffrs': has_diffrs,
        'rej_percentile': args.rej_percentile,
        'backsteps': args.backsteps,
        'max_iter': args.max_iter,
        'iter_warmup': args.iter_warmup,
        'num_configs': len(eval_configs),
        'configs': [{'name': c.name, 'method': c.method, 'steps': c.steps}
                    for c in eval_configs],
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(experiment_config, f, indent=2)

    logger.info("=" * 70)
    logger.info("DiffRS Evaluation")
    logger.info("=" * 70)
    logger.info("Model: %s", args.bravo_model)
    logger.info("DiffRS head: %s", args.diffrs_head or "NONE (baseline only)")
    logger.info("Volume: %dx%dx%d (trim %d)", image_size, image_size, depth, args.trim_slices)
    logger.info("Volumes per config: %d", args.num_volumes)
    logger.info("Configs: %d (%d baseline + %d DiffRS)",
                len(eval_configs),
                sum(1 for c in eval_configs if c.method == "baseline"),
                sum(1 for c in eval_configs if c.method == "diffrs"))
    logger.info("Seed: %d", args.seed)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 70)

    # ── Discover splits ──────────────────────────────────────────────────
    data_root = Path(args.data_root)
    logger.info("Discovering dataset splits in %s...", data_root)
    splits = discover_splits(data_root)

    if args.cond_split not in splits:
        raise ValueError(
            f"Conditioning split '{args.cond_split}' not found. "
            f"Available: {list(splits.keys())}"
        )

    # ── Load conditioning masks ──────────────────────────────────────────
    logger.info("Loading %d conditioning masks from '%s'...", args.num_volumes, args.cond_split)
    cond_list = load_conditioning(splits[args.cond_split], args.num_volumes, depth)

    logger.info("Saving conditioning masks...")
    save_conditioning(cond_list, output_dir, voxel_size, args.trim_slices)

    # ── Extract/cache reference features ─────────────────────────────────
    logger.info("Preparing reference features...")
    cache_dir = output_dir / "reference_features"
    ref_features = get_or_cache_reference_features(
        splits, cache_dir, device, depth, args.trim_slices, image_size,
    )

    # ── Load bravo model ─────────────────────────────────────────────────
    from medgen.diffusion import RFlowStrategy, load_diffusion_model

    logger.info("Loading bravo model...")
    bravo_model = load_diffusion_model(
        args.bravo_model, device=device,
        in_channels=2, out_channels=1, compile_model=False, spatial_dims=3,
    )

    # ── Setup strategy ───────────────────────────────────────────────────
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000,
        image_size=image_size,
        depth_size=depth,
        spatial_dims=3,
    )

    # ── Load DiffRS head if provided ─────────────────────────────────────
    diffrs_disc = None
    if has_diffrs:
        from medgen.diffusion.diffrs import DiffRSDiscriminator, load_diffrs_head

        logger.info("Loading DiffRS head from %s...", args.diffrs_head)
        head = load_diffrs_head(args.diffrs_head, device)
        diffrs_disc = DiffRSDiscriminator(bravo_model, head, device)
        logger.info("DiffRS discriminator loaded")

    # ── Pre-generate noise tensors ───────────────────────────────────────
    logger.info("Pre-generating %d noise tensors (seed=%d)...", args.num_volumes, args.seed)
    noise_list = generate_noise_tensors(args.num_volumes, depth, image_size, device, args.seed)

    # ── Load existing results if resuming ────────────────────────────────
    existing_results: list[EvalResult] = []
    completed_names: set[str] = set()
    if args.resume and results_json_path.exists():
        existing_results = load_results_json(results_json_path)
        completed_names = {r.name for r in existing_results}
        logger.info("Resuming: %d configs already completed", len(existing_results))

    # ── Evaluate each config ─────────────────────────────────────────────
    all_results = list(existing_results)
    total_start = time.time()

    for i, eval_cfg in enumerate(eval_configs):
        if eval_cfg.name in completed_names:
            logger.info("[%d/%d] SKIP %s (done)", i + 1, len(eval_configs), eval_cfg.name)
            continue

        logger.info("\n%s", "=" * 70)
        logger.info("[%d/%d] %s", i + 1, len(eval_configs), eval_cfg.label)
        logger.info("%s", "=" * 70)

        # Clear DiffRS threshold cache between configs with different step counts
        if hasattr(strategy, '_diffrs_adaptive_cache'):
            strategy._diffrs_adaptive_cache.clear()

        # Generate
        logger.info("  Generating %d volumes...", args.num_volumes)
        volumes, total_nfe, wall_time = generate_volumes(
            bravo_model, strategy, noise_list, cond_list,
            eval_cfg, device,
            diffrs_discriminator=diffrs_disc if eval_cfg.method == "diffrs" else None,
        )
        nfe_per_vol = total_nfe / args.num_volumes
        logger.info("  Done: %.1fs total, %.1fs/vol, NFE=%d (%.0f/vol)",
                     wall_time, wall_time / args.num_volumes, total_nfe, nfe_per_vol)

        # Save volumes
        vol_dir = output_dir / "generated" / eval_cfg.dir_name
        logger.info("  Saving volumes to %s...", vol_dir)
        save_volumes(volumes, cond_list, vol_dir, voxel_size, args.trim_slices)

        # Compute metrics
        logger.info("  Computing metrics...")
        split_metrics = compute_all_metrics(volumes, ref_features, device, args.trim_slices)

        result = EvalResult(
            name=eval_cfg.name,
            method=eval_cfg.method,
            steps=eval_cfg.steps,
            nfe_total=total_nfe,
            nfe_per_volume=nfe_per_vol,
            wall_time_s=wall_time,
            time_per_volume_s=wall_time / args.num_volumes,
            num_volumes=args.num_volumes,
            metrics={s: asdict(m) for s, m in split_metrics.items()},
        )
        all_results.append(result)

        # Save intermediate results (for resume)
        save_results_json(all_results, results_json_path)
        save_results_csv(all_results, results_csv_path)

        del volumes
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # ── Final output ─────────────────────────────────────────────────────
    print_results_table(all_results, primary_split='all')

    for split_name in splits:
        print_results_table(all_results, primary_split=split_name)

    save_results_json(all_results, results_json_path)
    save_results_csv(all_results, results_csv_path)

    logger.info("\nTotal evaluation time: %.1f hours", total_time / 3600)
    logger.info("Results: %s", results_json_path)
    logger.info("CSV:     %s", results_csv_path)
    logger.info("Volumes: %s", output_dir / "generated")
    logger.info("Done!")


if __name__ == "__main__":
    main()
