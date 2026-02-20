#!/usr/bin/env python3
"""Evaluate Restart Sampling against baseline Euler generation.

Generates 3D BRAVO volumes with baseline Euler sampler and Restart Sampling,
using identical noise and real segmentation masks as conditioning. Computes
FID, KID, CMMD against reference data (per-split and combined) to measure
whether Restart Sampling improves generation quality.

Restart Sampling (Xu et al., NeurIPS 2023) alternates between adding forward
noise and running backward ODE within a restart interval [tmin, tmax] to
contract accumulated discretization errors. No auxiliary model needed.

Experimental design:
  - Baseline: Standard Euler sampler at multiple step counts [10, 25, 50]
  - Restart: Same baseline steps + restart iterations (higher NFE, hopefully better quality)
  - All configs use the SAME noise tensors and SAME real seg masks
  - NFE counted per config for fair comparison

Usage:
    # Full evaluation
    python -m medgen.scripts.eval_restart \\
        --bravo-model runs/checkpoint_bravo.pt \\
        --data-root ~/data/brainmetshare-3 \\
        --num-volumes 25 --output-dir results/eval_restart

    # Quick test (fewer configs)
    python -m medgen.scripts.eval_restart \\
        --bravo-model runs/checkpoint_bravo.pt \\
        --data-root ~/data/brainmetshare-3 \\
        --num-volumes 5 --output-dir results/eval_restart_quick --quick

    # Resume interrupted run
    python -m medgen.scripts.eval_restart \\
        --bravo-model runs/checkpoint_bravo.pt \\
        --data-root ~/data/brainmetshare-3 \\
        --num-volumes 25 --output-dir results/eval_restart --resume

    # Smoke test (no checkpoint needed)
    python -m medgen.scripts.eval_restart --smoke-test
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
QUICK_STEPS = [10, 25]

# Default restart hyperparameter grid (25 main steps)
RESTART_GRID = [
    # (K, n_restart, tmin, tmax)
    (1, 3,  0.1,  0.3),   # Light: NFE = 25 + 3 = 28
    (1, 5,  0.1,  0.3),   # Medium: NFE = 25 + 5 = 30
    (2, 3,  0.1,  0.3),   # K=2: NFE = 25 + 6 = 31
    (2, 5,  0.1,  0.3),   # Standard: NFE = 25 + 10 = 35
    (2, 10, 0.1,  0.3),   # Heavy: NFE = 25 + 20 = 45
    (1, 5,  0.05, 0.2),   # Narrow interval
    (1, 5,  0.1,  0.5),   # Wide interval
]

QUICK_RESTART_GRID = [
    (1, 5,  0.1, 0.3),    # Medium
    (2, 5,  0.1, 0.3),    # Standard
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    """A single evaluation configuration."""
    name: str          # e.g., "baseline_025", "restart_025_K2_n05"
    method: str        # "baseline" or "restart"
    steps: int         # Number of main Euler steps

    # Restart hyperparameters (only used when method="restart")
    tmin: float = 0.1
    tmax: float = 0.3
    K: int = 2
    n_restart: int = 5

    @property
    def dir_name(self) -> str:
        return self.name

    @property
    def label(self) -> str:
        if self.method == "restart":
            return (f"Restart/{self.steps} "
                    f"(K={self.K}, n={self.n_restart}, "
                    f"[{self.tmin:.2f},{self.tmax:.2f}])")
        return f"Euler/{self.steps}"

    @property
    def expected_nfe(self) -> int:
        """Expected NFE per volume."""
        if self.method == "restart":
            return self.steps + self.K * self.n_restart
        return self.steps


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
    # Restart hyperparams (None for baseline)
    tmin: float | None
    tmax: float | None
    K: int | None
    n_restart: int | None
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
    quick: bool = False,
    tmin: float = 0.1,
    tmax: float = 0.3,
    K: int = 2,
    n_restart: int = 5,
) -> list[EvalConfig]:
    """Build evaluation configurations.

    Includes baseline configs + restart grid configs. The CLI args (tmin, tmax,
    K, n_restart) are only used when --quick mode is off and override the grid.
    """
    steps_list = QUICK_STEPS if quick else BASELINE_STEPS
    configs = []

    # Baseline configs
    for steps in steps_list:
        configs.append(EvalConfig(
            name=f"baseline_{steps:03d}",
            method="baseline",
            steps=steps,
        ))

    # Restart configs
    restart_grid = QUICK_RESTART_GRID if quick else RESTART_GRID
    main_steps = 25  # Restart always uses 25 main steps (our best baseline)

    for K_g, n_g, tmin_g, tmax_g in restart_grid:
        configs.append(EvalConfig(
            name=f"restart_{main_steps:03d}_K{K_g}_n{n_g:02d}_t{tmin_g:.2f}_{tmax_g:.2f}",
            method="restart",
            steps=main_steps,
            tmin=tmin_g,
            tmax=tmax_g,
            K=K_g,
            n_restart=n_g,
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
) -> tuple[list[np.ndarray], int, float]:
    """Generate BRAVO volumes for one evaluation configuration.

    Args:
        model: BRAVO model (will be wrapped with NFECounter).
        strategy: RFlowStrategy instance.
        noise_list: Pre-generated noise tensors.
        cond_list: (patient_id, seg_tensor) pairs.
        eval_cfg: Evaluation configuration.
        device: CUDA device.

    Returns:
        (volumes_list, total_nfe, wall_time_seconds)
    """
    counter = NFECounter(model)
    counter.reset()

    # Build restart kwargs if needed
    gen_kwargs: dict = {}
    if eval_cfg.method == "restart":
        gen_kwargs['restart_config'] = {
            'tmin': eval_cfg.tmin,
            'tmax': eval_cfg.tmax,
            'K': eval_cfg.K,
            'n_restart': eval_cfg.n_restart,
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
    print("\n" + "=" * 130)
    print(f"Restart Sampling Evaluation Results (vs '{primary_split}' reference)")
    print("=" * 130)
    print(f"{'Config':<50} {'Method':<10} {'Steps':>5} {'NFE/vol':>8} {'Time/vol':>9} "
          f"{'FID':>10} {'KID':>14} {'CMMD':>10}")
    print("-" * 130)

    # Sort by NFE for easy comparison
    sorted_results = sorted(results, key=lambda r: r.nfe_per_volume)

    for r in sorted_results:
        m = r.metrics.get(primary_split, {})
        if not m:
            continue
        kid_str = f"{m['kid_mean']:.6f}\u00b1{m['kid_std']:.4f}"
        print(f"{r.name:<50} {r.method:<10} {r.steps:>5} {r.nfe_per_volume:>8.0f} "
              f"{r.time_per_volume_s:>8.1f}s "
              f"{m['fid']:>10.2f} {kid_str:>14} {m['cmmd']:>10.6f}")

    # Show improvement: best restart vs best baseline at same step count
    baselines = [r for r in results if r.method == "baseline"]
    restarts = [r for r in results if r.method == "restart"]

    if baselines and restarts:
        # Compare against baseline_025 (our standard)
        baseline_25 = next((b for b in baselines if b.steps == 25), None)
        if baseline_25:
            bm = baseline_25.metrics.get(primary_split, {})
            if bm:
                print()
                print(f"  {'Baseline 25-step reference:':<50} "
                      f"FID={bm['fid']:.2f}  KID={bm['kid_mean']:.6f}  CMMD={bm['cmmd']:.6f}")
                for r in sorted(restarts, key=lambda x: x.nfe_per_volume):
                    rm = r.metrics.get(primary_split, {})
                    if rm:
                        fid_delta = rm['fid'] - bm['fid']
                        kid_delta = rm['kid_mean'] - bm['kid_mean']
                        cmmd_delta = rm['cmmd'] - bm['cmmd']
                        nfe_ratio = r.nfe_per_volume / max(baseline_25.nfe_per_volume, 1)
                        print(f"  {r.name:<50} "
                              f"FID={fid_delta:>+.2f}  KID={kid_delta:>+.6f}  "
                              f"CMMD={cmmd_delta:>+.6f}  ({nfe_ratio:.1f}x NFE)")

    print("=" * 130)


def save_results_csv(results: list[EvalResult], path: Path) -> None:
    """Save results as CSV for thesis tables/plots."""
    all_splits = set()
    for r in results:
        all_splits.update(r.metrics.keys())
    all_splits = sorted(all_splits)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['name', 'method', 'steps', 'nfe_per_volume',
                  'wall_time_s', 'time_per_volume_s', 'num_volumes',
                  'tmin', 'tmax', 'K', 'n_restart']
        for split in all_splits:
            header.extend([f'fid_{split}', f'kid_mean_{split}',
                          f'kid_std_{split}', f'cmmd_{split}'])
        writer.writerow(header)

        for r in sorted(results, key=lambda x: (x.method, x.nfe_per_volume)):
            row = [r.name, r.method, r.steps, f"{r.nfe_per_volume:.1f}",
                   f"{r.wall_time_s:.1f}", f"{r.time_per_volume_s:.1f}", r.num_volumes,
                   r.tmin if r.tmin is not None else '',
                   r.tmax if r.tmax is not None else '',
                   r.K if r.K is not None else '',
                   r.n_restart if r.n_restart is not None else '']
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(tempfile.mkdtemp(prefix="eval_restart_smoke_"))

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

    # Test configs: baseline and restart (10 steps needed for fine-enough grid)
    configs = [
        EvalConfig("baseline_010", "baseline", 10),
        EvalConfig("restart_010_K1_n02", "restart", 10, tmin=0.1, tmax=0.3, K=1, n_restart=2),
    ]

    all_results: list[EvalResult] = []
    for eval_cfg in configs:
        logger.info("[%s] %s", eval_cfg.name, eval_cfg.label)

        volumes, total_nfe, wall_time = generate_volumes(
            model, strategy, noise_list, cond_list, eval_cfg, device,
        )

        split_metrics = compute_all_metrics(volumes, ref_features, device, trim_slices)

        result = EvalResult(
            name=eval_cfg.name, method=eval_cfg.method, steps=eval_cfg.steps,
            nfe_total=total_nfe, nfe_per_volume=total_nfe / num_volumes,
            wall_time_s=wall_time, time_per_volume_s=wall_time / num_volumes,
            num_volumes=num_volumes,
            tmin=eval_cfg.tmin if eval_cfg.method == "restart" else None,
            tmax=eval_cfg.tmax if eval_cfg.method == "restart" else None,
            K=eval_cfg.K if eval_cfg.method == "restart" else None,
            n_restart=eval_cfg.n_restart if eval_cfg.method == "restart" else None,
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
        description="Evaluate Restart Sampling against baseline generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--bravo-model', default=None,
                        help='Path to trained bravo model checkpoint')
    parser.add_argument('--data-root', default=None,
                        help='Root of dataset (e.g., /path/to/brainmetshare-3)')
    parser.add_argument('--output-dir', default='results/eval_restart',
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
                        help='Quick mode: fewer configs')
    parser.add_argument('--resume', action='store_true',
                        help='Resume: skip configs with existing results')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Smoke test with tiny dummy model')
    # Restart hyperparameters (override grid for single-config runs)
    parser.add_argument('--tmin', type=float, default=0.1,
                        help='Restart interval start as fraction of T (default: 0.1)')
    parser.add_argument('--tmax', type=float, default=0.3,
                        help='Restart interval end as fraction of T (default: 0.3)')
    parser.add_argument('--restart-K', type=int, default=2,
                        help='Number of restart iterations (default: 2)')
    parser.add_argument('--n-restart', type=int, default=5,
                        help='Euler steps per restart backward pass (default: 5)')
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
    eval_configs = build_eval_configs(
        quick=args.quick,
        tmin=args.tmin,
        tmax=args.tmax,
        K=args.restart_K,
        n_restart=args.n_restart,
    )

    # ── Save experiment config ───────────────────────────────────────────
    experiment_config = {
        'bravo_model': str(Path(args.bravo_model).resolve()),
        'data_root': str(Path(args.data_root).resolve()),
        'cond_split': args.cond_split,
        'num_volumes': args.num_volumes,
        'image_size': image_size,
        'depth': depth,
        'trim_slices': args.trim_slices,
        'seed': args.seed,
        'num_configs': len(eval_configs),
        'configs': [{'name': c.name, 'method': c.method, 'steps': c.steps,
                     'expected_nfe': c.expected_nfe}
                    for c in eval_configs],
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(experiment_config, f, indent=2)

    logger.info("=" * 70)
    logger.info("Restart Sampling Evaluation")
    logger.info("=" * 70)
    logger.info("Model: %s", args.bravo_model)
    logger.info("Volume: %dx%dx%d (trim %d)", image_size, image_size, depth, args.trim_slices)
    logger.info("Volumes per config: %d", args.num_volumes)
    logger.info("Configs: %d (%d baseline + %d restart)",
                len(eval_configs),
                sum(1 for c in eval_configs if c.method == "baseline"),
                sum(1 for c in eval_configs if c.method == "restart"))
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
        logger.info("[%d/%d] %s  (expected NFE=%d/vol)",
                     i + 1, len(eval_configs), eval_cfg.label, eval_cfg.expected_nfe)
        logger.info("%s", "=" * 70)

        # Generate
        logger.info("  Generating %d volumes...", args.num_volumes)
        volumes, total_nfe, wall_time = generate_volumes(
            bravo_model, strategy, noise_list, cond_list,
            eval_cfg, device,
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
            tmin=eval_cfg.tmin if eval_cfg.method == "restart" else None,
            tmax=eval_cfg.tmax if eval_cfg.method == "restart" else None,
            K=eval_cfg.K if eval_cfg.method == "restart" else None,
            n_restart=eval_cfg.n_restart if eval_cfg.method == "restart" else None,
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
