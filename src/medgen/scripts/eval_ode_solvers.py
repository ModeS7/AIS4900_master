#!/usr/bin/env python3
"""Evaluate ODE solvers for RFlow generation quality.

Generates 3D BRAVO volumes with different ODE solvers and step counts using
identical noise and real segmentation masks as conditioning. Computes FID, KID,
CMMD against reference data (per-split and combined) to determine optimal solver
configurations for quality and efficiency.

Experimental design:
  - Fixed-step solvers (euler, midpoint, heun2, heun3, rk4) tested at 5 step counts
  - Adaptive solvers (fehlberg2, bosh3, dopri5, dopri8) tested at 5 tolerances
  - All configs use the SAME noise tensors and SAME real seg masks
  - NFE (Number of Function Evaluations) counted per config for fair comparison

NFE per step:
  euler=1, midpoint=2, heun2=2, heun3=3, rk4=4, adaptive=varies

Usage:
    # Full evaluation
    python -m medgen.scripts.eval_ode_solvers \\
        --bravo-model runs/checkpoint_bravo.pt \\
        --data-root ~/NTNU/MedicalDataSets/brainmetshare-3 \\
        --num-volumes 25 --output-dir eval_ode_solvers

    # Quick test (fewer configs)
    python -m medgen.scripts.eval_ode_solvers \\
        --bravo-model runs/checkpoint_bravo.pt \\
        --data-root ~/NTNU/MedicalDataSets/brainmetshare-3 \\
        --num-volumes 5 --output-dir eval_ode_quick --quick

    # Resume interrupted run
    python -m medgen.scripts.eval_ode_solvers \\
        --bravo-model runs/checkpoint_bravo.pt \\
        --data-root ~/NTNU/MedicalDataSets/brainmetshare-3 \\
        --num-volumes 25 --output-dir eval_ode_solvers --resume
"""
import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

ADAPTIVE_SOLVERS = frozenset({'fehlberg2', 'bosh3', 'dopri5', 'dopri8'})

NFE_PER_STEP: dict[str, int] = {
    'euler': 1,
    'midpoint': 2,
    'heun2': 2,
    'heun3': 3,
    'rk4': 4,
}

FIXED_STEP_SOLVERS = ['euler', 'midpoint', 'heun2', 'heun3', 'rk4']
ADAPTIVE_SOLVER_LIST = ['fehlberg2', 'bosh3', 'dopri5', 'dopri8']

FIXED_STEPS = [5, 10, 25, 50, 100]
ADAPTIVE_TOLS = [1e-2, 1e-3, 1e-4, 1e-5]

# Quick mode: reduced grid for fast iteration
QUICK_FIXED_STEPS = [10, 50]
QUICK_ADAPTIVE_TOLS = [1e-3, 1e-5]


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolverConfig:
    """A single ODE solver configuration to evaluate."""
    solver: str
    steps: int | None = None    # For fixed-step solvers
    atol: float = 1e-5          # For adaptive solvers
    rtol: float = 1e-5          # For adaptive solvers

    @property
    def is_adaptive(self) -> bool:
        return self.solver in ADAPTIVE_SOLVERS

    @property
    def dir_name(self) -> str:
        """Directory name for saving volumes."""
        if self.is_adaptive:
            return f"{self.solver}_tol{self.atol:.0e}"
        return f"{self.solver}_steps{self.steps:03d}"

    @property
    def label(self) -> str:
        """Human-readable label for tables."""
        if self.is_adaptive:
            return f"{self.solver}(tol={self.atol:.0e})"
        return f"{self.solver}/{self.steps}"

    @property
    def expected_nfe_per_volume(self) -> int | None:
        """Expected NFE per volume (None for adaptive)."""
        if self.is_adaptive:
            return None
        return self.steps * NFE_PER_STEP[self.solver]


@dataclass
class SplitMetrics:
    """Metrics computed against one reference split."""
    fid: float
    kid_mean: float
    kid_std: float
    cmmd: float


@dataclass
class EvalResult:
    """Full result for one solver configuration."""
    solver: str
    steps: int | None
    atol: float | None
    rtol: float | None
    dir_name: str
    nfe_total: int                            # Total NFE across all volumes
    nfe_per_volume: float                     # Average NFE per volume
    wall_time_s: float                        # Total generation time
    time_per_volume_s: float                  # Average time per volume
    num_volumes: int
    metrics: dict[str, dict[str, float]]      # split -> {fid, kid_mean, kid_std, cmmd}


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
# Build solver configs
# ═══════════════════════════════════════════════════════════════════════════════

def build_solver_configs(quick: bool = False) -> list[SolverConfig]:
    """Build the full grid of solver configurations."""
    configs = []
    steps_list = QUICK_FIXED_STEPS if quick else FIXED_STEPS
    tols_list = QUICK_ADAPTIVE_TOLS if quick else ADAPTIVE_TOLS

    for solver in FIXED_STEP_SOLVERS:
        for steps in steps_list:
            configs.append(SolverConfig(solver=solver, steps=steps))

    for solver in ADAPTIVE_SOLVER_LIST:
        for tol in tols_list:
            configs.append(SolverConfig(solver=solver, atol=tol, rtol=tol))

    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def discover_splits(data_root: Path) -> dict[str, Path]:
    """Auto-discover dataset splits (directories containing */bravo.nii.gz).

    Returns:
        Dict of split_name -> split_path, e.g. {'train': Path(...), 'val': Path(...)}.
    """
    splits = {}
    for subdir in sorted(data_root.iterdir()):
        if not subdir.is_dir():
            continue
        bravo_files = list(subdir.glob("*/bravo.nii.gz"))
        if bravo_files:
            splits[subdir.name] = subdir
            logger.info(f"  Found split '{subdir.name}': {len(bravo_files)} volumes")
    if not splits:
        raise FileNotFoundError(f"No splits with bravo.nii.gz found in {data_root}")
    return splits


def load_conditioning(
    val_dir: Path,
    num_volumes: int,
    depth: int,
) -> list[tuple[str, torch.Tensor]]:
    """Load real segmentation masks from validation set as conditioning.

    Args:
        val_dir: Validation directory with patient_id/seg.nii.gz.
        num_volumes: Number of conditioning masks to load.
        depth: Target depth (pad if needed).

    Returns:
        List of (patient_id, seg_tensor [1, 1, D, H, W]) tuples.
    """
    seg_files = sorted(val_dir.glob("*/seg.nii.gz"))
    if len(seg_files) < num_volumes:
        raise ValueError(
            f"Requested {num_volumes} volumes but val set only has "
            f"{len(seg_files)} seg masks in {val_dir}"
        )
    seg_files = seg_files[:num_volumes]

    result = []
    for seg_path in seg_files:
        patient_id = seg_path.parent.name
        seg_np = nib.load(str(seg_path)).get_fdata().astype(np.float32)

        # Binarize
        seg_np = (seg_np > 0.5).astype(np.float32)

        # [H, W, D] -> [D, H, W]
        seg_np = np.transpose(seg_np, (2, 0, 1))

        # Pad to target depth
        d = seg_np.shape[0]
        if d < depth:
            pad = np.zeros((depth - d, seg_np.shape[1], seg_np.shape[2]), dtype=np.float32)
            seg_np = np.concatenate([seg_np, pad], axis=0)
        elif d > depth:
            seg_np = seg_np[:depth]

        tensor = torch.from_numpy(seg_np).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        result.append((patient_id, tensor))
        logger.info(f"  Loaded seg mask: {patient_id} (tumor pct={seg_np.mean():.4%})")

    return result


def save_conditioning(
    cond_list: list[tuple[str, torch.Tensor]],
    output_dir: Path,
    voxel_size: tuple[float, float, float],
    trim_slices: int,
) -> None:
    """Save conditioning masks to output directory for reference."""
    cond_dir = output_dir / "conditioning"
    cond_dir.mkdir(parents=True, exist_ok=True)
    for i, (patient_id, seg_tensor) in enumerate(cond_list):
        seg_np = seg_tensor[0, 0].numpy()  # [D, H, W]
        if trim_slices > 0:
            seg_np = seg_np[:-trim_slices]
        seg_np = np.transpose(seg_np, (1, 2, 0))  # [H, W, D]
        affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])
        nifti = nib.Nifti1Image(seg_np, affine)
        nib.save(nifti, str(cond_dir / f"{i:02d}_{patient_id}_seg.nii.gz"))


# ═══════════════════════════════════════════════════════════════════════════════
# Reference feature extraction and caching
# ═══════════════════════════════════════════════════════════════════════════════

def extract_split_features(
    split_dir: Path,
    extractor: nn.Module,
    depth: int,
    trim_slices: int,
    image_size: int,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Extract features from all volumes in a split, one volume at a time.

    Memory-efficient: loads one volume, extracts slice features, frees volume.

    Args:
        split_dir: Directory with patient_id/bravo.nii.gz.
        extractor: Feature extractor (ResNet50Features or BiomedCLIPFeatures).
        depth: Generation depth (for padding).
        trim_slices: Number of end slices to exclude (padding slices).
        image_size: Expected H/W dimension.
        chunk_size: Slices per feature extraction batch.

    Returns:
        Feature tensor [total_slices, feat_dim].
    """
    bravo_files = sorted(split_dir.glob("*/bravo.nii.gz"))
    all_features = []
    effective_depth = depth - trim_slices

    for path in bravo_files:
        vol = nib.load(str(path)).get_fdata().astype(np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)

        # [H, W, D_orig] -> [D, H, W]
        vol = np.transpose(vol, (2, 0, 1))

        # Trim or pad to effective_depth
        d = vol.shape[0]
        if d > effective_depth:
            vol = vol[:effective_depth]
        elif d < effective_depth:
            pad = np.zeros((effective_depth - d, vol.shape[1], vol.shape[2]), dtype=np.float32)
            vol = np.concatenate([vol, pad], axis=0)

        # Resize if needed (model may use different resolution than data)
        # vol is [D, H, W], we need [D, 1, H, W] for feature extraction
        slices_tensor = torch.from_numpy(vol).unsqueeze(1)  # [D, 1, H, W]

        # Extract features in chunks
        for start in range(0, slices_tensor.shape[0], chunk_size):
            end = min(start + chunk_size, slices_tensor.shape[0])
            chunk = slices_tensor[start:end]
            features = extractor.extract_features(chunk)
            all_features.append(features.cpu())
            del features

        del vol, slices_tensor

    return torch.cat(all_features, dim=0)


def get_or_cache_reference_features(
    splits: dict[str, Path],
    cache_dir: Path,
    device: torch.device,
    depth: int,
    trim_slices: int,
    image_size: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract and cache reference features for all splits.

    Returns:
        Dict: split_name -> {'resnet': Tensor, 'clip': Tensor}
        Also includes 'all' key with concatenated features.
    """
    from medgen.metrics.feature_extractors import BiomedCLIPFeatures, ResNet50Features

    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_features: dict[str, dict[str, torch.Tensor]] = {}

    # Check if all features are cached
    all_cached = True
    for split_name in splits:
        resnet_path = cache_dir / f"{split_name}_resnet.pt"
        clip_path = cache_dir / f"{split_name}_clip.pt"
        if not resnet_path.exists() or not clip_path.exists():
            all_cached = False
            break

    if all_cached:
        logger.info("Loading cached reference features...")
        for split_name in splits:
            ref_features[split_name] = {
                'resnet': torch.load(cache_dir / f"{split_name}_resnet.pt", weights_only=True),
                'clip': torch.load(cache_dir / f"{split_name}_clip.pt", weights_only=True),
            }
            logger.info(f"  {split_name}: resnet={ref_features[split_name]['resnet'].shape}, "
                         f"clip={ref_features[split_name]['clip'].shape}")
    else:
        logger.info("Extracting reference features (this is a one-time cost)...")

        # Extract ResNet features
        logger.info("  Loading ResNet50...")
        resnet = ResNet50Features(device, compile_model=False)
        for split_name, split_dir in splits.items():
            logger.info(f"  Extracting ResNet50 features for '{split_name}'...")
            features = extract_split_features(
                split_dir, resnet, depth, trim_slices, image_size,
            )
            torch.save(features, cache_dir / f"{split_name}_resnet.pt")
            ref_features.setdefault(split_name, {})['resnet'] = features
            logger.info(f"    {split_name}: {features.shape}")
        resnet.unload()

        # Extract CLIP features
        logger.info("  Loading BiomedCLIP...")
        clip = BiomedCLIPFeatures(device, compile_model=False)
        for split_name, split_dir in splits.items():
            logger.info(f"  Extracting BiomedCLIP features for '{split_name}'...")
            features = extract_split_features(
                split_dir, clip, depth, trim_slices, image_size,
            )
            torch.save(features, cache_dir / f"{split_name}_clip.pt")
            ref_features[split_name]['clip'] = features
            logger.info(f"    {split_name}: {features.shape}")
        clip.unload()

    # Build 'all' by concatenating
    all_resnet = torch.cat([ref_features[s]['resnet'] for s in splits], dim=0)
    all_clip = torch.cat([ref_features[s]['clip'] for s in splits], dim=0)
    ref_features['all'] = {'resnet': all_resnet, 'clip': all_clip}
    logger.info(f"  all (combined): resnet={all_resnet.shape}, clip={all_clip.shape}")

    return ref_features


# ═══════════════════════════════════════════════════════════════════════════════
# Noise generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_noise_tensors(
    num: int,
    depth: int,
    image_size: int,
    device: torch.device,
    seed: int,
) -> list[torch.Tensor]:
    """Pre-generate deterministic noise tensors (shared across all configs).

    Each noise tensor is generated with a unique sub-seed for reproducibility.

    Returns:
        List of [1, 1, D, H, W] tensors on the given device.
    """
    noise_list = []
    for i in range(num):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + i)
        noise = torch.randn(1, 1, depth, image_size, image_size,
                             device=device, generator=gen)
        noise_list.append(noise)
    return noise_list


# ═══════════════════════════════════════════════════════════════════════════════
# Volume generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_volumes(
    model: nn.Module,
    strategy,
    noise_list: list[torch.Tensor],
    cond_list: list[tuple[str, torch.Tensor]],
    solver_cfg: SolverConfig,
    device: torch.device,
) -> tuple[list[np.ndarray], int, float]:
    """Generate BRAVO volumes for one solver configuration.

    Args:
        model: BRAVO model (will be wrapped with NFECounter).
        strategy: RFlowStrategy instance (ode_solver will be set).
        noise_list: Pre-generated noise tensors.
        cond_list: (patient_id, seg_tensor) pairs.
        solver_cfg: Solver configuration.
        device: CUDA device.

    Returns:
        (volumes_list, total_nfe, wall_time_seconds)
        volumes_list contains numpy arrays [D, H, W] in [0, 1].
    """
    # Configure strategy
    strategy.ode_solver = solver_cfg.solver
    strategy.ode_atol = solver_cfg.atol
    strategy.ode_rtol = solver_cfg.rtol

    # Wrap for NFE counting
    counter = NFECounter(model)
    counter.reset()

    # For adaptive solvers, num_steps is unused but required by the API
    num_steps = solver_cfg.steps if solver_cfg.steps is not None else 50

    volumes = []
    start_time = time.time()

    for i, (noise, (_patient_id, seg_tensor)) in enumerate(zip(noise_list, cond_list)):
        seg_on_device = seg_tensor.to(device)
        model_input = torch.cat([noise, seg_on_device], dim=1)  # [1, 2, D, H, W]

        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            with torch.no_grad():
                result = strategy.generate(counter, model_input, num_steps, device)

        vol_np = torch.clamp(result[0, 0], 0, 1).cpu().float().numpy()  # [D, H, W]
        volumes.append(vol_np)

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - start_time
            logger.info(f"    {i+1}/{len(noise_list)} volumes "
                         f"({elapsed:.0f}s, NFE={counter.nfe})")

    wall_time = time.time() - start_time
    return volumes, counter.nfe, wall_time


def save_volumes(
    volumes: list[np.ndarray],
    cond_list: list[tuple[str, torch.Tensor]],
    output_dir: Path,
    voxel_size: tuple[float, float, float],
    trim_slices: int,
) -> None:
    """Save generated volumes as NIfTI files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])

    for i, (vol_np, (patient_id, _)) in enumerate(zip(volumes, cond_list)):
        # Trim padded slices
        if trim_slices > 0:
            vol_np = vol_np[:-trim_slices]
        # [D, H, W] -> [H, W, D] for NIfTI
        vol_nifti = np.transpose(vol_np, (1, 2, 0))
        nifti = nib.Nifti1Image(vol_nifti.astype(np.float32), affine)
        nib.save(nifti, str(output_dir / f"{i:02d}_{patient_id}_bravo.nii.gz"))


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction for generated volumes
# ═══════════════════════════════════════════════════════════════════════════════

def extract_generated_features(
    volumes: list[np.ndarray],
    extractor: nn.Module,
    trim_slices: int,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Extract features from generated volumes (list of numpy arrays).

    Args:
        volumes: List of [D, H, W] numpy arrays in [0, 1].
        extractor: Feature extractor.
        trim_slices: Number of end slices to exclude.
        chunk_size: Slices per extraction batch.

    Returns:
        Feature tensor [total_slices, feat_dim].
    """
    all_features = []
    for vol_np in volumes:
        if trim_slices > 0:
            vol_np = vol_np[:-trim_slices]
        # [D, H, W] -> [D, 1, H, W]
        slices_tensor = torch.from_numpy(vol_np).unsqueeze(1)

        for start in range(0, slices_tensor.shape[0], chunk_size):
            end = min(start + chunk_size, slices_tensor.shape[0])
            chunk = slices_tensor[start:end]
            features = extractor.extract_features(chunk)
            all_features.append(features.cpu())
            del features

    return torch.cat(all_features, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    gen_volumes: list[np.ndarray],
    ref_features: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    trim_slices: int,
) -> dict[str, SplitMetrics]:
    """Compute FID, KID, CMMD for generated volumes against all reference splits.

    Args:
        gen_volumes: List of generated [D, H, W] numpy arrays.
        ref_features: Dict of split_name -> {'resnet': Tensor, 'clip': Tensor}.
        device: CUDA device for feature extraction.
        trim_slices: Slices to trim from generated volumes.

    Returns:
        Dict of split_name -> SplitMetrics.
    """
    from medgen.metrics.feature_extractors import BiomedCLIPFeatures, ResNet50Features
    from medgen.metrics.generation import compute_cmmd, compute_fid, compute_kid

    results = {}

    # Extract ResNet features from generated volumes
    logger.info("    Extracting ResNet50 features from generated...")
    resnet = ResNet50Features(device, compile_model=False)
    gen_resnet = extract_generated_features(gen_volumes, resnet, trim_slices)
    logger.info(f"    Generated ResNet features: {gen_resnet.shape}")
    resnet.unload()

    # Extract CLIP features from generated volumes
    logger.info("    Extracting BiomedCLIP features from generated...")
    clip = BiomedCLIPFeatures(device, compile_model=False)
    gen_clip = extract_generated_features(gen_volumes, clip, trim_slices)
    logger.info(f"    Generated CLIP features: {gen_clip.shape}")
    clip.unload()

    # Compute metrics against each split
    for split_name, split_feats in ref_features.items():
        ref_resnet = split_feats['resnet']
        ref_clip = split_feats['clip']

        fid = compute_fid(ref_resnet, gen_resnet)

        # Cap KID subset_size to min available
        min_n = min(ref_resnet.shape[0], gen_resnet.shape[0])
        kid_subset = min(100, min_n)
        kid_mean, kid_std = compute_kid(ref_resnet, gen_resnet, subset_size=kid_subset)

        cmmd = compute_cmmd(ref_clip, gen_clip)

        results[split_name] = SplitMetrics(
            fid=fid, kid_mean=kid_mean, kid_std=kid_std, cmmd=cmmd,
        )
        logger.info(f"    vs {split_name}: FID={fid:.2f}  KID={kid_mean:.6f}  CMMD={cmmd:.6f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Results output
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_table(results: list[EvalResult], primary_split: str = 'all') -> None:
    """Print formatted results table sorted by NFE."""
    print("\n" + "=" * 130)
    print(f"ODE Solver Evaluation Results (vs '{primary_split}' reference)")
    print("=" * 130)
    print(f"{'Config':<28} {'Steps':>5} {'NFE/vol':>8} {'Time/vol':>9} "
          f"{'FID':>10} {'KID':>14} {'CMMD':>10}")
    print("-" * 130)

    sorted_results = sorted(results, key=lambda r: r.nfe_per_volume)

    for r in sorted_results:
        m = r.metrics.get(primary_split, {})
        if not m:
            continue
        steps_str = str(r.steps) if r.steps is not None else "adapt"
        kid_str = f"{m['kid_mean']:.6f}±{m['kid_std']:.4f}"
        label = f"{r.solver}({r.dir_name.split('_', 1)[1]})"
        print(f"{label:<28} {steps_str:>5} {r.nfe_per_volume:>8.0f} "
              f"{r.time_per_volume_s:>8.1f}s "
              f"{m['fid']:>10.2f} {kid_str:>14} {m['cmmd']:>10.6f}")

    print("=" * 130)

    # Group by NFE bucket for comparison
    print(f"\nNFE-Normalized Comparison (vs '{primary_split}'):")
    print("-" * 100)

    nfe_buckets: dict[str, list[EvalResult]] = {}
    for r in sorted_results:
        nfe = r.nfe_per_volume
        if nfe <= 15:
            bucket = "NFE ~10"
        elif nfe <= 35:
            bucket = "NFE ~25"
        elif nfe <= 75:
            bucket = "NFE ~50"
        elif nfe <= 150:
            bucket = "NFE ~100"
        elif nfe <= 350:
            bucket = "NFE ~200"
        else:
            bucket = "NFE ~500+"
        nfe_buckets.setdefault(bucket, []).append(r)

    for bucket in ['NFE ~10', 'NFE ~25', 'NFE ~50', 'NFE ~100', 'NFE ~200', 'NFE ~500+']:
        bucket_results = nfe_buckets.get(bucket, [])
        if not bucket_results:
            continue
        # Sort by FID within bucket
        bucket_results.sort(key=lambda r: r.metrics.get(primary_split, {}).get('fid', 999))
        best = bucket_results[0]

        print(f"\n  {bucket}:")
        for r in bucket_results:
            m = r.metrics.get(primary_split, {})
            if not m:
                continue
            marker = " *" if r is best and len(bucket_results) > 1 else ""
            print(f"    {r.dir_name:<30} NFE={r.nfe_per_volume:<6.0f} "
                  f"FID={m['fid']:<10.2f} KID={m['kid_mean']:<10.6f} "
                  f"CMMD={m['cmmd']:<10.6f} {r.time_per_volume_s:.1f}s{marker}")


def save_results_csv(results: list[EvalResult], path: Path) -> None:
    """Save results as CSV for plotting in thesis."""
    # Collect all split names
    all_splits = set()
    for r in results:
        all_splits.update(r.metrics.keys())
    all_splits = sorted(all_splits)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['solver', 'steps', 'tol', 'dir_name', 'nfe_per_volume',
                  'wall_time_s', 'time_per_volume_s', 'num_volumes']
        for split in all_splits:
            header.extend([f'fid_{split}', f'kid_mean_{split}', f'kid_std_{split}',
                          f'cmmd_{split}'])
        writer.writerow(header)

        # Data rows
        for r in sorted(results, key=lambda r: (r.solver, r.nfe_per_volume)):
            row = [
                r.solver,
                r.steps if r.steps is not None else '',
                r.atol if r.atol is not None else '',
                r.dir_name,
                f"{r.nfe_per_volume:.1f}",
                f"{r.wall_time_s:.1f}",
                f"{r.time_per_volume_s:.1f}",
                r.num_volumes,
            ]
            for split in all_splits:
                m = r.metrics.get(split, {})
                row.extend([
                    f"{m.get('fid', ''):.4f}" if m else '',
                    f"{m.get('kid_mean', ''):.6f}" if m else '',
                    f"{m.get('kid_std', ''):.6f}" if m else '',
                    f"{m.get('cmmd', ''):.6f}" if m else '',
                ])
            writer.writerow(row)

    logger.info(f"CSV results saved to {path}")


def save_results_json(results: list[EvalResult], path: Path) -> None:
    """Save full structured results as JSON."""
    data = []
    for r in results:
        entry = {
            'solver': r.solver,
            'steps': r.steps,
            'atol': r.atol,
            'rtol': r.rtol,
            'dir_name': r.dir_name,
            'nfe_total': r.nfe_total,
            'nfe_per_volume': r.nfe_per_volume,
            'wall_time_s': r.wall_time_s,
            'time_per_volume_s': r.time_per_volume_s,
            'num_volumes': r.num_volumes,
            'metrics': {
                split: asdict(m) if isinstance(m, SplitMetrics) else m
                for split, m in r.metrics.items()
            },
        }
        data.append(entry)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_results_json(path: Path) -> list[EvalResult]:
    """Load results from JSON for resume support."""
    with open(path) as f:
        data = json.load(f)
    results = []
    for entry in data:
        results.append(EvalResult(
            solver=entry['solver'],
            steps=entry['steps'],
            atol=entry['atol'],
            rtol=entry['rtol'],
            dir_name=entry['dir_name'],
            nfe_total=entry['nfe_total'],
            nfe_per_volume=entry['nfe_per_volume'],
            wall_time_s=entry['wall_time_s'],
            time_per_volume_s=entry['time_per_volume_s'],
            num_volumes=entry['num_volumes'],
            metrics=entry['metrics'],
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def _create_tiny_unet(device: torch.device) -> nn.Module:
    """Create a tiny 3D UNet for smoke testing (no checkpoint needed)."""
    from monai.networks.nets import DiffusionModelUNet

    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2,       # noise + seg conditioning
        out_channels=1,
        channels=[8, 16],
        attention_levels=[False, False],
        num_res_blocks=1,
        num_head_channels=8,
        norm_num_groups=8,
    ).to(device)
    return model


def _run_smoke_test(args) -> None:
    """Fast end-to-end pipeline verification with tiny dummy model.

    Exercises ALL code paths (generation, save, metrics, CSV/JSON, resume)
    with a tiny model and fake data. Should complete in seconds.
    """
    import tempfile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(tempfile.mkdtemp(prefix="eval_ode_smoke_"))
    results_json_path = output_dir / "results.json"
    results_csv_path = output_dir / "results.csv"

    # Tiny dimensions
    image_size = 16
    depth = 8
    trim_slices = 2
    num_volumes = 2
    seed = 42
    voxel_size = (1.0, 1.0, 1.0)

    logger.info(f"=== SMOKE TEST (output: {output_dir}) ===")

    # ── Minimal solver configs: euler, midpoint (torchdiffeq fixed), dopri5 (adaptive)
    smoke_configs = [
        SolverConfig(solver='euler', steps=3),
        SolverConfig(solver='midpoint', steps=3),
        SolverConfig(solver='dopri5', atol=1e-2, rtol=1e-2),
    ]

    # ── Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump({'smoke_test': True, 'num_configs': len(smoke_configs)}, f)

    # ── Create tiny model
    logger.info("Creating tiny dummy model...")
    model = _create_tiny_unet(device)

    # ── Setup strategy
    from medgen.diffusion import RFlowStrategy
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000, image_size=image_size,
        depth_size=depth, spatial_dims=3,
    )

    # ── Fake conditioning (random binary masks)
    cond_list = [
        (f"smoke_{i:03d}", torch.randint(0, 2, (1, 1, depth, image_size, image_size)).float())
        for i in range(num_volumes)
    ]
    save_conditioning(cond_list, output_dir, voxel_size, trim_slices)

    # ── Fake reference features (random — metrics will be meaningless but code paths exercised)
    effective_slices = depth - trim_slices
    ref_features = {
        'fake_split': {
            'resnet': torch.randn(num_volumes * effective_slices, 2048),
            'clip': torch.randn(num_volumes * effective_slices, 512),
        },
    }

    # ── Pre-generate noise
    noise_list = generate_noise_tensors(num_volumes, depth, image_size, device, seed)

    # ── Run each solver config
    all_results: list[EvalResult] = []

    for i, solver_cfg in enumerate(smoke_configs):
        logger.info(f"[{i+1}/{len(smoke_configs)}] {solver_cfg.label}")

        volumes, total_nfe, wall_time = generate_volumes(
            model, strategy, noise_list, cond_list, solver_cfg, device,
        )
        nfe_per_vol = total_nfe / num_volumes
        logger.info(f"  Done: {wall_time:.2f}s, NFE={total_nfe} ({nfe_per_vol:.0f}/vol)")

        # Save volumes
        vol_dir = output_dir / "generated" / solver_cfg.dir_name
        save_volumes(volumes, cond_list, vol_dir, voxel_size, trim_slices)

        # Compute metrics (with fake refs — exercises the metric pipeline)
        split_metrics = compute_all_metrics(
            volumes, ref_features, device, trim_slices,
        )

        result = EvalResult(
            solver=solver_cfg.solver,
            steps=solver_cfg.steps,
            atol=solver_cfg.atol if solver_cfg.is_adaptive else None,
            rtol=solver_cfg.rtol if solver_cfg.is_adaptive else None,
            dir_name=solver_cfg.dir_name,
            nfe_total=total_nfe,
            nfe_per_volume=nfe_per_vol,
            wall_time_s=wall_time,
            time_per_volume_s=wall_time / num_volumes,
            num_volumes=num_volumes,
            metrics={s: asdict(m) for s, m in split_metrics.items()},
        )
        all_results.append(result)

        save_results_json(all_results, results_json_path)
        save_results_csv(all_results, results_csv_path)

        del volumes
        torch.cuda.empty_cache()

    # ── Verify resume: reload results and check
    loaded = load_results_json(results_json_path)
    assert len(loaded) == len(smoke_configs), \
        f"Resume check failed: {len(loaded)} != {len(smoke_configs)}"

    # ── Print final table
    print_results_table(all_results, primary_split='fake_split')

    # ── Verify output files exist
    checks = [
        output_dir / "config.json",
        results_json_path,
        results_csv_path,
        output_dir / "conditioning",
    ]
    for solver_cfg in smoke_configs:
        checks.append(output_dir / "generated" / solver_cfg.dir_name)

    for path in checks:
        assert path.exists(), f"Missing: {path}"

    logger.info(f"\n=== SMOKE TEST PASSED ({output_dir}) ===")
    logger.info(f"  Configs tested: {len(smoke_configs)}")
    logger.info(f"  Volumes per config: {num_volumes}")
    logger.info(f"  Output verified: config.json, results.json, results.csv, "
                 f"conditioning/, {len(smoke_configs)} generated/ dirs")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ODE solvers for RFlow generation quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--bravo-model', default=None,
                        help='Path to trained bravo model checkpoint (required unless --smoke-test)')
    parser.add_argument('--data-root', default=None,
                        help='Root of dataset (required unless --smoke-test)')
    parser.add_argument('--output-dir', default='eval_ode_solvers',
                        help='Output directory (default: eval_ode_solvers)')
    parser.add_argument('--num-volumes', type=int, default=25,
                        help='Volumes per solver config (default: 25)')
    parser.add_argument('--cond-split', default='val',
                        help='Split to use for conditioning seg masks (default: val)')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Image H/W (auto-detected from checkpoint)')
    parser.add_argument('--depth', type=int, default=None,
                        help='Generation depth (auto-detected from checkpoint)')
    parser.add_argument('--trim-slices', type=int, default=10,
                        help='Slices to trim from end (default: 10)')
    parser.add_argument('--fov-mm', type=float, default=240.0,
                        help='Field of view in mm (default: 240.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for noise generation (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer configs for testing')
    parser.add_argument('--resume', action='store_true',
                        help='Resume: skip configs with existing results')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Smoke test: tiny dummy model, fast end-to-end verification')
    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test(args)
        return

    # Validate required args for real runs
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

    # ── Save experiment config ───────────────────────────────────────────
    solver_configs = build_solver_configs(quick=args.quick)
    experiment_config = {
        'bravo_model': str(Path(args.bravo_model).resolve()),
        'data_root': str(Path(args.data_root).resolve()),
        'cond_split': args.cond_split,
        'num_volumes': args.num_volumes,
        'image_size': image_size,
        'depth': depth,
        'trim_slices': args.trim_slices,
        'fov_mm': args.fov_mm,
        'voxel_size': list(voxel_size),
        'seed': args.seed,
        'quick': args.quick,
        'num_configs': len(solver_configs),
        'solver_configs': [
            {'solver': c.solver, 'steps': c.steps, 'atol': c.atol, 'rtol': c.rtol,
             'dir_name': c.dir_name}
            for c in solver_configs
        ],
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(experiment_config, f, indent=2)

    logger.info("=" * 70)
    logger.info("ODE Solver Evaluation")
    logger.info("=" * 70)
    logger.info(f"Model: {args.bravo_model}")
    logger.info(f"Volume: {image_size}x{image_size}x{depth} (trim {args.trim_slices})")
    logger.info(f"Volumes per config: {args.num_volumes}")
    logger.info(f"Solver configs: {len(solver_configs)}")
    logger.info(f"Total volumes to generate: {len(solver_configs) * args.num_volumes}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)

    # ── Discover splits ──────────────────────────────────────────────────
    data_root = Path(args.data_root)
    logger.info(f"Discovering dataset splits in {data_root}...")
    splits = discover_splits(data_root)

    if args.cond_split not in splits:
        raise ValueError(
            f"Conditioning split '{args.cond_split}' not found. "
            f"Available: {list(splits.keys())}"
        )

    # ── Load conditioning masks ──────────────────────────────────────────
    logger.info(f"Loading {args.num_volumes} conditioning masks from '{args.cond_split}'...")
    cond_list = load_conditioning(splits[args.cond_split], args.num_volumes, depth)

    logger.info("Saving conditioning masks to output dir...")
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
    logger.info(f"Pre-generating {args.num_volumes} noise tensors (seed={args.seed})...")
    noise_list = generate_noise_tensors(
        args.num_volumes, depth, image_size, device, args.seed,
    )

    # ── Load existing results if resuming ────────────────────────────────
    existing_results: list[EvalResult] = []
    completed_dirs: set[str] = set()
    if args.resume and results_json_path.exists():
        existing_results = load_results_json(results_json_path)
        completed_dirs = {r.dir_name for r in existing_results}
        logger.info(f"Resuming: {len(existing_results)} configs already completed")

    # ── Evaluate each solver config ──────────────────────────────────────
    all_results = list(existing_results)
    total_start = time.time()

    for i, solver_cfg in enumerate(solver_configs):
        if solver_cfg.dir_name in completed_dirs:
            logger.info(f"[{i+1}/{len(solver_configs)}] SKIP {solver_cfg.dir_name} (done)")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"[{i+1}/{len(solver_configs)}] {solver_cfg.label}")
        logger.info(f"{'='*70}")

        # Generate
        logger.info(f"  Generating {args.num_volumes} volumes...")
        volumes, total_nfe, wall_time = generate_volumes(
            bravo_model, strategy, noise_list, cond_list, solver_cfg, device,
        )
        nfe_per_vol = total_nfe / args.num_volumes
        logger.info(f"  Done: {wall_time:.1f}s total, {wall_time/args.num_volumes:.1f}s/vol, "
                     f"NFE={total_nfe} ({nfe_per_vol:.0f}/vol)")

        # Save volumes
        vol_dir = output_dir / "generated" / solver_cfg.dir_name
        logger.info(f"  Saving volumes to {vol_dir}...")
        save_volumes(volumes, cond_list, vol_dir, voxel_size, args.trim_slices)

        # Compute metrics
        logger.info("  Computing metrics...")
        split_metrics = compute_all_metrics(
            volumes, ref_features, device, args.trim_slices,
        )

        # Build result
        result = EvalResult(
            solver=solver_cfg.solver,
            steps=solver_cfg.steps,
            atol=solver_cfg.atol if solver_cfg.is_adaptive else None,
            rtol=solver_cfg.rtol if solver_cfg.is_adaptive else None,
            dir_name=solver_cfg.dir_name,
            nfe_total=total_nfe,
            nfe_per_volume=nfe_per_vol,
            wall_time_s=wall_time,
            time_per_volume_s=wall_time / args.num_volumes,
            num_volumes=args.num_volumes,
            metrics={
                split: asdict(m) for split, m in split_metrics.items()
            },
        )
        all_results.append(result)

        # Save intermediate results (for resume)
        save_results_json(all_results, results_json_path)
        save_results_csv(all_results, results_csv_path)

        # Clean up
        del volumes
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # ── Final output ─────────────────────────────────────────────────────
    print_results_table(all_results, primary_split='all')

    # Also print per-split summaries
    for split_name in splits:
        print_results_table(all_results, primary_split=split_name)

    save_results_json(all_results, results_json_path)
    save_results_csv(all_results, results_csv_path)

    logger.info(f"\nTotal evaluation time: {total_time/3600:.1f} hours")
    logger.info(f"Results: {results_json_path}")
    logger.info(f"CSV:     {results_csv_path}")
    logger.info(f"Volumes: {output_dir / 'generated'}/")
    logger.info("Done!")


if __name__ == "__main__":
    main()
