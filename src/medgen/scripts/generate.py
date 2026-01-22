"""
Image/volume generation script using trained diffusion models.

Supports both 2D images and 3D volumes, with multiple generation modes:
- bravo: Generate BRAVO images conditioned on seg masks
- dual: Generate T1 pre+gd images conditioned on seg masks
- seg_conditioned: Generate seg masks conditioned on size bins (3D only)

Usage:
    # 2D: seg -> bravo pipeline (local, default)
    python -m medgen.scripts.generate mode=bravo \\
        seg_model=runs/seg/model.pt image_model=runs/bravo/model.pt

    # 3D: size_bins -> seg -> bravo pipeline (cluster)
    python -m medgen.scripts.generate paths=cluster spatial_dims=3 mode=bravo \\
        seg_model=runs/seg/checkpoint.pt image_model=runs/bravo/checkpoint.pt

    # Custom output subdirectory
    python -m medgen.scripts.generate mode=bravo output_subdir=experiment1 \\
        seg_model=... image_model=...
"""
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import nibabel as nib
import numpy as np
import torch
from omegaconf import DictConfig
from torch.amp import autocast
from tqdm import tqdm

from medgen.core import (
    MAX_WHITE_PERCENTAGE, BINARY_THRESHOLD_GEN,
    setup_cuda_optimizations,
)
from medgen.diffusion import DDPMStrategy, RFlowStrategy, DiffusionStrategy, load_diffusion_model
from medgen.data import make_binary

setup_cuda_optimizations()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
log = logging.getLogger(__name__)


def sample_random_size_bins(min_tumors: int = 1, max_tumors: int = 5) -> List[int]:
    """Sample random size bins for tumor generation."""
    num_tumors = random.randint(min_tumors, max_tumors)
    bins = [0] * 7
    # Weight towards clinically relevant sizes (bins 2-5: 6-30mm)
    weights = [0.05, 0.1, 0.2, 0.25, 0.2, 0.15, 0.05]
    for _ in range(num_tumors):
        bin_idx = random.choices(range(7), weights=weights)[0]
        bins[bin_idx] += 1
    return bins


def auto_adjust_batch_size(base_batch_size: int, spatial_dims: int, device: torch.device) -> int:
    """Automatically adjust batch size based on available VRAM and dimensions."""
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    if spatial_dims == 3:
        # 3D needs much smaller batches
        if total_vram_gb >= 75:
            adjusted = min(base_batch_size, 2)
        else:
            adjusted = 1
    else:
        # 2D can use larger batches
        if total_vram_gb >= 75:
            adjusted = int(base_batch_size * 2.0)
        else:
            adjusted = base_batch_size

    log.info(f"GPU: {total_vram_gb:.1f}GB | Batch size: {base_batch_size} -> {adjusted}")
    return adjusted


def is_valid_mask(binary_mask: np.ndarray, max_white_percentage: float = MAX_WHITE_PERCENTAGE) -> bool:
    """Check if segmentation mask is valid (not empty, not too large)."""
    white_pixels = np.sum(binary_mask == 1.0)
    total_pixels = binary_mask.size
    white_percentage = white_pixels / total_pixels
    return 0.0 < white_percentage < max_white_percentage


def save_nifti(data: np.ndarray, output_path: str, affine: Optional[np.ndarray] = None) -> None:
    """Save numpy array as NIfTI file."""
    if affine is None:
        affine = np.eye(4)
    nifti = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(nifti, output_path)


def get_noise_shape(batch_size: int, channels: int, spatial_dims: int,
                    image_size: int, depth: int) -> Tuple[int, ...]:
    """Get noise tensor shape based on spatial dimensions."""
    if spatial_dims == 2:
        return (batch_size, channels, image_size, image_size)
    else:
        return (batch_size, channels, depth, image_size, image_size)


def generate_batch(
    model: torch.nn.Module,
    strategy: DiffusionStrategy,
    noise: torch.Tensor,
    num_steps: int,
    device: torch.device,
    conditioning: Optional[torch.Tensor] = None,
    size_bins: Optional[torch.Tensor] = None,
    use_progress: bool = False,
) -> torch.Tensor:
    """Generate a batch using diffusion model.

    Args:
        model: Diffusion model (may be SizeBinModelWrapper for seg_conditioned).
        strategy: Diffusion strategy (DDPM or RFlow).
        noise: Initial noise tensor.
        num_steps: Number of denoising steps.
        device: Torch device.
        conditioning: Optional conditioning tensor (e.g., seg mask for bravo).
        size_bins: Optional size bin tensor [B, num_bins] for seg_conditioned mode.
        use_progress: Show progress bar.

    Returns:
        Generated tensor.
    """
    if conditioning is not None:
        model_input = torch.cat([noise, conditioning], dim=1)
    else:
        model_input = noise

    with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            return strategy.generate(
                model, model_input, num_steps, device,
                size_bins=size_bins,
                use_progress_bars=use_progress
            )


def run_2d_pipeline(cfg: DictConfig, output_dir: Path) -> None:
    """Run 2D generation pipeline: seg -> bravo/dual."""
    # Validate gen_mode
    VALID_2D_MODES = {'bravo', 'dual'}
    if cfg.gen_mode not in VALID_2D_MODES:
        raise ValueError(f"Invalid gen_mode '{cfg.gen_mode}' for 2D. Valid: {VALID_2D_MODES}")

    device = torch.device("cuda")
    batch_size = auto_adjust_batch_size(cfg.batch_size, 2, device)

    # Initialize strategy
    strategy: DiffusionStrategy = RFlowStrategy() if cfg.strategy == 'rflow' else DDPMStrategy()
    strategy.setup_scheduler(1000, cfg.image_size)

    # Load models
    log.info("Loading segmentation model...")
    seg_model = load_diffusion_model(
        cfg.seg_model, device=device,
        in_channels=1, out_channels=1, compile_model=True
    )

    log.info(f"Loading image model ({cfg.gen_mode})...")
    if cfg.gen_mode == 'bravo':
        in_ch, out_ch = 2, 1
    else:  # dual
        in_ch, out_ch = 3, 2

    image_model = load_diffusion_model(
        cfg.image_model, device=device,
        in_channels=in_ch, out_channels=out_ch, compile_model=True
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Generating {cfg.num_images} samples...")

    current_image = cfg.current_image
    mask_cache: List[Tuple[np.ndarray, int]] = []
    use_progress = cfg.verbose

    # Infinite loop protection
    MAX_CONSECUTIVE_FAILURES = 100
    consecutive_failures = 0

    while current_image < cfg.num_images:
        # Generate seg masks
        noise = torch.randn(get_noise_shape(batch_size, 1, 2, cfg.image_size, 0), device=device)
        seg_masks = generate_batch(seg_model, strategy, noise, cfg.num_steps, device)

        # Validate and cache masks
        valid_in_batch = 0
        for j in range(len(seg_masks)):
            if current_image >= cfg.num_images:
                break

            mask = seg_masks[j, 0].cpu().numpy()
            mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
            mask = make_binary(mask, threshold=BINARY_THRESHOLD_GEN)

            if is_valid_mask(mask):
                mask_cache.append((mask, current_image))
                current_image += 1
                valid_in_batch += 1

        # Track consecutive failures to prevent infinite loop
        if valid_in_batch == 0:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log.error(
                    f"No valid masks in {MAX_CONSECUTIVE_FAILURES} consecutive batches. "
                    f"Check seg model quality or adjust MAX_WHITE_PERCENTAGE threshold."
                )
                raise RuntimeError(
                    f"Generation stuck: no valid masks in {MAX_CONSECUTIVE_FAILURES} batches"
                )
        else:
            consecutive_failures = 0  # Reset on success

        # Process cached masks in batches
        while len(mask_cache) >= batch_size:
            batch_masks = [mask_cache[i][0] for i in range(batch_size)]
            batch_counters = [mask_cache[i][1] for i in range(batch_size)]
            mask_cache = mask_cache[batch_size:]

            # Convert to tensor
            masks_tensor = torch.stack([
                torch.from_numpy(m).unsqueeze(0) for m in batch_masks
            ], dim=0).to(device, dtype=torch.float32)

            # Generate images
            out_ch = 1 if cfg.gen_mode == 'bravo' else 2
            noise = torch.randn(get_noise_shape(batch_size, out_ch, 2, cfg.image_size, 0), device=device)
            images = generate_batch(image_model, strategy, noise, cfg.num_steps, device, masks_tensor)

            # Save
            for i, counter in enumerate(batch_counters):
                output_path = output_dir / f"{counter:05d}.nii.gz"
                if cfg.gen_mode == 'bravo':
                    combined = np.stack([images[i, 0].cpu().numpy(), batch_masks[i]], axis=-1)
                else:
                    combined = np.stack([
                        images[i, 0].cpu().numpy(),
                        images[i, 1].cpu().numpy(),
                        batch_masks[i]
                    ], axis=-1)
                save_nifti(combined, str(output_path))

        torch.cuda.empty_cache()

    # Process remaining
    if mask_cache:
        for mask, counter in mask_cache:
            masks_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
            out_ch = 1 if cfg.gen_mode == 'bravo' else 2
            noise = torch.randn(get_noise_shape(1, out_ch, 2, cfg.image_size, 0), device=device)
            images = generate_batch(image_model, strategy, noise, cfg.num_steps, device, masks_tensor)

            output_path = output_dir / f"{counter:05d}.nii.gz"
            if cfg.gen_mode == 'bravo':
                combined = np.stack([images[0, 0].cpu().numpy(), mask], axis=-1)
            else:
                combined = np.stack([images[0, 0].cpu().numpy(), images[0, 1].cpu().numpy(), mask], axis=-1)
            save_nifti(combined, str(output_path))

    log.info(f"Saved {current_image} samples to {output_dir}")


def save_bins_csv(bins_data: List[Tuple[int, List[int]]], output_path: Path) -> None:
    """Save all bins information to a single CSV file.

    Args:
        bins_data: List of (sample_id, bins) tuples
        output_path: Path to save CSV file
    """
    with open(output_path, 'w') as f:
        # Header: id, bin_0, bin_1, ..., bin_6, total_tumors
        f.write('id,bin_0,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,total_tumors\n')
        for sample_id, bins in bins_data:
            bins_str = ','.join(map(str, bins))
            total = sum(bins)
            f.write(f'{sample_id:05d},{bins_str},{total}\n')


def run_3d_pipeline(cfg: DictConfig, output_dir: Path) -> None:
    """Run 3D generation pipeline: size_bins -> seg -> bravo.

    Output format:
        - bins.csv: All size bin information for all samples
        - {id}.nii.gz: Combined volume [D, H, W, 2] where channel 0=seg, channel 1=bravo
          (or just [D, H, W] for seg_conditioned mode)
    """
    # Validate gen_mode (fail-fast before loading models)
    VALID_3D_MODES = {'bravo', 'seg_conditioned'}
    if cfg.gen_mode not in VALID_3D_MODES:
        raise ValueError(f"Invalid gen_mode '{cfg.gen_mode}' for 3D. Valid: {VALID_3D_MODES}")

    device = torch.device("cuda")
    batch_size = auto_adjust_batch_size(cfg.batch_size, 3, device)

    strategy = RFlowStrategy() if cfg.strategy == 'rflow' else DDPMStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000,
        image_size=cfg.image_size,
        depth_size=cfg.depth,
        spatial_dims=3,
    )

    # Parse fixed size bins if provided
    fixed_bins = None
    if cfg.size_bins:
        fixed_bins = [int(x) for x in cfg.size_bins.split(',')]
        assert len(fixed_bins) == 7, "Size bins must have exactly 7 values"
        log.info(f"Using fixed size bins: {fixed_bins}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all bins info for CSV
    all_bins: List[Tuple[int, List[int]]] = []

    # Mode: seg_conditioned only (just generate seg masks)
    if cfg.gen_mode == 'seg_conditioned':
        log.info("Loading seg_conditioned model...")
        seg_model = load_diffusion_model(
            cfg.seg_model, device=device,
            in_channels=1, out_channels=1, compile_model=True, spatial_dims=3
        )

        log.info(f"Generating {cfg.num_images} seg masks...")
        for i in tqdm(range(cfg.num_images), disable=not cfg.verbose):
            bins = fixed_bins if fixed_bins else sample_random_size_bins(cfg.min_tumors, cfg.max_tumors)
            size_bins = torch.tensor([bins], dtype=torch.long, device=device)

            noise = torch.randn(get_noise_shape(1, 1, 3, cfg.image_size, cfg.depth), device=device)
            seg = generate_batch(seg_model, strategy, noise, cfg.num_steps, device, size_bins=size_bins)

            # Binarize and save
            seg_np = seg[0, 0].cpu().numpy()
            seg_np = (seg_np - seg_np.min()) / (seg_np.max() - seg_np.min() + 1e-8)
            seg_binary = make_binary(seg_np, threshold=0.5)

            # Save single-channel seg volume
            save_nifti(seg_binary, str(output_dir / f"{i:05d}.nii.gz"))
            all_bins.append((i, bins))

            if i % 10 == 0:
                torch.cuda.empty_cache()

    # Mode: bravo (full pipeline: seg -> bravo)
    elif cfg.gen_mode == 'bravo':
        log.info("Loading seg_conditioned model...")
        seg_model = load_diffusion_model(
            cfg.seg_model, device=device,
            in_channels=1, out_channels=1, compile_model=True, spatial_dims=3
        )

        log.info("Loading bravo model...")
        bravo_model = load_diffusion_model(
            cfg.image_model, device=device,
            in_channels=2, out_channels=1, compile_model=True, spatial_dims=3
        )

        log.info(f"Generating {cfg.num_images} seg+bravo pairs...")
        for i in tqdm(range(cfg.num_images), disable=not cfg.verbose):
            bins = fixed_bins if fixed_bins else sample_random_size_bins(cfg.min_tumors, cfg.max_tumors)
            size_bins = torch.tensor([bins], dtype=torch.long, device=device)

            # Generate seg with size bin conditioning
            noise = torch.randn(get_noise_shape(1, 1, 3, cfg.image_size, cfg.depth), device=device)
            seg = generate_batch(seg_model, strategy, noise, cfg.num_steps, device, size_bins=size_bins)

            # Binarize seg
            seg_np = seg[0, 0].cpu().numpy()
            seg_np = (seg_np - seg_np.min()) / (seg_np.max() - seg_np.min() + 1e-8)
            seg_binary = make_binary(seg_np, threshold=0.5)
            seg_tensor = torch.from_numpy(seg_binary).float().unsqueeze(0).unsqueeze(0).to(device)

            # Generate bravo
            noise = torch.randn(get_noise_shape(1, 1, 3, cfg.image_size, cfg.depth), device=device)
            bravo = generate_batch(bravo_model, strategy, noise, cfg.num_steps, device, seg_tensor)
            bravo_np = bravo[0, 0].cpu().numpy()

            # Stack seg + bravo into combined volume [D, H, W, 2]
            # Channel 0 = seg (binary), Channel 1 = bravo (continuous)
            # NOTE: NIfTI tools will interpret dim 4 as time series (2 frames).
            # This is intentional - load with nibabel and index [..., 0] for seg, [..., 1] for bravo
            combined = np.stack([seg_binary, bravo_np], axis=-1)
            save_nifti(combined, str(output_dir / f"{i:05d}.nii.gz"))
            all_bins.append((i, bins))

            if i % 10 == 0:
                torch.cuda.empty_cache()

    else:
        raise ValueError(f"Mode '{cfg.gen_mode}' not supported for 3D. Use 'seg_conditioned' or 'bravo'.")

    # Save all bins to single CSV file
    save_bins_csv(all_bins, output_dir / "bins.csv")
    log.info(f"Saved {len(all_bins)} samples to {output_dir}")
    log.info(f"Bins info saved to {output_dir / 'bins.csv'}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="generate")
def main(cfg: DictConfig) -> None:
    """Main entry point for generation."""
    # Build output directory from paths config
    generated_dir = Path(cfg.paths.generated_dir)
    if cfg.output_subdir:
        output_dir = generated_dir / cfg.output_subdir
    else:
        # Default subdirectory based on mode and spatial dims
        output_dir = generated_dir / f"{cfg.spatial_dims}d_{cfg.gen_mode}"

    log.info("=" * 60)
    log.info(f"Generation: {cfg.spatial_dims}D | Mode: {cfg.gen_mode} | Strategy: {cfg.strategy}")
    log.info(f"Output: {output_dir}")
    log.info(f"Paths: {cfg.paths.name}")
    log.info("=" * 60)

    if cfg.spatial_dims == 2:
        if not cfg.seg_model or not cfg.image_model:
            raise ValueError("2D mode requires seg_model and image_model")
        run_2d_pipeline(cfg, output_dir)
    else:
        if cfg.gen_mode == 'seg_conditioned' and not cfg.seg_model:
            raise ValueError("seg_conditioned mode requires seg_model")
        if cfg.gen_mode == 'bravo' and (not cfg.seg_model or not cfg.image_model):
            raise ValueError("3D bravo mode requires seg_model and image_model")
        run_3d_pipeline(cfg, output_dir)

    log.info("Generation complete!")


if __name__ == "__main__":
    main()
