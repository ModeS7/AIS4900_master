"""
Image generation script using trained diffusion models.

This module generates synthetic brain MRI images using trained segmentation
and image generation models. Supports BRAVO single-image and T1 dual-image
generation modes.

Usage:
    python -m medgen.scripts.generate --strategy ddpm --mode bravo \\
        --seg_model seg/model.pt --image_model bravo/model.pt \\
        --num_images 15000 --output_dir gen_bravo

    python -m medgen.scripts.generate --strategy rflow --mode dual \\
        --seg_model seg/model.pt --image_model dual/model.pt \\
        --num_images 10000 --output_dir gen_dual
"""
import argparse
import logging
import os
import time
from typing import Callable, List, Literal, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import DiffusionModelUNet
from torch.amp import autocast

from medgen.core import (
    MAX_WHITE_PERCENTAGE, BINARY_THRESHOLD_GEN,
    DEFAULT_CHANNELS, DEFAULT_ATTENTION_LEVELS,
    DEFAULT_NUM_RES_BLOCKS, DEFAULT_NUM_HEAD_CHANNELS,
    ModeType, setup_cuda_optimizations,
)
from medgen.diffusion import DDPMStrategy, RFlowStrategy, DiffusionStrategy
from medgen.pipeline.utils import get_vram_usage
from medgen.data import make_binary
from medgen.data import create_conditioning_wrapper

# Enable CUDA optimizations
setup_cuda_optimizations()

log = logging.getLogger(__name__)


def _flush_mask_cache(
    mask_cache: List[Tuple[np.ndarray, int]],
    batch_size: int,
    process_fn: Callable[[List[np.ndarray], List[int]], None],
) -> List[Tuple[np.ndarray, int]]:
    """Process mask cache in batches until insufficient items remain.

    Args:
        mask_cache: List of (mask_array, counter) tuples.
        batch_size: Number of masks to process per batch.
        process_fn: Function that takes (batch_masks, batch_counters) and processes them.

    Returns:
        Remaining mask_cache with < batch_size items.
    """
    while len(mask_cache) >= batch_size:
        batch_masks = [mask_cache[i][0] for i in range(batch_size)]
        batch_counters = [mask_cache[i][1] for i in range(batch_size)]
        mask_cache = mask_cache[batch_size:]
        process_fn(batch_masks, batch_counters)
    return mask_cache


GenerationMode = Literal['bravo', 'dual']
StrategyType = Literal['ddpm', 'rflow']


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for generation."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic medical images using trained diffusion models'
    )
    parser.add_argument(
        '--strategy', type=str, choices=['ddpm', 'rflow'], default='ddpm',
        help='Generation strategy: ddpm or rflow (default: ddpm)'
    )
    parser.add_argument(
        '--mode', type=str, choices=['bravo', 'dual'], default='bravo',
        help='Generation mode: bravo (single image) or dual (T1 pre+gd)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size for generation (default: 8)'
    )
    parser.add_argument(
        '--image_size', type=int, default=128,
        help='Image size (default: 128)'
    )
    parser.add_argument(
        '--num_images', type=int, default=15000,
        help='Number of images to generate (default: 15000)'
    )
    parser.add_argument(
        '--current_image', type=int, default=0,
        help='Starting image counter for resuming (default: 0)'
    )
    parser.add_argument(
        '--num_steps', type=int, default=1000,
        help='Number of diffusion steps (default: 1000)'
    )
    parser.add_argument(
        '--seg_model', type=str, required=True,
        help='Path to trained segmentation model'
    )
    parser.add_argument(
        '--image_model', type=str, required=True,
        help='Path to trained image model (bravo or dual)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for generated images'
    )
    parser.add_argument(
        '--no_progress', action='store_true',
        help='Disable progress bars (for cluster)'
    )
    return parser.parse_args()


def auto_adjust_batch_size(base_batch_size: int, device: torch.device) -> int:
    """Automatically adjust batch size based on available VRAM."""
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    if total_vram_gb >= 75:
        multiplier = 2.0
        gpu_tier = ">75GB"
    else:
        multiplier = 1.0
        gpu_tier = "<75GB"

    adjusted_batch_size = int(base_batch_size * multiplier)

    log.info(f"GPU tier detected: {gpu_tier} ({total_vram_gb:.1f}GB)")
    log.info(f"Batch size auto-adjusted: {base_batch_size} -> {adjusted_batch_size} (x{multiplier})")

    return adjusted_batch_size


def is_valid_mask(binary_mask: np.ndarray, max_white_percentage: float = MAX_WHITE_PERCENTAGE) -> bool:
    """Check if segmentation mask is valid (not empty, not too large)."""
    white_pixels = np.sum(binary_mask == 1.0)
    total_pixels = binary_mask.size
    white_percentage = white_pixels / total_pixels
    return 0.0 < white_percentage < max_white_percentage


def _detect_wrapper_type(state_dict: dict) -> str:
    """Detect wrapper type from checkpoint state dict keys.

    Args:
        state_dict: Model state dictionary from checkpoint.

    Returns:
        Wrapper type: 'raw', 'score_aug', 'mode_embed', or 'combined'.
    """
    keys = list(state_dict.keys())
    has_inner_model = any('inner_model' in k for k in keys)

    if not has_inner_model:
        return 'raw'

    has_omega_mlp = any('omega_mlp' in k for k in keys)
    has_mode_mlp = any('mode_mlp' in k for k in keys)

    if has_omega_mlp and has_mode_mlp:
        return 'combined'
    elif has_omega_mlp:
        return 'score_aug'
    elif has_mode_mlp:
        return 'mode_embed'

    return 'raw'


def load_model(
    model_path: str,
    in_channels: int,
    out_channels: int,
    device: torch.device
) -> torch.nn.Module:
    """Load and compile trained diffusion model.

    Automatically detects and handles wrapped models (ScoreAug, ModeEmbed, Combined).
    """
    # Load checkpoint first to detect wrapper type
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    wrapper_type = _detect_wrapper_type(state_dict)
    log.info(f"Detected model type: {wrapper_type}")

    # Create base model
    base_model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=DEFAULT_CHANNELS,
        attention_levels=DEFAULT_ATTENTION_LEVELS,
        num_res_blocks=DEFAULT_NUM_RES_BLOCKS,
        num_head_channels=DEFAULT_NUM_HEAD_CHANNELS,
    )

    # Wrap model if needed
    # MONAI DiffusionModelUNet time_embed output dim is 4 * channels[0]
    embed_dim = 4 * DEFAULT_CHANNELS[0]

    if wrapper_type == 'raw':
        model = base_model
    elif wrapper_type in ('score_aug', 'mode_embed', 'combined'):
        # Use factory function to create appropriate wrapper
        model, _ = create_conditioning_wrapper(
            model=base_model,
            use_omega=(wrapper_type in ('score_aug', 'combined')),
            use_mode=(wrapper_type in ('mode_embed', 'combined')),
            embed_dim=embed_dim,
        )
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper_type}")

    # Load state dict
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    opt_model = torch.compile(model, mode="reduce-overhead")

    return opt_model


class ValidationTracker:
    """Track mask generation validation statistics."""

    def __init__(self) -> None:
        self.total_generated: int = 0
        self.total_valid: int = 0
        self.batch_start_time: float = 0.0

    def start_batch(self) -> None:
        self.batch_start_time = time.time()

    def record_generation(self, valid_count: int, total_count: int) -> None:
        self.total_generated += total_count
        self.total_valid += valid_count

    def get_success_rate(self) -> float:
        if self.total_generated == 0:
            return 0.0
        return (self.total_valid / self.total_generated) * 100

    def get_batch_time(self) -> float:
        return time.time() - self.batch_start_time


def log_mask_batch(
    tracker: ValidationTracker,
    batch_size: int,
    valid_count: int,
    cache_count: int,
    device: torch.device
) -> None:
    """Log mask generation batch statistics."""
    timestamp = time.strftime("%H:%M:%S")
    success_rate = tracker.get_success_rate()
    batch_time = tracker.get_batch_time()
    vram_info = get_vram_usage(device)

    log.info(
        f"[{timestamp}] Mask: {valid_count:3d}/{batch_size:3d} valid | "
        f"Success: {success_rate:5.1f}% | Cache: {cache_count:4d} | "
        f"Time: {batch_time:6.1f}s | {vram_info}"
    )


def log_image_batch(
    current_image: int,
    target_images: int,
    batch_size: int,
    batch_time: float,
    mode: GenerationMode,
    device: torch.device
) -> None:
    """Log image generation batch statistics."""
    timestamp = time.strftime("%H:%M:%S")
    progress_pct = (current_image / target_images) * 100
    vram_info = get_vram_usage(device)

    mode_label = "Bravo" if mode == ModeType.BRAVO else "Dual"
    log.info(
        f"[{timestamp}] Image {current_image:6d}/{target_images:6d} ({progress_pct:5.1f}%) | "
        f"{mode_label}: {batch_size:3d} saved | Time: {batch_time:6.1f}s | {vram_info}"
    )


def _process_mask_batch(
    batch_masks: List[np.ndarray],
    batch_counters: List[int],
    image_model: torch.nn.Module,
    strategy: DiffusionStrategy,
    num_steps: int,
    image_size: int,
    save_dir: str,
    mode: GenerationMode,
    device: torch.device,
    use_progress_bars: bool,
    num_images: int
) -> int:
    """Process a batch of masks through image generation.

    Args:
        batch_masks: List of mask arrays to condition image generation.
        batch_counters: List of image indices for saving.
        image_model: Trained diffusion model for image generation.
        strategy: Diffusion strategy (DDPM or RFlow).
        num_steps: Number of denoising steps.
        image_size: Image size in pixels.
        save_dir: Directory to save generated images.
        mode: Generation mode ('bravo' or 'dual').
        device: Target device.
        use_progress_bars: Whether to show progress bars.
        num_images: Total target number of images (for logging).

    Returns:
        saved_up_to: Highest image counter saved + 1
    """
    image_start = time.time()
    generate_images_from_masks(
        batch_masks, batch_counters, image_model,
        strategy, num_steps, image_size,
        save_dir, mode, device, use_progress_bars=use_progress_bars
    )
    image_time = time.time() - image_start

    saved_up_to = max(batch_counters) + 1
    log_image_batch(saved_up_to, num_images, len(batch_masks), image_time, mode, device)
    return saved_up_to


def _run_diffusion_generation(
    model_input: torch.Tensor,
    image_model: torch.nn.Module,
    strategy: DiffusionStrategy,
    num_steps: int,
    device: torch.device,
    use_progress_bars: bool,
) -> torch.Tensor:
    """Run diffusion generation with autocast and no_grad.

    Args:
        model_input: Input tensor [B, C, H, W] (noise + conditioning).
        image_model: Trained diffusion model.
        strategy: Diffusion strategy (DDPM or RFlow).
        num_steps: Number of denoising steps.
        device: Target device.
        use_progress_bars: Whether to show progress bars.

    Returns:
        Generated images tensor [B, C, H, W].
    """
    with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            return strategy.generate(
                image_model, model_input, num_steps,
                device, use_progress_bars=use_progress_bars
            )


def _save_nifti(data: torch.Tensor, output_path: str, counter: int) -> None:
    """Save tensor as NIfTI file.

    Args:
        data: Image data to save.
        output_path: Full path to output file.
        counter: Image counter (for error messages).
    """
    try:
        nifti_image = nib.Nifti1Image(np.array(data), np.eye(4))
        nib.save(nifti_image, output_path)
    except Exception as e:
        log.error(f"Failed to save image {counter:05d}: {e}")
        raise


def generate_images_from_masks(
    mask_images: List[np.ndarray],
    start_counters: List[int],
    image_model: torch.nn.Module,
    strategy: DiffusionStrategy,
    num_steps: int,
    image_size: int,
    save_dir: str,
    mode: GenerationMode,
    device: torch.device,
    use_progress_bars: bool = False
) -> bool:
    """Generate images from segmentation masks.

    Creates noise, runs diffusion generation, and saves results as NIfTI files.
    Supports both BRAVO (single channel) and DUAL (two channel) modes.
    """
    batch_size_local = len(mask_images)

    # Convert masks to tensor
    masks = torch.stack([
        torch.from_numpy(mask).unsqueeze(0)
        for mask in mask_images
    ], dim=0).to(device, dtype=torch.float32)

    # Determine output channels based on mode
    out_channels = 1 if mode == ModeType.BRAVO else 2

    # Create noise and model input
    noise = torch.randn(
        (batch_size_local, out_channels, image_size, image_size), device=device
    )
    model_input = torch.cat([noise, masks], dim=1)

    # Run generation
    generated_images = _run_diffusion_generation(
        model_input, image_model, strategy, num_steps, device, use_progress_bars
    )

    # Save results (mode-specific output formatting)
    for i in range(batch_size_local):
        output_path = os.path.join(save_dir, f"{start_counters[i]:05d}.nii.gz")
        mask_cpu = masks[i:i + 1].cpu().unsqueeze(-1)

        if mode == ModeType.BRAVO:
            # BRAVO: [image, mask]
            image_cpu = generated_images[i:i + 1].cpu().unsqueeze(-1)
            combined = torch.cat((image_cpu[0, 0], mask_cpu[0, 0]), dim=2)
        else:
            # DUAL: [t1_pre, t1_gd, mask]
            t1_pre_cpu = torch.clamp(generated_images[i, 0:1].cpu().unsqueeze(-1), 0, 1)
            t1_gd_cpu = torch.clamp(generated_images[i, 1:2].cpu().unsqueeze(-1), 0, 1)
            mask_cpu = torch.clamp(mask_cpu, 0, 1)
            combined = torch.cat((t1_pre_cpu[0], t1_gd_cpu[0], mask_cpu[0, 0]), dim=2)

        _save_nifti(combined, output_path, start_counters[i])

    return True


def main() -> None:
    """Main entry point for image generation."""
    args = parse_arguments()

    device = torch.device("cuda")
    batch_size = auto_adjust_batch_size(args.batch_size, device)

    # Initialize strategy
    if args.strategy == 'ddpm':
        strategy: DiffusionStrategy = DDPMStrategy()
    else:
        strategy = RFlowStrategy()

    strategy.setup_scheduler(1000, args.image_size)

    tracker = ValidationTracker()
    mask_cache: List[Tuple[np.ndarray, int]] = []

    mode_description = "BRAVO images" if args.mode == ModeType.BRAVO else "T1 pre+gd images"
    log.info("")
    log.info("=" * 60)
    log.info(f"Generating {mode_description} with {args.strategy}")
    log.info(f"Target: {args.num_images} image pairs | Batch size: {batch_size}")
    log.info(f"Steps: {args.num_steps} | {get_vram_usage(device)}")
    log.info("=" * 60)
    log.info("")

    # Load models
    log.info("Loading segmentation model...")
    seg_model = load_model(args.seg_model, in_channels=1, out_channels=1, device=device)

    log.info(f"Loading image model ({args.mode})...")

    if args.mode == ModeType.BRAVO:
        in_channels = 2
        out_channels = 1
    else:
        in_channels = 3
        out_channels = 2

    image_model = load_model(
        args.image_model, in_channels=in_channels,
        out_channels=out_channels, device=device
    )

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Saving to: {save_dir}")
    log.info("")

    # Generation loop
    start_time = time.time()
    current_image = args.current_image

    use_progress_bars = not args.no_progress

    # Define batch processing function (used by _flush_mask_cache)
    def process_batch(masks: List[np.ndarray], counters: List[int]) -> None:
        _process_mask_batch(
            masks, counters, image_model, strategy, args.num_steps,
            args.image_size, save_dir, args.mode, device,
            use_progress_bars, args.num_images
        )

    while current_image < args.num_images:
        tracker.start_batch()

        noise = torch.randn(
            (batch_size, 1, args.image_size, args.image_size), device=device
        )

        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            with torch.no_grad():
                seg_masks = strategy.generate(
                    seg_model, noise, args.num_steps,
                    device, use_progress_bars=use_progress_bars
                )

        valid_count = 0
        for j in range(len(seg_masks)):
            if current_image >= args.num_images:
                break

            mask = seg_masks[j, 0].detach().cpu().numpy()
            mask_min, mask_max = np.min(mask), np.max(mask)
            if mask_max > mask_min:
                mask = (mask - mask_min) / (mask_max - mask_min)
            else:
                mask = np.zeros_like(mask)  # Constant mask → empty (filtered by is_valid_mask)
            mask = make_binary(mask, threshold=BINARY_THRESHOLD_GEN)

            if is_valid_mask(mask):
                mask_cache.append((mask, current_image))
                valid_count += 1
                current_image += 1

        tracker.record_generation(valid_count, batch_size)
        log_mask_batch(tracker, batch_size, valid_count, len(mask_cache), device)

        # Process full batches from cache
        mask_cache = _flush_mask_cache(mask_cache, batch_size, process_batch)
        torch.cuda.empty_cache()

    # Process remaining masks in cache (filter out any past target)
    mask_cache = [(m, c) for m, c in mask_cache if c < args.num_images]
    mask_cache = _flush_mask_cache(mask_cache, batch_size, process_batch)

    # Final partial batch
    if len(mask_cache) > 0:
        log.info(f"Processing final partial batch: {len(mask_cache)} masks")
        batch_masks = [mask_cache[i][0] for i in range(len(mask_cache))]
        batch_counters = [mask_cache[i][1] for i in range(len(mask_cache))]

        _process_mask_batch(
            batch_masks, batch_counters, image_model,
            strategy, args.num_steps, args.image_size,
            save_dir, args.mode, device, use_progress_bars, args.num_images
        )

    # Summary
    total_time = time.time() - start_time
    final_success_rate = tracker.get_success_rate()

    log.info("")
    log.info("=" * 60)
    log.info(f"Generation complete | {get_vram_usage(device)}")
    log.info(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    log.info(f"Generated: {current_image}/{args.num_images} valid image pairs")
    log.info(f"Final success rate: {final_success_rate:.1f}%")
    log.info(f"Average time per valid image: {total_time / max(current_image, 1):.2f}s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
