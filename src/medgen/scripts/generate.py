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

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.amp import autocast

from medgen.core import (
    MAX_WHITE_PERCENTAGE,
    setup_cuda_optimizations,
)
from medgen.data import binarize_seg
from medgen.data.loaders.datasets import create_size_bin_maps
from medgen.data.loaders.seg import DEFAULT_BIN_EDGES, compute_size_bins_3d
from medgen.data.utils import save_nifti
from medgen.diffusion import DDPMStrategy, DiffusionStrategy, RFlowStrategy, load_diffusion_model
from medgen.metrics.brain_mask import (
    create_brain_mask,
    is_seg_inside_atlas,
    load_brain_atlas,
    remove_tumors_outside_brain,
)

setup_cuda_optimizations()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


def sample_random_size_bins(min_tumors: int = 1, max_tumors: int = 5) -> list[int]:
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

    logger.info(f"GPU: {total_vram_gb:.1f}GB | Batch size: {base_batch_size} -> {adjusted}")
    return adjusted


def is_valid_mask(binary_mask: np.ndarray, max_white_percentage: float = MAX_WHITE_PERCENTAGE) -> bool:
    """Check if segmentation mask is valid (not empty, not too large).

    For 2D masks: checks overall percentage.
    For 3D masks: checks per-slice to maintain consistency with 2D threshold.
    """
    if binary_mask.ndim == 2:
        # 2D: simple percentage check
        white_percentage = np.mean(binary_mask)
        return 0.0 < white_percentage < max_white_percentage
    elif binary_mask.ndim == 3:
        # 3D: check per-slice (assumes [D, H, W] format)
        # Must have some tumor overall
        overall_pct = np.mean(binary_mask)
        if overall_pct == 0.0:
            return False
        # No single slice should exceed the threshold
        for d in range(binary_mask.shape[0]):
            slice_pct = np.mean(binary_mask[d])
            if slice_pct >= max_white_percentage:
                return False
        return True
    else:
        # Fallback for other dimensions
        white_percentage = np.mean(binary_mask)
        return 0.0 < white_percentage < max_white_percentage


def compute_voxel_size(image_size: int, fov_mm: float = 240.0,
                       z_spacing_mm: float = 1.0) -> tuple[float, float, float]:
    """Compute voxel size based on resolution and field of view.

    Returns (xy_mm, xy_mm, z_mm) — used for NIfTI affine diagonals.

    **Convention note:** This returns (x, y, z) order suitable for NIfTI
    affine matrices and ``save_nifti()``. For 3D size-bin functions like
    ``compute_feret_diameter_3d`` which expect (depth, height, width) order,
    the caller must reorder: ``(z, xy, xy)`` instead of ``(xy, xy, z)``.
    The training dataloader (``seg.py``) reads voxel_spacing from config in
    the correct (D, H, W) order; ``generate.py`` currently passes this
    tuple directly, which works because xy ≈ xy (isotropic in-plane).

    Args:
        image_size: Image height/width in pixels.
        fov_mm: Field of view in millimeters (default 240.0).
        z_spacing_mm: Z-axis spacing in mm (default 1.0).

    Returns:
        Tuple of (x_mm, y_mm, z_mm) voxel dimensions.
    """
    xy_spacing = fov_mm / image_size
    return (xy_spacing, xy_spacing, z_spacing_mm)


def get_noise_shape(batch_size: int, channels: int, spatial_dims: int,
                    image_size: int, depth: int) -> tuple[int, ...]:
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
    conditioning: torch.Tensor | None = None,
    size_bins: torch.Tensor | None = None,
    bin_maps: torch.Tensor | None = None,
    cfg_scale: float = 1.0,
    cfg_scale_end: float | None = None,
    use_progress: bool = False,
    latent_channels: int = 1,
    diffrs_discriminator: object | None = None,
    diffrs_config: dict | None = None,
) -> torch.Tensor:
    """Generate a batch using diffusion model.

    Args:
        model: Diffusion model (may be SizeBinModelWrapper for seg_conditioned).
        strategy: Diffusion strategy (DDPM or RFlow).
        noise: Initial noise tensor.
        num_steps: Number of denoising steps.
        device: Torch device.
        conditioning: Optional conditioning tensor (e.g., seg mask for bravo).
        size_bins: Optional size bin tensor [B, num_bins] for seg_conditioned mode (FiLM).
        bin_maps: Optional spatial bin maps [B, num_bins, ...] for seg_conditioned_input mode.
                  These are concatenated with noise for input channel conditioning.
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance, >1.0 = stronger).
                   For dynamic CFG, this is the starting scale (at t=T).
        cfg_scale_end: Optional ending CFG scale (at t=0). If None, uses constant cfg_scale.
        use_progress: Show progress bar.
        latent_channels: Number of noise channels (1 for pixel, 4 for latent space).
        diffrs_discriminator: Optional DiffRSDiscriminator for rejection sampling.
        diffrs_config: Optional DiffRS config dict (rej_percentile, backsteps, etc.).

    Returns:
        Generated tensor.
    """
    if conditioning is not None:
        model_input = torch.cat([noise, conditioning], dim=1)
    else:
        model_input = noise

    # Build kwargs for strategy.generate()
    gen_kwargs: dict = dict(
        size_bins=size_bins,
        bin_maps=bin_maps,
        cfg_scale=cfg_scale,
        cfg_scale_end=cfg_scale_end,
        use_progress_bars=use_progress,
        latent_channels=latent_channels,
    )
    if diffrs_discriminator is not None:
        gen_kwargs['diffrs_discriminator'] = diffrs_discriminator
        gen_kwargs['diffrs_config'] = diffrs_config

    with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            return strategy.generate(
                model, model_input, num_steps, device,
                **gen_kwargs,
            )


def _build_diffrs(cfg: DictConfig, model: torch.nn.Module, device: torch.device):
    """Build DiffRS discriminator and config from generate config.

    Returns (discriminator, config_dict) or (None, None) if disabled.
    """
    diffrs_ckpt = cfg.get('diffrs_checkpoint', None)
    if not diffrs_ckpt:
        return None, None

    from medgen.diffusion.diffrs import DiffRSDiscriminator, load_diffrs_head

    head = load_diffrs_head(diffrs_ckpt, device)
    disc = DiffRSDiscriminator(model, head, device)
    diffrs_config = {
        'rej_percentile': cfg.get('diffrs_rej_percentile', 0.75),
        'backsteps': cfg.get('diffrs_backsteps', 1),
        'max_iter': cfg.get('diffrs_max_iter', 999999),
        'iter_warmup': cfg.get('diffrs_iter_warmup', 10),
    }
    logger.info("DiffRS enabled: %s", diffrs_ckpt)
    return disc, diffrs_config


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

    # ODE solver config (RFlow only)
    if hasattr(strategy, 'ode_solver'):
        strategy.ode_solver = cfg.get('ode_solver', 'euler')
        strategy.ode_atol = cfg.get('ode_atol', 1e-5)
        strategy.ode_rtol = cfg.get('ode_rtol', 1e-5)

    # EDM preconditioning (loaded from image model checkpoint)
    _img_ckpt = torch.load(cfg.image_model, map_location='cpu', weights_only=False)
    _img_cfg = _img_ckpt.get('config', {})
    _sigma_data = _img_cfg.get('sigma_data', 0.0)
    _out_ch = _img_cfg.get('out_channels', 1)
    del _img_ckpt
    if _sigma_data > 0 and hasattr(strategy, 'set_preconditioning'):
        strategy.set_preconditioning(_sigma_data, _out_ch)

    # Load models
    logger.info("Loading segmentation model...")
    seg_model = load_diffusion_model(
        cfg.seg_model, device=device,
        in_channels=1, out_channels=1, compile_model=True
    )

    logger.info(f"Loading image model ({cfg.gen_mode})...")
    if cfg.gen_mode == 'bravo':
        in_ch, out_ch = 2, 1
    else:  # dual
        in_ch, out_ch = 3, 2

    image_model = load_diffusion_model(
        cfg.image_model, device=device,
        in_channels=in_ch, out_channels=out_ch, compile_model=True
    )

    # DiffRS (opt-in, applied to image model only)
    diffrs_disc, diffrs_cfg = _build_diffrs(cfg, image_model, device)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve per-model step counts (fallback to num_steps)
    steps_seg = cfg.get('num_steps_seg', None) or cfg.num_steps
    steps_bravo = cfg.get('num_steps_bravo', None) or cfg.num_steps

    logger.info(f"Generating {cfg.num_images} samples...")

    current_image = cfg.current_image
    mask_cache: list[tuple[np.ndarray, int]] = []

    # Infinite loop protection
    MAX_CONSECUTIVE_FAILURES = 100
    consecutive_failures = 0

    while current_image < cfg.num_images:
        # Generate seg masks
        noise = torch.randn(get_noise_shape(batch_size, 1, 2, cfg.image_size, 0), device=device)
        seg_masks = generate_batch(seg_model, strategy, noise, steps_seg, device)

        # Validate and cache masks
        valid_in_batch = 0
        for j in range(len(seg_masks)):
            if current_image >= cfg.num_images:
                break

            mask = binarize_seg(seg_masks[j, 0]).cpu().numpy()

            if is_valid_mask(mask):
                mask_cache.append((mask, current_image))
                current_image += 1
                valid_in_batch += 1

        # Track consecutive failures to prevent infinite loop
        if valid_in_batch == 0:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(
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
            images = generate_batch(
                image_model, strategy, noise, steps_bravo, device, masks_tensor,
                diffrs_discriminator=diffrs_disc, diffrs_config=diffrs_cfg,
            )

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
            images = generate_batch(
                image_model, strategy, noise, steps_bravo, device, masks_tensor,
                diffrs_discriminator=diffrs_disc, diffrs_config=diffrs_cfg,
            )

            output_path = output_dir / f"{counter:05d}.nii.gz"
            if cfg.gen_mode == 'bravo':
                combined = np.stack([images[0, 0].cpu().numpy(), mask], axis=-1)
            else:
                combined = np.stack([images[0, 0].cpu().numpy(), images[0, 1].cpu().numpy(), mask], axis=-1)
            save_nifti(combined, str(output_path))

    logger.info(f"Saved {current_image} samples to {output_dir}")


def save_bins_csv(bins_data: list[tuple[int, list[int]]], output_path: Path) -> None:
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


def _generate_bravo(
    seg_binary: np.ndarray,
    bravo_model: torch.nn.Module,
    strategy: DiffusionStrategy,
    steps_bravo: int,
    device: torch.device,
    cfg: DictConfig,
    bravo_space: object | None,
    diffrs_disc: object | None = None,
    diffrs_cfg: dict | None = None,
) -> np.ndarray:
    """Generate a BRAVO volume conditioned on a binary seg mask.

    Returns:
        BRAVO image as numpy array [D, H, W] in [0, 1].
    """
    seg_tensor = torch.from_numpy(seg_binary).float().unsqueeze(0).unsqueeze(0).to(device)

    # Encode conditioning to match training (pixel normalization encodes labels)
    if bravo_space is not None:
        seg_tensor = bravo_space.encode(seg_tensor)

    noise = torch.randn(get_noise_shape(1, 1, 3, cfg.image_size, cfg.depth), device=device)
    bravo = generate_batch(bravo_model, strategy, noise, steps_bravo, device,
                           conditioning=seg_tensor,
                           cfg_scale=cfg.cfg_scale_bravo,
                           cfg_scale_end=cfg.get('cfg_scale_bravo_end', None),
                           diffrs_discriminator=diffrs_disc,
                           diffrs_config=diffrs_cfg)
    # Decode from diffusion space to pixel space, then clamp
    if bravo_space is not None:
        bravo = bravo_space.decode(bravo)
    return torch.clamp(bravo[0, 0], 0, 1).cpu().numpy()  # [D, H, W]


def run_3d_pipeline(cfg: DictConfig, output_dir: Path) -> None:
    """Run 3D generation pipeline: size_bins -> seg -> bravo.

    Output format:
        - bins.csv: All size bin information for all samples
        - {id}/seg.nii.gz: Segmentation mask volume [H, W, D]
        - {id}/bravo.nii.gz: BRAVO image volume [H, W, D] (bravo mode only)
    """
    # Validate gen_mode (fail-fast before loading models)
    VALID_3D_MODES = {'bravo', 'seg_conditioned', 'seg_conditioned_input'}
    if cfg.gen_mode not in VALID_3D_MODES:
        raise ValueError(f"Invalid gen_mode '{cfg.gen_mode}' for 3D. Valid: {VALID_3D_MODES}")

    device = torch.device("cuda")

    strategy = RFlowStrategy() if cfg.strategy == 'rflow' else DDPMStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000,
        image_size=cfg.image_size,
        depth_size=cfg.depth,
        spatial_dims=3,
    )

    # ODE solver config (RFlow only)
    if hasattr(strategy, 'ode_solver'):
        strategy.ode_solver = cfg.get('ode_solver', 'euler')
        strategy.ode_atol = cfg.get('ode_atol', 1e-5)
        strategy.ode_rtol = cfg.get('ode_rtol', 1e-5)

    # EDM preconditioning (loaded from bravo/image model checkpoint)
    # Determine which model to check based on gen_mode
    _precond_model = cfg.get('image_model', cfg.get('seg_model', None))
    if _precond_model:
        _pc_ckpt = torch.load(_precond_model, map_location='cpu', weights_only=False)
        _pc_cfg = _pc_ckpt.get('config', {})
        _sigma_data = _pc_cfg.get('sigma_data', 0.0)
        _out_ch = _pc_cfg.get('out_channels', 1)
        del _pc_ckpt
        if _sigma_data > 0 and hasattr(strategy, 'set_preconditioning'):
            strategy.set_preconditioning(_sigma_data, _out_ch)

    # Parse fixed size bins if provided
    fixed_bins = None
    if cfg.size_bins:
        fixed_bins = [int(x) for x in cfg.size_bins.split(',')]
        assert len(fixed_bins) == 7, "Size bins must have exactly 7 values"
        logger.info(f"Using fixed size bins: {fixed_bins}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load brain atlas
    brain_atlas = None
    atlas_path = cfg.get('brain_atlas_path', None)
    if atlas_path == 'auto':
        # Bundled atlas: data/brain_atlas_{H}x{W}x{D}.nii.gz relative to repo root
        repo_root = Path(__file__).resolve().parents[3]  # src/medgen/scripts -> repo root
        atlas_path = repo_root / 'data' / f'brain_atlas_{cfg.image_size}x{cfg.image_size}x{cfg.depth}.nii.gz'
        if not atlas_path.exists():
            logger.warning(f"Bundled brain atlas not found: {atlas_path} (skipping atlas validation)")
            atlas_path = None
    if atlas_path:
        expected_shape = (cfg.depth, cfg.image_size, cfg.image_size)
        brain_atlas = load_brain_atlas(atlas_path, expected_shape=expected_shape)
        logger.info(f"Brain atlas loaded: {atlas_path} (coverage: {brain_atlas.mean():.1%})")

    # Resolve per-model step counts (fallback to num_steps)
    steps_seg = cfg.get('num_steps_seg', None) or cfg.num_steps
    steps_bravo = cfg.get('num_steps_bravo', None) or cfg.num_steps

    # Log output dimensions
    trim_slices = cfg.get('trim_slices', 10)
    output_depth = cfg.depth - trim_slices if trim_slices > 0 else cfg.depth
    logger.info(f"Output volume: {cfg.image_size}x{cfg.image_size}x{output_depth} (gen {cfg.depth}, trim {trim_slices})")
    if steps_seg != steps_bravo:
        logger.info(f"Steps: seg={steps_seg}, bravo={steps_bravo}")
    else:
        logger.info(f"Steps: {steps_seg}")

    # Collect all bins info for CSV
    all_bins: list[tuple[int, list[int]]] = []

    # Mode: seg_conditioned only (just generate seg masks)
    if cfg.gen_mode == 'seg_conditioned':
        logger.info("Loading seg_conditioned model...")
        seg_model = load_diffusion_model(
            cfg.seg_model, device=device,
            in_channels=1, out_channels=1, compile_model=True, spatial_dims=3
        )

        # Size bin validation settings
        validate_size_bins = cfg.get('validate_size_bins', True)
        voxel_spacing = compute_voxel_size(cfg.image_size, cfg.get('fov_mm', 240.0))
        bin_edges = list(cfg.get('bin_edges', DEFAULT_BIN_EDGES))
        num_bins = cfg.get('num_bins', 7)
        max_retries = cfg.get('max_retries', 10)

        # Atlas validation settings
        brain_tolerance = cfg.get('brain_tolerance', 0.0)
        brain_dilate = cfg.get('brain_dilate_pixels', 0)

        logger.info(f"Generating {cfg.num_images} seg masks...")
        if validate_size_bins:
            logger.info("Size bin validation: enabled (verify generated seg matches conditioning)")
        if brain_atlas is not None:
            logger.info(f"Atlas validation: enabled (tolerance={brain_tolerance:.0%}, dilate={brain_dilate}px)")

        generated = 0
        total_retries = 0

        while generated < cfg.num_images:
            bins = fixed_bins if fixed_bins else sample_random_size_bins(cfg.min_tumors, cfg.max_tumors)
            size_bins = torch.tensor([bins], dtype=torch.long, device=device)

            # Retry loop for valid seg mask
            valid_mask = False
            retries = 0
            while not valid_mask and retries < max_retries:
                noise = torch.randn(get_noise_shape(1, 1, 3, cfg.image_size, cfg.depth), device=device)
                seg = generate_batch(seg_model, strategy, noise, steps_seg, device,
                                     size_bins=size_bins,
                                     cfg_scale=cfg.cfg_scale_seg,
                                     cfg_scale_end=cfg.get('cfg_scale_seg_end', None))

                # Binarize
                seg_binary = binarize_seg(seg[0, 0]).cpu().numpy()

                # Validate size bins match conditioning
                if validate_size_bins:
                    actual_bins = compute_size_bins_3d(seg_binary, bin_edges, voxel_spacing, num_bins)
                    if not np.array_equal(actual_bins, np.array(bins)):
                        retries += 1
                        total_retries += 1
                        if cfg.verbose and retries == 1:
                            logger.warning(f"Sample {generated}: size bins mismatch "
                                       f"(requested={bins}, got={actual_bins.tolist()}), retrying...")
                        continue

                # Atlas validation: check tumors are inside brain atlas
                if brain_atlas is not None:
                    if not is_seg_inside_atlas(seg_binary, brain_atlas,
                                              tolerance=brain_tolerance,
                                              dilate_pixels=brain_dilate):
                        retries += 1
                        total_retries += 1
                        if cfg.verbose and retries == 1:
                            logger.warning(f"Sample {generated}: seg outside brain atlas, retrying...")
                        continue

                valid_mask = True

            if not valid_mask:
                logger.warning(f"Sample {generated}: failed after {max_retries} retries, using last attempt")

            # Transpose [D, H, W] -> [H, W, D] for NIfTI (slices should be HxW)
            seg_binary = np.transpose(seg_binary, (1, 2, 0))

            # Trim last N slices to match training data (training pads, so we remove padding)
            trim_slices = cfg.get('trim_slices', 10)
            if trim_slices > 0:
                seg_binary = seg_binary[:, :, :-trim_slices]  # [H, W, D-trim]

            # Save in subdirectory: 00000/seg.nii.gz
            sample_dir = output_dir / f"{generated:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            voxel = compute_voxel_size(cfg.image_size, cfg.get('fov_mm', 240.0))
            save_nifti(seg_binary, str(sample_dir / "seg.nii.gz"), voxel_size=voxel)
            all_bins.append((generated, bins))
            generated += 1

            # Log progress and clear cache periodically
            if generated % 10 == 0 or generated == cfg.num_images:
                logger.info(f"Progress: {generated}/{cfg.num_images}")
                torch.cuda.empty_cache()

        if total_retries > 0:
            logger.info(f"Generation complete. Total retries: {total_retries}")

    # Mode: bravo (full pipeline: seg -> bravo)
    elif cfg.gen_mode == 'bravo':
        logger.info("Loading seg_conditioned model...")
        seg_model = load_diffusion_model(
            cfg.seg_model, device=device,
            in_channels=1, out_channels=1, compile_model=True, spatial_dims=3
        )

        logger.info("Loading bravo model...")
        bravo_model = load_diffusion_model(
            cfg.image_model, device=device,
            in_channels=2, out_channels=1, compile_model=True, spatial_dims=3
        )

        # Pixel normalization from bravo checkpoint (exp1b rescale, exp1c shift/scale)
        bravo_ckpt = torch.load(cfg.image_model, map_location='cpu', weights_only=False)
        bravo_pixel_cfg = bravo_ckpt.get('config', {}).get('pixel', {})
        del bravo_ckpt
        bravo_space = None
        _bravo_pixel_shift = bravo_pixel_cfg.get('pixel_shift')
        _bravo_pixel_scale = bravo_pixel_cfg.get('pixel_scale')
        _bravo_pixel_rescale = bravo_pixel_cfg.get('rescale', False)
        if _bravo_pixel_shift is not None or _bravo_pixel_rescale:
            from medgen.diffusion.spaces import PixelSpace
            bravo_space = PixelSpace(
                rescale=_bravo_pixel_rescale,
                shift=_bravo_pixel_shift,
                scale=_bravo_pixel_scale,
            )
            if _bravo_pixel_shift is not None:
                logger.info(f"Bravo pixel normalization: shift={_bravo_pixel_shift}, scale={_bravo_pixel_scale}")
            if _bravo_pixel_rescale:
                logger.info("Bravo pixel rescale: [-1, 1]")

        # DiffRS (opt-in, applied to bravo model only)
        diffrs_disc, diffrs_cfg = _build_diffrs(cfg, bravo_model, device)

        # Validation thresholds for 3D seg masks (per-slice, same as 2D)
        max_white_pct = cfg.get('max_white_percentage', MAX_WHITE_PERCENTAGE)
        max_retries = cfg.get('max_retries', 10)

        # Brain mask validation settings
        validate_brain_mask = cfg.get('validate_brain_mask', True)
        brain_threshold = cfg.get('brain_threshold', 0.05)
        brain_tolerance = cfg.get('brain_tolerance', 0.0)
        brain_dilate = cfg.get('brain_dilate_pixels', 0)

        # Size bin validation settings (verify generated seg matches conditioning)
        validate_size_bins = cfg.get('validate_size_bins', True)
        voxel_spacing = compute_voxel_size(cfg.image_size, cfg.get('fov_mm', 240.0))
        bin_edges = list(cfg.get('bin_edges', DEFAULT_BIN_EDGES))
        num_bins = cfg.get('num_bins', 7)

        logger.info(f"Generating {cfg.num_images} seg+bravo pairs...")
        logger.info(f"Seg validation: per-slice max {max_white_pct:.2%} (same as 2D threshold)")
        if brain_atlas is not None:
            logger.info(f"Stage 1 — Atlas validation: enabled (tolerance={brain_tolerance:.0%}, dilate={brain_dilate}px)")
        if validate_brain_mask:
            logger.info(f"Stage 2 — Brain mask validation: enabled (per-tumor cleanup, threshold={brain_threshold})")
        if validate_size_bins:
            logger.info("Size bin validation: enabled (verify generated seg matches conditioning)")

        generated = 0
        total_retries = 0
        brain_retries = 0
        max_brain_retries = cfg.get('max_brain_retries', 5)

        while generated < cfg.num_images:
            bins = fixed_bins if fixed_bins else sample_random_size_bins(cfg.min_tumors, cfg.max_tumors)
            size_bins = torch.tensor([bins], dtype=torch.long, device=device)

            # Retry loop for valid seg mask
            valid_mask = False
            retries = 0
            while not valid_mask and retries < max_retries:
                # Generate seg with size bin conditioning
                noise = torch.randn(get_noise_shape(1, 1, 3, cfg.image_size, cfg.depth), device=device)
                seg = generate_batch(seg_model, strategy, noise, steps_seg, device,
                                     size_bins=size_bins,
                                     cfg_scale=cfg.cfg_scale_seg,
                                     cfg_scale_end=cfg.get('cfg_scale_seg_end', None))

                # Binarize seg
                seg_binary = binarize_seg(seg[0, 0]).cpu().numpy()

                # Validate mask (per-slice check for 3D)
                if not is_valid_mask(seg_binary, max_white_pct):
                    retries += 1
                    total_retries += 1
                    if cfg.verbose and retries == 1:
                        overall_pct = np.mean(seg_binary)
                        logger.warning(f"Sample {generated}: invalid seg mask ({overall_pct:.2%} overall), retrying...")
                    continue

                # Validate size bins match conditioning
                if validate_size_bins:
                    actual_bins = compute_size_bins_3d(seg_binary, bin_edges, voxel_spacing, num_bins)
                    if not np.array_equal(actual_bins, np.array(bins)):
                        retries += 1
                        total_retries += 1
                        if cfg.verbose and retries == 1:
                            logger.warning(f"Sample {generated}: size bins mismatch "
                                       f"(requested={bins}, got={actual_bins.tolist()}), retrying...")
                        continue

                valid_mask = True

            if not valid_mask:
                logger.warning(f"Sample {generated}: failed after {max_retries} retries, using last attempt")

            # Stage 1 — Atlas check (before BRAVO generation)
            if brain_atlas is not None:
                cleaned_seg, n_removed = remove_tumors_outside_brain(seg_binary, brain_atlas)
                if n_removed > 0:
                    logger.info(f"Sample {generated}: atlas check removed {n_removed} tumor(s)")
                    if cleaned_seg.sum() == 0:
                        # All tumors outside atlas — retry with new seg
                        total_retries += 1
                        continue
                    seg_binary = cleaned_seg

            # Generate BRAVO conditioned on seg mask
            bravo_np = _generate_bravo(
                seg_binary, bravo_model, strategy, steps_bravo, device, cfg,
                bravo_space, diffrs_disc, diffrs_cfg,
            )

            # Stage 2 — BRAVO brain mask check (per-tumor cleanup)
            if validate_brain_mask:
                brain_mask = create_brain_mask(
                    bravo_np, threshold=brain_threshold,
                    dilate_pixels=brain_dilate,
                )
                cleaned_seg, n_removed = remove_tumors_outside_brain(seg_binary, brain_mask)

                if n_removed > 0:
                    logger.info(f"Sample {generated}: bravo check removed {n_removed} tumor(s)")

                    if cleaned_seg.sum() == 0:
                        # All tumors outside — retry with new seg entirely
                        brain_retries += 1
                        total_retries += 1
                        if brain_retries < max_brain_retries:
                            if cfg.verbose:
                                logger.warning(f"Sample {generated}: all tumors outside brain, retrying...")
                            continue
                        else:
                            logger.warning(f"Sample {generated}: max brain retries ({max_brain_retries}) reached, using anyway")
                            brain_retries = 0
                    else:
                        # Re-generate BRAVO with cleaned seg mask
                        seg_binary = cleaned_seg
                        bravo_np = _generate_bravo(
                            seg_binary, bravo_model, strategy, steps_bravo, device, cfg,
                            bravo_space, diffrs_disc, diffrs_cfg,
                        )

            # Reset brain retries on success
            brain_retries = 0

            # Transpose [D, H, W] -> [H, W, D] for NIfTI (slices should be HxW)
            seg_binary_save = np.transpose(seg_binary, (1, 2, 0))
            bravo_np = np.transpose(bravo_np, (1, 2, 0))

            # Trim last N slices to match training data (training pads, so we remove padding)
            trim_slices = cfg.get('trim_slices', 10)
            if trim_slices > 0:
                seg_binary_save = seg_binary_save[:, :, :-trim_slices]  # [H, W, D-trim]
                bravo_np = bravo_np[:, :, :-trim_slices]

            # Save in subdirectory: 00000/seg.nii.gz and 00000/bravo.nii.gz
            sample_dir = output_dir / f"{generated:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            voxel = compute_voxel_size(cfg.image_size, cfg.get('fov_mm', 240.0))
            save_nifti(seg_binary_save, str(sample_dir / "seg.nii.gz"), voxel_size=voxel)
            save_nifti(bravo_np, str(sample_dir / "bravo.nii.gz"), voxel_size=voxel)
            all_bins.append((generated, bins))

            generated += 1

            # Log progress and clear cache periodically
            if generated % 10 == 0 or generated == cfg.num_images:
                logger.info(f"Progress: {generated}/{cfg.num_images}")
                torch.cuda.empty_cache()

        if total_retries > 0:
            logger.info(f"Total retries: {total_retries} (seg validation + brain mask)")

    # Mode: seg_conditioned_input (input channel conditioning - stronger than FiLM)
    elif cfg.gen_mode == 'seg_conditioned_input':
        logger.info("Loading seg_conditioned_input model...")
        # Model has 8 input channels: 1 noisy_seg + 7 bin_maps
        seg_model = load_diffusion_model(
            cfg.seg_model, device=device,
            in_channels=8, out_channels=1, compile_model=True, spatial_dims=3
        )

        # Size bin validation settings
        validate_size_bins = cfg.get('validate_size_bins', True)
        voxel_spacing = compute_voxel_size(cfg.image_size, cfg.get('fov_mm', 240.0))
        bin_edges = list(cfg.get('bin_edges', DEFAULT_BIN_EDGES))
        num_bins = cfg.get('num_bins', 7)
        max_count = cfg.get('max_count', 10)
        max_retries = cfg.get('max_retries', 10)

        # Atlas validation settings
        brain_tolerance = cfg.get('brain_tolerance', 0.0)
        brain_dilate = cfg.get('brain_dilate_pixels', 0)

        logger.info(f"Generating {cfg.num_images} seg masks with input conditioning...")
        if validate_size_bins:
            logger.info("Size bin validation: enabled (verify generated seg matches conditioning)")
        if brain_atlas is not None:
            logger.info(f"Atlas validation: enabled (tolerance={brain_tolerance:.0%}, dilate={brain_dilate}px)")

        generated = 0
        total_retries = 0

        while generated < cfg.num_images:
            bins = fixed_bins if fixed_bins else sample_random_size_bins(cfg.min_tumors, cfg.max_tumors)
            size_bins_tensor = torch.tensor(bins, dtype=torch.long, device=device)

            # Convert size_bins to spatial bin_maps [1, 7, D, H, W]
            spatial_shape = (cfg.depth, cfg.image_size, cfg.image_size)
            bin_maps = create_size_bin_maps(
                size_bins_tensor, spatial_shape, normalize=True, max_count=max_count
            ).unsqueeze(0).to(device)  # [1, 7, D, H, W]

            # Retry loop for valid seg mask
            valid_mask = False
            retries = 0
            while not valid_mask and retries < max_retries:
                noise = torch.randn(get_noise_shape(1, 1, 3, cfg.image_size, cfg.depth), device=device)
                seg = generate_batch(seg_model, strategy, noise, steps_seg, device,
                                     bin_maps=bin_maps,
                                     cfg_scale=cfg.cfg_scale_seg,
                                     cfg_scale_end=cfg.get('cfg_scale_seg_end', None))

                # Binarize
                seg_binary = binarize_seg(seg[0, 0]).cpu().numpy()

                # Validate size bins match conditioning
                if validate_size_bins:
                    actual_bins = compute_size_bins_3d(seg_binary, bin_edges, voxel_spacing, num_bins)
                    if not np.array_equal(actual_bins, np.array(bins)):
                        retries += 1
                        total_retries += 1
                        if cfg.verbose and retries == 1:
                            logger.warning(f"Sample {generated}: size bins mismatch "
                                       f"(requested={bins}, got={actual_bins.tolist()}), retrying...")
                        continue

                # Atlas validation: check tumors are inside brain atlas
                if brain_atlas is not None:
                    if not is_seg_inside_atlas(seg_binary, brain_atlas,
                                              tolerance=brain_tolerance,
                                              dilate_pixels=brain_dilate):
                        retries += 1
                        total_retries += 1
                        if cfg.verbose and retries == 1:
                            logger.warning(f"Sample {generated}: seg outside brain atlas, retrying...")
                        continue

                valid_mask = True

            if not valid_mask:
                logger.warning(f"Sample {generated}: failed after {max_retries} retries, using last attempt")

            # Transpose [D, H, W] -> [H, W, D] for NIfTI (slices should be HxW)
            seg_binary = np.transpose(seg_binary, (1, 2, 0))

            # Trim last N slices to match training data (training pads, so we remove padding)
            trim_slices = cfg.get('trim_slices', 10)
            if trim_slices > 0:
                seg_binary = seg_binary[:, :, :-trim_slices]  # [H, W, D-trim]

            # Save in subdirectory: 00000/seg.nii.gz
            sample_dir = output_dir / f"{generated:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            voxel = compute_voxel_size(cfg.image_size, cfg.get('fov_mm', 240.0))
            save_nifti(seg_binary, str(sample_dir / "seg.nii.gz"), voxel_size=voxel)
            all_bins.append((generated, bins))
            generated += 1

            # Log progress and clear cache periodically
            if generated % 10 == 0 or generated == cfg.num_images:
                logger.info(f"Progress: {generated}/{cfg.num_images}")
                torch.cuda.empty_cache()

        if total_retries > 0:
            logger.info(f"Generation complete. Total retries: {total_retries}")

    else:
        raise ValueError(f"Mode '{cfg.gen_mode}' not supported for 3D. Use 'seg_conditioned', 'seg_conditioned_input', or 'bravo'.")

    # Save all bins to single CSV file
    save_bins_csv(all_bins, output_dir / "bins.csv")
    logger.info(f"Saved {len(all_bins)} samples to {output_dir}")
    logger.info(f"Bins info saved to {output_dir / 'bins.csv'}")


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

    # Compute voxel size from resolution
    fov_mm = cfg.get('fov_mm', 240.0)
    voxel = compute_voxel_size(cfg.image_size, fov_mm)

    logger.info("=" * 60)
    logger.info(f"Generation: {cfg.spatial_dims}D | Mode: {cfg.gen_mode} | Strategy: {cfg.strategy}")
    logger.info(f"Resolution: {cfg.image_size}x{cfg.image_size} | FOV: {fov_mm}mm | Voxel: {voxel[0]:.3f}mm")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Paths: {cfg.paths.name}")
    logger.info("=" * 60)

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

    logger.info("Generation complete!")


if __name__ == "__main__":
    main()
