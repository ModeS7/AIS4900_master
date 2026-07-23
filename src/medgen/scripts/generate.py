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
import nibabel as nib
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
from medgen.diffusion import (
    BridgeStrategy,
    DDPMStrategy,
    DiffusionStrategy,
    IRSDEStrategy,
    ResfusionStrategy,
    RFlowStrategy,
    load_diffusion_model,
)


def _create_strategy(name: str) -> DiffusionStrategy:
    """Create diffusion strategy by name. Keep in sync with DiffusionTrainerBase._create_strategy."""
    strategies: dict[str, type] = {
        'ddpm': DDPMStrategy,
        'rflow': RFlowStrategy,
        'bridge': BridgeStrategy,
        'irsde': IRSDEStrategy,
        'resfusion': ResfusionStrategy,
    }
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Choose from {list(strategies.keys())}")
    return strategies[name]()


from medgen.metrics.brain_mask import (
    BrainPCAModel,
    create_brain_mask,
    evaluate_conditioning_brain_containment,
    has_single_brain_component,
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


def _generated_seg_rejection_reason(
    binary_mask: np.ndarray,
    *,
    max_white_percentage: float | None,
    brain_atlas: np.ndarray | None = None,
    brain_tolerance: float = 0.0,
    brain_dilate_pixels: int = 0,
) -> str | None:
    """Return a stable rejection reason for one generated ``[D, H, W]`` mask.

    This is a coarse artifact filter, not a distribution-matching score. It
    leaves accepted lesion morphology unchanged.
    """
    if binary_mask.ndim != 3:
        return 'wrong_shape'
    if not np.isfinite(binary_mask).all():
        return 'non_finite'
    if max_white_percentage is not None and not 0.0 < max_white_percentage <= 1.0:
        raise ValueError("max_white_percentage must be in (0, 1]")
    if not 0.0 <= brain_tolerance <= 1.0:
        raise ValueError("brain_tolerance must be in [0, 1]")
    if brain_dilate_pixels < 0:
        raise ValueError("brain_dilate_pixels must be non-negative")

    foreground = binary_mask > 0.5
    if not foreground.any():
        return 'empty'
    if max_white_percentage is not None:
        slice_fractions = foreground.reshape(foreground.shape[0], -1).mean(axis=1)
        if float(slice_fractions.max()) >= max_white_percentage:
            return 'slice_foreground_limit'

    if brain_atlas is not None:
        if brain_atlas.shape != foreground.shape:
            raise ValueError(
                f"Brain atlas shape {brain_atlas.shape} does not match mask shape "
                f"{foreground.shape}"
            )
        if not is_seg_inside_atlas(
            foreground,
            brain_atlas,
            tolerance=brain_tolerance,
            dilate_pixels=brain_dilate_pixels,
        ):
            return 'outside_brain_atlas'
    return None


def _xyz_to_dhw(voxel_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    """Reorder (x, y, z) voxel spacing to (D, H, W) expected by size-bin / Feret functions.

    compute_voxel_size returns NIfTI-style (x, y, z); compute_size_bins_3d and
    compute_feret_diameter_3d expect (D, H, W). In practice H == W for isotropic
    in-plane data, so the primary effect is swapping D (was x) with W (was z).
    """
    x, y, z = voxel_xyz
    return (z, y, x)


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


_SEED_STREAM_OFFSETS = {
    # Keep the first BRAVO draw exactly base_seed + sample_index. This is the
    # pre-specified stream used by the fixed-mask paper evaluation.
    'bravo': 0,
    'seg': 1_000_000_000,
    'seg_input': 2_000_000_000,
}
_SEED_ATTEMPT_STRIDE = 1_000_000


def _derive_sample_seed(
    base_seed: int | None,
    sample_index: int,
    *,
    stream: str = 'bravo',
    attempt: int = 0,
) -> int | None:
    """Derive an order-independent seed for one sample and retry stream.

    ``None`` deliberately returns ``None`` so historical stochastic generation
    remains the default. The first BRAVO attempt is ``base_seed + sample_index``;
    other stages and retries occupy disjoint deterministic streams.
    """
    if base_seed is None:
        return None
    if stream not in _SEED_STREAM_OFFSETS:
        raise ValueError(f"Unknown generation seed stream: {stream}")
    if sample_index < 0 or attempt < 0:
        raise ValueError("sample_index and attempt must be non-negative")
    seed = int(base_seed) + sample_index + _SEED_STREAM_OFFSETS[stream] + attempt * _SEED_ATTEMPT_STRIDE
    if not 0 <= seed < 2**63:
        raise ValueError(f"Derived generation seed is outside [0, 2**63): {seed}")
    return seed


def _torch_generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    """Create a local generator without perturbing PyTorch's global RNG."""
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _randn(
    shape: tuple[int, ...],
    device: torch.device,
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw noise, optionally from a local deterministic generator."""
    if seed is not None and generator is not None:
        raise ValueError("Pass either seed or generator, not both")
    local_generator = generator if generator is not None else _torch_generator(device, seed)
    return torch.randn(shape, device=device, generator=local_generator)


def _seed_process_rngs(seed: int | None) -> None:
    """Seed non-sample RNGs when deterministic generation is requested."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_image_model_with_optional_handoff(
    cfg: DictConfig,
    device: torch.device,
    *,
    in_channels: int,
    out_channels: int,
    spatial_dims: int,
    compile_model: bool = True,
) -> torch.nn.Module:
    """Load one image model or a high-t/low-t handoff pair.

    Both the 2D and 3D generation paths call this helper, preventing the 3D
    path from silently ignoring ``image_model_high_t``.
    """
    low_path = cfg.image_model
    high_path = cfg.get('image_model_high_t', None)
    if not high_path:
        return load_diffusion_model(
            low_path,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            compile_model=compile_model,
            spatial_dims=spatial_dims,
        )

    from medgen.models.handoff import HandoffWrapper

    logger.info(
        "Two-model handoff: high-t ← %s, low-t ← %s, handoff_t=%s",
        high_path, low_path, cfg.handoff_t,
    )
    if compile_model:
        logger.info(
            "Disabling torch.compile for the two-model handoff to avoid retaining "
            "CUDA-graph private pools for one model while the other model runs"
        )
    low_model = load_diffusion_model(
        low_path,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        compile_model=False,
        spatial_dims=spatial_dims,
    )
    high_model = load_diffusion_model(
        high_path,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        compile_model=False,
        spatial_dims=spatial_dims,
    )
    return HandoffWrapper(
        high_t_model=high_model,
        low_t_model=low_model,
        handoff_t=float(cfg.handoff_t),
        num_train_timesteps=1000,
    )


def _get_offset_noise_config(checkpoint_path: str) -> tuple[bool, float]:
    """Extract adjusted offset noise config from a checkpoint.

    Returns (adjusted, strength). If not configured, returns (False, 0.0).
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        offset_cfg = cfg.get('offset_noise', {})
        enabled = offset_cfg.get('enabled', False)
        adjusted = offset_cfg.get('adjusted', False)
        strength = offset_cfg.get('strength', 0.1)
        del ckpt
        if enabled and adjusted:
            return True, strength
    except Exception as e:
        logger.warning(
            f"Could not load offset noise config from {checkpoint_path}: {type(e).__name__}: {e}. "
            "Falling back to offset-disabled — generated samples may diverge from training distribution."
        )
    return False, 0.0


def _maybe_add_generation_offset(
    noise: torch.Tensor,
    adjusted: bool,
    strength: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply generation offset noise, using the sample-local RNG when provided."""
    if not adjusted:
        return noise
    offset_shape = list(noise.shape[:2]) + [1] * (noise.ndim - 2)
    offset = torch.randn(
        offset_shape,
        device=noise.device,
        dtype=noise.dtype,
        generator=generator,
    )
    return noise + strength * offset


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
    cfg_mode: str = 'standard',
    cfg_zero_init_steps: int = 1,
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
        cfg_mode: CFG mode ('standard' or 'zero_star' for CFG-Zero*).
        cfg_zero_init_steps: Zero-velocity steps for zero_star mode.

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
        cfg_mode=cfg_mode,
        cfg_zero_init_steps=cfg_zero_init_steps,
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
    _seed_process_rngs(cfg.get('seed', None))
    batch_size = auto_adjust_batch_size(cfg.batch_size, 2, device)

    # Initialize strategy
    strategy: DiffusionStrategy = _create_strategy(cfg.strategy)
    strategy.setup_scheduler(1000, cfg.image_size)

    # ODE solver config (RFlow only)
    if hasattr(strategy, 'ode_solver'):
        strategy.ode_solver = cfg.get('ode_solver', 'euler')
        strategy.ode_atol = cfg.get('ode_atol', 1e-5)
        strategy.ode_rtol = cfg.get('ode_rtol', 1e-5)

    # EDM preconditioning + pixel normalization (loaded from image model checkpoint)
    _img_ckpt = torch.load(cfg.image_model, map_location='cpu', weights_only=False)
    _img_cfg = _img_ckpt.get('config', {})
    _sigma_data = _img_cfg.get('sigma_data', 0.0)
    _out_ch = _img_cfg.get('out_channels', 1)

    # Pixel normalization from bravo checkpoint (same as 3D pipeline)
    _pixel_cfg = _img_cfg.get('pixel', {})
    _pixel_shift = _pixel_cfg.get('pixel_shift')
    _pixel_scale = _pixel_cfg.get('pixel_scale')
    _pixel_rescale = _pixel_cfg.get('rescale', False)
    bravo_space = None
    if _pixel_shift is not None or _pixel_rescale:
        from medgen.diffusion.spaces import PixelSpace
        bravo_space = PixelSpace(
            rescale=_pixel_rescale,
            shift=_pixel_shift,
            scale=_pixel_scale,
        )
        if _pixel_shift is not None:
            logger.info(f"Bravo pixel normalization: shift={_pixel_shift}, scale={_pixel_scale}")
        if _pixel_rescale:
            logger.info("Bravo pixel rescale: [-1, 1]")

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

    image_model = _load_image_model_with_optional_handoff(
        cfg,
        device,
        in_channels=in_ch,
        out_channels=out_ch,
        spatial_dims=2,
        compile_model=True,
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

            # Encode conditioning to match training pixel normalization
            if bravo_space is not None:
                masks_tensor = bravo_space.encode(masks_tensor)

            # Generate images
            out_ch = 1 if cfg.gen_mode == 'bravo' else 2
            noise = torch.randn(get_noise_shape(batch_size, out_ch, 2, cfg.image_size, 0), device=device)
            images = generate_batch(
                image_model, strategy, noise, steps_bravo, device, masks_tensor,
                diffrs_discriminator=diffrs_disc, diffrs_config=diffrs_cfg,
            )

            # Decode from pixel normalization space
            if bravo_space is not None:
                images = bravo_space.decode(images)

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
            if bravo_space is not None:
                masks_tensor = bravo_space.encode(masks_tensor)
            out_ch = 1 if cfg.gen_mode == 'bravo' else 2
            noise = torch.randn(get_noise_shape(1, out_ch, 2, cfg.image_size, 0), device=device)
            images = generate_batch(
                image_model, strategy, noise, steps_bravo, device, masks_tensor,
                diffrs_discriminator=diffrs_disc, diffrs_config=diffrs_cfg,
            )
            if bravo_space is not None:
                images = bravo_space.decode(images)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        # Header: id, bin_0, bin_1, ..., bin_6, total_tumors
        f.write('id,bin_0,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,total_tumors\n')
        for sample_id, bins in bins_data:
            bins_str = ','.join(map(str, bins))
            total = sum(bins)
            f.write(f'{sample_id:05d},{bins_str},{total}\n')


def _discover_real_seg_files(
    real_seg_dir: str | Path,
    *,
    num_images: int,
    expected_cases: int | None = None,
    require_bravo_pairs: bool = False,
) -> list[Path]:
    """Find and strictly validate a real conditioning-mask pool."""
    directory = Path(real_seg_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Real seg directory does not exist: {directory}")
    files = sorted(directory.glob('*/seg.nii.gz'))
    if not files:
        raise FileNotFoundError(f"No seg.nii.gz files found in {directory}")

    patient_ids = [path.parent.name for path in files]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError(f"Duplicate patient identifiers found under {directory}")
    if expected_cases is not None and len(files) != int(expected_cases):
        raise ValueError(
            f"Expected exactly {expected_cases} real masks under {directory}, found {len(files)}"
        )
    if num_images > len(files):
        raise ValueError(
            f"Requested {num_images} samples but only {len(files)} unique real masks are available. "
            "Real masks are not cycled."
        )
    if require_bravo_pairs:
        bravo_files = sorted(directory.glob('*/bravo.nii.gz'))
        bravo_ids = {path.parent.name for path in bravo_files}
        mask_ids = set(patient_ids)
        missing = sorted(mask_ids - bravo_ids)
        extra = sorted(bravo_ids - mask_ids)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing BRAVO for {', '.join(missing[:5])}")
            if extra:
                details.append(f"BRAVO without mask for {', '.join(extra[:5])}")
            raise ValueError(
                "Real mask/BRAVO patient identifiers do not match: " + '; '.join(details)
            )
    return files


def _validate_real_seg_volume(
    volume: np.ndarray,
    *,
    source: Path,
    image_size: int,
    expected_depth: int | None = None,
) -> None:
    """Reject malformed real conditioning masks before generation."""
    if volume.ndim != 3:
        raise ValueError(f"Real mask {source} must be 3D, got shape {volume.shape}")
    if volume.shape[:2] != (image_size, image_size):
        raise ValueError(
            f"Real mask {source} has in-plane shape {volume.shape[:2]}, "
            f"expected {(image_size, image_size)}"
        )
    if expected_depth is not None and volume.shape[2] != expected_depth:
        raise ValueError(
            f"Real mask {source} has depth {volume.shape[2]}, expected {expected_depth}"
        )
    if not np.isfinite(volume).all():
        raise ValueError(f"Real mask {source} contains non-finite values")
    close_to_binary = np.isclose(volume, 0.0, atol=1e-6) | np.isclose(volume, 1.0, atol=1e-6)
    if not close_to_binary.all():
        invalid = np.unique(volume[~close_to_binary])
        raise ValueError(
            f"Real mask {source} is not binary; values include {invalid[:10]}"
        )
    if not np.any(volume > 0.5):
        raise ValueError(f"Real mask {source} is empty after thresholding at 0.5")


def _preflight_real_seg_files(
    files: list[Path],
    *,
    num_images: int,
    image_size: int,
    expected_depth: int | None = None,
) -> None:
    """Validate every selected real mask before loading a GPU model."""
    for path in files[:num_images]:
        volume = nib.load(str(path)).get_fdata().astype(np.float32)  # type: ignore[attr-defined]
        _validate_real_seg_volume(
            volume,
            source=path,
            image_size=image_size,
            expected_depth=expected_depth,
        )


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
    offset_adjusted: bool = False,
    offset_strength: float = 0.0,
    noise_seed: int | None = None,
) -> np.ndarray:
    """Generate a BRAVO volume conditioned on a binary seg mask.

    Returns:
        BRAVO image as numpy array [D, H, W] in [0, 1].
    """
    seg_tensor = torch.from_numpy(seg_binary).float().unsqueeze(0).unsqueeze(0).to(device)

    # Encode conditioning to match training (pixel normalization encodes labels)
    if bravo_space is not None:
        seg_tensor = bravo_space.encode(seg_tensor)

    generator = _torch_generator(device, noise_seed)
    noise = _randn(
        get_noise_shape(1, 1, 3, cfg.image_size, cfg.depth),
        device,
        generator=generator,
    )
    noise = _maybe_add_generation_offset(
        noise,
        offset_adjusted,
        offset_strength,
        generator=generator,
    )
    bravo = generate_batch(bravo_model, strategy, noise, steps_bravo, device,
                           conditioning=seg_tensor,
                           cfg_scale=cfg.cfg_scale_bravo,
                           cfg_scale_end=cfg.get('cfg_scale_bravo_end', None),
                           diffrs_discriminator=diffrs_disc,
                           diffrs_config=diffrs_cfg,
                           cfg_mode=cfg.get('cfg_mode', 'standard'),
                           cfg_zero_init_steps=cfg.get('cfg_zero_init_steps', 1))
    # Decode from diffusion space to pixel space, then clamp
    if bravo_space is not None:
        bravo = bravo_space.decode(bravo)
    return torch.clamp(bravo[0, 0], 0, 1).cpu().numpy()  # [D, H, W]


def _generate_dual(
    seg_binary: np.ndarray,
    dual_model: torch.nn.Module,
    strategy: DiffusionStrategy,
    steps_bravo: int,
    device: torch.device,
    cfg: DictConfig,
    dual_space: object | None,
    offset_adjusted: bool = False,
    offset_strength: float = 0.0,
    noise_seed: int | None = None,
) -> np.ndarray:
    """Generate a dual-modality (T1pre + T1gd) volume conditioned on a seg mask.

    Returns:
        Dual image as numpy array [2, D, H, W] in [0, 1], where channel 0 is
        T1pre and channel 1 is T1gd.
    """
    seg_tensor = torch.from_numpy(seg_binary).float().unsqueeze(0).unsqueeze(0).to(device)
    if dual_space is not None:
        seg_tensor = dual_space.encode(seg_tensor)

    # Dual: 2 noise channels (T1pre + T1gd) concatenated with 1 seg channel → 3 input
    generator = _torch_generator(device, noise_seed)
    noise = _randn(
        get_noise_shape(1, 2, 3, cfg.image_size, cfg.depth),
        device,
        generator=generator,
    )
    noise = _maybe_add_generation_offset(
        noise,
        offset_adjusted,
        offset_strength,
        generator=generator,
    )
    dual = generate_batch(dual_model, strategy, noise, steps_bravo, device,
                          conditioning=seg_tensor,
                          cfg_scale=cfg.cfg_scale_bravo,
                          cfg_scale_end=cfg.get('cfg_scale_bravo_end', None),
                          cfg_mode=cfg.get('cfg_mode', 'standard'),
                          cfg_zero_init_steps=cfg.get('cfg_zero_init_steps', 1))
    if dual_space is not None:
        dual = dual_space.decode(dual)
    return torch.clamp(dual[0], 0, 1).cpu().numpy()  # [2, D, H, W]


def run_3d_pipeline(cfg: DictConfig, output_dir: Path) -> None:
    """Run 3D generation pipeline: size_bins -> seg -> bravo.

    Output format:
        - bins.csv: All size bin information for all samples
        - {id}/seg.nii.gz: Segmentation mask volume [H, W, D]
        - {id}/bravo.nii.gz: BRAVO image volume [H, W, D] (bravo mode only)
    """
    # Validate gen_mode (fail-fast before loading models)
    VALID_3D_MODES = {'bravo', 'dual', 'seg_conditioned', 'seg_conditioned_input'}
    if cfg.gen_mode not in VALID_3D_MODES:
        raise ValueError(f"Invalid gen_mode '{cfg.gen_mode}' for 3D. Valid: {VALID_3D_MODES}")

    device = torch.device("cuda")
    base_seed = cfg.get('seed', None)
    _seed_process_rngs(base_seed)

    strategy = _create_strategy(cfg.strategy)
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
    _precond_model = cfg.get('image_model', None) or cfg.get('seg_model', None)
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

    # Load brain PCA shape model for validation
    brain_pca = None
    brain_pca_path = cfg.get('brain_pca_path', None)
    if brain_pca_path == 'auto':
        repo_root = Path(__file__).resolve().parents[3]
        brain_pca_path = repo_root / 'data' / f'brain_pca_{cfg.image_size}x{cfg.image_size}x{cfg.depth}.npz'
        if not brain_pca_path.exists():
            logger.warning(f"Brain PCA model not found: {brain_pca_path} (skipping shape validation)")
            brain_pca_path = None
    if brain_pca_path:
        brain_pca = BrainPCAModel(brain_pca_path)
        logger.info(f"Brain PCA model loaded: {brain_pca_path} "
                     f"({brain_pca.n_samples} ref volumes, threshold={brain_pca.error_threshold:.6f})")

    # Load the train-derived PCA model used only to reconstruct anatomical
    # support for fixed-mask containment QC. This is separate from the legacy
    # global reconstruction-error filter above.
    brain_support_pca = None
    brain_support_pca_path = cfg.get('brain_support_pca_path', None)
    if brain_support_pca_path:
        brain_support_pca = BrainPCAModel(brain_support_pca_path)
        logger.info(
            f"Brain-support PCA model loaded: {brain_support_pca_path} "
            f"({brain_support_pca.n_samples} training volumes, "
            f"{brain_support_pca.components.shape[0]} components, "
            f"support threshold={brain_support_pca.support_threshold:.1f})"
        )

    # Load seg PCA shape model for seg mask validation
    seg_pca = None
    seg_pca_path = cfg.get('seg_pca_path', None)
    if seg_pca_path == 'auto':
        repo_root = Path(__file__).resolve().parents[3]
        seg_pca_path = repo_root / 'data' / f'seg_pca_{cfg.image_size}x{cfg.image_size}x{cfg.depth}.npz'
        if not seg_pca_path.exists():
            logger.warning(f"Seg PCA model not found: {seg_pca_path} (skipping seg shape validation)")
            seg_pca_path = None
    if seg_pca_path:
        seg_pca = BrainPCAModel(seg_pca_path)
        logger.info(f"Seg PCA model loaded: {seg_pca_path} "
                     f"({seg_pca.n_samples} ref volumes, threshold={seg_pca.error_threshold:.8f})")

    # Resolve per-model step counts (fallback to num_steps)
    steps_seg = cfg.get('num_steps_seg', None) or cfg.num_steps
    steps_bravo = cfg.get('num_steps_bravo', None) or cfg.num_steps

    # Resolve per-model time-shift ratios
    default_shift = cfg.get('shift_ratio', 1.0)
    shift_seg = cfg.get('shift_ratio_seg', None) or default_shift
    shift_bravo = cfg.get('shift_ratio_bravo', None) or default_shift

    input_numel = cfg.image_size * cfg.image_size * cfg.depth

    def _apply_shift(ratio: float) -> None:
        """Set scheduler time-shift by computing base_img_size_numel from ratio."""
        if hasattr(strategy, 'scheduler') and hasattr(strategy.scheduler, 'base_img_size_numel'):
            base_numel = max(1, int(input_numel / (ratio ** 3)))
            strategy.scheduler.base_img_size_numel = base_numel

    # Log output dimensions
    trim_slices = int(cfg.get('trim_slices', 10))
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
        _apply_shift(shift_seg)
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
        component_connectivity = int(cfg.get('seg_component_connectivity', 6))
        if component_connectivity not in (6, 18, 26):
            raise ValueError(
                "seg_component_connectivity must be one of 6, 18, or 26"
            )
        max_attempts = int(cfg.get('max_retries', 10))
        max_white_value = cfg.get('max_white_percentage', MAX_WHITE_PERCENTAGE)
        max_white_pct = None if max_white_value is None else float(max_white_value)
        if max_attempts < 1:
            raise ValueError("max_retries must allow at least one generation attempt")

        # Atlas validation settings
        brain_tolerance = float(cfg.get('brain_tolerance', 0.0))
        brain_dilate = int(cfg.get('brain_dilate_pixels', 0))

        trim_slices = int(trim_slices)
        if trim_slices < 0 or trim_slices >= cfg.depth:
            raise ValueError(
                f"trim_slices must be in [0, {cfg.depth}), got {trim_slices}"
            )
        output_depth = cfg.depth - trim_slices
        output_atlas = brain_atlas[:output_depth] if brain_atlas is not None else None

        logger.info(f"Generating {cfg.num_images} seg masks...")
        if max_white_pct is None:
            logger.info("Mask artifact filter: non-empty")
        else:
            logger.info(
                f"Mask artifact filter: non-empty and per-slice foreground < {max_white_pct:.2%}"
            )
        logger.info(
            f"Measured lesion components: {component_connectivity}-connectivity"
        )
        if validate_size_bins:
            logger.info("Size bin validation: enabled (verify generated seg matches conditioning)")
        if brain_atlas is not None:
            logger.info(
                f"Atlas validation: reject draw (tolerance={brain_tolerance:.0%}, "
                f"dilate={brain_dilate}px)"
            )
        logger.info(f"Fail-closed attempt cap: {max_attempts} draws per accepted mask")

        generated = cfg.get('current_image', 0)
        if generated != 0:
            raise ValueError(
                "Mask-only generation requires current_image=0 so bins.csv "
                "describes the complete generated pool"
            )
        total_retries = 0

        while generated < cfg.num_images:
            bins = fixed_bins if fixed_bins else sample_random_size_bins(cfg.min_tumors, cfg.max_tumors)
            size_bins = torch.tensor([bins], dtype=torch.long, device=device)

            # Deterministic, fail-closed retry loop for one accepted mask.
            valid_mask = False
            rejection_counts: dict[str, int] = {}
            actual_bins: np.ndarray | None = None
            output_seg: np.ndarray | None = None
            for attempt in range(max_attempts):
                seg_seed = _derive_sample_seed(base_seed, generated, stream='seg', attempt=attempt)
                noise = _randn(
                    get_noise_shape(1, 1, 3, cfg.image_size, cfg.depth),
                    device,
                    seed=seg_seed,
                )
                seg = generate_batch(seg_model, strategy, noise, steps_seg, device,
                                     size_bins=size_bins,
                                     cfg_scale=cfg.cfg_scale_seg,
                                     cfg_scale_end=cfg.get('cfg_scale_seg_end', None),
                                     cfg_mode=cfg.get('cfg_mode', 'standard'),
                                     cfg_zero_init_steps=cfg.get('cfg_zero_init_steps', 1))

                # Binarize
                seg_binary = binarize_seg(seg[0, 0]).cpu().numpy()

                # Validate the depth that will be published. Foreground only
                # in padded slices must not become an empty saved mask.
                candidate = seg_binary[:output_depth]
                rejection_reason = _generated_seg_rejection_reason(
                    candidate,
                    max_white_percentage=max_white_pct,
                    brain_atlas=output_atlas,
                    brain_tolerance=brain_tolerance,
                    brain_dilate_pixels=brain_dilate,
                )
                if rejection_reason is not None:
                    rejection_counts[rejection_reason] = (
                        rejection_counts.get(rejection_reason, 0) + 1
                    )
                    total_retries += 1
                    if cfg.verbose and attempt == 0:
                        logger.warning(
                            f"Sample {generated}: rejected mask ({rejection_reason}), retrying..."
                        )
                    continue

                actual_bins = compute_size_bins_3d(
                    candidate,
                    bin_edges,
                    _xyz_to_dhw(voxel_spacing),
                    num_bins,
                    connectivity=component_connectivity,
                )

                # Validate size bins match conditioning
                if validate_size_bins:
                    if not np.array_equal(actual_bins, np.array(bins)):
                        rejection_counts['size_bin_mismatch'] = (
                            rejection_counts.get('size_bin_mismatch', 0) + 1
                        )
                        total_retries += 1
                        if cfg.verbose and attempt == 0:
                            logger.warning(f"Sample {generated}: size bins mismatch "
                                       f"(requested={bins}, got={actual_bins.tolist()}), retrying...")
                        continue

                output_seg = candidate
                valid_mask = True
                break

            if not valid_mask:
                raise RuntimeError(
                    f"Sample {generated}: no mask passed quality control after "
                    f"{max_attempts} deterministic draws; rejections={rejection_counts}"
                )
            assert output_seg is not None and actual_bins is not None

            # Transpose [D, H, W] -> [H, W, D] for NIfTI (slices should be HxW)
            seg_binary = np.transpose(output_seg, (1, 2, 0))

            # Save in subdirectory: 00000/seg.nii.gz
            sample_dir = output_dir / f"{generated:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            voxel = compute_voxel_size(cfg.image_size, cfg.get('fov_mm', 240.0))
            save_nifti(seg_binary, str(sample_dir / "seg.nii.gz"), voxel_size=voxel)
            # Requested bins do not condition an unconditional checkpoint. Save
            # measured bins so bins.csv never implies adherence that was absent.
            all_bins.append((generated, actual_bins.astype(int).tolist()))
            generated += 1

            # Log progress and clear cache periodically
            if generated % 10 == 0 or generated == cfg.num_images:
                logger.info(f"Progress: {generated}/{cfg.num_images}")
                torch.cuda.empty_cache()

        if total_retries > 0:
            logger.info(f"Generation complete. Total retries: {total_retries}")

    # Mode: bravo (full pipeline: seg -> bravo)
    elif cfg.gen_mode in ('bravo', 'dual'):
        # Dual mode: 2 image channels (T1pre + T1gd) + 1 seg conditioning = 3 in, 2 out.
        is_dual = (cfg.gen_mode == 'dual')
        image_in_ch = 3 if is_dual else 2
        image_out_ch = 2 if is_dual else 1

        # Load real seg masks if provided (skip seg generation)
        real_seg_dir = cfg.get('real_seg_dir', None)
        real_seg_files = None
        if real_seg_dir:
            real_seg_dir = Path(real_seg_dir)
            real_seg_files = _discover_real_seg_files(
                real_seg_dir,
                num_images=int(cfg.num_images),
                expected_cases=cfg.get('expected_real_cases', None),
                require_bravo_pairs=bool(cfg.get('require_real_bravo_pairs', False)),
            )
            if cfg.get('validate_real_seg_masks', True):
                _preflight_real_seg_files(
                    real_seg_files,
                    num_images=int(cfg.num_images),
                    image_size=int(cfg.image_size),
                    expected_depth=cfg.get('expected_real_depth', None),
                )
            logger.info(f"Using real seg masks from {real_seg_dir} ({len(real_seg_files)} available)")
        else:
            logger.info("Loading seg_conditioned model...")
            seg_model = load_diffusion_model(
                cfg.seg_model, device=device,
                in_channels=1, out_channels=1, compile_model=True, spatial_dims=3
            )

        logger.info(f"Loading {cfg.gen_mode} model (in={image_in_ch}, out={image_out_ch})...")
        bravo_model = _load_image_model_with_optional_handoff(
            cfg,
            device,
            in_channels=image_in_ch,
            out_channels=image_out_ch,
            spatial_dims=3,
            compile_model=True,
        )

        # Config from bravo checkpoint
        bravo_ckpt = torch.load(cfg.image_model, map_location='cpu', weights_only=False)
        _bravo_train_cfg = bravo_ckpt.get('config', {})
        bravo_pixel_cfg = _bravo_train_cfg.get('pixel', {})
        # Adjusted offset noise: generation must start from N(strength*xi, I)
        _bravo_offset_cfg = _bravo_train_cfg.get('offset_noise', {})
        _bravo_offset_adjusted = (_bravo_offset_cfg.get('enabled', False)
                                  and _bravo_offset_cfg.get('adjusted', False))
        _bravo_offset_strength = _bravo_offset_cfg.get('strength', 0.1) if _bravo_offset_adjusted else 0.0
        if _bravo_offset_adjusted:
            logger.info(f"Adjusted offset noise: strength={_bravo_offset_strength}")
        del bravo_ckpt, _bravo_train_cfg
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
        conditioning_brain_qc_mode = str(
            cfg.get('conditioning_brain_qc_mode', 'cleanup')
        )
        if conditioning_brain_qc_mode not in {'cleanup', 'reject'}:
            raise ValueError(
                "conditioning_brain_qc_mode must be 'cleanup' or 'reject'"
            )
        if (
            brain_support_pca is not None
            and conditioning_brain_qc_mode != 'reject'
        ):
            raise ValueError(
                "brain_support_pca_path is only supported with "
                "conditioning_brain_qc_mode=reject"
            )
        brain_containment_margin_mm = float(
            cfg.get('brain_containment_margin_mm', 3.0)
        )
        if brain_containment_margin_mm < 0.0:
            raise ValueError("brain_containment_margin_mm must be non-negative")
        max_image_attempts_raw = cfg.get('max_image_attempts_per_mask', None)
        max_image_attempts_per_mask = (
            None if max_image_attempts_raw is None else int(max_image_attempts_raw)
        )
        skip_failed_fixed_masks = bool(cfg.get('skip_failed_fixed_masks', False))
        stop_after_accepted_raw = cfg.get('stop_after_accepted_images', None)
        stop_after_accepted_images = (
            None
            if stop_after_accepted_raw is None
            else int(stop_after_accepted_raw)
        )
        if max_image_attempts_per_mask is not None:
            if max_image_attempts_per_mask < 1:
                raise ValueError("max_image_attempts_per_mask must be positive")
            if real_seg_files is None:
                raise ValueError(
                    "max_image_attempts_per_mask is only supported with a fixed real_seg_dir pool"
                )
            if conditioning_brain_qc_mode != 'reject':
                raise ValueError(
                    "max_image_attempts_per_mask requires conditioning_brain_qc_mode=reject"
                )
        if skip_failed_fixed_masks and max_image_attempts_per_mask is None:
            raise ValueError(
                "skip_failed_fixed_masks requires max_image_attempts_per_mask"
            )
        if stop_after_accepted_images is not None:
            if stop_after_accepted_images < 1:
                raise ValueError("stop_after_accepted_images must be positive")
            if real_seg_files is None:
                raise ValueError(
                    "stop_after_accepted_images is only supported with a fixed real_seg_dir pool"
                )
            remaining_candidates = int(cfg.num_images) - int(
                cfg.get('current_image', 0)
            )
            if stop_after_accepted_images > remaining_candidates:
                raise ValueError(
                    "stop_after_accepted_images exceeds the remaining fixed-mask candidates"
                )
        if conditioning_brain_qc_mode == 'reject' and not validate_brain_mask:
            raise ValueError(
                "conditioning_brain_qc_mode=reject requires validate_brain_mask=true"
            )

        # Size bin validation settings (verify generated seg matches conditioning)
        validate_size_bins = cfg.get('validate_size_bins', True)
        voxel_spacing = compute_voxel_size(cfg.image_size, cfg.get('fov_mm', 240.0))
        bin_edges = list(cfg.get('bin_edges', DEFAULT_BIN_EDGES))
        num_bins = cfg.get('num_bins', 7)

        if shift_seg != 1.0:
            logger.info(f"Seg time-shift: ratio={shift_seg:.2f}")
        if shift_bravo != 1.0:
            logger.info(f"Bravo time-shift: ratio={shift_bravo:.2f}")
        logger.info(f"Generating {cfg.num_images} seg+bravo pairs...")
        if real_seg_files is None:
            if max_white_pct is None:
                logger.info("Generated-seg per-slice foreground limit: disabled")
            else:
                logger.info(
                    f"Generated-seg validation: per-slice max {max_white_pct:.2%}"
                )
        else:
            logger.info(
                "Fixed conditioning masks: input validation only; "
                "no per-slice foreground limit is applied"
            )
        if brain_atlas is not None:
            logger.info(f"Stage 1 — Atlas validation: enabled (tolerance={brain_tolerance:.0%}, dilate={brain_dilate}px)")
        if seg_pca is not None:
            logger.info(f"Stage 1b — Seg PCA validation: enabled (threshold={seg_pca.error_threshold:.8f})")
        if validate_brain_mask:
            if conditioning_brain_qc_mode == 'reject':
                if brain_support_pca is not None:
                    logger.info(
                        "Stage 2 — Train-PCA brain-support containment: reject image and "
                        "preserve mask "
                        f"(foreground threshold={brain_threshold}, "
                        f"support threshold={brain_support_pca.support_threshold:.1f}, "
                        f"margin={brain_containment_margin_mm:.3f} mm)"
                    )
                else:
                    logger.info(
                        "Stage 2 — Brain mask validation: reject image and preserve mask "
                        f"(threshold={brain_threshold}, "
                        f"margin={brain_containment_margin_mm:.3f} mm)"
                    )
            else:
                logger.info(
                    "Stage 2 — Brain mask validation: enabled "
                    f"(historical per-tumor cleanup, threshold={brain_threshold})"
                )
        if validate_size_bins:
            logger.info("Size bin validation: enabled (verify generated seg matches conditioning)")

        generated = cfg.get('current_image', 0)
        if generated > 0:
            logger.info(f"Resuming from sample {generated}/{cfg.num_images}")
        total_retries = 0
        max_brain_retries = cfg.get('max_brain_retries', 5)
        outer_retries = 0  # retries per sample (reset on successful save)
        max_outer_retries = cfg.get('max_outer_retries', 20)
        mask_outside_brain = cfg.get('mask_outside_brain', True)
        mask_outside_dilate = cfg.get('mask_outside_brain_dilate_pixels', 2)
        if mask_outside_brain:
            logger.info(f"Final brain-mask zeroing: enabled (threshold={brain_threshold}, dilate={mask_outside_dilate}px)")
        if max_image_attempts_per_mask is not None:
            logger.info(
                f"Fixed-mask image attempt cap: {max_image_attempts_per_mask} deterministic draws"
            )
            if skip_failed_fixed_masks:
                logger.info("Fixed masks that exhaust the cap will be skipped")
        if stop_after_accepted_images is not None:
            logger.info(
                "Fixed-mask accepted-image target: "
                f"{stop_after_accepted_images} new images"
            )

        while generated < cfg.num_images and (
            stop_after_accepted_images is None
            or len(all_bins) < stop_after_accepted_images
        ):
            if outer_retries >= max_outer_retries:
                raise RuntimeError(
                    f"Sample {generated}: exhausted {max_outer_retries} outer retries. "
                    f"Likely pathological atlas/bin/PCA combination. "
                    f"Consider: increase max_outer_retries, relax brain_tolerance, "
                    f"disable seg_pca, or use real_seg_dir."
                )
            if real_seg_files is not None:
                # Load real seg mask from dataset
                seg_path = real_seg_files[generated]
                seg_vol = nib.load(str(seg_path)).get_fdata().astype(np.float32)  # type: ignore[attr-defined]
                if cfg.get('validate_real_seg_masks', True):
                    _validate_real_seg_volume(
                        seg_vol,
                        source=seg_path,
                        image_size=int(cfg.image_size),
                        expected_depth=cfg.get('expected_real_depth', None),
                    )
                seg_vol = np.transpose(seg_vol, (2, 0, 1))  # [H, W, D] -> [D, H, W]
                # Pad/crop depth
                d = seg_vol.shape[0]
                if d < cfg.depth:
                    seg_vol = np.concatenate([seg_vol, np.zeros((cfg.depth - d, *seg_vol.shape[1:]), dtype=np.float32)], axis=0)
                elif d > cfg.depth:
                    seg_vol = seg_vol[:cfg.depth]
                seg_binary = (seg_vol > 0.5).astype(np.float32)
                bins = [0] * 7  # Placeholder — real masks don't have bin info
                if cfg.verbose and generated < 3:
                    logger.info(f"Sample {generated}: real seg from {seg_path.parent.name} "
                                f"(tumor voxels: {int(seg_binary.sum())})")
            else:
                bins = fixed_bins if fixed_bins else sample_random_size_bins(cfg.min_tumors, cfg.max_tumors)
                size_bins = torch.tensor([bins], dtype=torch.long, device=device)

                # Retry loop for valid seg mask
                valid_mask = False
                retries = 0
                while not valid_mask and retries < max_retries:
                    # Generate seg with size bin conditioning
                    _apply_shift(shift_seg)
                    seg_attempt = outer_retries * max_retries + retries
                    seg_seed = _derive_sample_seed(
                        base_seed,
                        generated,
                        stream='seg',
                        attempt=seg_attempt,
                    )
                    noise = _randn(
                        get_noise_shape(1, 1, 3, cfg.image_size, cfg.depth),
                        device,
                        seed=seg_seed,
                    )
                    seg = generate_batch(seg_model, strategy, noise, steps_seg, device,
                                         size_bins=size_bins,
                                         cfg_scale=cfg.cfg_scale_seg,
                                         cfg_scale_end=cfg.get('cfg_scale_seg_end', None),
                                         cfg_mode=cfg.get('cfg_mode', 'standard'),
                                         cfg_zero_init_steps=cfg.get('cfg_zero_init_steps', 1))

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
                        # All tumors outside atlas — restart with new seg
                        total_retries += 1
                        outer_retries += 1
                        continue
                    seg_binary = cleaned_seg

            # Stage 1b — Seg PCA check (is tumor pattern realistic?)
            if seg_pca is not None:
                seg_valid, seg_error = seg_pca.is_valid(seg_binary)
                if not seg_valid:
                    total_retries += 1
                    outer_retries += 1
                    if cfg.verbose:
                        logger.warning(
                            f"Sample {generated}: seg pattern unrealistic "
                            f"(PCA error={seg_error:.8f}, threshold={seg_pca.error_threshold:.8f}), retrying seg..."
                        )
                    continue

            # Inner loop: generate bravo, validate, retry bravo or restart seg
            bravo_accepted = False
            dual_channels = None   # Holds [2, D, H, W] array when dual mode; else None
            image_attempts_this_round = (
                max_image_attempts_per_mask
                if max_image_attempts_per_mask is not None
                else max_brain_retries
            )
            for bravo_attempt in range(image_attempts_this_round):
                # Generate BRAVO (or dual) conditioned on seg mask
                _apply_shift(shift_bravo)
                image_attempt = (
                    bravo_attempt
                    if max_image_attempts_per_mask is not None
                    else outer_retries * max_brain_retries + bravo_attempt
                )
                image_seed = _derive_sample_seed(
                    base_seed,
                    generated,
                    stream='bravo',
                    attempt=image_attempt,
                )
                if is_dual:
                    dual_channels = _generate_dual(
                        seg_binary, bravo_model, strategy, steps_bravo, device, cfg,
                        bravo_space,
                        offset_adjusted=_bravo_offset_adjusted,
                        offset_strength=_bravo_offset_strength,
                        noise_seed=image_seed,
                    )
                    # Use channel 0 (T1pre) for brain-mask / PCA validation — same
                    # anatomy is visible across both channels, T1pre has more uniform
                    # intensity so the brain threshold works consistently.
                    bravo_np = dual_channels[0]
                else:
                    bravo_np = _generate_bravo(
                        seg_binary, bravo_model, strategy, steps_bravo, device, cfg,
                        bravo_space, diffrs_disc, diffrs_cfg,
                        offset_adjusted=_bravo_offset_adjusted,
                        offset_strength=_bravo_offset_strength,
                        noise_seed=image_seed,
                    )

                # Validate the retained output volume. The padded tail is
                # discarded before saving and should not affect acceptance.
                qc_bravo_np = bravo_np
                qc_seg_binary = seg_binary
                if conditioning_brain_qc_mode == 'reject' and trim_slices > 0:
                    qc_bravo_np = bravo_np[:-trim_slices]
                    qc_seg_binary = seg_binary[:-trim_slices]
                    if brain_support_pca is not None:
                        # The PCA was fitted on training volumes padded from
                        # 150 to 160 slices. Remove the generated tail that is
                        # never saved, then restore that known-zero padding.
                        pad_width = ((0, trim_slices), (0, 0), (0, 0))
                        qc_bravo_np = np.pad(qc_bravo_np, pad_width)
                        qc_seg_binary = np.pad(qc_seg_binary, pad_width)

                # Preserve the historical component check for cleanup mode.
                # Fixed-mask rejection uses only the explicit containment rule.
                if (
                    validate_brain_mask
                    and conditioning_brain_qc_mode == 'cleanup'
                    and not has_single_brain_component(
                        qc_bravo_np,
                        threshold=brain_threshold,
                    )
                ):
                    total_retries += 1
                    if cfg.verbose:
                        logger.warning(f"Sample {generated}: multiple brain components detected "
                                       f"(attempt {bravo_attempt + 1}/{image_attempts_this_round})")
                    continue

                # Stage 2b — Either preserve and reject (controlled fixed-mask
                # protocol) or retain the historical cleanup behavior.
                if validate_brain_mask:
                    if conditioning_brain_qc_mode == 'reject':
                        containment = evaluate_conditioning_brain_containment(
                            qc_bravo_np,
                            qc_seg_binary,
                            brain_threshold=brain_threshold,
                            margin_mm=brain_containment_margin_mm,
                            voxel_spacing_mm=_xyz_to_dhw(voxel_spacing),
                            brain_support_pca=brain_support_pca,
                        )
                        if not containment['valid']:
                            total_retries += 1
                            if cfg.verbose:
                                support_name = (
                                    "PCA-reconstructed brain support"
                                    if brain_support_pca is not None
                                    else "main brain"
                                )
                                logger.warning(
                                    f"Sample {generated}: conditioning lesion outside "
                                    f"{support_name} "
                                    f"(max distance={containment['max_distance_mm']}, "
                                    f"attempt {bravo_attempt + 1}/{image_attempts_this_round})"
                                )
                            continue
                    else:
                        brain_mask = create_brain_mask(
                            bravo_np, threshold=brain_threshold,
                            dilate_pixels=brain_dilate,
                        )
                        cleaned_seg, n_removed = remove_tumors_outside_brain(seg_binary, brain_mask)

                        if n_removed > 0:
                            total_retries += 1
                            if cleaned_seg.sum() == 0:
                                if cfg.verbose:
                                    logger.warning(f"Sample {generated}: all tumors outside brain "
                                                   f"(attempt {bravo_attempt + 1}/{image_attempts_this_round})")
                            else:
                                if cfg.verbose:
                                    logger.warning(f"Sample {generated}: removed {n_removed} tumor(s) outside brain, "
                                                   f"retrying bravo (attempt {bravo_attempt + 1}/{image_attempts_this_round})")
                                seg_binary = cleaned_seg
                            continue

                # Stage 2c — PCA brain shape validation
                if brain_pca is not None:
                    gen_brain_mask = create_brain_mask(bravo_np, threshold=brain_threshold, dilate_pixels=0)
                    shape_valid, recon_error = brain_pca.is_valid(gen_brain_mask)
                    if not shape_valid:
                        total_retries += 1
                        if cfg.verbose:
                            logger.warning(
                                f"Sample {generated}: brain shape invalid "
                                f"(PCA error={recon_error:.6f}, threshold={brain_pca.error_threshold:.6f}, "
                                f"attempt {bravo_attempt + 1}/{image_attempts_this_round})"
                            )
                        continue

                bravo_accepted = True
                break

            if not bravo_accepted:
                if max_image_attempts_per_mask is not None:
                    if skip_failed_fixed_masks:
                        logger.warning(
                            f"Sample {generated}: no image passed quality control after "
                            f"{max_image_attempts_per_mask} deterministic draws; "
                            "skipping fixed mask"
                        )
                        generated += 1
                        outer_retries = 0
                        if generated % 10 == 0 or generated == cfg.num_images:
                            logger.info(f"Progress: {generated}/{cfg.num_images}")
                            torch.cuda.empty_cache()
                        continue
                    raise RuntimeError(
                        f"Sample {generated}: no image passed quality control after "
                        f"{max_image_attempts_per_mask} deterministic draws"
                    )
                # All bravo attempts failed — restart with new seg mask
                logger.warning(f"Sample {generated}: bravo failed {max_brain_retries} times, restarting with new seg...")
                total_retries += 1
                outer_retries += 1
                continue

            # Stage 3 — Zero everything outside the predicted brain mask so
            # downstream segmentation is not distracted by background noise.
            if mask_outside_brain:
                final_brain_mask = create_brain_mask(
                    bravo_np, threshold=brain_threshold,
                    dilate_pixels=mask_outside_dilate,
                )
                bravo_np = bravo_np * final_brain_mask.astype(bravo_np.dtype)
                if is_dual:
                    dual_channels = dual_channels * final_brain_mask[None].astype(dual_channels.dtype)

            # Transpose [D, H, W] -> [H, W, D] for NIfTI (slices should be HxW)
            seg_binary_save = np.transpose(seg_binary, (1, 2, 0))
            bravo_np = np.transpose(bravo_np, (1, 2, 0))
            if is_dual:
                # dual_channels is [2, D, H, W]; permute each channel to [H, W, D]
                dual_channels_save = np.transpose(dual_channels, (0, 2, 3, 1))

            # Trim last N slices to match training data (training pads, so we remove padding)
            if trim_slices > 0:
                seg_binary_save = seg_binary_save[:, :, :-trim_slices]  # [H, W, D-trim]
                bravo_np = bravo_np[:, :, :-trim_slices]
                if is_dual:
                    dual_channels_save = dual_channels_save[:, :, :, :-trim_slices]

            # Save in subdirectory: 00000/{seg,bravo}.nii.gz for bravo mode
            # or 00000/{seg,t1_pre,t1_gd}.nii.gz for dual mode.
            sample_dir = output_dir / f"{generated:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            voxel = compute_voxel_size(cfg.image_size, cfg.get('fov_mm', 240.0))
            save_nifti(seg_binary_save, str(sample_dir / "seg.nii.gz"), voxel_size=voxel)
            if is_dual:
                save_nifti(
                    dual_channels_save[0],
                    str(sample_dir / "t1_pre.nii.gz"),
                    voxel_size=voxel,
                )
                save_nifti(
                    dual_channels_save[1],
                    str(sample_dir / "t1_gd.nii.gz"),
                    voxel_size=voxel,
                )
            else:
                save_nifti(bravo_np, str(sample_dir / "bravo.nii.gz"), voxel_size=voxel)
            all_bins.append((generated, bins))

            generated += 1
            outer_retries = 0  # reset on successful sample

            if (
                stop_after_accepted_images is not None
                and len(all_bins) == stop_after_accepted_images
            ):
                logger.info(
                    "Reached fixed-mask accepted-image target after candidate "
                    f"{generated - 1}"
                )

            # Log progress and clear cache periodically
            if generated % 10 == 0 or generated == cfg.num_images:
                logger.info(f"Progress: {generated}/{cfg.num_images}")
                torch.cuda.empty_cache()

        if total_retries > 0:
            logger.info(f"Total retries: {total_retries} (seg validation + brain mask)")
        if (
            stop_after_accepted_images is not None
            and len(all_bins) < stop_after_accepted_images
        ):
            logger.warning(
                "Fixed-mask pool exhausted before accepted-image target: "
                f"accepted {len(all_bins)}/{stop_after_accepted_images}"
            )

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

        generated = cfg.get('current_image', 0)
        if generated > 0:
            logger.info(f"Resuming from sample {generated}/{cfg.num_images}")
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
                seg_seed = _derive_sample_seed(
                    base_seed,
                    generated,
                    stream='seg_input',
                    attempt=retries,
                )
                noise = _randn(
                    get_noise_shape(1, 1, 3, cfg.image_size, cfg.depth),
                    device,
                    seed=seg_seed,
                )
                seg = generate_batch(seg_model, strategy, noise, steps_seg, device,
                                     bin_maps=bin_maps,
                                     cfg_scale=cfg.cfg_scale_seg,
                                     cfg_scale_end=cfg.get('cfg_scale_seg_end', None),
                                     cfg_mode=cfg.get('cfg_mode', 'standard'),
                                     cfg_zero_init_steps=cfg.get('cfg_zero_init_steps', 1))

                # Binarize
                seg_binary = binarize_seg(seg[0, 0]).cpu().numpy()

                # Validate size bins match conditioning
                if validate_size_bins:
                    actual_bins = compute_size_bins_3d(seg_binary, bin_edges, _xyz_to_dhw(voxel_spacing), num_bins)
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
        raise ValueError(f"Mode '{cfg.gen_mode}' not supported for 3D. Use 'seg_conditioned', 'seg_conditioned_input', 'bravo', or 'dual'.")

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
        if cfg.gen_mode == 'bravo' and not cfg.image_model:
            raise ValueError("3D bravo mode requires image_model")
        if cfg.gen_mode == 'bravo' and not cfg.seg_model and not cfg.get('real_seg_dir'):
            raise ValueError("3D bravo mode requires seg_model or real_seg_dir")
        run_3d_pipeline(cfg, output_dir)

    logger.info("Generation complete!")


if __name__ == "__main__":
    main()
