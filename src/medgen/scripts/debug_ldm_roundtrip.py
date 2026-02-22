#!/usr/bin/env python3
"""Diagnose LDM generation pipeline by testing partial denoising round-trips.

Tests whether the diffusion model can denoise latents at various noise levels.
If the model works well at low noise but fails at high noise, the model is
functional but hasn't learned the full generation trajectory. If it fails even
at low noise, there's likely a pipeline bug (normalization, channel ordering, etc.).

Test levels:
  1. VQ-VAE round-trip: encode → decode (no diffusion, sanity check)
  2. Low noise (t=100/1000): add small noise → denoise → decode
  3. Medium noise (t=500/1000): add medium noise → denoise → decode
  4. High noise (t=900/1000): add heavy noise → denoise → decode
  5. Full generation: pure noise → denoise → decode (same as actual generation)

Saves NIfTI volumes and per-level PSNR/SSIM for visual inspection.

Usage:
    python -m medgen.scripts.debug_ldm_roundtrip \
        --checkpoint runs/diffusion_3d/.../checkpoint_latest.pt \
        --compression-checkpoint runs/compression_3d/.../checkpoint_latest.pt \
        --compression-type vqvae \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --output-dir debug_ldm_roundtrip
"""
import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from medgen.data.utils import save_nifti

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
)
logger = logging.getLogger(__name__)


def load_volume(path: Path, depth: int) -> np.ndarray:
    """Load NIfTI, normalize to [0,1], transpose to [D,H,W], pad/crop depth."""
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    vol = np.transpose(vol, (2, 0, 1))  # [H,W,D] -> [D,H,W]
    d = vol.shape[0]
    if d < depth:
        pad = np.zeros((depth - d, vol.shape[1], vol.shape[2]), dtype=np.float32)
        vol = np.concatenate([vol, pad], axis=0)
    elif d > depth:
        vol = vol[:depth]
    return vol


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(1.0 / mse))


def save_volume_nifti(vol_np: np.ndarray, path: str, voxel_size: tuple):
    """Save [D,H,W] volume as NIfTI."""
    vol_hw = np.transpose(vol_np, (1, 2, 0))  # [D,H,W] -> [H,W,D]
    save_nifti(vol_hw, path, voxel_size)


@torch.no_grad()
def denoise_single_step(
    model: nn.Module,
    strategy,
    noisy_latent: torch.Tensor,
    conditioning_latent: torch.Tensor | None,
    num_steps: int,
    device: torch.device,
    latent_channels: int,
) -> torch.Tensor:
    """Run full Euler denoising from a noisy latent using the strategy's generate().

    For partial noise tests, we still run the full generation from the given noisy_latent
    because the Euler integrator expects to run the full trajectory. Instead, we test
    by checking if the model CAN recover signal from various noise levels.
    """
    if conditioning_latent is not None:
        model_input = torch.cat([noisy_latent, conditioning_latent], dim=1)
    else:
        model_input = noisy_latent

    result = strategy.generate(
        model, model_input, num_steps, device,
        latent_channels=latent_channels,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Debug LDM round-trip at various noise levels")
    parser.add_argument('--checkpoint', required=True, help='Diffusion model checkpoint')
    parser.add_argument('--compression-checkpoint', required=True, help='Compression model checkpoint')
    parser.add_argument('--compression-type', default='auto')
    parser.add_argument('--latent-cache-dir', default=None,
                        help='Latent cache dir with train/metadata.json for normalization stats')
    parser.add_argument('--data-root', required=True, help='Dataset root')
    parser.add_argument('--output-dir', default='debug_ldm_roundtrip')
    parser.add_argument('--num-volumes', type=int, default=3, help='Volumes to test (default: 3)')
    parser.add_argument('--num-steps', type=int, default=25, help='Euler steps for denoising')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--image-size', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint config ──
    logger.info("Loading checkpoint metadata...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_cfg = ckpt.get('config', {})

    base_in_channels = ckpt_cfg.get('in_channels', 2)
    base_out_channels = ckpt_cfg.get('out_channels', 1)
    spatial_dims = ckpt_cfg.get('spatial_dims', 3)
    mode = ckpt_cfg.get('mode')
    if isinstance(mode, dict):
        mode = mode.get('name')
    strategy_name = ckpt_cfg.get('strategy', 'rflow')

    model_type = ckpt_cfg.get('model_type', 'unet')
    is_transformer = model_type in ('dit', 'sit', 'uvit', 'hdit')
    if is_transformer:
        pixel_image_size = args.image_size or 256
        pixel_depth = args.depth or 160
    else:
        pixel_image_size = args.image_size or ckpt_cfg.get('image_size', 256)
        pixel_depth = args.depth or ckpt_cfg.get('depth_size', 160)
    del ckpt

    # ── Load compression model + latent space ──
    import json as json_mod

    from medgen.data.loaders.compression_detection import load_compression_model
    from medgen.diffusion.spaces import LatentSpace

    comp_model, detected_type, comp_spatial_dims, scale_factor, latent_channels = (
        load_compression_model(args.compression_checkpoint, args.compression_type, device, spatial_dims=spatial_dims)
    )
    logger.info(f"Compression: {detected_type} {scale_factor}x, {latent_channels}ch")

    # Latent normalization
    latent_shift = None
    latent_scale = None
    if args.latent_cache_dir:
        meta_path = Path(args.latent_cache_dir) / 'train' / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json_mod.load(f)
            latent_shift = metadata.get('latent_shift')
            latent_scale = metadata.get('latent_scale')
            if latent_shift is not None:
                logger.info(f"Latent normalization: shift={latent_shift}, scale={latent_scale}")

    slicewise = (comp_spatial_dims == 2 and spatial_dims == 3)
    depth_sf = 1 if slicewise else scale_factor

    space = LatentSpace(
        compression_model=comp_model,
        device=device,
        deterministic=True,
        spatial_dims=comp_spatial_dims,
        compression_type=detected_type,
        scale_factor=scale_factor,
        latent_channels=latent_channels,
        slicewise_encoding=slicewise,
        latent_shift=latent_shift,
        latent_scale=latent_scale,
    )

    model_out_channels = base_out_channels * latent_channels
    model_in_channels = base_in_channels * latent_channels
    cond_channels = base_in_channels - base_out_channels
    noise_image_size = pixel_image_size // scale_factor
    noise_depth = pixel_depth // depth_sf

    logger.info(f"Mode: {mode} | Strategy: {strategy_name}")
    logger.info(f"Pixel: {pixel_image_size}x{pixel_image_size}x{pixel_depth}")
    logger.info(f"Latent: {noise_image_size}x{noise_image_size}x{noise_depth} "
                f"({model_out_channels}ch out, {model_in_channels}ch in)")
    logger.info(f"Conditioning channels: {cond_channels} (pixel-level)")

    # ── Load diffusion model ──
    from medgen.diffusion import load_diffusion_model

    model = load_diffusion_model(
        args.checkpoint, device=device,
        in_channels=model_in_channels, out_channels=model_out_channels,
        compile_model=False, spatial_dims=spatial_dims,
    )

    # ── Setup strategy ──
    if strategy_name == 'ddpm':
        from medgen.diffusion import DDPMStrategy
        strategy = DDPMStrategy()
        prediction_type = ckpt_cfg.get('prediction_type', 'sample') if 'ckpt_cfg' in dir() else 'sample'
        strategy.setup_scheduler(
            num_timesteps=1000, image_size=noise_image_size,
            depth_size=noise_depth, spatial_dims=spatial_dims,
            prediction_type=prediction_type, schedule='linear_beta',
        )
    else:
        from medgen.diffusion import RFlowStrategy
        strategy = RFlowStrategy()
        strategy.setup_scheduler(
            num_timesteps=1000, image_size=noise_image_size,
            depth_size=noise_depth, spatial_dims=spatial_dims,
        )

    # ── Load real volumes ──
    data_root = Path(args.data_root)
    val_dir = data_root / 'val'
    if not val_dir.exists():
        # fallback to any split
        for d in sorted(data_root.iterdir()):
            if d.is_dir() and list(d.glob("*/bravo.nii.gz")):
                val_dir = d
                break

    bravo_files = sorted(val_dir.glob("*/bravo.nii.gz"))[:args.num_volumes]
    seg_files = sorted(val_dir.glob("*/seg.nii.gz"))[:args.num_volumes]
    logger.info(f"Loading {len(bravo_files)} volumes from {val_dir}")

    voxel_size = (240.0 / pixel_image_size, 240.0 / pixel_image_size, 1.0)
    trim_slices = 10

    # ── Run tests ──
    noise_levels = [0, 100, 250, 500, 750, 900, 1000]
    # 0 = VQ-VAE only, 1000 = full generation from noise

    all_results = []

    for vol_idx, bravo_path in enumerate(bravo_files):
        patient_id = bravo_path.parent.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Volume {vol_idx}: {patient_id}")
        logger.info(f"{'='*60}")

        # Load original
        orig_np = load_volume(bravo_path, pixel_depth)
        orig_tensor = torch.from_numpy(orig_np).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]

        # Save original
        save_volume_nifti(orig_np, str(output_dir / f"{vol_idx:02d}_{patient_id}_original.nii.gz"), voxel_size)

        # Load conditioning if needed
        cond_latent = None
        if cond_channels > 0 and vol_idx < len(seg_files):
            seg_np = load_volume(seg_files[vol_idx], pixel_depth)
            seg_np = (seg_np > 0.5).astype(np.float32)
            seg_tensor = torch.from_numpy(seg_np).unsqueeze(0).unsqueeze(0).to(device)
            with autocast('cuda', dtype=torch.bfloat16):
                cond_latent = space.encode(seg_tensor)
            logger.info(f"  Conditioning latent shape: {cond_latent.shape}")

        # Encode to latent
        with autocast('cuda', dtype=torch.bfloat16):
            clean_latent = space.encode(orig_tensor)  # raw (unnormalized)
        logger.info(f"  Clean latent: shape={clean_latent.shape}, "
                    f"range=[{clean_latent.min():.3f}, {clean_latent.max():.3f}], "
                    f"mean={clean_latent.mean():.3f}, std={clean_latent.std():.3f}")

        # Normalize latent (same as training pipeline)
        if space._shift is not None:
            normalized_latent = (clean_latent.float() - space._shift) / space._scale
            logger.info(f"  Normalized latent: "
                        f"range=[{normalized_latent.min():.3f}, {normalized_latent.max():.3f}], "
                        f"mean={normalized_latent.mean():.3f}, std={normalized_latent.std():.3f}")
        else:
            normalized_latent = clean_latent.float()
            logger.info("  No latent normalization applied")

        if cond_latent is not None and space._shift is not None:
            cond_latent_norm = (cond_latent.float() - space._shift) / space._scale
        elif cond_latent is not None:
            cond_latent_norm = cond_latent.float()
        else:
            cond_latent_norm = None

        vol_results = {'patient_id': patient_id, 'levels': {}}

        # Fixed noise for this volume
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + vol_idx)
        noise = torch.randn_like(normalized_latent, generator=gen)

        for noise_t in noise_levels:
            logger.info(f"\n  --- Noise level t={noise_t}/1000 ---")

            if noise_t == 0:
                # Pure VQ-VAE round-trip (encode → decode, no diffusion)
                with autocast('cuda', dtype=torch.bfloat16):
                    recon_pixel = space.decode(clean_latent)
                recon_np = recon_pixel[0, 0].cpu().float().clamp(0, 1).numpy()
                label = "vqvae_only"

            elif noise_t == 1000:
                # Full generation from pure noise
                denoised = denoise_single_step(
                    model, strategy, noise, cond_latent_norm,
                    args.num_steps, device, model_out_channels,
                )
                with autocast('cuda', dtype=torch.bfloat16):
                    recon_pixel = space.decode(denoised)
                recon_np = recon_pixel[0, 0].cpu().float().clamp(0, 1).numpy()
                label = "full_gen"

            else:
                # Partial noise: interpolate between clean and noise
                # RFlow: x_t = (1 - t/1000) * x_0 + (t/1000) * noise
                t_frac = noise_t / 1000.0
                noisy_latent = (1 - t_frac) * normalized_latent + t_frac * noise

                logger.info(f"    Noisy latent: range=[{noisy_latent.min():.3f}, {noisy_latent.max():.3f}]")

                # Denoise using full Euler trajectory
                denoised = denoise_single_step(
                    model, strategy, noisy_latent, cond_latent_norm,
                    args.num_steps, device, model_out_channels,
                )

                logger.info(f"    Denoised latent: range=[{denoised.min():.3f}, {denoised.max():.3f}]")

                # Latent-space MSE (before decode)
                latent_mse = torch.mean((denoised - normalized_latent) ** 2).item()
                logger.info(f"    Latent MSE vs clean: {latent_mse:.6f}")

                with autocast('cuda', dtype=torch.bfloat16):
                    recon_pixel = space.decode(denoised)
                recon_np = recon_pixel[0, 0].cpu().float().clamp(0, 1).numpy()
                label = f"t{noise_t:04d}"

            psnr = compute_psnr(orig_np, recon_np)
            logger.info(f"    PSNR vs original: {psnr:.2f} dB")

            vol_results['levels'][str(noise_t)] = {
                'psnr': psnr,
                'label': label,
            }

            # Save reconstruction
            save_volume_nifti(
                recon_np,
                str(output_dir / f"{vol_idx:02d}_{patient_id}_{label}.nii.gz"),
                voxel_size,
            )

        all_results.append(vol_results)

    # ── Summary ──
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY: Round-trip PSNR at each noise level")
    logger.info(f"{'='*70}")

    header = f"{'Patient':<20}"
    for t in noise_levels:
        if t == 0:
            header += f"{'VQVAE':>10}"
        elif t == 1000:
            header += f"{'FullGen':>10}"
        else:
            header += f"{'t='+str(t):>10}"
    logger.info(header)
    logger.info("-" * len(header))

    for r in all_results:
        row = f"{r['patient_id']:<20}"
        for t in noise_levels:
            psnr = r['levels'][str(t)]['psnr']
            row += f"{psnr:>10.2f}"
        logger.info(row)

    # Average
    row = f"{'AVERAGE':<20}"
    for t in noise_levels:
        avg = np.mean([r['levels'][str(t)]['psnr'] for r in all_results])
        row += f"{avg:>10.2f}"
    logger.info(row)

    logger.info("")
    logger.info("Interpretation:")
    logger.info("  VQVAE ~= t=0: Compression model quality (should be ~37+ dB)")
    logger.info("  t=100: Light noise — should recover almost perfectly if model works")
    logger.info("  t=500: Medium noise — tests denoising capability")
    logger.info("  t=900: Heavy noise — close to full generation difficulty")
    logger.info("  FullGen: Pure noise generation — matches eval pipeline quality")
    logger.info("")
    logger.info("  If VQVAE is good but t=100 drops sharply → pipeline bug")
    logger.info("  If gradual degradation t=100 → FullGen → model capability issue")
    logger.info("  If all levels are bad → fundamental pipeline/normalization bug")

    # Save results
    results_path = output_dir / 'roundtrip_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
