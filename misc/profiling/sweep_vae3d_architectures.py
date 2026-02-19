#!/usr/bin/env python3
"""Sweep 3D VAE architectures to find what fits in 80GB VRAM.

Tests multiple channel configurations with full training pipeline
(forward + backward + discriminator + optimizers) to find the largest
architecture that fits in an A100/H100 80GB GPU.

Usage:
    python misc/profiling/sweep_vae3d_architectures.py
    python misc/profiling/sweep_vae3d_architectures.py --volume 256 256 160
    python misc/profiling/sweep_vae3d_architectures.py --volume 128 128 80 --no-checkpoint
"""

import argparse
import gc
import os
import sys

os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from monai.data import set_track_meta
set_track_meta(False)

from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from medgen.losses.losses import PerceptualLoss


# =============================================================================
# Architecture configs to sweep
# =============================================================================

ARCHITECTURES = [
    # --- Baseline: exp14 (current, ~50GB allocated) ---
    {
        'name': 'exp14 (current)',
        'channels': (32, 64, 128),
        'latent_channels': 4,
        'mid_block': False,
    },
    # --- Same arch + mid block ---
    {
        'name': 'exp14 +mid',
        'channels': (32, 64, 128),
        'latent_channels': 4,
        'mid_block': True,
    },
    # --- 4-level variants (extra downsample = smaller activations at deep levels) ---
    {
        'name': '4L 32ch',
        'channels': (32, 64, 128, 128),
        'latent_channels': 3,
        'mid_block': False,
    },
    {
        'name': '4L 32ch lc=4',
        'channels': (32, 64, 128, 128),
        'latent_channels': 4,
        'mid_block': False,
    },
    {
        'name': '4L 32ch +mid',
        'channels': (32, 64, 128, 128),
        'latent_channels': 4,
        'mid_block': True,
    },
    # --- Moderate scale-up (2x base channels) ---
    {
        'name': '3L 48ch',
        'channels': (48, 96, 192),
        'latent_channels': 4,
        'mid_block': False,
    },
    {
        'name': '3L 64ch',
        'channels': (64, 128, 256),
        'latent_channels': 4,
        'mid_block': False,
    },
    {
        'name': '4L 48ch',
        'channels': (48, 96, 192, 192),
        'latent_channels': 4,
        'mid_block': False,
    },
    {
        'name': '4L 64ch',
        'channels': (64, 128, 256, 256),
        'latent_channels': 4,
        'mid_block': False,
    },
    # --- Latent channel variants ---
    {
        'name': '3L 32ch lc=8',
        'channels': (32, 64, 128),
        'latent_channels': 8,
        'mid_block': False,
    },
    {
        'name': '4L 32ch lc=8',
        'channels': (32, 64, 128, 128),
        'latent_channels': 8,
        'mid_block': False,
    },
]


# =============================================================================
# Gradient checkpointing wrapper
# =============================================================================

class CheckpointedAutoencoder(nn.Module):
    """Wraps AutoencoderKL with gradient checkpointing."""

    def __init__(self, model: AutoencoderKL):
        super().__init__()
        self.model = model

    def forward(self, x):
        def encode_mu(x):
            return self.model.quant_conv_mu(self.model.encoder(x))

        def encode_logvar(x):
            return self.model.quant_conv_log_sigma(self.model.encoder(x))

        z_mu = grad_checkpoint(encode_mu, x, use_reentrant=False)
        z_log_var = grad_checkpoint(encode_logvar, x, use_reentrant=False)
        z = self.model.sampling(z_mu, z_log_var)

        def decode(z):
            return self.model.decoder(self.model.post_quant_conv(z))

        reconstruction = grad_checkpoint(decode, z, use_reentrant=False)
        return reconstruction, z_mu, z_log_var


# =============================================================================
# Profiling
# =============================================================================

def reset_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1024**3


def profile_architecture(
    arch: dict,
    volume: tuple[int, int, int],
    batch_size: int,
    use_checkpointing: bool,
    use_disc: bool,
    disc_channels: int,
    disc_layers: int,
    perceptual_weight: float = 0.001,
    perceptual_slice_fraction: float = 0.25,
) -> dict:
    """Profile one architecture config with full training pipeline.

    Includes: VAE forward/backward, perceptual loss (2.5D), KL loss,
    discriminator forward/backward, and optimizer steps — matching
    the actual compression trainer.
    """
    device = torch.device('cuda')
    reset_gpu()

    name = arch['name']
    channels = arch['channels']
    latent_channels = arch['latent_channels']
    mid_block = arch['mid_block']
    n_levels = len(channels)
    attn = tuple(False for _ in channels)

    try:
        # Create VAE
        base_model = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=channels,
            latent_channels=latent_channels,
            num_res_blocks=2,
            attention_levels=attn,
            with_encoder_nonlocal_attn=mid_block,
            with_decoder_nonlocal_attn=mid_block,
            norm_num_groups=32,
        ).to(device)

        if use_checkpointing:
            model = CheckpointedAutoencoder(base_model)
        else:
            model = base_model

        num_params = sum(p.numel() for p in model.parameters()) / 1e6

        # Perceptual loss (2D network for 2.5D computation, same as trainer)
        percep_fn = None
        if perceptual_weight > 0:
            percep_fn = PerceptualLoss(spatial_dims=2, device=device)

        # Discriminator
        disc = None
        disc_params = 0
        if use_disc:
            disc = PatchDiscriminator(
                spatial_dims=3,
                in_channels=1,
                channels=disc_channels,
                num_layers_d=disc_layers,
            ).to(device)
            disc_params = sum(p.numel() for p in disc.parameters()) / 1e6

        # Optimizers
        opt_g = torch.optim.AdamW(model.parameters(), lr=5e-5)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4) if disc else None

        # Input
        D, H, W = volume
        x = torch.randn(batch_size, 1, D, H, W, device=device)

        reset_gpu()

        # Forward + loss computation (matching actual trainer)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            recon, z_mu, z_logvar = model(x)
            loss_recon = nn.functional.l1_loss(recon.float(), x.float())

            # KL loss (same as trainer)
            kl_loss = 0.5 * torch.mean(
                z_mu.pow(2) + z_logvar.exp() - z_logvar - 1
            )

            # 2.5D perceptual loss: sample 25% of depth slices
            percep_loss = torch.tensor(0.0, device=device)
            if percep_fn is not None:
                n_slices = max(1, int(D * perceptual_slice_fraction))
                indices = torch.randperm(D)[:n_slices]
                for idx in indices:
                    recon_slice = recon[:, :, idx, :, :].float()
                    target_slice = x[:, :, idx, :, :].float()
                    percep_loss = percep_loss + percep_fn(recon_slice, target_slice)
                percep_loss = percep_loss / n_slices

            total_loss = loss_recon + 1e-6 * kl_loss + perceptual_weight * percep_loss

        # Generator backward
        opt_g.zero_grad()
        total_loss.backward()
        opt_g.step()

        gen_peak = peak_gb()

        # Discriminator step
        disc_peak = 0
        if disc:
            reset_gpu()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                recon_det = recon.detach()
                d_real = disc(x)
                d_fake = disc(recon_det)
                loss_d = (nn.functional.relu(1 - d_real).mean() +
                          nn.functional.relu(1 + d_fake).mean())

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            disc_peak = peak_gb()

        total_peak = max(gen_peak, disc_peak)
        latent_shape = list(z_mu.shape[1:])

        del model, base_model, disc, percep_fn, opt_g, opt_d, x, recon, z_mu, z_logvar
        reset_gpu()

        return {
            'name': name,
            'channels': list(channels),
            'latent_ch': latent_channels,
            'mid_block': mid_block,
            'params_m': num_params,
            'disc_params_m': disc_params,
            'latent_shape': latent_shape,
            'gen_peak_gb': gen_peak,
            'disc_peak_gb': disc_peak,
            'total_peak_gb': total_peak,
            'fits_80gb': total_peak < 76,  # 4GB headroom
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
            raise
        # Variables go out of scope here; gc.collect in reset_gpu handles the rest
        reset_gpu()
        return {
            'name': name,
            'channels': list(channels),
            'latent_ch': latent_channels,
            'mid_block': mid_block,
            'params_m': 0,
            'total_peak_gb': float('inf'),
            'fits_80gb': False,
            'oom': True,
        }
    except Exception as e:
        print(f"    Unexpected error: {e}")
        reset_gpu()
        return {
            'name': name,
            'channels': list(channels),
            'latent_ch': latent_channels,
            'mid_block': mid_block,
            'params_m': 0,
            'total_peak_gb': float('inf'),
            'fits_80gb': False,
            'oom': True,
        }


def main():
    parser = argparse.ArgumentParser(description='Sweep 3D VAE architectures for VRAM')
    parser.add_argument('--volume', type=int, nargs=3, default=[160, 256, 256],
                        metavar=('D', 'H', 'W'), help='Volume size (depth height width)')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Disable gradient checkpointing')
    parser.add_argument('--no-disc', action='store_true',
                        help='Skip discriminator')
    parser.add_argument('--disc-channels', type=int, default=64,
                        help='Discriminator base channels')
    parser.add_argument('--disc-layers', type=int, default=3,
                        help='Discriminator layers')
    parser.add_argument('--no-perceptual', action='store_true',
                        help='Skip perceptual loss')
    parser.add_argument('--perceptual-weight', type=float, default=0.001,
                        help='Perceptual loss weight')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('CUDA not available')
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    volume = tuple(args.volume)

    print('=' * 90)
    print('3D VAE Architecture Sweep')
    print('=' * 90)
    print(f'GPU: {gpu_name} ({gpu_mem:.0f} GB)')
    print(f'Volume: {volume[0]}x{volume[1]}x{volume[2]} ({volume[0]*volume[1]*volume[2]/1e6:.1f}M voxels)')
    print(f'Batch size: {args.batch_size}')
    perceptual_weight = 0.0 if args.no_perceptual else args.perceptual_weight
    print(f'Gradient checkpointing: {not args.no_checkpoint}')
    print(f'Discriminator: {not args.no_disc}')
    print(f'Perceptual loss (2.5D): {perceptual_weight > 0} (weight={perceptual_weight})')
    print('=' * 90)
    print('Pipeline: VAE fwd/bwd + KL + 2.5D perceptual + discriminator + optimizer steps')
    print('=' * 90)

    results = []
    for arch in ARCHITECTURES:
        print(f"\nProfiling: {arch['name']} ...")
        result = profile_architecture(
            arch, volume, args.batch_size,
            use_checkpointing=not args.no_checkpoint,
            use_disc=not args.no_disc,
            disc_channels=args.disc_channels,
            disc_layers=args.disc_layers,
            perceptual_weight=perceptual_weight,
        )
        results.append(result)

        if result.get('oom'):
            print(f'  OOM!')
        else:
            print(f"  Params: {result['params_m']:.1f}M | "
                  f"Gen: {result['gen_peak_gb']:.1f}GB | "
                  f"Total: {result['total_peak_gb']:.1f}GB | "
                  f"{'OK' if result['fits_80gb'] else 'TOO BIG'}")

    # Summary table
    print(f"\n\n{'=' * 90}")
    print('SUMMARY')
    print(f"{'=' * 90}")
    header = (f"{'Architecture':<25} {'Channels':<20} {'LC':>3} {'Mid':>4} "
              f"{'Params':>7} {'Gen GB':>7} {'Total GB':>9} {'Fits':>5}")
    print(header)
    print('-' * 90)

    for r in results:
        if r.get('oom'):
            print(f"{r['name']:<25} {str(r['channels']):<20} {r['latent_ch']:>3} "
                  f"{'Y' if r['mid_block'] else 'N':>4} {'':>7} {'':>7} {'OOM':>9} {'':>5}")
        else:
            fit = 'YES' if r['fits_80gb'] else 'NO'
            latent = 'x'.join(str(x) for x in r.get('latent_shape', []))
            print(f"{r['name']:<25} {str(r['channels']):<20} {r['latent_ch']:>3} "
                  f"{'Y' if r['mid_block'] else 'N':>4} "
                  f"{r['params_m']:>6.1f}M {r['gen_peak_gb']:>6.1f}G "
                  f"{r['total_peak_gb']:>8.1f}G {fit:>5}")

    print(f"\n{'=' * 90}")
    print(f"Volume: {volume[0]}x{volume[1]}x{volume[2]}, "
          f"batch_size={args.batch_size}, "
          f"checkpointing={'ON' if not args.no_checkpoint else 'OFF'}, "
          f"disc={'ON' if not args.no_disc else 'OFF'}")


if __name__ == '__main__':
    main()
