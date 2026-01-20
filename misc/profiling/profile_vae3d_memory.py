#!/usr/bin/env python3
"""Profile 3D VAE memory usage for different volume sizes.

Usage:
    python misc/profile_vae3d_memory.py
    python misc/profile_vae3d_memory.py --size 128 128 80
    python misc/profile_vae3d_memory.py --size 256 256 160 --no-backward
    python misc/profile_vae3d_memory.py --size 256 256 160 --checkpoint
"""

import argparse
import gc
import os
import torch
from torch.utils.checkpoint import checkpoint

# Set CUDA memory config for reduced fragmentation
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# Disable MONAI MetaTensor tracking (fixes torch.compile issues)
from monai.data import set_track_meta
set_track_meta(False)

from monai.networks.nets import AutoencoderKL, PatchDiscriminator


def get_gpu_memory_gb():
    """Get current and peak GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0, 0
    current = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return current, peak


def reset_memory():
    """Reset memory tracking."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class CheckpointedAutoencoder(torch.nn.Module):
    """Wrapper that applies gradient checkpointing to AutoencoderKL."""

    def __init__(self, model: AutoencoderKL):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Checkpoint encoder
        def encode_fn(x):
            h = self.model.encoder(x)
            h = self.model.quant_conv_mu(h)
            return h

        def encode_logvar_fn(x):
            h = self.model.encoder(x)
            h = self.model.quant_conv_log_sigma(h)
            return h

        # Use checkpointing for encoder (saves most memory)
        z_mu = checkpoint(encode_fn, x, use_reentrant=False)
        z_log_var = checkpoint(encode_logvar_fn, x, use_reentrant=False)

        # Sample from latent
        z = self.model.sampling(z_mu, z_log_var)

        # Checkpoint decoder
        def decode_fn(z):
            z = self.model.post_quant_conv(z)
            return self.model.decoder(z)

        reconstruction = checkpoint(decode_fn, z, use_reentrant=False)

        return reconstruction, z_mu, z_log_var


def profile_vae3d(
    height: int = 256,
    width: int = 256,
    depth: int = 160,
    channels: tuple = (32, 64, 128, 128),
    latent_channels: int = 3,
    batch_size: int = 1,
    include_backward: bool = True,
    include_discriminator: bool = True,
    use_amp: bool = True,
    use_checkpointing: bool = False,
    use_compile: bool = False,
):
    """Profile memory usage for 3D VAE."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, cannot profile GPU memory")
        return

    print(f"\n{'='*60}")
    print(f"Profiling 3D VAE: {height}x{width}x{depth}")
    print(f"{'='*60}")
    print(f"Channels: {channels}")
    print(f"Latent channels: {latent_channels}")
    print(f"Batch size: {batch_size}")
    print(f"AMP (bfloat16): {use_amp}")
    print(f"Gradient checkpointing: {use_checkpointing}")
    print(f"torch.compile: {use_compile}")
    print(f"Include backward: {include_backward}")
    print(f"Include discriminator: {include_discriminator}")

    voxels = height * width * depth
    print(f"Total voxels: {voxels:,} ({voxels/1e6:.2f}M)")

    reset_memory()

    # Create model
    print("\n[1] Creating AutoencoderKL...")
    raw_model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=latent_channels,
        channels=channels,
        num_res_blocks=2,
        attention_levels=(False, False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    ).to(device)

    # Wrap with checkpointing if requested
    if use_checkpointing:
        model = CheckpointedAutoencoder(raw_model)
        print("   Using gradient checkpointing wrapper")
    else:
        model = raw_model

    # Apply torch.compile if requested
    if use_compile:
        print("   Applying torch.compile (this may take a moment)...")
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        print("   torch.compile applied")

    num_params = sum(p.numel() for p in model.parameters())
    _, peak_after_model = get_gpu_memory_gb()
    print(f"   Parameters: {num_params/1e6:.1f}M")
    print(f"   VRAM after model: {peak_after_model:.2f} GB")

    # Create discriminator if needed
    disc = None
    if include_discriminator:
        print("\n[2] Creating PatchDiscriminator...")
        disc = PatchDiscriminator(
            spatial_dims=3,
            in_channels=1,
            channels=64,
            num_layers_d=3,
        ).to(device)
        disc_params = sum(p.numel() for p in disc.parameters())
        _, peak_after_disc = get_gpu_memory_gb()
        print(f"   Parameters: {disc_params/1e6:.1f}M")
        print(f"   VRAM after discriminator: {peak_after_disc:.2f} GB")

    # Create optimizers
    print("\n[3] Creating optimizers...")
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=5e-5)
    if disc:
        optimizer_d = torch.optim.AdamW(disc.parameters(), lr=1e-4)
    _, peak_after_opt = get_gpu_memory_gb()
    print(f"   VRAM after optimizers: {peak_after_opt:.2f} GB")

    # Create dummy input
    print("\n[4] Creating input tensor...")
    x = torch.randn(batch_size, 1, depth, height, width, device=device)
    input_size_gb = x.numel() * 4 / 1024**3  # float32
    _, peak_after_input = get_gpu_memory_gb()
    print(f"   Input shape: {list(x.shape)}")
    print(f"   Input size: {input_size_gb:.3f} GB")
    print(f"   VRAM after input: {peak_after_input:.2f} GB")

    # Forward pass
    print("\n[5] Forward pass...")
    try:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            reconstruction, z_mu, z_sigma = model(x)

        _, peak_after_forward = get_gpu_memory_gb()
        print(f"   Output shape: {list(reconstruction.shape)}")
        print(f"   Latent shape: {list(z_mu.shape)}")
        print(f"   VRAM after forward: {peak_after_forward:.2f} GB")

        # Compute loss
        loss = torch.nn.functional.l1_loss(reconstruction, x)

        # Discriminator forward
        if disc and include_discriminator:
            print("\n[6] Discriminator forward...")
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                disc_real = disc(x)
                disc_fake = disc(reconstruction.detach())
            _, peak_after_disc_fwd = get_gpu_memory_gb()
            print(f"   VRAM after disc forward: {peak_after_disc_fwd:.2f} GB")

        # Backward pass
        if include_backward:
            print("\n[7] Backward pass (generator)...")
            optimizer_g.zero_grad()
            loss.backward()
            _, peak_after_backward = get_gpu_memory_gb()
            print(f"   VRAM after backward: {peak_after_backward:.2f} GB")

            optimizer_g.step()
            _, peak_after_step = get_gpu_memory_gb()
            print(f"   VRAM after optimizer step: {peak_after_step:.2f} GB")

        # Final summary
        _, final_peak = get_gpu_memory_gb()
        print(f"\n{'='*60}")
        print(f"PEAK VRAM USAGE: {final_peak:.2f} GB")
        print(f"{'='*60}")

        # Estimate for different batch sizes
        print(f"\nEstimated VRAM for different batch sizes:")
        # Rough estimate: memory scales ~linearly with batch size for activations
        activation_mem = final_peak - peak_after_opt
        base_mem = peak_after_opt
        for bs in [1, 2, 4]:
            estimated = base_mem + activation_mem * bs
            fits = "✓" if estimated < 80 else "✗"
            print(f"   batch_size={bs}: ~{estimated:.1f} GB {fits}")

        return final_peak

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n   *** OUT OF MEMORY ***")
        print(f"   {e}")
        _, peak_at_oom = get_gpu_memory_gb()
        print(f"   Peak before OOM: {peak_at_oom:.2f} GB")
        return None


def main():
    parser = argparse.ArgumentParser(description="Profile 3D VAE memory usage")
    parser.add_argument("--size", type=int, nargs=3, default=[256, 256, 160],
                        metavar=("H", "W", "D"), help="Volume size (height width depth)")
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 128],
                        help="Channel configuration")
    parser.add_argument("--latent-channels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-backward", action="store_true", help="Skip backward pass")
    parser.add_argument("--no-disc", action="store_true", help="Skip discriminator")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP (use fp32)")
    parser.add_argument("--checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--sweep", action="store_true", help="Test multiple sizes")
    args = parser.parse_args()

    if args.sweep:
        # Test multiple common sizes
        sizes = [
            (64, 64, 64),
            (96, 96, 64),
            (128, 128, 80),
            (128, 128, 128),
            (160, 160, 96),
            (192, 192, 128),
            (256, 256, 160),
        ]
        results = []
        for h, w, d in sizes:
            reset_memory()
            peak = profile_vae3d(
                height=h, width=w, depth=d,
                channels=tuple(args.channels),
                latent_channels=args.latent_channels,
                batch_size=args.batch_size,
                include_backward=not args.no_backward,
                include_discriminator=not args.no_disc,
                use_amp=not args.no_amp,
                use_checkpointing=args.checkpoint,
                use_compile=args.compile,
            )
            results.append((h, w, d, peak))

        print(f"\n\n{'='*60}")
        print("SUMMARY - Peak VRAM by volume size")
        print(f"{'='*60}")
        print(f"{'Size':<20} {'Voxels':<12} {'VRAM':<10} {'Fits 80GB'}")
        print("-" * 60)
        for h, w, d, peak in results:
            voxels = h * w * d
            if peak:
                fits = "✓" if peak < 80 else "✗"
                print(f"{h}x{w}x{d:<10} {voxels/1e6:>6.2f}M     {peak:>6.1f} GB   {fits}")
            else:
                print(f"{h}x{w}x{d:<10} {voxels/1e6:>6.2f}M     OOM")
    else:
        profile_vae3d(
            height=args.size[0],
            width=args.size[1],
            depth=args.size[2],
            channels=tuple(args.channels),
            latent_channels=args.latent_channels,
            batch_size=args.batch_size,
            include_backward=not args.no_backward,
            include_discriminator=not args.no_disc,
            use_amp=not args.no_amp,
            use_checkpointing=args.checkpoint,
            use_compile=args.compile,
        )


if __name__ == "__main__":
    main()
