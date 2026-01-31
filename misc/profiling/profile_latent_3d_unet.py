#!/usr/bin/env python3
"""Profile 3D UNet VRAM usage for latent diffusion.

Tests UNet configurations for 80GB GPU with ~60GB training budget.

Final configs:
- LDM_4x: For VQ-VAE 4x compression (64x64x40 latent), ~55GB training
- LDM_8x: For VQ-VAE 8x compression (32x32x20 latent), ~60GB training

Usage:
    python misc/profiling/profile_latent_3d_unet.py
"""

import gc
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Optional
from monai.networks.nets import DiffusionModelUNet


@dataclass
class UNetConfig:
    """UNet configuration to test."""
    name: str
    channels: List[int]
    attention_levels: List[bool]
    num_res_blocks: List[int]
    num_head_channels: int = 32


# Final UNet configurations for latent diffusion on 80GB GPU (~60GB training budget)
CONFIGS = [
    # === REFERENCE: MAISI (what NVIDIA uses) ===
    UNetConfig(
        name="MAISI",
        channels=[64, 128, 256, 512],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
        num_head_channels=32,
    ),

    # === CHOSEN CONFIGS ===

    # For VQ-VAE 4x compression (latent: 64x64x40x4, decoder: ~5GB)
    # Training VRAM: ~55GB, leaves ~20GB for validation w/ decoder
    UNetConfig(
        name="LDM_4x",
        channels=[256, 512, 1024, 2048],
        attention_levels=[False, False, True, True],
        num_res_blocks=[3, 3, 3, 3],
        num_head_channels=64,
    ),

    # For VQ-VAE 8x compression (latent: 32x32x20x4, decoder: ~10GB)
    # Smaller latent = 8× less activation memory = can fit MUCH larger UNet
    # Target: ~60GB training VRAM
    UNetConfig(
        name="LDM_8x",
        channels=[512, 1024, 2048, 4096],
        attention_levels=[False, False, True, True],
        num_res_blocks=[3, 3, 3, 3],
        num_head_channels=64,
    ),
]

# Latent shapes to test: [C, D, H, W]
# Original input: 256x256x160
LATENT_SHAPES = [
    ("64x64x40x4", (4, 40, 64, 64)),   # VQ-VAE 4x compression (decoder: ~5GB)
    ("32x32x20x4", (4, 20, 32, 32)),   # VQ-VAE 8x compression (decoder: ~10GB)
]


def get_vram_mb() -> float:
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_peak_vram_mb() -> float:
    """Get peak VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_vram():
    """Reset VRAM tracking."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_config(
    config: UNetConfig,
    latent_shape: Tuple[int, int, int, int],
    device: torch.device,
    use_amp: bool = True,
    gradient_checkpointing: bool = True,
) -> Optional[dict]:
    """Profile a single UNet configuration.

    Returns dict with VRAM measurements or None if OOM.
    """
    reset_vram()

    in_channels, depth, height, width = latent_shape

    try:
        # Create model
        model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=in_channels,  # Predict noise/velocity same channels
            channels=config.channels,
            attention_levels=config.attention_levels,
            num_res_blocks=config.num_res_blocks,
            num_head_channels=config.num_head_channels,
        ).to(device)

        if gradient_checkpointing:
            # Enable gradient checkpointing if available
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()

        model.train()

        model_vram = get_vram_mb()
        num_params = count_parameters(model)

        # Create dummy inputs
        x = torch.randn(1, in_channels, depth, height, width, device=device)
        timesteps = torch.randint(0, 1000, (1,), device=device)

        # Forward pass with AMP
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            output = model(x, timesteps)
            loss = output.mean()

        forward_vram = get_peak_vram_mb()

        # Backward pass
        loss.backward()

        backward_vram = get_peak_vram_mb()

        # Optimizer step simulation
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()

        total_vram = get_peak_vram_mb()

        # Cleanup
        del model, x, timesteps, output, loss, optimizer
        reset_vram()

        return {
            'params_m': num_params / 1e6,
            'model_vram_mb': model_vram,
            'forward_vram_mb': forward_vram,
            'backward_vram_mb': backward_vram,
            'total_vram_mb': total_vram,
            'total_vram_gb': total_vram / 1024,
        }

    except torch.cuda.OutOfMemoryError:
        reset_vram()
        return None
    except Exception as e:
        print(f"    Error: {e}")
        reset_vram()
        return None


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print("=" * 80)
    print("3D UNet Latent Diffusion VRAM Profiling")
    print("=" * 80)
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {gpu_total_mem:.1f} GB")
    print(f"Settings: batch_size=1, AMP=bfloat16, gradient_checkpointing=True")
    print("=" * 80)

    results = []

    for latent_name, latent_shape in LATENT_SHAPES:
        print(f"\n{'='*80}")
        print(f"Latent Shape: {latent_name} -> {latent_shape}")
        print(f"{'='*80}")
        print(f"{'Config':<20} {'Params':>10} {'Model':>10} {'Forward':>10} {'Backward':>10} {'Total':>10} {'Status'}")
        print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for config in CONFIGS:
            result = profile_config(config, latent_shape, device)

            if result:
                status = "✓ OK"
                if result['total_vram_gb'] > 40:
                    status = "⚠ >40GB"
                elif result['total_vram_gb'] > 24:
                    status = "⚠ >24GB"

                print(f"{config.name:<20} {result['params_m']:>9.1f}M "
                      f"{result['model_vram_mb']:>9.0f}MB "
                      f"{result['forward_vram_mb']:>9.0f}MB "
                      f"{result['backward_vram_mb']:>9.0f}MB "
                      f"{result['total_vram_gb']:>9.1f}GB "
                      f"{status}")

                results.append({
                    'latent': latent_name,
                    'config': config.name,
                    **result
                })
            else:
                print(f"{config.name:<20} {'OOM':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10} ✗ OOM")

    # Summary for target range (20-40GB)
    print(f"\n{'='*80}")
    print("Summary: Configs fitting in 20-40GB range")
    print(f"{'='*80}")

    for latent_name, _ in LATENT_SHAPES:
        print(f"\n{latent_name}:")
        matching = [r for r in results
                   if r['latent'] == latent_name
                   and 20 <= r['total_vram_gb'] <= 40]

        if matching:
            for r in sorted(matching, key=lambda x: x['total_vram_gb']):
                print(f"  {r['config']:<20} {r['params_m']:>6.1f}M params, {r['total_vram_gb']:>5.1f}GB")
        else:
            # Show closest options
            all_for_latent = [r for r in results if r['latent'] == latent_name]
            if all_for_latent:
                under = [r for r in all_for_latent if r['total_vram_gb'] < 20]
                over = [r for r in all_for_latent if r['total_vram_gb'] > 40]

                if under:
                    best_under = max(under, key=lambda x: x['total_vram_gb'])
                    print(f"  Largest under 20GB: {best_under['config']} ({best_under['total_vram_gb']:.1f}GB)")
                if over:
                    best_over = min(over, key=lambda x: x['total_vram_gb'])
                    print(f"  Smallest over 40GB: {best_over['config']} ({best_over['total_vram_gb']:.1f}GB)")


if __name__ == "__main__":
    main()
