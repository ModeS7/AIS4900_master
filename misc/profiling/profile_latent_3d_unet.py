#!/usr/bin/env python3
"""Profile 3D UNet VRAM usage for latent diffusion.

Tests various UNet configurations for 80GB GPU to find optimal architecture.
Handles OOM gracefully and reports results for all tested configurations.

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


# Comprehensive UNet configurations to test - from tiny to massive
CONFIGS = [
    # === TINY (baseline) ===
    UNetConfig(
        name="tiny",
        channels=[32, 64, 128, 256],
        attention_levels=[False, False, True, True],
        num_res_blocks=[1, 1, 1, 1],
        num_head_channels=32,
    ),

    # === SMALL ===
    UNetConfig(
        name="small",
        channels=[64, 128, 256, 512],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
        num_head_channels=32,
    ),

    # === MAISI (NVIDIA reference) ===
    UNetConfig(
        name="maisi",
        channels=[64, 128, 256, 512],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
        num_head_channels=32,
    ),

    # === MEDIUM ===
    UNetConfig(
        name="medium",
        channels=[128, 256, 512, 1024],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
        num_head_channels=64,
    ),

    # === MEDIUM-DEEP (more res blocks) ===
    UNetConfig(
        name="medium_deep",
        channels=[128, 256, 512, 1024],
        attention_levels=[False, False, True, True],
        num_res_blocks=[3, 3, 3, 3],
        num_head_channels=64,
    ),

    # === LARGE ===
    UNetConfig(
        name="large",
        channels=[256, 512, 1024, 2048],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
        num_head_channels=64,
    ),

    # === LARGE-DEEP (LDM_4x config) ===
    UNetConfig(
        name="large_deep",
        channels=[256, 512, 1024, 2048],
        attention_levels=[False, False, True, True],
        num_res_blocks=[3, 3, 3, 3],
        num_head_channels=64,
    ),

    # === XLARGE ===
    UNetConfig(
        name="xlarge",
        channels=[384, 768, 1536, 3072],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
        num_head_channels=64,
    ),

    # === XLARGE-DEEP ===
    UNetConfig(
        name="xlarge_deep",
        channels=[384, 768, 1536, 3072],
        attention_levels=[False, False, True, True],
        num_res_blocks=[3, 3, 3, 3],
        num_head_channels=64,
    ),

    # === HUGE (LDM_8x config - user's original) ===
    UNetConfig(
        name="huge",
        channels=[512, 1024, 2048, 4096],
        attention_levels=[False, False, True, True],
        num_res_blocks=[3, 3, 3, 3],
        num_head_channels=64,
    ),

    # === 3-LEVEL variants (less downsampling) ===
    UNetConfig(
        name="3lvl_medium",
        channels=[128, 256, 512],
        attention_levels=[False, True, True],
        num_res_blocks=[2, 2, 2],
        num_head_channels=64,
    ),

    UNetConfig(
        name="3lvl_large",
        channels=[256, 512, 1024],
        attention_levels=[False, True, True],
        num_res_blocks=[3, 3, 3],
        num_head_channels=64,
    ),

    UNetConfig(
        name="3lvl_xlarge",
        channels=[512, 1024, 2048],
        attention_levels=[False, True, True],
        num_res_blocks=[3, 3, 3],
        num_head_channels=64,
    ),
]

# Latent shapes to test: [C, D, H, W]
# Based on actual VQ-VAE outputs with different compression factors
LATENT_SHAPES = [
    # 4x compression from 160x256x256 input
    ("4x_40x64x64", (4, 40, 64, 64)),
    ("4x_40x40x32", (4, 40, 40, 32)),   # From 160x160x128

    # 8x compression from 160x256x256 input
    ("8x_20x32x32", (4, 20, 32, 32)),
    ("8x_20x20x16", (4, 20, 20, 16)),   # From 160x160x128

    # bravo_seg_cond mode: 8 channels (bravo_latent + seg_latent)
    ("8x_cond_20x32x32", (8, 20, 32, 32)),
    ("8x_cond_20x20x16", (8, 20, 20, 16)),
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
            out_channels=in_channels // 2 if in_channels == 8 else in_channels,  # For cond mode, output is 4ch
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
        # Handle OOM gracefully
        reset_vram()
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            reset_vram()
            return None
        print(f"    RuntimeError: {e}")
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

    print("=" * 100)
    print("3D UNet Latent Diffusion VRAM Profiling")
    print("=" * 100)
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {gpu_total_mem:.1f} GB")
    print(f"Settings: batch_size=1, AMP=bfloat16, gradient_checkpointing=True")
    print("=" * 100)

    results = []

    for latent_name, latent_shape in LATENT_SHAPES:
        print(f"\n{'='*100}")
        print(f"Latent Shape: {latent_name} -> {latent_shape} (C, D, H, W)")
        print(f"{'='*100}")
        print(f"{'Config':<15} {'Channels':<25} {'Params':>10} {'Model':>8} {'Fwd':>8} {'Bwd':>8} {'Total':>8} {'Status'}")
        print(f"{'-'*15} {'-'*25} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

        for config in CONFIGS:
            result = profile_config(config, latent_shape, device)

            channels_str = str(config.channels)

            if result:
                status = "✓ OK"
                if result['total_vram_gb'] > 70:
                    status = "⚠ >70GB"
                elif result['total_vram_gb'] > 60:
                    status = "⚠ >60GB"
                elif result['total_vram_gb'] > 40:
                    status = "⚠ >40GB"

                print(f"{config.name:<15} {channels_str:<25} {result['params_m']:>9.1f}M "
                      f"{result['model_vram_mb']/1024:>7.1f}G "
                      f"{result['forward_vram_mb']/1024:>7.1f}G "
                      f"{result['backward_vram_mb']/1024:>7.1f}G "
                      f"{result['total_vram_gb']:>7.1f}G "
                      f"{status}")

                results.append({
                    'latent': latent_name,
                    'config': config.name,
                    'channels': channels_str,
                    **result
                })
            else:
                print(f"{config.name:<15} {channels_str:<25} {'---':>10} {'---':>8} {'---':>8} {'---':>8} {'OOM':>8} ✗ OOM")

    # Summary tables
    print(f"\n{'='*100}")
    print("SUMMARY: Best configs for each latent shape (within 60GB budget)")
    print(f"{'='*100}")

    for latent_name, _ in LATENT_SHAPES:
        print(f"\n{latent_name}:")
        matching = [r for r in results
                   if r['latent'] == latent_name
                   and r['total_vram_gb'] <= 60]

        if matching:
            # Sort by params (largest first within budget)
            for r in sorted(matching, key=lambda x: -x['params_m'])[:5]:
                print(f"  {r['config']:<15} {r['channels']:<25} {r['params_m']:>6.1f}M params, {r['total_vram_gb']:>5.1f}GB")
        else:
            # Show smallest OOM
            all_for_latent = [r for r in results if r['latent'] == latent_name]
            if all_for_latent:
                smallest = min(all_for_latent, key=lambda x: x['total_vram_gb'])
                print(f"  Smallest tested: {smallest['config']} ({smallest['total_vram_gb']:.1f}GB) - need smaller config")
            else:
                print(f"  All configs OOM - need even smaller configs")

    # Recommendations
    print(f"\n{'='*100}")
    print("RECOMMENDATIONS for 80GB GPU (60GB training budget)")
    print(f"{'='*100}")

    for latent_name, _ in LATENT_SHAPES:
        matching = [r for r in results
                   if r['latent'] == latent_name
                   and 40 <= r['total_vram_gb'] <= 60]
        if matching:
            best = max(matching, key=lambda x: x['params_m'])
            print(f"\n{latent_name}:")
            print(f"  Recommended: {best['config']}")
            print(f"  Channels: {best['channels']}")
            print(f"  Params: {best['params_m']:.1f}M, VRAM: {best['total_vram_gb']:.1f}GB")


if __name__ == "__main__":
    main()
