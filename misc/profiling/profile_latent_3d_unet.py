#!/usr/bin/env python3
"""Profile 3D UNet VRAM usage for latent diffusion.

Tests various UNet configurations for 80GB GPU to find optimal architecture.
Handles OOM gracefully and reports results for all tested configurations.

Key insight: 4-level UNets require latent dimensions divisible by 16 (2^4).
For 8x compression with small latents, use 3-level UNets instead.

Usage:
    python misc/profiling/profile_latent_3d_unet.py
"""

import gc
import sys
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
    num_levels: int = 4  # Computed from channels length

    def __post_init__(self):
        self.num_levels = len(self.channels)


# 4-level UNet configs (require latent dims divisible by 16)
CONFIGS_4LEVEL = [
    UNetConfig(
        name="tiny",
        channels=[32, 64, 128, 256],
        attention_levels=[False, False, True, True],
        num_res_blocks=[1, 1, 1, 1],
        num_head_channels=32,
    ),
    UNetConfig(
        name="small",
        channels=[64, 128, 256, 512],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
        num_head_channels=32,
    ),
    UNetConfig(
        name="maisi",
        channels=[64, 128, 256, 512],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
        num_head_channels=32,
    ),
    UNetConfig(
        name="medium",
        channels=[128, 256, 512, 1024],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
        num_head_channels=64,
    ),
    UNetConfig(
        name="medium_deep",
        channels=[128, 256, 512, 1024],
        attention_levels=[False, False, True, True],
        num_res_blocks=[3, 3, 3, 3],
        num_head_channels=64,
    ),
    UNetConfig(
        name="large",
        channels=[256, 512, 1024, 2048],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
        num_head_channels=64,
    ),
    UNetConfig(
        name="large_deep",
        channels=[256, 512, 1024, 2048],
        attention_levels=[False, False, True, True],
        num_res_blocks=[3, 3, 3, 3],
        num_head_channels=64,
    ),
]

# 3-level UNet configs (work with smaller latents, dims divisible by 8)
CONFIGS_3LEVEL = [
    UNetConfig(
        name="3lvl_tiny",
        channels=[64, 128, 256],
        attention_levels=[False, True, True],
        num_res_blocks=[1, 1, 1],
        num_head_channels=32,
    ),
    UNetConfig(
        name="3lvl_small",
        channels=[128, 256, 512],
        attention_levels=[False, True, True],
        num_res_blocks=[2, 2, 2],
        num_head_channels=64,
    ),
    UNetConfig(
        name="3lvl_medium",
        channels=[192, 384, 768],
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
        channels=[384, 768, 1536],
        attention_levels=[False, True, True],
        num_res_blocks=[3, 3, 3],
        num_head_channels=64,
    ),
    UNetConfig(
        name="3lvl_huge",
        channels=[512, 1024, 2048],
        attention_levels=[False, True, True],
        num_res_blocks=[3, 3, 3],
        num_head_channels=64,
    ),
]

# Latent shapes to test: (name, (C, D, H, W), compatible_levels)
# Based on actual VQ-VAE outputs
# Original volumes: 160x256x256 (DxHxW)
LATENT_SHAPES = [
    # 4x compression: 160x256x256 -> 40x64x64 (divisible by 16, supports 4-level)
    ("4x_latent", (4, 40, 64, 64), [3, 4]),

    # 8x compression: 160x256x256 -> 20x32x32 (divisible by 8 but not 16, 3-level only)
    ("8x_latent", (4, 20, 32, 32), [3]),

    # 8x with seg conditioning: 8 input channels (bravo_latent + seg_latent)
    ("8x_cond", (8, 20, 32, 32), [3]),

    # Smaller volume: 160x160x128 with 8x compression -> 20x20x16
    ("8x_small", (4, 20, 20, 16), [3]),
    ("8x_small_cond", (8, 20, 20, 16), [3]),
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


def estimate_model_params(config: UNetConfig, in_channels: int, out_channels: int) -> float:
    """Rough estimate of model parameters in millions."""
    channels = config.channels
    total = 0
    total += in_channels * channels[0] * 27
    for i, ch in enumerate(channels):
        num_res = config.num_res_blocks[i]
        total += num_res * ch * ch * 27 * 2
        if config.attention_levels[i]:
            total += ch * ch * 3
        if i < len(channels) - 1:
            total += ch * channels[i + 1] * 27
    total += channels[0] * out_channels * 27
    return total / 1e6


def profile_config(
    config: UNetConfig,
    latent_shape: Tuple[int, int, int, int],
    device: torch.device,
    use_amp: bool = True,
    gradient_checkpointing: bool = True,
) -> Optional[dict]:
    """Profile a single UNet configuration."""
    reset_vram()

    in_channels, depth, height, width = latent_shape
    # For conditioned mode (8ch input), output is 4ch (noise prediction for bravo only)
    out_channels = 4 if in_channels == 8 else in_channels

    try:
        model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=config.channels,
            attention_levels=config.attention_levels,
            num_res_blocks=config.num_res_blocks,
            num_head_channels=config.num_head_channels,
        ).to(device)

        if gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()

        model.train()
        model_vram = get_vram_mb()
        num_params = count_parameters(model)

        x = torch.randn(1, in_channels, depth, height, width, device=device)
        timesteps = torch.randint(0, 1000, (1,), device=device)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            output = model(x, timesteps)
            loss = output.mean()

        forward_vram = get_peak_vram_mb()
        loss.backward()
        backward_vram = get_peak_vram_mb()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()
        total_vram = get_peak_vram_mb()

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
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "size" in str(e).lower():
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

    for latent_name, latent_shape, compatible_levels in LATENT_SHAPES:
        print(f"\n{'='*100}")
        print(f"Latent: {latent_name} -> {latent_shape} (C, D, H, W)")
        print(f"Compatible UNet levels: {compatible_levels}")
        print(f"{'='*100}")
        print(f"{'Config':<15} {'Channels':<25} {'Params':>10} {'Model':>8} {'Fwd':>8} {'Bwd':>8} {'Total':>8} {'Status'}")
        print(f"{'-'*15} {'-'*25} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

        # Select configs based on compatible levels
        configs_to_test = []
        if 4 in compatible_levels:
            configs_to_test.extend(CONFIGS_4LEVEL)
        if 3 in compatible_levels:
            configs_to_test.extend(CONFIGS_3LEVEL)

        for config in configs_to_test:
            channels_str = str(config.channels)
            out_ch = 4 if latent_shape[0] == 8 else latent_shape[0]
            est_params = estimate_model_params(config, latent_shape[0], out_ch)

            # Skip configs estimated >2500M params
            if est_params > 2500:
                print(f"{config.name:<15} {channels_str:<25} {'~'+str(int(est_params))+'M':>10} {'---':>8} {'---':>8} {'---':>8} {'SKIP':>8} ⏭ Too large")
                sys.stdout.flush()
                continue

            result = profile_config(config, latent_shape, device)

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
                sys.stdout.flush()

                results.append({
                    'latent': latent_name,
                    'config': config.name,
                    'channels': channels_str,
                    'num_levels': config.num_levels,
                    **result
                })
            else:
                print(f"{config.name:<15} {channels_str:<25} {'---':>10} {'---':>8} {'---':>8} {'---':>8} {'OOM':>8} ✗ OOM")
                sys.stdout.flush()

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY: Best configs for each latent shape (within 60GB budget)")
    print(f"{'='*100}")

    for latent_name, latent_shape, _ in LATENT_SHAPES:
        print(f"\n{latent_name} {latent_shape}:")
        matching = [r for r in results
                   if r['latent'] == latent_name
                   and r['total_vram_gb'] <= 60]

        if matching:
            for r in sorted(matching, key=lambda x: -x['params_m'])[:5]:
                print(f"  {r['config']:<15} {r['channels']:<25} {r['params_m']:>6.1f}M params, {r['total_vram_gb']:>5.1f}GB")
        else:
            all_for_latent = [r for r in results if r['latent'] == latent_name]
            if all_for_latent:
                smallest = min(all_for_latent, key=lambda x: x['total_vram_gb'])
                print(f"  Smallest: {smallest['config']} ({smallest['total_vram_gb']:.1f}GB)")
            else:
                print(f"  All configs failed")

    # Recommendations for 8x compression (the user's case)
    print(f"\n{'='*100}")
    print("RECOMMENDATIONS for 8x compression latent diffusion")
    print(f"{'='*100}")

    for latent_name, latent_shape, _ in LATENT_SHAPES:
        if '8x' not in latent_name:
            continue
        matching = [r for r in results
                   if r['latent'] == latent_name
                   and 20 <= r['total_vram_gb'] <= 60]
        if matching:
            best = max(matching, key=lambda x: x['params_m'])
            print(f"\n{latent_name}:")
            print(f"  Config: {best['config']}")
            print(f"  Channels: {best['channels']}")
            print(f"  Params: {best['params_m']:.1f}M, VRAM: {best['total_vram_gb']:.1f}GB")
            print(f"  Hydra override: model.channels={best['channels']}")


if __name__ == "__main__":
    main()
