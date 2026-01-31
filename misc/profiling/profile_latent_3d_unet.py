#!/usr/bin/env python3
"""Profile 3D UNet VRAM usage for latent diffusion with seg conditioning.

Tests UNet configurations for two latent shapes:
- 4x compression: (8, 40, 64, 64) - supports 4-level UNets
- 8x compression: (8, 20, 32, 32) - requires 3-level UNets

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

    @property
    def num_levels(self) -> int:
        return len(self.channels)


# 4-level UNet configs (for 4x compression latents)
CONFIGS_4LEVEL = [
    UNetConfig("4L_tiny", [32, 64, 128, 256], [False, False, True, True], [1, 1, 1, 1], 32),
    UNetConfig("4L_small", [64, 128, 256, 512], [False, False, True, True], [2, 2, 2, 2], 32),
    UNetConfig("4L_maisi", [64, 128, 256, 512], [False, False, True, True], [2, 2, 2, 2], 32),
    UNetConfig("4L_medium", [128, 256, 512, 1024], [False, False, True, True], [2, 2, 2, 2], 64),
    UNetConfig("4L_medium_d", [128, 256, 512, 1024], [False, False, True, True], [3, 3, 3, 3], 64),
    UNetConfig("4L_large", [256, 512, 1024, 2048], [False, False, True, True], [2, 2, 2, 2], 64),
    UNetConfig("4L_large_d", [256, 512, 1024, 2048], [False, False, True, True], [3, 3, 3, 3], 64),
]

# 3-level UNet configs (for 8x compression latents)
CONFIGS_3LEVEL = [
    UNetConfig("3L_tiny", [32, 64, 128], [False, True, True], [1, 1, 1], 32),
    UNetConfig("3L_small", [64, 128, 256], [False, True, True], [2, 2, 2], 32),
    UNetConfig("3L_medium", [128, 256, 512], [False, True, True], [2, 2, 2], 64),
    UNetConfig("3L_medium_d", [128, 256, 512], [False, True, True], [3, 3, 3], 64),
    UNetConfig("3L_large", [256, 512, 1024], [False, True, True], [2, 2, 2], 64),
    UNetConfig("3L_large_d", [256, 512, 1024], [False, True, True], [3, 3, 3], 64),
    UNetConfig("3L_xlarge", [384, 768, 1536], [False, True, True], [2, 2, 2], 64),
    UNetConfig("3L_xlarge_d", [384, 768, 1536], [False, True, True], [3, 3, 3], 64),
    UNetConfig("3L_huge", [512, 1024, 2048], [False, True, True], [2, 2, 2], 64),
    UNetConfig("3L_huge_d", [512, 1024, 2048], [False, True, True], [3, 3, 3], 64),
]

# The two latent shapes to test
LATENT_SHAPES = [
    # 4x compression with seg conditioning: 8 channels, 40x64x64 spatial
    ("4x_cond", (8, 40, 64, 64), CONFIGS_4LEVEL + CONFIGS_3LEVEL),

    # 8x compression with seg conditioning: 8 channels, 20x32x32 spatial
    ("8x_cond", (8, 20, 32, 32), CONFIGS_3LEVEL),
]


def get_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_config(
    config: UNetConfig,
    latent_shape: Tuple[int, int, int, int],
    device: torch.device,
) -> Optional[dict]:
    """Profile a single UNet configuration. Returns None if fails."""
    reset_vram()

    in_channels, depth, height, width = latent_shape
    out_channels = 4  # Output is noise prediction for bravo only (4ch)

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

        model.train()
        model_vram = get_vram_mb()
        num_params = count_parameters(model)

        x = torch.randn(1, in_channels, depth, height, width, device=device)
        timesteps = torch.randint(0, 1000, (1,), device=device)

        # Forward with AMP bfloat16
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
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
            'model_gb': model_vram / 1024,
            'forward_gb': forward_vram / 1024,
            'backward_gb': backward_vram / 1024,
            'total_gb': total_vram / 1024,
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError):
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
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print("=" * 95)
    print("3D UNet VRAM Profiling for Seg-Conditioned Latent Diffusion")
    print("=" * 95)
    print(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    print("Settings: batch_size=1, AMP=bfloat16, in_ch=8, out_ch=4")
    print("=" * 95)

    all_results = {}

    for latent_name, latent_shape, configs in LATENT_SHAPES:
        print(f"\n{'='*95}")
        print(f"Latent: {latent_name} -> (C={latent_shape[0]}, D={latent_shape[1]}, H={latent_shape[2]}, W={latent_shape[3]})")
        print(f"{'='*95}")
        print(f"{'Config':<12} {'Channels':<22} {'Params':>9} {'Model':>7} {'Fwd':>7} {'Bwd':>7} {'Total':>7} {'Status':<10}")
        print("-" * 95)

        results = []

        for config in configs:
            result = profile_config(config, latent_shape, device)
            ch_str = str(config.channels)

            if result:
                if result['total_gb'] > 70:
                    status = "⚠ >70GB"
                elif result['total_gb'] > 60:
                    status = "⚠ >60GB"
                elif result['total_gb'] > 40:
                    status = "⚠ >40GB"
                else:
                    status = "✓ OK"

                print(f"{config.name:<12} {ch_str:<22} {result['params_m']:>8.1f}M "
                      f"{result['model_gb']:>6.1f}G {result['forward_gb']:>6.1f}G "
                      f"{result['backward_gb']:>6.1f}G {result['total_gb']:>6.1f}G {status:<10}")
                results.append({'config': config, **result})
            else:
                print(f"{config.name:<12} {ch_str:<22} {'---':>9} {'---':>7} {'---':>7} {'---':>7} {'OOM':>7} ✗ FAIL")

            sys.stdout.flush()

        all_results[latent_name] = results

    # Summary
    print(f"\n{'='*95}")
    print("SUMMARY")
    print("=" * 95)

    for latent_name, results in all_results.items():
        print(f"\n{latent_name}:")

        # Within 60GB budget
        within_budget = [r for r in results if r['total_gb'] <= 60]
        if within_budget:
            print("  Within 60GB budget (sorted by params):")
            for r in sorted(within_budget, key=lambda x: -x['params_m']):
                cfg = r['config']
                print(f"    {cfg.name:<12} {str(cfg.channels):<22} {r['params_m']:>7.1f}M  {r['total_gb']:>5.1f}GB")

        # Best recommendation
        optimal = [r for r in results if 30 <= r['total_gb'] <= 60]
        if optimal:
            best = max(optimal, key=lambda x: x['params_m'])
            cfg = best['config']
            print(f"\n  RECOMMENDED: {cfg.name}")
            print(f"    model.channels={list(cfg.channels)}")
            print(f"    model.attention_levels={list(cfg.attention_levels)}")
            print(f"    model.num_res_blocks={list(cfg.num_res_blocks)}")
            print(f"    model.num_head_channels={cfg.num_head_channels}")
            print(f"    Params: {best['params_m']:.1f}M, VRAM: {best['total_gb']:.1f}GB")


if __name__ == "__main__":
    main()
