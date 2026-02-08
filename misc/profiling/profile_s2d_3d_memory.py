#!/usr/bin/env python3
"""VRAM profiler for space-to-depth 3D diffusion architectures.

Comprehensive benchmark of UNet architectures for s2d experiments (exp11/exp11_1).
Designed for 80 GB cluster GPUs. Handles OOM gracefully per-config.

Covers:
- 3, 4, and 5-level UNet architectures
- Channel width variations per level count
- WDM-style architectures (Friedrich et al., 2024)
- Attention placement: none, single-level, multi-level, full
- Head channel variations: 16, 32, 64
- Res block depth variations
- Resolution scaling: 64x64x80 (from 128) and 128x128x80 (from 256)

Usage:
    python misc/profiling/profile_s2d_3d_memory.py
"""

import gc
import time
from dataclasses import dataclass

import torch
from torch.amp import autocast
from monai.networks.nets import DiffusionModelUNet


# ── Helpers ──────────────────────────────────────────────────────────────────

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@dataclass
class ProfileResult:
    name: str
    params_m: float
    input_shape: list[int]
    peak_gb: float
    oom: bool = False

    def summary_line(self) -> str:
        if self.oom:
            return f"  {self.name:<58s}  {'OOM':>6s}  {self.peak_gb:>7.1f}G*"
        return (
            f"  {self.name:<58s}  "
            f"{self.params_m:>5.0f}M  "
            f"{self.peak_gb:>7.2f}G"
        )


# ── Core profiler ────────────────────────────────────────────────────────────

def profile_one(
    label: str,
    channels: list[int],
    attention_levels: list[bool],
    num_res_blocks: list[int],
    input_shape: tuple[int, ...],  # (B, C, D, H, W)
    num_head_channels: int = 16,
    norm_num_groups: int = 16,
) -> ProfileResult:
    """Run a full bravo-mode training step and measure peak VRAM."""
    clear_gpu()
    torch.cuda.reset_peak_memory_stats()

    B, C, D, H, W = input_shape
    in_ch = C * 2   # bravo: cat(image, seg)
    out_ch = C

    try:
        model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=tuple(channels),
            attention_levels=tuple(attention_levels),
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
        ).to("cuda")

        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        images = torch.randn(B, C, D, H, W, device="cuda")
        labels = torch.randn(B, C, D, H, W, device="cuda")

        with autocast("cuda", dtype=torch.bfloat16):
            noise = torch.randn_like(images)
            t = torch.rand(B, device="cuda")
            t_exp = t.view(B, 1, 1, 1, 1)
            noisy = (1 - t_exp) * images + t_exp * noise
            model_input = torch.cat([noisy, labels], dim=1)
            timesteps = (t * 1000).float()
            pred = model(model_input, timesteps=timesteps)
            target = images - noise
            loss = ((pred.float() - target.float()) ** 2).mean()

        loss.backward()
        optimizer.step()

        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        return ProfileResult(
            name=label, params_m=n_params / 1e6,
            input_shape=list(input_shape), peak_gb=peak_gb,
        )

    except torch.cuda.OutOfMemoryError:
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        return ProfileResult(
            name=label, params_m=0,
            input_shape=list(input_shape), peak_gb=peak_gb, oom=True,
        )

    finally:
        for v in list(locals().values()):
            if isinstance(v, (torch.Tensor, torch.nn.Module, torch.optim.Optimizer)):
                del v
        clear_gpu()


# ── Architecture configs ─────────────────────────────────────────────────────

def get_configs() -> list[dict]:
    """All architecture configs to profile."""
    configs = []

    # =================================================================
    # 3-LEVEL ARCHITECTURES (2 downsampling steps)
    # =================================================================
    # exp11:   64x64x80   -> 32x32x40 -> 16x16x20
    # exp11_1: 128x128x80 -> 64x64x40 -> 32x32x20

    configs.append(dict(
        name="3L [128,256,512] no-attn",
        channels=[128, 256, 512],
        attention_levels=[False, False, False],
        num_res_blocks=[2, 2, 2],
    ))

    configs.append(dict(
        name="3L [128,256,512] attn-L2",
        channels=[128, 256, 512],
        attention_levels=[False, False, True],
        num_res_blocks=[2, 2, 2],
    ))

    configs.append(dict(
        name="3L [128,512,1024] attn-L2",
        channels=[128, 512, 1024],
        attention_levels=[False, False, True],
        num_res_blocks=[2, 2, 2],
    ))

    configs.append(dict(
        name="3L [256,512,512] attn-L12",
        channels=[256, 512, 512],
        attention_levels=[False, True, True],
        num_res_blocks=[2, 2, 2],
    ))

    configs.append(dict(
        name="3L [256,512,1024] attn-L2",
        channels=[256, 512, 1024],
        attention_levels=[False, False, True],
        num_res_blocks=[2, 2, 2],
    ))

    configs.append(dict(
        name="3L [256,512,1024] attn-L12",
        channels=[256, 512, 1024],
        attention_levels=[False, True, True],
        num_res_blocks=[2, 2, 2],
    ))

    configs.append(dict(
        name="3L [256,512,1024] r=3,3,3 attn-L12",
        channels=[256, 512, 1024],
        attention_levels=[False, True, True],
        num_res_blocks=[3, 3, 3],
    ))

    # =================================================================
    # 4-LEVEL ARCHITECTURES (3 downsampling steps)
    # =================================================================
    # exp11:   64x64x80   -> 32x32x40 -> 16x16x20 -> 8x8x10
    # exp11_1: 128x128x80 -> 64x64x40 -> 32x32x20 -> 16x16x10

    configs.append(dict(
        name="4L [64,128,256,512] no-attn",
        channels=[64, 128, 256, 512],
        attention_levels=[False, False, False, False],
        num_res_blocks=[1, 2, 2, 2],
    ))

    configs.append(dict(
        name="4L [64,128,256,512] attn-L3",
        channels=[64, 128, 256, 512],
        attention_levels=[False, False, False, True],
        num_res_blocks=[1, 2, 2, 2],
    ))

    configs.append(dict(
        name="4L [64,128,256,512] attn-L23",
        channels=[64, 128, 256, 512],
        attention_levels=[False, False, True, True],
        num_res_blocks=[1, 2, 2, 2],
    ))

    configs.append(dict(
        name="4L [64,256,512,512] attn-L23",
        channels=[64, 256, 512, 512],
        attention_levels=[False, False, True, True],
        num_res_blocks=[1, 2, 2, 2],
    ))

    configs.append(dict(
        name="4L [64,256,512,1024] attn-L23",
        channels=[64, 256, 512, 1024],
        attention_levels=[False, False, True, True],
        num_res_blocks=[1, 2, 2, 2],
    ))

    configs.append(dict(
        name="4L [128,256,512,512] attn-L23",
        channels=[128, 256, 512, 512],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
    ))

    configs.append(dict(
        name="4L [128,256,512,1024] attn-L23",
        channels=[128, 256, 512, 1024],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 2, 2],
    ))

    configs.append(dict(
        name="4L [128,256,512,1024] attn-L3",
        channels=[128, 256, 512, 1024],
        attention_levels=[False, False, False, True],
        num_res_blocks=[2, 2, 2, 2],
    ))

    configs.append(dict(
        name="4L [128,256,512,1024] attn-ALL",
        channels=[128, 256, 512, 1024],
        attention_levels=[True, True, True, True],
        num_res_blocks=[2, 2, 2, 2],
    ))

    configs.append(dict(
        name="4L [128,256,512,1024] r=2,2,3,3 attn-L23",
        channels=[128, 256, 512, 1024],
        attention_levels=[False, False, True, True],
        num_res_blocks=[2, 2, 3, 3],
    ))

    # =================================================================
    # 5-LEVEL ARCHITECTURES — Channel width variations
    # =================================================================
    # exp11:   64x64x80   -> ... -> 4x4x5
    # exp11_1: 128x128x80 -> ... -> 8x8x5

    std_res = [1, 1, 2, 2, 2]
    std_attn = [False, False, False, True, True]

    # Narrow / small
    configs.append(dict(
        name="5L narrow [16,32,128,256,256]",
        channels=[16, 32, 128, 256, 256],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    configs.append(dict(
        name="5L narrow-deep [16,32,128,256,512]",
        channels=[16, 32, 128, 256, 512],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    # Baseline (current default_3d_5lvl)
    configs.append(dict(
        name="5L baseline [32,64,256,512,512]",
        channels=[32, 64, 256, 512, 512],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    # Progressive doubling
    configs.append(dict(
        name="5L progressive [32,64,128,256,512]",
        channels=[32, 64, 128, 256, 512],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    configs.append(dict(
        name="5L progressive-wide [64,128,256,512,1024]",
        channels=[64, 128, 256, 512, 1024],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    # Wider deep (keep early layers small)
    configs.append(dict(
        name="5L wider-deep [32,64,256,512,1024]",
        channels=[32, 64, 256, 512, 1024],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    configs.append(dict(
        name="5L wider-deep-v2 [32,64,256,1024,1024]",
        channels=[32, 64, 256, 1024, 1024],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    # Wider early (boost shallow layers)
    configs.append(dict(
        name="5L wider-early [64,128,256,512,512]",
        channels=[64, 128, 256, 512, 512],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    configs.append(dict(
        name="5L wider-early-v2 [128,128,256,512,512]",
        channels=[128, 128, 256, 512, 512],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    # Wider both
    configs.append(dict(
        name="5L wider-both [64,128,512,512,1024]",
        channels=[64, 128, 512, 512, 1024],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    # Flat / uniform
    configs.append(dict(
        name="5L flat [256,256,256,256,256]",
        channels=[256, 256, 256, 256, 256],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    configs.append(dict(
        name="5L flat-deep [256,256,256,512,512]",
        channels=[256, 256, 256, 512, 512],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    # Large (for 80 GB)
    configs.append(dict(
        name="5L large [64,128,512,1024,1024]",
        channels=[64, 128, 512, 1024, 1024],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    configs.append(dict(
        name="5L xlarge [128,256,512,1024,1024]",
        channels=[128, 256, 512, 1024, 1024],
        attention_levels=std_attn, num_res_blocks=std_res,
    ))

    # =================================================================
    # 5-LEVEL — Res block depth variations
    # =================================================================

    configs.append(dict(
        name="5L deep-res [32,64,256,512,512] r=2,2,3,3,3",
        channels=[32, 64, 256, 512, 512],
        attention_levels=std_attn, num_res_blocks=[2, 2, 3, 3, 3],
    ))

    configs.append(dict(
        name="5L deep-res+wide [64,128,256,512,1024] r=1,1,2,3,3",
        channels=[64, 128, 256, 512, 1024],
        attention_levels=std_attn, num_res_blocks=[1, 1, 2, 3, 3],
    ))

    configs.append(dict(
        name="5L uniform-r2 [32,64,256,512,1024] r=2,2,2,2,2",
        channels=[32, 64, 256, 512, 1024],
        attention_levels=std_attn, num_res_blocks=[2, 2, 2, 2, 2],
    ))

    configs.append(dict(
        name="5L large-deep [64,128,512,1024,1024] r=2,2,3,3,3",
        channels=[64, 128, 512, 1024, 1024],
        attention_levels=std_attn, num_res_blocks=[2, 2, 3, 3, 3],
    ))

    # =================================================================
    # WDM-style (Friedrich et al. 2024)
    # =================================================================
    # base=64, mult=(1,2,2,4,4), 2 res blocks, no attention
    # Haar wavelet = our s2d 2x2x2

    configs.append(dict(
        name="WDM-128 [64,128,128,256,256] no-attn",
        channels=[64, 128, 128, 256, 256],
        attention_levels=[False, False, False, False, False],
        num_res_blocks=[2, 2, 2, 2, 2],
    ))

    configs.append(dict(
        name="WDM-128+attn34 [64,128,128,256,256]",
        channels=[64, 128, 128, 256, 256],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[2, 2, 2, 2, 2],
    ))

    configs.append(dict(
        name="WDM-256 [64,128,256,512,512] no-attn",
        channels=[64, 128, 256, 512, 512],
        attention_levels=[False, False, False, False, False],
        num_res_blocks=[2, 2, 2, 2, 2],
    ))

    configs.append(dict(
        name="WDM-256+attn34 [64,128,256,512,512]",
        channels=[64, 128, 256, 512, 512],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[2, 2, 2, 2, 2],
    ))

    configs.append(dict(
        name="WDM-256-wider [64,128,256,512,1024] no-attn",
        channels=[64, 128, 256, 512, 1024],
        attention_levels=[False, False, False, False, False],
        num_res_blocks=[2, 2, 2, 2, 2],
    ))

    # =================================================================
    # 5-LEVEL — Attention variations on [32,64,256,512,1024]
    # =================================================================

    base_ch = [32, 64, 256, 512, 1024]

    configs.append(dict(
        name="5L attn-none [32,64,256,512,1024]",
        channels=base_ch, num_res_blocks=std_res,
        attention_levels=[False, False, False, False, False],
    ))

    configs.append(dict(
        name="5L attn-L4 [32,64,256,512,1024]",
        channels=base_ch, num_res_blocks=std_res,
        attention_levels=[False, False, False, False, True],
    ))

    configs.append(dict(
        name="5L attn-L3 [32,64,256,512,1024]",
        channels=base_ch, num_res_blocks=std_res,
        attention_levels=[False, False, False, True, False],
    ))

    # L3+L4 already covered as "5L wider-deep"

    configs.append(dict(
        name="5L attn-L2+L4 [32,64,256,512,1024]",
        channels=base_ch, num_res_blocks=std_res,
        attention_levels=[False, False, True, False, True],
    ))

    configs.append(dict(
        name="5L attn-L234 [32,64,256,512,1024]",
        channels=base_ch, num_res_blocks=std_res,
        attention_levels=[False, False, True, True, True],
    ))

    configs.append(dict(
        name="5L attn-L1234 [32,64,256,512,1024]",
        channels=base_ch, num_res_blocks=std_res,
        attention_levels=[False, True, True, True, True],
    ))

    configs.append(dict(
        name="5L attn-ALL [32,64,256,512,1024]",
        channels=base_ch, num_res_blocks=std_res,
        attention_levels=[True, True, True, True, True],
    ))

    # =================================================================
    # 5-LEVEL — Attention variations on [64,128,256,512,512]
    # =================================================================

    early_ch = [64, 128, 256, 512, 512]

    configs.append(dict(
        name="5L attn-none [64,128,256,512,512]",
        channels=early_ch, num_res_blocks=std_res,
        attention_levels=[False, False, False, False, False],
    ))

    configs.append(dict(
        name="5L attn-L4 [64,128,256,512,512]",
        channels=early_ch, num_res_blocks=std_res,
        attention_levels=[False, False, False, False, True],
    ))

    configs.append(dict(
        name="5L attn-L3 [64,128,256,512,512]",
        channels=early_ch, num_res_blocks=std_res,
        attention_levels=[False, False, False, True, False],
    ))

    configs.append(dict(
        name="5L attn-L234 [64,128,256,512,512]",
        channels=early_ch, num_res_blocks=std_res,
        attention_levels=[False, False, True, True, True],
    ))

    configs.append(dict(
        name="5L attn-ALL [64,128,256,512,512]",
        channels=early_ch, num_res_blocks=std_res,
        attention_levels=[True, True, True, True, True],
    ))

    # =================================================================
    # 5-LEVEL — Attention variations on [64,128,256,512,1024]
    # =================================================================

    prog_ch = [64, 128, 256, 512, 1024]

    configs.append(dict(
        name="5L attn-none [64,128,256,512,1024]",
        channels=prog_ch, num_res_blocks=std_res,
        attention_levels=[False, False, False, False, False],
    ))

    configs.append(dict(
        name="5L attn-L4 [64,128,256,512,1024]",
        channels=prog_ch, num_res_blocks=std_res,
        attention_levels=[False, False, False, False, True],
    ))

    configs.append(dict(
        name="5L attn-L34 [64,128,256,512,1024]",
        channels=prog_ch, num_res_blocks=std_res,
        attention_levels=[False, False, False, True, True],
    ))

    configs.append(dict(
        name="5L attn-L234 [64,128,256,512,1024]",
        channels=prog_ch, num_res_blocks=std_res,
        attention_levels=[False, False, True, True, True],
    ))

    configs.append(dict(
        name="5L attn-ALL [64,128,256,512,1024]",
        channels=prog_ch, num_res_blocks=std_res,
        attention_levels=[True, True, True, True, True],
    ))

    # =================================================================
    # Head channel variations
    # =================================================================

    configs.append(dict(
        name="5L heads=32 [32,64,256,512,1024] attn-L34",
        channels=base_ch, num_res_blocks=std_res,
        attention_levels=std_attn, num_head_channels=32,
    ))

    configs.append(dict(
        name="5L heads=64 [32,64,256,512,1024] attn-L34",
        channels=base_ch, num_res_blocks=std_res,
        attention_levels=std_attn, num_head_channels=64,
    ))

    configs.append(dict(
        name="5L heads=32 [64,128,256,512,512] attn-L34",
        channels=early_ch, num_res_blocks=std_res,
        attention_levels=std_attn, num_head_channels=32,
    ))

    configs.append(dict(
        name="5L heads=64 [64,128,256,512,512] attn-L34",
        channels=early_ch, num_res_blocks=std_res,
        attention_levels=std_attn, num_head_channels=64,
    ))

    configs.append(dict(
        name="5L heads=32 [64,128,256,512,1024] attn-L34",
        channels=prog_ch, num_res_blocks=std_res,
        attention_levels=std_attn, num_head_channels=32,
    ))

    return configs


# ── Main ─────────────────────────────────────────────────────────────────────

def run_suite(input_shape: tuple[int, ...], suite_label: str) -> list[ProfileResult]:
    """Run all configs at a given input shape."""
    print(f"\n{'#' * 76}")
    print(f"# {suite_label}")
    print(f"# Input shape: {list(input_shape)}")
    print(f"# (bravo: model sees {input_shape[1]*2}ch input, {input_shape[1]}ch output)")
    print(f"{'#' * 76}")

    configs = get_configs()
    results = []

    for i, cfg in enumerate(configs, 1):
        name = cfg.pop("name")
        label = f"[{i}/{len(configs)}] {name}"
        print(f"\n--- {label} ---")

        t0 = time.time()
        result = profile_one(label=name, input_shape=input_shape, **cfg)
        elapsed = time.time() - t0

        if result.oom:
            print(f"  OOM (peaked at {result.peak_gb:.1f} GB) [{elapsed:.0f}s]")
        else:
            print(f"  {result.params_m:.0f}M params | {result.peak_gb:.2f} GB [{elapsed:.0f}s]")

        results.append(result)

    return results


def print_summary_table(results: list[ProfileResult], title: str):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Config':<58s}  {'Params':>5s}  {'Peak VRAM':>8s}")
    print(f"  {'─' * 58}  {'─' * 5}  {'─' * 8}")
    for r in results:
        print(r.summary_line())
    print()


def main():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    n_configs = len(get_configs())

    print("=" * 80)
    print("  S2D 3D UNet VRAM Profiler")
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  Configs: {n_configs} x 2 resolutions = {n_configs * 2} runs")
    print("=" * 80)

    # exp11: s2d from 128x128x160 -> 8ch @ 64x64x80
    results_exp11 = run_suite(
        input_shape=(1, 8, 80, 64, 64),
        suite_label="exp11: s2d from 128x128x160 (encoded 64x64x80, 8ch)",
    )

    # exp11_1: s2d from 256x256x160 -> 8ch @ 128x128x80
    results_exp11_1 = run_suite(
        input_shape=(1, 8, 80, 128, 128),
        suite_label="exp11_1: s2d from 256x256x160 (encoded 128x128x80, 8ch)",
    )

    # Summary tables
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)

    print_summary_table(results_exp11, "exp11 (64x64x80) — target: ~13 GB isolated")
    print_summary_table(results_exp11_1, "exp11_1 (128x128x80) — target: fit 80 GB GPU")

    # Side-by-side scaling
    print(f"\n{'=' * 80}")
    print(f"  SCALING: exp11 -> exp11_1 (4x spatial voxels)")
    print(f"{'=' * 80}")
    print(f"  {'Config':<48s}  {'exp11':>8s}  {'exp11_1':>8s}  {'Ratio':>6s}")
    print(f"  {'─' * 48}  {'─' * 8}  {'─' * 8}  {'─' * 6}")
    for r11, r11_1 in zip(results_exp11, results_exp11_1):
        if r11.oom or r11_1.oom:
            s11 = "OOM" if r11.oom else f"{r11.peak_gb:.1f}G"
            s11_1 = "OOM" if r11_1.oom else f"{r11_1.peak_gb:.1f}G"
            print(f"  {r11.name:<48s}  {s11:>8s}  {s11_1:>8s}  {'N/A':>6s}")
        else:
            ratio = r11_1.peak_gb / r11.peak_gb if r11.peak_gb > 0 else 0
            print(f"  {r11.name:<48s}  {r11.peak_gb:>7.1f}G  {r11_1.peak_gb:>7.1f}G  {ratio:>5.2f}x")
    print()

    # Recommendations
    print(f"{'=' * 80}")
    print("  RECOMMENDATIONS")
    print(f"{'=' * 80}")
    print("  Target: exp11 ~13 GB isolated (~20 GB actual training)")
    print("  Target: exp11_1 fits 80 GB GPU (~65 GB usable after overhead)")
    print()

    viable = [
        (r11, r11_1)
        for r11, r11_1 in zip(results_exp11, results_exp11_1)
        if not r11.oom and not r11_1.oom
        and r11.peak_gb <= 20
        and r11_1.peak_gb <= 65
    ]
    if viable:
        print("  Viable configs (exp11 <= 20 GB, exp11_1 <= 65 GB):")
        for r11, r11_1 in viable:
            print(f"    {r11.name:<53s}  {r11.peak_gb:.1f} / {r11_1.peak_gb:.1f} GB")
    else:
        print("  No configs met both targets. Review results above.")

    print()


if __name__ == "__main__":
    main()
