#!/usr/bin/env python3
"""VRAM profiler for pixel-space 3D diffusion UNet architectures.

Profiles UNet configs with gradient checkpointing enabled (matching actual
training conditions) for bravo-mode pixel-space generation.

Tests architectures starting at 32 first-level channels to find what fits
on 80 GB A100 at 256x256x160 and 128x128x160.

Usage:
    python misc/profiling/profile_pixel_3d_unet.py
"""

import gc
import time
from dataclasses import dataclass

import torch
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from monai.networks.nets import DiffusionModelUNet


# ── Helpers ──────────────────────────────────────────────────────────────────

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def enable_gradient_checkpointing(model: DiffusionModelUNet) -> int:
    """Patch block-level forward methods with gradient checkpointing."""
    def _make_ckpt_forward(original_forward):
        def ckpt_forward(*args, **kwargs):
            return grad_checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
        return ckpt_forward

    count = 0
    for block in model.down_blocks:
        block.forward = _make_ckpt_forward(block.forward)
        count += 1
    model.middle_block.forward = _make_ckpt_forward(model.middle_block.forward)
    count += 1
    for block in model.up_blocks:
        block.forward = _make_ckpt_forward(block.forward)
        count += 1
    return count


@dataclass
class ProfileResult:
    name: str
    params_m: float
    input_shape: list[int]
    peak_gb: float
    oom: bool = False

    def summary_line(self) -> str:
        if self.oom:
            return f"  {self.name:<60s}  {'OOM':>6s}  {self.peak_gb:>7.1f}G*"
        return (
            f"  {self.name:<60s}  "
            f"{self.params_m:>5.0f}M  "
            f"{self.peak_gb:>7.2f}G"
        )


# ── Core profiler ────────────────────────────────────────────────────────────

def profile_one(
    label: str,
    channels: list[int],
    attention_levels: list[bool],
    num_res_blocks: list[int],
    input_shape: tuple[int, ...],  # (B, C_out, D, H, W)
    num_head_channels: int = 32,
    norm_num_groups: int = 16,
) -> ProfileResult:
    """Run a full bravo-mode pixel-space training step and measure peak VRAM.

    Bravo mode: in_channels=2 (noisy_bravo + seg_mask), out_channels=1.
    Gradient checkpointing is enabled to match actual training.
    """
    clear_gpu()
    torch.cuda.reset_peak_memory_stats()

    B, C_out, D, H, W = input_shape
    in_ch = C_out + 1  # bravo: cat(noisy_image, seg_mask)
    out_ch = C_out

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

        enable_gradient_checkpointing(model)

        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Simulate bravo training: image + seg conditioning
        images = torch.randn(B, C_out, D, H, W, device="cuda")
        seg_masks = torch.randn(B, 1, D, H, W, device="cuda")

        with autocast("cuda", dtype=torch.bfloat16):
            noise = torch.randn_like(images)
            t = torch.rand(B, device="cuda")
            t_exp = t.view(B, 1, 1, 1, 1)
            noisy = (1 - t_exp) * images + t_exp * noise
            model_input = torch.cat([noisy, seg_masks], dim=1)  # [B, 2, D, H, W]
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
    """Architecture configs to profile — all start with 32 first-level channels."""
    configs = []

    # =================================================================
    # REFERENCE: Current config (16 first channel)
    # =================================================================
    configs.append(dict(
        name="REF 6L [16,32,64,256,512,512] r=1,1,1,2,2,2 h=16",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=16,
    ))

    # =================================================================
    # 6-LEVEL: 32 first channel — direct upgrades of current config
    # =================================================================

    configs.append(dict(
        name="6L [32,64,128,256,512,512] r=1,1,1,2,2,2",
        channels=[32, 64, 128, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        name="6L [32,64,128,256,512,512] r=1,1,2,2,2,2",
        channels=[32, 64, 128, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 2, 2],
    ))

    configs.append(dict(
        name="6L [32,64,128,256,512,512] attn-L345",
        channels=[32, 64, 128, 256, 512, 512],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    # Narrower at deep levels to save memory
    configs.append(dict(
        name="6L [32,64,128,256,256,512] r=1,1,1,2,2,2",
        channels=[32, 64, 128, 256, 256, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        name="6L [32,64,128,256,256,256] r=1,1,1,2,2,2",
        channels=[32, 64, 128, 256, 256, 256],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    # Fewer deep channels but more res blocks
    configs.append(dict(
        name="6L [32,64,128,256,256,512] r=1,1,2,2,3,3",
        channels=[32, 64, 128, 256, 256, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 3, 3],
    ))

    configs.append(dict(
        name="6L [32,64,128,256,512,512] r=2,2,2,2,2,2",
        channels=[32, 64, 128, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[2, 2, 2, 2, 2, 2],
    ))

    # =================================================================
    # 5-LEVEL: 32 first channel — fewer levels, more capacity per level
    # =================================================================

    configs.append(dict(
        name="5L [32,64,256,512,512] r=1,1,2,2,2",
        channels=[32, 64, 256, 512, 512],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        name="5L [32,64,128,256,512] r=2,2,2,2,2",
        channels=[32, 64, 128, 256, 512],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[2, 2, 2, 2, 2],
    ))

    configs.append(dict(
        name="5L [32,64,128,512,512] r=1,1,2,2,2",
        channels=[32, 64, 128, 512, 512],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        name="5L [32,64,256,512,1024] r=1,1,2,2,2",
        channels=[32, 64, 256, 512, 1024],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        name="5L [32,128,256,512,512] r=1,2,2,2,2",
        channels=[32, 128, 256, 512, 512],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 2, 2, 2, 2],
    ))

    configs.append(dict(
        name="5L [32,128,256,512,512] r=2,2,2,3,3",
        channels=[32, 128, 256, 512, 512],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[2, 2, 2, 3, 3],
    ))

    # =================================================================
    # ATTENTION VARIATIONS on best 6L candidate
    # =================================================================

    configs.append(dict(
        name="6L [32,64,128,256,512,512] attn-L45 h=16",
        channels=[32, 64, 128, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=16,
    ))

    configs.append(dict(
        name="6L [32,64,128,256,512,512] attn-L45 h=64",
        channels=[32, 64, 128, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=64,
    ))

    configs.append(dict(
        name="6L [32,64,128,256,512,512] attn-L345 h=64",
        channels=[32, 64, 128, 256, 512, 512],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=64,
    ))

    # =================================================================
    # NORM_NUM_GROUPS=32: channels must all be divisible by 32
    # =================================================================

    configs.append(dict(
        name="6L [32,64,128,256,512,512] r=1,1,1,2,2,2 ng=32",
        channels=[32, 64, 128, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        norm_num_groups=32,
    ))

    return configs


# ── Main ─────────────────────────────────────────────────────────────────────

def run_suite(input_shape: tuple[int, ...], suite_label: str) -> list[ProfileResult]:
    """Run all configs at a given input shape."""
    print(f"\n{'#' * 84}")
    print(f"# {suite_label}")
    print(f"# Input: {list(input_shape)} -> model sees in_ch=2, out_ch=1")
    print(f"# Gradient checkpointing: ENABLED")
    print(f"{'#' * 84}")

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
    print(f"\n{'=' * 84}")
    print(f"  {title}")
    print(f"{'=' * 84}")
    print(f"  {'Config':<60s}  {'Params':>5s}  {'Peak VRAM':>8s}")
    print(f"  {'─' * 60}  {'─' * 5}  {'─' * 8}")
    for r in results:
        print(r.summary_line())
    print()


def main():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    n_configs = len(get_configs())

    print("=" * 84)
    print("  Pixel-Space 3D UNet VRAM Profiler")
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  Configs: {n_configs} x 2 resolutions = {n_configs * 2} runs")
    print(f"  Mode: bravo (in=2, out=1), gradient checkpointing ON")
    print("=" * 84)

    # 128x128x160 pixel space (original exp1 resolution)
    results_128 = run_suite(
        input_shape=(1, 1, 160, 128, 128),
        suite_label="128x128x160 pixel space (batch=1)",
    )

    # 256x256x160 pixel space (exp1_1 resolution)
    results_256 = run_suite(
        input_shape=(1, 1, 160, 256, 256),
        suite_label="256x256x160 pixel space (batch=1)",
    )

    # Summary tables
    print("\n" + "=" * 84)
    print("  FINAL SUMMARY")
    print("=" * 84)

    print_summary_table(results_128, "128x128x160 — target: <25 GB (headroom for metrics)")
    print_summary_table(results_256, "256x256x160 — target: <65 GB (fit on 80 GB)")

    # Side-by-side scaling
    print(f"\n{'=' * 84}")
    print(f"  SCALING: 128x128 -> 256x256 (4x spatial voxels)")
    print(f"{'=' * 84}")
    print(f"  {'Config':<52s}  {'128x128':>8s}  {'256x256':>8s}  {'Ratio':>6s}")
    print(f"  {'─' * 52}  {'─' * 8}  {'─' * 8}  {'─' * 6}")
    for r128, r256 in zip(results_128, results_256):
        if r128.oom or r256.oom:
            s128 = "OOM" if r128.oom else f"{r128.peak_gb:.1f}G"
            s256 = "OOM" if r256.oom else f"{r256.peak_gb:.1f}G"
            print(f"  {r128.name:<52s}  {s128:>8s}  {s256:>8s}  {'N/A':>6s}")
        else:
            ratio = r256.peak_gb / r128.peak_gb if r128.peak_gb > 0 else 0
            print(f"  {r128.name:<52s}  {r128.peak_gb:>7.1f}G  {r256.peak_gb:>7.1f}G  {ratio:>5.2f}x")
    print()

    # Viable configs
    print(f"{'=' * 84}")
    print("  VIABLE CONFIGS (fit 80 GB at 256x256x160)")
    print(f"{'=' * 84}")
    viable = [
        (r128, r256)
        for r128, r256 in zip(results_128, results_256)
        if not r256.oom and r256.peak_gb <= 70
    ]
    if viable:
        for r128, r256 in viable:
            s128 = "OOM" if r128.oom else f"{r128.peak_gb:.1f}G"
            print(f"  {r256.name:<52s}  128: {s128:>6s}  256: {r256.peak_gb:.1f}G  params: {r256.params_m:.0f}M")
    else:
        print("  No configs fit at 256x256x160. Consider 128x128x160 only.")
        viable_128 = [r for r in results_128 if not r.oom and r.peak_gb <= 25]
        if viable_128:
            print("\n  Viable at 128x128x160 (<25 GB):")
            for r in viable_128:
                print(f"    {r.name:<56s}  {r.peak_gb:.1f}G  {r.params_m:.0f}M")

    print()


if __name__ == "__main__":
    main()
