#!/usr/bin/env python3
"""VRAM profiler for pixel-space 3D UNet — architecture sweep v2.

Focused sweep around the current default_3d config [16,32,64,256,512,512]
to find better configurations that fit on 80 GB at 256x256x160.

Explores:
  1. Wider deep channels (cheap: spatial grids are tiny at deep levels)
  2. Deeper residual blocks (more capacity per level)
  3. More attention heads (num_head_channels 16 vs 32 vs 64)
  4. Attention at L3 (32x32x20 = 20K tokens — expensive but possible)
  5. 5-level alternative (fewer levels, wider per level)
  6. Combinations of the above

All configs use gradient checkpointing + AMP BF16 + AdamW optimizer
to match actual training conditions.

Usage:
    python misc/profiling/profile_pixel_3d_unet_v2.py
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
    group: str
    params_m: float
    input_shape: list[int]
    peak_gb: float
    oom: bool = False

    def summary_line(self) -> str:
        if self.oom:
            return f"  {self.name:<65s}  {'OOM':>6s}  {self.peak_gb:>7.1f}G*"
        return (
            f"  {self.name:<65s}  "
            f"{self.params_m:>6.0f}M  "
            f"{self.peak_gb:>7.2f}G"
        )


# ── Core profiler ────────────────────────────────────────────────────────────

def profile_one(
    label: str,
    group: str,
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
            name=label, group=group, params_m=n_params / 1e6,
            input_shape=list(input_shape), peak_gb=peak_gb,
        )

    except torch.cuda.OutOfMemoryError:
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        return ProfileResult(
            name=label, group=group, params_m=0,
            input_shape=list(input_shape), peak_gb=peak_gb, oom=True,
        )

    finally:
        for v in list(locals().values()):
            if isinstance(v, (torch.Tensor, torch.nn.Module, torch.optim.Optimizer)):
                del v
        clear_gpu()


# ── Architecture configs ─────────────────────────────────────────────────────

def get_configs() -> list[dict]:
    """Architecture configs to profile.

    Spatial grids at each level for 256x256x160:
      6L: L0=256x256x160  L1=128x128x80  L2=64x64x40  L3=32x32x20  L4=16x16x10  L5=8x8x5
      5L: L0=256x256x160  L1=128x128x80  L2=64x64x40  L3=32x32x20  L4=16x16x10

    Key: L0-L2 dominate VRAM. L3-L5 are cheap. Put capacity deep.
    """
    configs = []

    # =================================================================
    # GROUP 1: REFERENCE (current default_3d)
    # =================================================================
    configs.append(dict(
        group="1_reference",
        name="REF 6L [16,32,64,256,512,512] r=1,1,1,2,2,2 h=16",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=16,
    ))

    # =================================================================
    # GROUP 2: WIDER DEEP CHANNELS (L4, L5 are tiny spatial grids)
    # =================================================================
    configs.append(dict(
        group="2_wider_deep",
        name="6L [16,32,64,256,512,1024] r=1,1,1,2,2,2",
        channels=[16, 32, 64, 256, 512, 1024],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="2_wider_deep",
        name="6L [16,32,64,256,768,768] r=1,1,1,2,2,2",
        channels=[16, 32, 64, 256, 768, 768],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="2_wider_deep",
        name="6L [16,32,64,384,512,512] r=1,1,1,2,2,2",
        channels=[16, 32, 64, 384, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="2_wider_deep",
        name="6L [16,32,64,384,768,768] r=1,1,1,2,2,2",
        channels=[16, 32, 64, 384, 768, 768],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="2_wider_deep",
        name="6L [16,32,64,512,512,512] r=1,1,1,2,2,2",
        channels=[16, 32, 64, 512, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="2_wider_deep",
        name="6L [16,32,64,256,512,1024] r=1,1,1,2,2,2 h=64",
        channels=[16, 32, 64, 256, 512, 1024],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=64,
    ))

    # =================================================================
    # GROUP 3: DEEPER RESIDUAL BLOCKS
    # =================================================================
    configs.append(dict(
        group="3_deeper_res",
        name="6L [16,32,64,256,512,512] r=1,1,1,2,3,3",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 3, 3],
    ))

    configs.append(dict(
        group="3_deeper_res",
        name="6L [16,32,64,256,512,512] r=1,1,1,3,3,3",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 3, 3, 3],
    ))

    configs.append(dict(
        group="3_deeper_res",
        name="6L [16,32,64,256,512,512] r=1,1,2,2,3,3",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 3, 3],
    ))

    configs.append(dict(
        group="3_deeper_res",
        name="6L [16,32,64,256,512,512] r=2,2,2,2,2,2",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[2, 2, 2, 2, 2, 2],
    ))

    configs.append(dict(
        group="3_deeper_res",
        name="6L [16,32,64,256,512,512] r=1,1,2,3,3,3",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 3, 3, 3],
    ))

    # =================================================================
    # GROUP 4: ATTENTION HEAD VARIATIONS
    # Current: h=16. Test h=32, h=64 on the reference config.
    # =================================================================
    configs.append(dict(
        group="4_attention_heads",
        name="6L [16,32,64,256,512,512] r=1,1,1,2,2,2 h=32",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=32,
    ))

    configs.append(dict(
        group="4_attention_heads",
        name="6L [16,32,64,256,512,512] r=1,1,1,2,2,2 h=64",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=64,
    ))

    configs.append(dict(
        group="4_attention_heads",
        name="6L [16,32,64,256,512,512] r=1,1,1,2,2,2 h=128",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=128,
    ))

    # =================================================================
    # GROUP 5: L3 ATTENTION (32x32x20 = 20,480 tokens — expensive)
    # =================================================================
    configs.append(dict(
        group="5_L3_attention",
        name="6L [16,32,64,256,512,512] attn-L345 h=16",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=16,
    ))

    configs.append(dict(
        group="5_L3_attention",
        name="6L [16,32,64,256,512,512] attn-L345 h=32",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=32,
    ))

    configs.append(dict(
        group="5_L3_attention",
        name="6L [16,32,64,256,512,512] attn-L345 h=64",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=64,
    ))

    configs.append(dict(
        group="5_L3_attention",
        name="6L [16,32,64,256,512,512] attn-L345 h=128",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=128,
    ))

    # =================================================================
    # GROUP 6: 5-LEVEL UNet (fewer downsamples, wider per level)
    # 256x256x160 -> 128x128x80 -> 64x64x40 -> 32x32x20 -> 16x16x10
    # =================================================================
    configs.append(dict(
        group="6_five_level",
        name="5L [16,64,256,512,512] r=1,1,2,2,2",
        channels=[16, 64, 256, 512, 512],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="6_five_level",
        name="5L [16,64,256,512,1024] r=1,1,2,2,2",
        channels=[16, 64, 256, 512, 1024],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="6_five_level",
        name="5L [16,64,128,512,512] r=1,2,2,2,2",
        channels=[16, 64, 128, 512, 512],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 2, 2, 2, 2],
    ))

    configs.append(dict(
        group="6_five_level",
        name="5L [16,64,256,512,512] r=1,1,2,3,3",
        channels=[16, 64, 256, 512, 512],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 3, 3],
    ))

    configs.append(dict(
        group="6_five_level",
        name="5L [16,64,256,768,768] r=1,1,2,2,2",
        channels=[16, 64, 256, 768, 768],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 2],
    ))

    # =================================================================
    # GROUP 7: COMBINATIONS (best ideas together)
    # =================================================================
    configs.append(dict(
        group="7_combos",
        name="6L [16,32,64,256,512,1024] r=1,1,1,2,3,3 h=32",
        channels=[16, 32, 64, 256, 512, 1024],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 3, 3],
        num_head_channels=32,
    ))

    configs.append(dict(
        group="7_combos",
        name="6L [16,32,64,256,512,1024] r=1,1,1,3,3,3 h=32",
        channels=[16, 32, 64, 256, 512, 1024],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 3, 3, 3],
        num_head_channels=32,
    ))

    configs.append(dict(
        group="7_combos",
        name="6L [16,32,64,384,768,768] r=1,1,1,2,3,3 h=32",
        channels=[16, 32, 64, 384, 768, 768],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 3, 3],
        num_head_channels=32,
    ))

    configs.append(dict(
        group="7_combos",
        name="6L [16,32,64,256,512,1024] attn-L345 h=64",
        channels=[16, 32, 64, 256, 512, 1024],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=64,
    ))

    configs.append(dict(
        group="7_combos",
        name="6L [16,32,64,256,512,512] r=1,1,1,2,3,3 attn-L345 h=64",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 3, 3],
        num_head_channels=64,
    ))

    configs.append(dict(
        group="7_combos",
        name="6L [16,32,64,256,768,768] r=1,1,1,2,3,3 attn-L345 h=64",
        channels=[16, 32, 64, 256, 768, 768],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 3, 3],
        num_head_channels=64,
    ))

    configs.append(dict(
        group="7_combos",
        name="5L [16,64,256,512,1024] r=1,1,2,3,3 h=32",
        channels=[16, 64, 256, 512, 1024],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 3, 3],
        num_head_channels=32,
    ))

    configs.append(dict(
        group="7_combos",
        name="6L [16,32,64,256,512,1024] r=1,1,2,3,3,3 h=64",
        channels=[16, 32, 64, 256, 512, 1024],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 3, 3, 3],
        num_head_channels=64,
    ))

    return configs


# ── Main ─────────────────────────────────────────────────────────────────────

def run_suite(input_shape: tuple[int, ...], suite_label: str) -> list[ProfileResult]:
    """Run all configs at a given input shape."""
    print(f"\n{'#' * 90}")
    print(f"# {suite_label}")
    print(f"# Input: {list(input_shape)} -> model sees in_ch=2, out_ch=1")
    print(f"# Gradient checkpointing: ENABLED | AMP BF16 | AdamW optimizer")
    print(f"{'#' * 90}")

    configs = get_configs()
    results = []
    current_group = None

    for i, cfg in enumerate(configs, 1):
        group = cfg.pop("group")
        name = cfg.pop("name")

        if group != current_group:
            current_group = group
            print(f"\n  ── {group.upper()} ──")

        label = f"[{i}/{len(configs)}] {name}"
        print(f"\n  {label}")

        t0 = time.time()
        result = profile_one(label=name, group=group, input_shape=input_shape, **cfg)
        elapsed = time.time() - t0

        if result.oom:
            print(f"    OOM (peaked at {result.peak_gb:.1f} GB) [{elapsed:.0f}s]")
        else:
            print(f"    {result.params_m:.0f}M params | {result.peak_gb:.2f} GB [{elapsed:.0f}s]")

        results.append(result)

    return results


def print_summary_table(results: list[ProfileResult], title: str):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(f"  {'Config':<65s}  {'Params':>6s}  {'Peak VRAM':>9s}")
    print(f"  {'─' * 65}  {'─' * 6}  {'─' * 9}")

    current_group = None
    for r in results:
        if r.group != current_group:
            current_group = r.group
            print(f"  ── {current_group} ──")
        print(r.summary_line())
    print()


def main():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    n_configs = len(get_configs())

    print("=" * 90)
    print("  Pixel-Space 3D UNet VRAM Profiler v2 — Architecture Sweep")
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  Configs: {n_configs} x 2 resolutions = {n_configs * 2} runs")
    print(f"  Mode: bravo (in=2, out=1)")
    print(f"  Gradient checkpointing: ON | AMP BF16 | AdamW optimizer")
    print("=" * 90)

    # 128x128x160 pixel space
    results_128 = run_suite(
        input_shape=(1, 1, 160, 128, 128),
        suite_label="128x128x160 pixel space (batch=1)",
    )

    # 256x256x160 pixel space
    results_256 = run_suite(
        input_shape=(1, 1, 160, 256, 256),
        suite_label="256x256x160 pixel space (batch=1)",
    )

    # ── Summary tables ──
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)

    print_summary_table(results_128, "128x128x160")
    print_summary_table(results_256, "256x256x160")

    # ── Scaling table ──
    print(f"\n{'=' * 90}")
    print(f"  SCALING: 128x128 -> 256x256")
    print(f"{'=' * 90}")
    print(f"  {'Config':<55s}  {'128':>7s}  {'256':>7s}  {'Ratio':>6s}  {'Free':>6s}")
    print(f"  {'─' * 55}  {'─' * 7}  {'─' * 7}  {'─' * 6}  {'─' * 6}")
    for r128, r256 in zip(results_128, results_256):
        s128 = "OOM" if r128.oom else f"{r128.peak_gb:.1f}G"
        s256 = "OOM" if r256.oom else f"{r256.peak_gb:.1f}G"
        if r128.oom or r256.oom:
            print(f"  {r128.name:<55s}  {s128:>7s}  {s256:>7s}  {'N/A':>6s}  {'N/A':>6s}")
        else:
            ratio = r256.peak_gb / r128.peak_gb if r128.peak_gb > 0 else 0
            free = gpu_mem - r256.peak_gb
            print(f"  {r128.name:<55s}  {r128.peak_gb:>6.1f}G  {r256.peak_gb:>6.1f}G  {ratio:>5.2f}x  {free:>5.1f}G")
    print()

    # ── Viable configs ranked by params ──
    print(f"{'=' * 90}")
    print("  VIABLE AT 256x256x160 (peak < 70 GB, sorted by params)")
    print(f"{'=' * 90}")
    viable = [
        (r128, r256)
        for r128, r256 in zip(results_128, results_256)
        if not r256.oom and r256.peak_gb <= 70
    ]
    viable.sort(key=lambda pair: pair[1].params_m, reverse=True)
    if viable:
        print(f"  {'Config':<55s}  {'Params':>6s}  {'256 VRAM':>8s}  {'Free':>6s}")
        print(f"  {'─' * 55}  {'─' * 6}  {'─' * 8}  {'─' * 6}")
        for r128, r256 in viable:
            free = gpu_mem - r256.peak_gb
            print(f"  {r256.name:<55s}  {r256.params_m:>5.0f}M  {r256.peak_gb:>7.1f}G  {free:>5.1f}G")
    else:
        print("  No configs fit at 256x256x160 under 70 GB.")
    print()

    # ── Viable at 128x128 for reference ──
    print(f"{'=' * 90}")
    print("  VIABLE AT 128x128x160 (peak < 30 GB, sorted by params)")
    print(f"{'=' * 90}")
    viable_128 = [r for r in results_128 if not r.oom and r.peak_gb <= 30]
    viable_128.sort(key=lambda r: r.params_m, reverse=True)
    if viable_128:
        print(f"  {'Config':<55s}  {'Params':>6s}  {'128 VRAM':>8s}  {'Free':>6s}")
        print(f"  {'─' * 55}  {'─' * 6}  {'─' * 8}  {'─' * 6}")
        for r in viable_128:
            free = gpu_mem - r.peak_gb
            print(f"  {r.name:<55s}  {r.params_m:>5.0f}M  {r.peak_gb:>7.1f}G  {free:>5.1f}G")
    print()


if __name__ == "__main__":
    main()
