#!/usr/bin/env python3
"""VRAM profiler for small 3D UNets — architecture sweep.

Focused on smaller architectures (10M-150M) that may generalize better
with limited data (105 training volumes). Profiles both WITH and WITHOUT
gradient checkpointing, since smaller models may not need it — allowing
torch.compile for faster training.

Explores:
  1. Reference small configs (exp20_4=67M, exp20_5=152M, exp20_6=~20M)
  2. More attention levels (affordable at small model sizes)
  3. Fewer levels / wider per level (4L, 5L alternatives)
  4. Uniform vs front-loaded res blocks
  5. Different depth/width tradeoffs at fixed param budget
  6. Tiny first channels (4ch) with attention on all levels
  7. Full-attention configs

Usage:
    python misc/profiling/profile_small_3d_unet.py
"""

import gc
import time
import traceback
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
    grad_ckpt: bool
    oom: bool = False
    error: str = ""

    def summary_line(self) -> str:
        ckpt_str = "GC" if self.grad_ckpt else "  "
        if self.error:
            return f"  {ckpt_str} {self.name:<60s}  {'ERR':>6s}  {self.error}"
        if self.oom:
            return f"  {ckpt_str} {self.name:<60s}  {'OOM':>6s}  {self.peak_gb:>7.1f}G*"
        return (
            f"  {ckpt_str} {self.name:<60s}  "
            f"{self.params_m:>6.0f}M  "
            f"{self.peak_gb:>7.2f}G"
        )


# ── Core profiler ────────────────────────────────────────────────────────────

def profile_one(
    label: str,
    group: str,
    channels: list[int],
    attention_levels: list[bool],
    num_res_blocks: list[int] | int,
    input_shape: tuple[int, ...],  # (B, C_out, D, H, W)
    num_head_channels: int = 8,
    norm_num_groups: int = 8,
    use_grad_ckpt: bool = False,
) -> ProfileResult:
    """Run a full bravo-mode training step and measure peak VRAM."""
    clear_gpu()
    torch.cuda.reset_peak_memory_stats()

    B, C_out, D, H, W = input_shape
    in_ch = C_out + 1  # bravo: cat(noisy_image, seg_mask)
    out_ch = C_out

    model = None
    optimizer = None

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

        if use_grad_ckpt:
            enable_gradient_checkpointing(model)

        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        images = torch.randn(B, C_out, D, H, W, device="cuda")
        seg_masks = torch.randn(B, 1, D, H, W, device="cuda")

        with autocast("cuda", dtype=torch.bfloat16):
            noise = torch.randn_like(images)
            t = torch.rand(B, device="cuda")
            t_exp = t.view(B, 1, 1, 1, 1)
            noisy = (1 - t_exp) * images + t_exp * noise
            model_input = torch.cat([noisy, seg_masks], dim=1)
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
            grad_ckpt=use_grad_ckpt,
        )

    except torch.cuda.OutOfMemoryError:
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        return ProfileResult(
            name=label, group=group, params_m=0,
            input_shape=list(input_shape), peak_gb=peak_gb,
            grad_ckpt=use_grad_ckpt, oom=True,
        )

    except (RuntimeError, Exception) as e:
        err_msg = str(e)
        # Some CUDA OOM errors come as RuntimeError
        if "out of memory" in err_msg.lower() or "CUDA" in err_msg:
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            return ProfileResult(
                name=label, group=group, params_m=0,
                input_shape=list(input_shape), peak_gb=peak_gb,
                grad_ckpt=use_grad_ckpt, oom=True,
            )
        # Non-OOM error (e.g., invalid config)
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        short_err = err_msg[:80]
        return ProfileResult(
            name=label, group=group, params_m=0,
            input_shape=list(input_shape), peak_gb=peak_gb,
            grad_ckpt=use_grad_ckpt, error=short_err,
        )

    finally:
        del model
        del optimizer
        clear_gpu()


# ── Architecture configs ─────────────────────────────────────────────────────

def get_configs() -> list[dict]:
    """Small architecture configs to profile.

    Spatial grids at each level for 128x128x160:
      6L: L0=128x128x160  L1=64x64x80  L2=32x32x40  L3=16x16x20  L4=8x8x10  L5=4x4x5
      5L: L0=128x128x160  L1=64x64x80  L2=32x32x40  L3=16x16x20  L4=8x8x10
      4L: L0=128x128x160  L1=64x64x80  L2=32x32x40  L3=16x16x20
    """
    configs = []

    # =================================================================
    # GROUP 1: REFERENCE — current experiments
    # =================================================================
    configs.append(dict(
        group="1_reference",
        name="REF 270M [16,32,64,256,512,512] r=1,1,1,2,2,2",
        channels=[16, 32, 64, 256, 512, 512],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=16,
        norm_num_groups=16,
    ))

    configs.append(dict(
        group="1_reference",
        name="exp20_5 152M [12,24,48,192,384,384] r=1,1,1,2,2,2",
        channels=[12, 24, 48, 192, 384, 384],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=12,
        norm_num_groups=12,
    ))

    configs.append(dict(
        group="1_reference",
        name="exp20_4 67M [8,16,32,128,256,256] r=1,1,1,2,2,2",
        channels=[8, 16, 32, 128, 256, 256],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="1_reference",
        name="exp20_6 ~20M [8,16,32,64,128,128] r=1,1,1,2,2,2",
        channels=[8, 16, 32, 64, 128, 128],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    # =================================================================
    # GROUP 2: MORE ATTENTION LEVELS (cheap at small sizes)
    # At 128x128x160: L3=16x16x20=5120, L2=32x32x40=40960, L1=64x64x80=327680
    # =================================================================
    configs.append(dict(
        group="2_more_attention",
        name="67M attn-L345 [8,16,32,128,256,256]",
        channels=[8, 16, 32, 128, 256, 256],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="2_more_attention",
        name="67M attn-L2345 [8,16,32,128,256,256]",
        channels=[8, 16, 32, 128, 256, 256],
        attention_levels=[False, False, True, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="2_more_attention",
        name="~20M attn-L345 [8,16,32,64,128,128]",
        channels=[8, 16, 32, 64, 128, 128],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="2_more_attention",
        name="~20M attn-L2345 [8,16,32,64,128,128]",
        channels=[8, 16, 32, 64, 128, 128],
        attention_levels=[False, False, True, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="2_more_attention",
        name="152M attn-L345 [12,24,48,192,384,384]",
        channels=[12, 24, 48, 192, 384, 384],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=12,
        norm_num_groups=12,
    ))

    # =================================================================
    # GROUP 3: FEWER LEVELS, WIDER (4L and 5L)
    # =================================================================

    # ~20M range, 5 levels
    configs.append(dict(
        group="3_fewer_levels",
        name="5L ~20M [8,16,64,128,128] r=1,1,2,2,2",
        channels=[8, 16, 64, 128, 128],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 2],
    ))

    # ~20M range, 4 levels
    configs.append(dict(
        group="3_fewer_levels",
        name="4L ~20M [8,32,128,256] r=1,2,2,2 attn-L23",
        channels=[8, 32, 128, 256],
        attention_levels=[False, False, True, True],
        num_res_blocks=[1, 2, 2, 2],
    ))

    # ~67M range, 5 levels
    configs.append(dict(
        group="3_fewer_levels",
        name="5L ~67M [8,32,128,256,256] r=1,1,2,2,2",
        channels=[8, 32, 128, 256, 256],
        attention_levels=[False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 2],
    ))

    # ~67M range, 4 levels
    configs.append(dict(
        group="3_fewer_levels",
        name="4L ~67M [8,64,256,384] r=1,2,2,2 attn-L23",
        channels=[8, 64, 256, 384],
        attention_levels=[False, False, True, True],
        num_res_blocks=[1, 2, 2, 2],
    ))

    # ~67M range, 5 levels, attention at L2
    configs.append(dict(
        group="3_fewer_levels",
        name="5L ~67M [8,32,128,256,256] r=1,1,2,2,2 attn-L234",
        channels=[8, 32, 128, 256, 256],
        attention_levels=[False, False, True, True, True],
        num_res_blocks=[1, 1, 2, 2, 2],
    ))

    # =================================================================
    # GROUP 4: UNIFORM RES BLOCKS
    # =================================================================
    configs.append(dict(
        group="4_uniform_res",
        name="67M uniform [8,16,32,128,256,256] r=2,2,2,2,2,2",
        channels=[8, 16, 32, 128, 256, 256],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[2, 2, 2, 2, 2, 2],
    ))

    configs.append(dict(
        group="4_uniform_res",
        name="~20M uniform [8,16,32,64,128,128] r=2,2,2,2,2,2",
        channels=[8, 16, 32, 64, 128, 128],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[2, 2, 2, 2, 2, 2],
    ))

    configs.append(dict(
        group="4_uniform_res",
        name="67M front-loaded [8,16,32,128,256,256] r=2,2,2,1,1,1",
        channels=[8, 16, 32, 128, 256, 256],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[2, 2, 2, 1, 1, 1],
    ))

    configs.append(dict(
        group="4_uniform_res",
        name="~20M deep [8,16,32,64,128,128] r=1,1,1,3,3,3",
        channels=[8, 16, 32, 64, 128, 128],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 3, 3, 3],
    ))

    # =================================================================
    # GROUP 5: WIDTH/DEPTH TRADEOFFS (~40-50M budget)
    # =================================================================
    configs.append(dict(
        group="5_mid_range",
        name="~40M [8,16,32,96,192,192] r=1,1,1,2,2,2",
        channels=[8, 16, 32, 96, 192, 192],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    configs.append(dict(
        group="5_mid_range",
        name="~40M wider-shallow [8,16,48,128,256,128] r=1,1,1,1,1,1",
        channels=[8, 16, 48, 128, 256, 128],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 1, 1, 1],
    ))

    configs.append(dict(
        group="5_mid_range",
        name="~40M deep-narrow [8,16,32,64,192,192] r=1,1,2,2,3,3",
        channels=[8, 16, 32, 64, 192, 192],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 2, 2, 3, 3],
    ))

    configs.append(dict(
        group="5_mid_range",
        name="~10M [8,16,24,48,96,96] r=1,1,1,2,2,2",
        channels=[8, 16, 24, 48, 96, 96],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
    ))

    # =================================================================
    # GROUP 6: TINY FIRST CHANNELS (4ch) — full attention possible?
    # norm_num_groups=4 to divide all channels
    # num_head_channels=4 to divide attention channels
    # =================================================================
    configs.append(dict(
        group="6_tiny_first_ch",
        name="4ch ~20M [4,8,32,64,128,128] attn-L45",
        channels=[4, 8, 32, 64, 128, 128],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=4,
        norm_num_groups=4,
    ))

    configs.append(dict(
        group="6_tiny_first_ch",
        name="4ch ~20M [4,8,32,64,128,128] attn-L345",
        channels=[4, 8, 32, 64, 128, 128],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=4,
        norm_num_groups=4,
    ))

    configs.append(dict(
        group="6_tiny_first_ch",
        name="4ch ~20M [4,8,32,64,128,128] attn-ALL",
        channels=[4, 8, 32, 64, 128, 128],
        attention_levels=[True, True, True, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=4,
        norm_num_groups=4,
    ))

    configs.append(dict(
        group="6_tiny_first_ch",
        name="4ch ~67M [4,8,32,128,256,256] attn-L45",
        channels=[4, 8, 32, 128, 256, 256],
        attention_levels=[False, False, False, False, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=4,
        norm_num_groups=4,
    ))

    configs.append(dict(
        group="6_tiny_first_ch",
        name="4ch ~67M [4,8,32,128,256,256] attn-L345",
        channels=[4, 8, 32, 128, 256, 256],
        attention_levels=[False, False, False, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=4,
        norm_num_groups=4,
    ))

    configs.append(dict(
        group="6_tiny_first_ch",
        name="4ch ~67M [4,8,32,128,256,256] attn-ALL",
        channels=[4, 8, 32, 128, 256, 256],
        attention_levels=[True, True, True, True, True, True],
        num_res_blocks=[1, 1, 1, 2, 2, 2],
        num_head_channels=4,
        norm_num_groups=4,
    ))

    configs.append(dict(
        group="6_tiny_first_ch",
        name="4ch 5L [4,16,64,128,256] attn-ALL r=1,1,2,2,2",
        channels=[4, 16, 64, 128, 256],
        attention_levels=[True, True, True, True, True],
        num_res_blocks=[1, 1, 2, 2, 2],
        num_head_channels=4,
        norm_num_groups=4,
    ))

    configs.append(dict(
        group="6_tiny_first_ch",
        name="4ch 4L [4,32,128,256] attn-ALL r=1,2,2,2",
        channels=[4, 32, 128, 256],
        attention_levels=[True, True, True, True],
        num_res_blocks=[1, 2, 2, 2],
        num_head_channels=4,
        norm_num_groups=4,
    ))

    return configs


# ── Main ─────────────────────────────────────────────────────────────────────

def run_suite(configs: list[dict], input_shape: tuple[int, ...], suite_label: str,
              use_grad_ckpt: bool = False) -> list[ProfileResult]:
    print(f"\n{'#' * 90}")
    print(f"# {suite_label}")
    print(f"# Input: {list(input_shape)} -> model sees in_ch=2, out_ch=1")
    ckpt_str = "ON" if use_grad_ckpt else "OFF"
    print(f"# Gradient checkpointing: {ckpt_str} | AMP BF16 | AdamW optimizer")
    print(f"{'#' * 90}")

    results = []
    current_group = None

    for i, cfg in enumerate(configs, 1):
        cfg = dict(cfg)  # copy
        group = cfg.pop("group")
        name = cfg.pop("name")

        if group != current_group:
            current_group = group
            print(f"\n  ── {group.upper()} ──")

        label = f"[{i}/{len(configs)}] {name}"
        print(f"\n  {label}")

        t0 = time.time()
        result = profile_one(
            label=name, group=group, input_shape=input_shape,
            use_grad_ckpt=use_grad_ckpt, **cfg,
        )
        elapsed = time.time() - t0

        if result.error:
            print(f"    ERROR: {result.error} [{elapsed:.0f}s]")
        elif result.oom:
            print(f"    OOM (peaked at {result.peak_gb:.1f} GB) [{elapsed:.0f}s]")
        else:
            print(f"    {result.params_m:.0f}M params | {result.peak_gb:.2f} GB [{elapsed:.0f}s]")

        results.append(result)

    return results


def print_summary_table(results: list[ProfileResult], title: str):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print(f"  GC {'Config':<60s}  {'Params':>6s}  {'Peak VRAM':>9s}")
    print(f"  -- {'─' * 60}  {'─' * 6}  {'─' * 9}")

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

    configs = get_configs()
    n_configs = len(configs)

    print("=" * 90)
    print("  Small 3D UNet VRAM Profiler — Architecture Sweep")
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  Configs: {n_configs} x 2 (with/without GC) = {n_configs * 2} runs")
    print(f"  Mode: bravo (in=2, out=1)")
    print("=" * 90)

    input_shape = (1, 1, 160, 128, 128)

    # Run WITHOUT gradient checkpointing (torch.compile compatible)
    results_no_gc = run_suite(
        configs, input_shape,
        "128x128x160 — NO gradient checkpointing (compile-compatible)",
        use_grad_ckpt=False,
    )

    # Run WITH gradient checkpointing for comparison
    results_gc = run_suite(
        configs, input_shape,
        "128x128x160 — WITH gradient checkpointing",
        use_grad_ckpt=True,
    )

    # ── Summary tables ──
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)

    print_summary_table(results_no_gc, "128x128x160 — NO gradient checkpointing")
    print_summary_table(results_gc, "128x128x160 — WITH gradient checkpointing")

    # ── Comparison: GC on vs off ──
    print(f"\n{'=' * 90}")
    print(f"  GRADIENT CHECKPOINTING SAVINGS")
    print(f"{'=' * 90}")
    print(f"  {'Config':<55s}  {'No GC':>7s}  {'GC':>7s}  {'Saved':>6s}  {'Params':>6s}")
    print(f"  {'─' * 55}  {'─' * 7}  {'─' * 7}  {'─' * 6}  {'─' * 6}")
    for r_no, r_gc in zip(results_no_gc, results_gc):
        if r_no.error or r_gc.error:
            continue
        if r_no.oom and r_gc.oom:
            print(f"  {r_no.name:<55s}  {'OOM':>7s}  {'OOM':>7s}  {'N/A':>6s}  {'N/A':>6s}")
        elif r_no.oom:
            print(f"  {r_no.name:<55s}  {'OOM':>7s}  {r_gc.peak_gb:>6.1f}G  {'N/A':>6s}  {r_gc.params_m:>5.0f}M")
        elif r_gc.oom:
            print(f"  {r_no.name:<55s}  {r_no.peak_gb:>6.1f}G  {'OOM':>7s}  {'N/A':>6s}  {r_no.params_m:>5.0f}M")
        else:
            saved = r_no.peak_gb - r_gc.peak_gb
            print(f"  {r_no.name:<55s}  {r_no.peak_gb:>6.1f}G  {r_gc.peak_gb:>6.1f}G  {saved:>5.1f}G  {r_no.params_m:>5.0f}M")
    print()

    # ── Compile-viable configs (no GC needed, fit in 80GB) ──
    print(f"{'=' * 90}")
    print("  COMPILE-VIABLE (no GC, peak < 70 GB, sorted by params)")
    print(f"{'=' * 90}")
    viable = [r for r in results_no_gc if not r.oom and not r.error and r.peak_gb <= 70]
    viable.sort(key=lambda r: r.params_m)
    if viable:
        print(f"  {'Config':<55s}  {'Params':>6s}  {'VRAM':>8s}  {'Free':>6s}")
        print(f"  {'─' * 55}  {'─' * 6}  {'─' * 8}  {'─' * 6}")
        for r in viable:
            free = gpu_mem - r.peak_gb
            print(f"  {r.name:<55s}  {r.params_m:>5.0f}M  {r.peak_gb:>7.1f}G  {free:>5.1f}G")
    print()


if __name__ == "__main__":
    main()
