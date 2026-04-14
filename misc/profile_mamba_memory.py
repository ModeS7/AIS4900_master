#!/usr/bin/env python3
"""Memory profiler for 3D MambaDiff (LaMamba) architectures.

Tests all MambaDiff variants (S/B/L/XL) across different spatial resolutions,
patch sizes, and depth configs. Reports peak VRAM for training (forward +
backward + optimizer) to find the maximum configuration that fits in 80GB.

OOM errors are caught gracefully — failed configs are marked in the table.

Usage:
    python misc/profile_mamba_memory.py                        # Run all tests
    python misc/profile_mamba_memory.py --quick                # Quick scan
    python misc/profile_mamba_memory.py --grad-ckpt            # With gradient checkpointing
    python misc/profile_mamba_memory.py --channels 2           # Bravo mode (noisy + seg)
    python misc/profile_mamba_memory.py --2d                   # 2D profiling only
    python misc/profile_mamba_memory.py --patch-sizes 2 4      # Specific patch sizes
    python misc/profile_mamba_memory.py --variants S B         # Specific variants
    python misc/profile_mamba_memory.py --depths 2,2,2,2       # Custom stage depths
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, "src")

from medgen.models.mamba_diff import MambaDiff, MAMBA_VARIANTS, create_mamba_diff
from medgen.models.mamba_blocks import MAMBA_CUDA_AVAILABLE


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ProfileResult:
    variant: str
    spatial_dims: int
    input_size: int
    depth_size: int | None
    patch_size: int
    in_channels: int
    tokens: int
    dims: list[int]
    params_m: float
    peak_vram_gb: float         # -1 = OOM/error
    optimizer_vram_gb: float    # model + optimizer baseline
    fwd_time_ms: float          # forward pass time
    bwd_time_ms: float          # backward pass time
    error: str | None = None


# 3D pixel-space resolutions (the main target)
PIXEL_3D_CONFIGS = [
    # (input_size, depth_size, label)
    (64,  40,  "64x64x40"),
    (64,  80,  "64x64x80"),
    (128, 80,  "128x128x80"),
    (128, 160, "128x128x160"),
    (256, 160, "256x256x160"),
]

# 2D resolutions (for verification)
PIXEL_2D_CONFIGS = [
    # (input_size, label)
    (64,  "64x64"),
    (128, "128x128"),
    (256, "256x256"),
]

PATCH_SIZES = [1, 2, 4]

# Quick mode
QUICK_3D_CONFIGS = [
    (128, 160, "128x128x160"),
    (256, 160, "256x256x160"),
]
QUICK_2D_CONFIGS = [
    (128, "128x128"),
]
QUICK_PATCHES = [2, 4]


# ── Profiling ─────────────────────────────────────────────────────────────────

def clean_gpu():
    """Aggressively free GPU memory between tests."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()


def compute_tokens_3d(input_size: int, depth_size: int, patch_size: int) -> int:
    return (depth_size // patch_size) * (input_size // patch_size) ** 2


def compute_tokens_2d(input_size: int, patch_size: int) -> int:
    return (input_size // patch_size) ** 2


def is_divisible(input_size: int, depth_size: int | None, patch_size: int, num_downsamples: int) -> bool:
    """Check if spatial dims are divisible by patch_size and allow num_downsamples halvings."""
    grid_h = input_size // patch_size
    if grid_h % (2 ** num_downsamples) != 0:
        return False
    if depth_size is not None:
        grid_d = depth_size // patch_size
        if grid_d % (2 ** num_downsamples) != 0:
            return False
    return input_size % patch_size == 0 and (depth_size is None or depth_size % patch_size == 0)


def profile_config(
    variant: str,
    spatial_dims: int,
    input_size: int,
    depth_size: int | None,
    patch_size: int,
    in_channels: int,
    out_channels: int,
    depths: list[int],
    bottleneck_depth: int,
    window_size: int,
    skip: int,
    grad_ckpt: bool = False,
    batch_size: int = 1,
) -> ProfileResult:
    """Profile a single MambaDiff configuration. OOM-safe."""

    if spatial_dims == 3:
        assert depth_size is not None
        tokens = compute_tokens_3d(input_size, depth_size, patch_size)
    else:
        tokens = compute_tokens_2d(input_size, patch_size)

    device = torch.device("cuda")
    cfg = MAMBA_VARIANTS[variant]
    embed_dim = cfg['embed_dim']
    num_stages = len(depths)
    dims = []
    for i in range(num_stages):
        scale = min(2 ** i, 2 ** (num_stages - skip))
        dims.append(embed_dim * scale)

    # Skip obviously impossible configs
    max_tokens = 2_000_000 if MAMBA_CUDA_AVAILABLE else 100_000
    if tokens > max_tokens:
        return ProfileResult(
            variant=variant, spatial_dims=spatial_dims, input_size=input_size,
            depth_size=depth_size, patch_size=patch_size, in_channels=in_channels,
            tokens=tokens, dims=dims, params_m=0, peak_vram_gb=-1,
            optimizer_vram_gb=-1, fwd_time_ms=-1, bwd_time_ms=-1,
            error=f"skipped (>{max_tokens//1000}K tokens)",
        )

    clean_gpu()

    model = None
    optimizer = None
    scaler = None

    try:
        # Phase 1: Create model + optimizer
        model = create_mamba_diff(
            variant=variant,
            spatial_dims=spatial_dims,
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            depth_size=depth_size,
            depths=depths,
            bottleneck_depth=bottleneck_depth,
            window_size=window_size,
            skip=skip,
        ).to(device)

        if grad_ckpt:
            model.enable_gradient_checkpointing()

        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler()

        clean_gpu()
        torch.cuda.reset_peak_memory_stats()
        optimizer_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)

        # Phase 2: Forward + backward
        if spatial_dims == 3:
            x = torch.randn(batch_size, in_channels, depth_size, input_size, input_size, device=device)
        else:
            x = torch.randn(batch_size, in_channels, input_size, input_size, device=device)
        t = torch.rand(batch_size, device=device) * 1000

        # Warmup (first call compiles things)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x, t)
            loss = pred.float().pow(2).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Timed run
        clean_gpu()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x, t)
            loss = pred.float().pow(2).mean()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        fwd_ms = (t1 - t0) * 1000
        bwd_ms = (t2 - t1) * 1000

        return ProfileResult(
            variant=variant, spatial_dims=spatial_dims, input_size=input_size,
            depth_size=depth_size, patch_size=patch_size, in_channels=in_channels,
            tokens=tokens, dims=dims, params_m=params_m, peak_vram_gb=peak_vram,
            optimizer_vram_gb=optimizer_vram, fwd_time_ms=fwd_ms, bwd_time_ms=bwd_ms,
        )

    except torch.cuda.OutOfMemoryError:
        return ProfileResult(
            variant=variant, spatial_dims=spatial_dims, input_size=input_size,
            depth_size=depth_size, patch_size=patch_size, in_channels=in_channels,
            tokens=tokens, dims=dims,
            params_m=sum(p.numel() for p in model.parameters()) / 1e6 if model else 0,
            peak_vram_gb=-1, optimizer_vram_gb=-1, fwd_time_ms=-1, bwd_time_ms=-1,
            error="OOM",
        )

    except Exception as e:
        return ProfileResult(
            variant=variant, spatial_dims=spatial_dims, input_size=input_size,
            depth_size=depth_size, patch_size=patch_size, in_channels=in_channels,
            tokens=tokens, dims=dims,
            params_m=sum(p.numel() for p in model.parameters()) / 1e6 if model else 0,
            peak_vram_gb=-1, optimizer_vram_gb=-1, fwd_time_ms=-1, bwd_time_ms=-1,
            error=str(e)[:60],
        )

    finally:
        del model, optimizer, scaler
        clean_gpu()


# ── Display ───────────────────────────────────────────────────────────────────

def vram_bar(gb: float, max_gb: float = 80.0, width: int = 20) -> str:
    if gb < 0:
        return "X" * width
    filled = min(int((gb / max_gb) * width), width)
    bar = "█" * filled + "░" * (width - filled)
    if gb > max_gb * 0.95:
        return f"\033[91m{bar}\033[0m"
    elif gb > max_gb * 0.75:
        return f"\033[93m{bar}\033[0m"
    elif gb > max_gb * 0.5:
        return f"\033[92m{bar}\033[0m"
    return bar


def print_results(results: list[ProfileResult], title: str, gpu_gb: float):
    print(f"\n{'=' * 140}")
    print(f"  {title}")
    print(f"  MAMBA_CUDA: {MAMBA_CUDA_AVAILABLE}")
    print(f"{'=' * 140}")
    hdr = (
        f"{'Var':>4} │ {'Resolution':>16} │ {'p':>2} │ "
        f"{'Tokens':>9} │ {'Dims':>20} │ {'Params':>7} │ "
        f"{'Mod+Opt':>7} │ {'Peak':>7} │ {'Fwd':>7} │ {'Bwd':>7} │ "
        f"{'':20} │ {'Status'}"
    )
    print(hdr)
    sep = "─" * 4 + "─┼─" + "─" * 16 + "─┼─" + "─" * 2 + "─┼─" + \
          "─" * 9 + "─┼─" + "─" * 20 + "─┼─" + "─" * 7 + "─┼─" + \
          "─" * 7 + "─┼─" + "─" * 7 + "─┼─" + "─" * 7 + "─┼─" + \
          "─" * 7 + "─┼─" + "─" * 20 + "─┼─" + "─" * 20
    print(sep)

    for r in results:
        if r.depth_size:
            res = f"{r.input_size}x{r.input_size}x{r.depth_size}"
        else:
            res = f"{r.input_size}x{r.input_size}"
        dims_str = str(r.dims) if r.dims else "---"
        if len(dims_str) > 20:
            dims_str = dims_str[:18] + ".."

        if r.error:
            status = f"\033[91m{r.error}\033[0m"
            peak = opt = fwd = bwd = "---"
            bar = vram_bar(-1, gpu_gb)
        else:
            peak = f"{r.peak_vram_gb:.1f}GB"
            opt = f"{r.optimizer_vram_gb:.1f}GB"
            fwd = f"{r.fwd_time_ms:.0f}ms"
            bwd = f"{r.bwd_time_ms:.0f}ms"
            bar = vram_bar(r.peak_vram_gb, gpu_gb)
            free = gpu_gb - r.peak_vram_gb
            if r.peak_vram_gb > gpu_gb:
                status = f"\033[91mover {gpu_gb:.0f}GB\033[0m"
            elif free < gpu_gb * 0.1:
                status = f"\033[93mtight ({free:.1f}GB free)\033[0m"
            else:
                status = f"\033[92mOK ({free:.1f}GB free)\033[0m"

        print(
            f"{r.variant:>4} │ {res:>16} │ {r.patch_size:>2} │ "
            f"{r.tokens:>9,} │ {dims_str:>20} │ {r.params_m:>6.1f}M │ "
            f"{opt:>7} │ {peak:>7} │ {fwd:>7} │ {bwd:>7} │ "
            f"{bar} │ {status}"
        )


def print_summary(all_results: list[ProfileResult], gpu_gb: float):
    viable = [r for r in all_results if r.error is None and r.peak_vram_gb <= gpu_gb]
    failed = [r for r in all_results if r.error is not None or (r.error is None and r.peak_vram_gb > gpu_gb)]

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY — MambaDiff on {gpu_gb:.0f}GB GPU")
    print(f"{'=' * 80}")
    print(f"  Total tested:  {len(all_results)}")
    print(f"  Viable:        {len(viable)}")
    print(f"  Failed/OOM:    {len(failed)}")

    if viable:
        print(f"\n  Best config per variant (most tokens that fit):")
        print(f"  {'Var':>4} │ {'Resolution':>16} │ {'p':>2} │ {'Tokens':>9} │ {'Params':>7} │ {'VRAM':>7} │ {'Fwd':>7} │ {'Bwd':>7}")
        print(f"  {'─'*4}─┼─{'─'*16}─┼─{'─'*2}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")
        for variant in ['S', 'B', 'L', 'XL']:
            v_results = [r for r in viable if r.variant == variant]
            if v_results:
                best = max(v_results, key=lambda r: r.tokens)
                if best.depth_size:
                    res = f"{best.input_size}x{best.input_size}x{best.depth_size}"
                else:
                    res = f"{best.input_size}x{best.input_size}"
                print(
                    f"  {variant:>4} │ {res:>16} │ {best.patch_size:>2} │ "
                    f"{best.tokens:>9,} │ {best.params_m:>6.1f}M │ "
                    f"{best.peak_vram_gb:>5.1f}GB │ {best.fwd_time_ms:>5.0f}ms │ {best.bwd_time_ms:>5.0f}ms"
                )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profile MambaDiff VRAM usage")
    parser.add_argument("--quick", action="store_true", help="Quick scan with fewer configs")
    parser.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--channels", type=int, default=2, help="Input channels (2=bravo+seg)")
    parser.add_argument("--out-channels", type=int, default=1, help="Output channels (1=bravo)")
    parser.add_argument("--gpu-gb", type=float, default=80.0, help="GPU memory in GB")
    parser.add_argument("--2d", action="store_true", dest="only_2d", help="Only test 2D configs")
    parser.add_argument("--3d", action="store_true", dest="only_3d", help="Only test 3D configs")
    parser.add_argument("--variants", nargs="+", default=["S", "B", "L", "XL"])
    parser.add_argument("--patch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--depths", type=str, default="2,2,2,2", help="Stage depths (comma-separated)")
    parser.add_argument("--bottleneck-depth", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    depths = [int(x) for x in args.depths.split(",")]
    num_downsamples = len(depths) - args.skip

    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"MAMBA_CUDA_AVAILABLE: {MAMBA_CUDA_AVAILABLE}")
    print(f"Config: channels={args.channels}, out={args.out_channels}, batch={args.batch_size}, "
          f"grad_ckpt={args.grad_ckpt}, bf16=True")
    print(f"Architecture: depths={depths}, bottleneck={args.bottleneck_depth}, "
          f"window={args.window_size}, skip={args.skip}")

    if not MAMBA_CUDA_AVAILABLE:
        print("\n⚠ WARNING: Using PyTorch fallback (no CUDA Mamba). Results will be SLOW and "
              "large configs will be skipped. Install mamba_ssm for accurate profiling.")

    # Select configs
    if args.quick:
        configs_3d = QUICK_3D_CONFIGS
        configs_2d = QUICK_2D_CONFIGS
        patch_sizes = args.patch_sizes or QUICK_PATCHES
    else:
        configs_3d = PIXEL_3D_CONFIGS
        configs_2d = PIXEL_2D_CONFIGS
        patch_sizes = args.patch_sizes or PATCH_SIZES

    variants = [v for v in args.variants if v in MAMBA_VARIANTS]

    all_results = []

    # ── 3D tests ──────────────────────────────────────────────────────────
    if not args.only_2d:
        results_3d = []
        configs = [
            (v, s, d, p)
            for v in variants
            for s, d, _ in configs_3d
            for p in patch_sizes
            if is_divisible(s, d, p, num_downsamples)
        ]
        for idx, (variant, input_size, depth_size, patch_size) in enumerate(configs, 1):
            tokens = compute_tokens_3d(input_size, depth_size, patch_size)
            print(f"\r  [{idx}/{len(configs)}] Mamba-{variant} "
                  f"{input_size}x{input_size}x{depth_size} p={patch_size} "
                  f"({tokens:,} tok)...", end="", flush=True)

            r = profile_config(
                variant=variant, spatial_dims=3, input_size=input_size,
                depth_size=depth_size, patch_size=patch_size,
                in_channels=args.channels, out_channels=args.out_channels,
                depths=depths, bottleneck_depth=args.bottleneck_depth,
                window_size=args.window_size, skip=args.skip,
                grad_ckpt=args.grad_ckpt, batch_size=args.batch_size,
            )
            results_3d.append(r)
            status = f"{r.peak_vram_gb:.1f}GB" if r.error is None else r.error
            print(f"\r  [{idx}/{len(configs)}] Mamba-{variant} "
                  f"{input_size}x{input_size}x{depth_size} p={patch_size} "
                  f"({tokens:,} tok) → {status}    ")

        print_results(results_3d, "3D PIXEL SPACE — MambaDiff", args.gpu_gb)
        all_results.extend(results_3d)

    # ── 2D tests ──────────────────────────────────────────────────────────
    if not args.only_3d:
        results_2d = []
        configs = [
            (v, s, p)
            for v in variants
            for s, _ in configs_2d
            for p in patch_sizes
            if is_divisible(s, None, p, num_downsamples)
        ]
        for idx, (variant, input_size, patch_size) in enumerate(configs, 1):
            tokens = compute_tokens_2d(input_size, patch_size)
            print(f"\r  [{idx}/{len(configs)}] Mamba-{variant} "
                  f"{input_size}x{input_size} p={patch_size} "
                  f"({tokens:,} tok)...", end="", flush=True)

            r = profile_config(
                variant=variant, spatial_dims=2, input_size=input_size,
                depth_size=None, patch_size=patch_size,
                in_channels=args.channels, out_channels=args.out_channels,
                depths=depths, bottleneck_depth=args.bottleneck_depth,
                window_size=args.window_size, skip=args.skip,
                grad_ckpt=args.grad_ckpt, batch_size=args.batch_size,
            )
            results_2d.append(r)
            status = f"{r.peak_vram_gb:.1f}GB" if r.error is None else r.error
            print(f"\r  [{idx}/{len(configs)}] Mamba-{variant} "
                  f"{input_size}x{input_size} p={patch_size} "
                  f"({tokens:,} tok) → {status}    ")

        print_results(results_2d, "2D PIXEL SPACE — MambaDiff", args.gpu_gb)
        all_results.extend(results_2d)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(all_results, args.gpu_gb)


if __name__ == "__main__":
    main()
