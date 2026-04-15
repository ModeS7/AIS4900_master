#!/usr/bin/env python3
"""Memory profiler for 3D MambaDiff architectures.

Tests various dim schedules, patch sizes, and stage counts at 256x256x160.
OOM errors are caught gracefully.

Usage:
    python misc/profile_mamba_memory.py                  # Full sweep
    python misc/profile_mamba_memory.py --grad-ckpt      # With gradient checkpointing
    python misc/profile_mamba_memory.py --quick           # Subset only
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass

import torch
from torch.amp import GradScaler, autocast

sys.path.insert(0, "src")

from medgen.models.mamba_diff import MambaDiff, create_mamba_diff
from medgen.models.mamba_blocks import MAMBA_CUDA_AVAILABLE


@dataclass
class ProfileResult:
    label: str
    dims: list[int]
    depths: list[int]
    patch_size: int
    skip: int
    tokens: int
    params_m: float
    peak_vram_gb: float
    fwd_ms: float
    bwd_ms: float
    error: str | None = None


# ── Architecture configs to test ─────────────────────────────────────────────
# Each entry: (label, dims, depths, skip, patch_sizes_to_test)

CONFIGS_4STAGE = [
    # Tiny: minimal footprint
    ("4s-16to128",     [16, 32, 64, 128],       [2,2,2,2], 2, [1,2,4]),
    ("4s-24to192",     [24, 48, 96, 192],       [2,2,2,2], 2, [1,2,4]),
    # Small
    ("4s-32to256",     [32, 64, 128, 256],      [2,2,2,2], 2, [1,2,4]),
    ("4s-48to384",     [48, 96, 192, 384],      [2,2,2,2], 2, [1,2,4]),
    # Medium
    ("4s-64to512",     [64, 128, 256, 512],     [2,2,2,2], 2, [1,2,4]),
    ("4s-96to384",     [96, 192, 384, 384],     [2,2,2,2], 2, [1,2,4]),
    # Large (reference)
    ("4s-128to512",    [128, 256, 512, 512],    [2,2,2,2], 2, [2,4]),
    ("4s-256to1024",   [256, 512, 1024, 1024],  [2,2,2,2], 2, [4]),
    # Flat early (more capacity at high res, slower growth)
    ("4s-64flat",      [64, 64, 128, 256],      [2,2,2,2], 2, [1,2,4]),
    ("4s-128flat",     [128, 128, 256, 256],    [2,2,2,2], 2, [1,2,4]),
    ("4s-192flat",     [192, 192, 384, 384],    [2,2,2,2], 2, [1,2,4]),
    # Deeper blocks at bottleneck (same params, more compute deep)
    ("4s-32to256-deep",[32, 64, 128, 256],      [1,1,3,3], 2, [1,2,4]),
    ("4s-64to512-deep",[64, 128, 256, 512],     [1,1,3,3], 2, [1,2,4]),
]

CONFIGS_3STAGE = [
    ("3s-32to128",     [32, 64, 128],           [2,2,2], 1, [1,2,4]),
    ("3s-64to256",     [64, 128, 256],          [2,2,2], 1, [1,2,4]),
    ("3s-128to512",    [128, 256, 512],         [2,2,2], 1, [1,2,4]),
    ("3s-64to256-dp",  [64, 128, 256],          [3,3,3], 1, [1,2,4]),
    ("3s-128to512-dp", [128, 256, 512],         [3,3,3], 1, [2,4]),
]

CONFIGS_5STAGE = [
    ("5s-16to256",     [16, 32, 64, 128, 256],  [1,1,2,2,2], 2, [1,2,4]),
    ("5s-32to512",     [32, 64, 128, 256, 512], [2,2,2,2,2], 2, [1,2,4]),
    ("5s-32to256",     [32, 64, 128, 256, 256], [1,1,2,2,2], 2, [1,2,4]),
    ("5s-64to512",     [64, 128, 256, 512, 512],[1,1,2,2,2], 2, [2,4]),
    # Tiny 5-stage
    ("5s-8to128",      [8, 16, 32, 64, 128],    [1,1,2,2,2], 2, [1,2]),
    ("5s-16to128",     [16, 32, 64, 128, 128],  [1,1,2,2,2], 2, [1,2,4]),
]

CONFIGS_6STAGE = [
    ("6s-8to256",      [8, 16, 32, 64, 128, 256],  [1,1,1,2,2,2], 2, [1,2]),
    ("6s-16to512",     [16, 32, 64, 128, 256, 512], [1,1,1,2,2,2], 2, [1,2]),
    ("6s-16to256",     [16, 32, 64, 128, 256, 256], [1,1,1,2,2,2], 2, [1,2]),
]

QUICK_CONFIGS = [
    ("4s-32to256",     [32, 64, 128, 256],      [2,2,2,2], 2, [1,2,4]),
    ("4s-64to512",     [64, 128, 256, 512],     [2,2,2,2], 2, [1,2,4]),
    ("3s-64to256",     [64, 128, 256],          [2,2,2], 1, [1,2,4]),
    ("5s-16to256",     [16, 32, 64, 128, 256],  [1,1,2,2,2], 2, [1,2]),
    ("6s-16to256",     [16, 32, 64, 128, 256, 256], [1,1,1,2,2,2], 2, [1,2]),
]


def clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()


def compute_tokens_3d(input_size, depth_size, patch_size, num_downsamples):
    """Tokens at stage 0 after patch embed."""
    g_h = input_size // patch_size
    g_d = depth_size // patch_size
    return g_d * g_h * g_h


def is_valid(input_size, depth_size, patch_size, num_downsamples):
    """Check divisibility through all downsamples."""
    g_h = input_size // patch_size
    g_d = depth_size // patch_size
    if input_size % patch_size != 0 or depth_size % patch_size != 0:
        return False
    for _ in range(num_downsamples):
        if g_h % 2 != 0 or g_d % 2 != 0:
            return False
        g_h //= 2
        g_d //= 2
    # Also check window_size=8 divides at each level
    return True


def profile_one(
    label: str, dims: list[int], depths: list[int], skip: int,
    patch_size: int, input_size: int, depth_size: int,
    grad_ckpt: bool, in_channels: int, out_channels: int,
) -> ProfileResult:
    num_downsamples = len(depths) - skip
    tokens = compute_tokens_3d(input_size, depth_size, patch_size, num_downsamples)

    if tokens > 12_000_000:
        return ProfileResult(label=label, dims=dims, depths=depths, patch_size=patch_size,
                             skip=skip, tokens=tokens, params_m=0, peak_vram_gb=-1,
                             fwd_ms=-1, bwd_ms=-1, error=f"skip ({tokens/1e6:.1f}M tok)")

    if not is_valid(input_size, depth_size, patch_size, num_downsamples):
        return ProfileResult(label=label, dims=dims, depths=depths, patch_size=patch_size,
                             skip=skip, tokens=tokens, params_m=0, peak_vram_gb=-1,
                             fwd_ms=-1, bwd_ms=-1, error="indivisible")

    clean_gpu()
    device = torch.device("cuda")
    model = None
    optimizer = None

    try:
        # Determine num_heads from max dim
        max_dim = max(dims)
        num_heads = max(4, min(16, max_dim // 64))

        model = MambaDiff(
            spatial_dims=3, input_size=input_size, patch_size=patch_size,
            in_channels=in_channels, out_channels=out_channels,
            embed_dim=dims[0], dims=dims, depths=depths,
            bottleneck_depth=2, num_heads=num_heads, window_size=8,
            skip=skip, ssm_d_state=1, ssm_ratio=2.0, mlp_ratio=4.0,
            depth_size=depth_size,
        ).to(device)

        if grad_ckpt:
            model.enable_gradient_checkpointing()

        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler()

        x = torch.randn(1, in_channels, depth_size, input_size, input_size, device=device)
        t = torch.rand(1, device=device) * 1000

        # Warmup
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

        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        return ProfileResult(label=label, dims=dims, depths=depths, patch_size=patch_size,
                             skip=skip, tokens=tokens, params_m=params_m, peak_vram_gb=peak,
                             fwd_ms=(t1 - t0) * 1000, bwd_ms=(t2 - t1) * 1000)

    except torch.cuda.OutOfMemoryError:
        return ProfileResult(label=label, dims=dims, depths=depths, patch_size=patch_size,
                             skip=skip, tokens=tokens,
                             params_m=sum(p.numel() for p in model.parameters()) / 1e6 if model else 0,
                             peak_vram_gb=-1, fwd_ms=-1, bwd_ms=-1, error="OOM")
    except Exception as e:
        return ProfileResult(label=label, dims=dims, depths=depths, patch_size=patch_size,
                             skip=skip, tokens=tokens,
                             params_m=sum(p.numel() for p in model.parameters()) / 1e6 if model else 0,
                             peak_vram_gb=-1, fwd_ms=-1, bwd_ms=-1, error=str(e)[:50])
    finally:
        del model, optimizer
        clean_gpu()


def vram_bar(gb, max_gb=80.0, w=20):
    if gb < 0:
        return "X" * w
    f = min(int((gb / max_gb) * w), w)
    bar = "█" * f + "░" * (w - f)
    if gb > max_gb * 0.95: return f"\033[91m{bar}\033[0m"
    if gb > max_gb * 0.75: return f"\033[93m{bar}\033[0m"
    if gb > max_gb * 0.50: return f"\033[92m{bar}\033[0m"
    return bar


def print_results(results, title, gpu_gb):
    print(f"\n{'='*150}")
    print(f"  {title}")
    print(f"{'='*150}")
    hdr = (f"{'Label':>18} │ {'Dims':>24} │ {'Depths':>14} │ {'p':>2} │ "
           f"{'Tokens':>10} │ {'Params':>7} │ {'VRAM':>7} │ {'Fwd':>7} │ {'Bwd':>7} │ "
           f"{'':20} │ {'Status'}")
    print(hdr)
    print("─" * 150)

    for r in results:
        dims_s = str(r.dims)
        if len(dims_s) > 24: dims_s = dims_s[:22] + ".."
        depths_s = str(r.depths)

        if r.error:
            print(f"{r.label:>18} │ {dims_s:>24} │ {depths_s:>14} │ {r.patch_size:>2} │ "
                  f"{r.tokens:>10,} │ {r.params_m:>6.1f}M │ {'---':>7} │ {'---':>7} │ {'---':>7} │ "
                  f"{vram_bar(-1, gpu_gb)} │ \033[91m{r.error}\033[0m")
        else:
            free = gpu_gb - r.peak_vram_gb
            if free < 0: status = f"\033[91mover\033[0m"
            elif free < gpu_gb * 0.1: status = f"\033[93mtight ({free:.0f}GB)\033[0m"
            else: status = f"\033[92mOK ({free:.0f}GB free)\033[0m"
            step_ms = r.fwd_ms + r.bwd_ms
            print(f"{r.label:>18} │ {dims_s:>24} │ {depths_s:>14} │ {r.patch_size:>2} │ "
                  f"{r.tokens:>10,} │ {r.params_m:>6.1f}M │ {r.peak_vram_gb:>5.1f}GB │ "
                  f"{r.fwd_ms:>5.0f}ms │ {r.bwd_ms:>5.0f}ms │ "
                  f"{vram_bar(r.peak_vram_gb, gpu_gb)} │ {status}")


def print_summary(results, gpu_gb):
    viable = [r for r in results if r.error is None and r.peak_vram_gb <= gpu_gb]
    print(f"\n{'='*100}")
    print(f"  SUMMARY — {len(viable)} viable / {len(results)} tested on {gpu_gb:.0f}GB GPU")
    print(f"{'='*100}")

    if not viable:
        print("  No viable configs found!")
        return

    # Best by different criteria
    print(f"\n  Most tokens (highest resolution):")
    by_tokens = sorted(viable, key=lambda r: r.tokens, reverse=True)[:5]
    for r in by_tokens:
        step = r.fwd_ms + r.bwd_ms
        print(f"    {r.label:>18} p={r.patch_size} │ {r.tokens:>10,} tok │ "
              f"{r.params_m:>6.1f}M │ {r.peak_vram_gb:>5.1f}GB │ {step:>6.0f}ms/step")

    print(f"\n  Most params (highest capacity):")
    by_params = sorted(viable, key=lambda r: r.params_m, reverse=True)[:5]
    for r in by_params:
        step = r.fwd_ms + r.bwd_ms
        print(f"    {r.label:>18} p={r.patch_size} │ {r.tokens:>10,} tok │ "
              f"{r.params_m:>6.1f}M │ {r.peak_vram_gb:>5.1f}GB │ {step:>6.0f}ms/step")

    print(f"\n  Fastest (lowest step time):")
    by_speed = sorted(viable, key=lambda r: r.fwd_ms + r.bwd_ms)[:5]
    for r in by_speed:
        step = r.fwd_ms + r.bwd_ms
        print(f"    {r.label:>18} p={r.patch_size} │ {r.tokens:>10,} tok │ "
              f"{r.params_m:>6.1f}M │ {r.peak_vram_gb:>5.1f}GB │ {step:>6.0f}ms/step")

    # Compare to UNet baseline
    print(f"\n  Reference: 270M UNet baseline uses ~43GB, ~284s/epoch (500 epochs)")


def main():
    parser = argparse.ArgumentParser(description="Profile MambaDiff architectures")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--grad-ckpt", action="store_true")
    parser.add_argument("--gpu-gb", type=float, default=80.0)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--depth-size", type=int, default=160)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--out-channels", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"MAMBA_CUDA: {MAMBA_CUDA_AVAILABLE}")
    print(f"Resolution: {args.input_size}x{args.input_size}x{args.depth_size}")
    print(f"Channels: in={args.channels}, out={args.out_channels}")
    print(f"Grad checkpointing: {args.grad_ckpt}")

    if not MAMBA_CUDA_AVAILABLE:
        print("\n⚠ mamba_ssm CUDA not available. Results will be slow/inaccurate.")

    if args.quick:
        all_configs = QUICK_CONFIGS
    else:
        all_configs = CONFIGS_4STAGE + CONFIGS_3STAGE + CONFIGS_5STAGE + CONFIGS_6STAGE

    # Build test list
    tests = []
    for label, dims, depths, skip, patch_sizes in all_configs:
        for ps in patch_sizes:
            tests.append((label, dims, depths, skip, ps))

    results = []
    for idx, (label, dims, depths, skip, ps) in enumerate(tests, 1):
        num_ds = len(depths) - skip
        tokens = compute_tokens_3d(args.input_size, args.depth_size, ps, num_ds)
        tag = f"{label} p={ps}"
        print(f"\r  [{idx}/{len(tests)}] {tag:>30} ({tokens:>10,} tok)...", end="", flush=True)

        r = profile_one(
            label=label, dims=dims, depths=depths, skip=skip,
            patch_size=ps, input_size=args.input_size, depth_size=args.depth_size,
            grad_ckpt=args.grad_ckpt, in_channels=args.channels, out_channels=args.out_channels,
        )
        results.append(r)

        status = f"{r.peak_vram_gb:.1f}GB" if r.error is None else r.error
        print(f"\r  [{idx}/{len(tests)}] {tag:>30} ({tokens:>10,} tok) → {status:>20}    ")

    ckpt_label = "WITH grad ckpt" if args.grad_ckpt else "NO grad ckpt"
    print_results(results,
                  f"MambaDiff 3D @ {args.input_size}x{args.input_size}x{args.depth_size} — {ckpt_label}",
                  args.gpu_gb)
    print_summary(results, args.gpu_gb)


if __name__ == "__main__":
    main()
