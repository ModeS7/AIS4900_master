#!/usr/bin/env python3
"""Memory profiler for 3D DiT architectures.

Tests all DiT variants (S/B/L/XL) across different spatial resolutions and
patch sizes. Reports peak VRAM for training (forward + backward + optimizer)
to find the maximum configuration that fits in 80GB A100.

OOM errors are caught gracefully — failed configs are marked in the table.

Usage:
    python misc/profile_dit_memory.py                # Run all tests
    python misc/profile_dit_memory.py --quick         # Quick scan (fewer configs)
    python misc/profile_dit_memory.py --grad-ckpt     # With gradient checkpointing
    python misc/profile_dit_memory.py --channels 1    # Single channel (seg mode)
"""

import argparse
import gc
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, "src")

from medgen.models.dit import DiT, DIT_VARIANTS


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ProfileResult:
    variant: str
    input_size: int
    depth_size: int
    patch_size: int
    in_channels: int
    tokens: int
    params_m: float        # millions
    peak_vram_gb: float    # GB, -1 = OOM
    optimizer_vram_gb: float  # model + optimizer (no grad)
    error: str | None = None


# Latent space resolutions (after VAE/DC-AE compression)
LATENT_CONFIGS = [
    # (input_size, depth_size, label)
    (16, 10,  "16x16x10  (32x comp)"),
    (16, 20,  "16x16x20  (16x comp)"),
    (32, 20,  "32x32x20  (8x comp)"),
    (32, 40,  "32x32x40  (4x comp)"),
    (64, 40,  "64x64x40  (4x comp)"),
    (64, 80,  "64x64x80  (2x comp)"),
]

# Pixel space resolutions (direct, no compression)
PIXEL_CONFIGS = [
    (64,  40,  "64x64x40   (pixel)"),
    (64,  80,  "64x64x80   (pixel)"),
    (128, 80,  "128x128x80 (pixel)"),
    (128, 160, "128x128x160 (pixel)"),
    (256, 160, "256x256x160 (pixel)"),
]

# Patch sizes to test (must divide all spatial dims)
PATCH_SIZES = [2, 4, 8]

# Quick mode: fewer configs for fast scan
QUICK_LATENT = [
    (16, 10, "16x16x10  (32x comp)"),
    (32, 20, "32x32x20  (8x comp)"),
    (64, 40, "64x64x40  (4x comp)"),
]
QUICK_PIXEL = [
    (128, 160, "128x128x160 (pixel)"),
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


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_tokens(input_size: int, depth_size: int, patch_size: int) -> int:
    return (depth_size // patch_size) * (input_size // patch_size) ** 2


def is_divisible(input_size: int, depth_size: int, patch_size: int) -> bool:
    return input_size % patch_size == 0 and depth_size % patch_size == 0


def enable_gradient_checkpointing(model: DiT):
    """Enable gradient checkpointing on DiT transformer blocks."""
    from torch.utils.checkpoint import checkpoint

    for block in model.blocks:
        original_forward = block.forward

        def make_ckpt_forward(orig_fn):
            def ckpt_forward(*args, **kwargs):
                # checkpoint requires at least one tensor arg with requires_grad
                return checkpoint(orig_fn, *args, use_reentrant=False, **kwargs)
            return ckpt_forward

        block.forward = make_ckpt_forward(original_forward)


def profile_config(
    variant: str,
    input_size: int,
    depth_size: int,
    patch_size: int,
    in_channels: int,
    grad_ckpt: bool = False,
    batch_size: int = 1,
) -> ProfileResult:
    """Profile a single DiT configuration. OOM-safe."""

    tokens = compute_tokens(input_size, depth_size, patch_size)
    device = torch.device("cuda")

    # Skip obviously impossible configs (>1M tokens)
    if tokens > 1_000_000:
        return ProfileResult(
            variant=variant, input_size=input_size, depth_size=depth_size,
            patch_size=patch_size, in_channels=in_channels, tokens=tokens,
            params_m=0, peak_vram_gb=-1, optimizer_vram_gb=-1,
            error="skipped (>1M tokens)",
        )

    clean_gpu()

    model = None
    optimizer = None
    scaler = None

    try:
        # ── Phase 1: Model creation + optimizer (no gradients) ────────────
        config = DIT_VARIANTS[variant]
        model = DiT(
            spatial_dims=3,
            input_size=input_size,
            depth_size=depth_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=config['hidden_size'],
            depth=config['depth'],
            num_heads=config['num_heads'],
        ).to(device)

        if grad_ckpt:
            enable_gradient_checkpointing(model)

        params_m = count_params(model) / 1e6

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = GradScaler()

        clean_gpu()
        torch.cuda.reset_peak_memory_stats()

        # Measure model + optimizer state (baseline without activations)
        optimizer_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)

        # ── Phase 2: Forward + backward (training step) ───────────────────
        x = torch.randn(batch_size, in_channels, depth_size, input_size, input_size, device=device)
        t = torch.rand(batch_size, device=device)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x, t)
            loss = pred.float().pow(2).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        return ProfileResult(
            variant=variant, input_size=input_size, depth_size=depth_size,
            patch_size=patch_size, in_channels=in_channels, tokens=tokens,
            params_m=params_m, peak_vram_gb=peak_vram,
            optimizer_vram_gb=optimizer_vram,
        )

    except torch.cuda.OutOfMemoryError:
        return ProfileResult(
            variant=variant, input_size=input_size, depth_size=depth_size,
            patch_size=patch_size, in_channels=in_channels, tokens=tokens,
            params_m=count_params(model) / 1e6 if model is not None else 0,
            peak_vram_gb=-1, optimizer_vram_gb=-1, error="OOM",
        )

    except Exception as e:
        return ProfileResult(
            variant=variant, input_size=input_size, depth_size=depth_size,
            patch_size=patch_size, in_channels=in_channels, tokens=tokens,
            params_m=count_params(model) / 1e6 if model is not None else 0,
            peak_vram_gb=-1, optimizer_vram_gb=-1, error=str(e)[:60],
        )

    finally:
        # Aggressively free everything
        del model, optimizer, scaler
        clean_gpu()


# ── Display ───────────────────────────────────────────────────────────────────

def vram_bar(gb: float, max_gb: float = 80.0, width: int = 25) -> str:
    """Visual VRAM usage bar."""
    if gb < 0:
        return "X" * width
    filled = int((gb / max_gb) * width)
    filled = min(filled, width)
    bar = "█" * filled + "░" * (width - filled)

    # Color thresholds
    if gb > 75:
        return f"\033[91m{bar}\033[0m"  # Red: danger
    elif gb > 60:
        return f"\033[93m{bar}\033[0m"  # Yellow: tight
    elif gb > 40:
        return f"\033[92m{bar}\033[0m"  # Green: comfortable
    return bar


def print_results(results: list[ProfileResult], title: str, gpu_gb: float):
    """Print formatted results table."""
    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"  GPU: {gpu_gb:.0f} GB | batch_size=1 | bf16 training")
    print(f"{'=' * 120}")
    print(
        f"{'Variant':>7} │ {'Resolution':>15} │ {'Patch':>5} │ "
        f"{'Tokens':>8} │ {'Params':>8} │ {'Model+Opt':>9} │ "
        f"{'Peak VRAM':>10} │ {'':25} │ {'Status'}"
    )
    print(f"{'─' * 7}─┼─{'─' * 15}─┼─{'─' * 5}─┼─{'─' * 8}─┼─{'─' * 8}─┼─{'─' * 9}─┼─{'─' * 10}─┼─{'─' * 25}─┼─{'─' * 20}")

    for r in results:
        res_str = f"{r.input_size}x{r.input_size}x{r.depth_size}"

        if r.error:
            status = f"\033[91m{r.error}\033[0m"
            peak_str = "---"
            opt_str = "---"
            bar = vram_bar(-1)
        else:
            if r.peak_vram_gb > gpu_gb:
                status = f"\033[91mover {gpu_gb:.0f}GB\033[0m"
            elif r.peak_vram_gb > gpu_gb * 0.9:
                status = f"\033[93mtight ({gpu_gb - r.peak_vram_gb:.1f}GB free)\033[0m"
            else:
                status = f"\033[92mOK ({gpu_gb - r.peak_vram_gb:.1f}GB free)\033[0m"
            peak_str = f"{r.peak_vram_gb:.1f} GB"
            opt_str = f"{r.optimizer_vram_gb:.1f} GB"
            bar = vram_bar(r.peak_vram_gb, gpu_gb)

        print(
            f"{r.variant:>7} │ {res_str:>15} │ {r.patch_size:>5} │ "
            f"{r.tokens:>8,} │ {r.params_m:>7.1f}M │ {opt_str:>9} │ "
            f"{peak_str:>10} │ {bar} │ {status}"
        )


def print_summary(all_results: list[ProfileResult], gpu_gb: float):
    """Print summary of viable configurations."""
    viable = [r for r in all_results if r.error is None and r.peak_vram_gb <= gpu_gb]
    tight = [r for r in viable if r.peak_vram_gb > gpu_gb * 0.85]
    failed = [r for r in all_results if r.error is not None or (r.error is None and r.peak_vram_gb > gpu_gb)]

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total configs tested: {len(all_results)}")
    print(f"  Viable (fits {gpu_gb:.0f}GB):  {len(viable)}")
    print(f"  Tight  (>85% used):   {len(tight)}")
    print(f"  Failed (OOM/over):    {len(failed)}")

    if viable:
        # Find max-token viable configs per variant
        print(f"\n  Best viable config per variant (most tokens that fit):")
        print(f"  {'Variant':>7} │ {'Resolution':>15} │ {'Patch':>5} │ {'Tokens':>8} │ {'Peak VRAM':>10} │ {'Headroom':>8}")
        print(f"  {'─' * 7}─┼─{'─' * 15}─┼─{'─' * 5}─┼─{'─' * 8}─┼─{'─' * 10}─┼─{'─' * 8}")
        for variant in ['S', 'B', 'L', 'XL']:
            variant_viable = [r for r in viable if r.variant == variant]
            if variant_viable:
                best = max(variant_viable, key=lambda r: r.tokens)
                res_str = f"{best.input_size}x{best.input_size}x{best.depth_size}"
                headroom = gpu_gb - best.peak_vram_gb
                print(
                    f"  {variant:>7} │ {res_str:>15} │ {best.patch_size:>5} │ "
                    f"{best.tokens:>8,} │ {best.peak_vram_gb:>8.1f}GB │ {headroom:>6.1f}GB"
                )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profile 3D DiT memory usage")
    parser.add_argument("--quick", action="store_true", help="Quick scan with fewer configs")
    parser.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--channels", type=int, default=1, help="Input channels (1=seg, 3=bravo+seg, 4=latent)")
    parser.add_argument("--gpu-gb", type=float, default=80.0, help="GPU memory in GB (default: 80 for A100)")
    parser.add_argument("--pixel-only", action="store_true", help="Only test pixel-space resolutions")
    parser.add_argument("--latent-only", action="store_true", help="Only test latent-space resolutions")
    parser.add_argument("--variants", nargs="+", default=["S", "B", "L", "XL"], help="Variants to test")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for profiling")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Config: channels={args.channels}, batch_size={args.batch_size}, "
          f"grad_ckpt={args.grad_ckpt}, bf16=True")

    # Select configs
    if args.quick:
        latent_configs = QUICK_LATENT
        pixel_configs = QUICK_PIXEL
        patch_sizes = QUICK_PATCHES
    else:
        latent_configs = LATENT_CONFIGS
        pixel_configs = PIXEL_CONFIGS
        patch_sizes = PATCH_SIZES

    if args.pixel_only:
        latent_configs = []
    if args.latent_only:
        pixel_configs = []

    variants = [v for v in args.variants if v in DIT_VARIANTS]

    # ── Run latent-space tests ────────────────────────────────────────────
    if latent_configs:
        results = []
        total = sum(
            1 for v in variants
            for s, d, _ in latent_configs
            for p in patch_sizes
            if is_divisible(s, d, p)
        )
        done = 0
        for variant in variants:
            for input_size, depth_size, label in latent_configs:
                for patch_size in patch_sizes:
                    if not is_divisible(input_size, depth_size, patch_size):
                        continue
                    done += 1
                    tokens = compute_tokens(input_size, depth_size, patch_size)
                    print(f"\r  [{done}/{total}] DiT-{variant} {input_size}x{input_size}x{depth_size} "
                          f"patch={patch_size} ({tokens:,} tokens)...", end="", flush=True)
                    r = profile_config(
                        variant, input_size, depth_size, patch_size,
                        args.channels, args.grad_ckpt, args.batch_size,
                    )
                    results.append(r)
                    status = f"{r.peak_vram_gb:.1f}GB" if r.error is None else r.error
                    print(f"\r  [{done}/{total}] DiT-{variant} {input_size}x{input_size}x{depth_size} "
                          f"patch={patch_size} ({tokens:,} tokens) → {status}    ")
        print_results(results, "LATENT SPACE (after VAE/DC-AE compression)", args.gpu_gb)

    # ── Run pixel-space tests ─────────────────────────────────────────────
    if pixel_configs:
        results_pixel = []
        total = sum(
            1 for v in variants
            for s, d, _ in pixel_configs
            for p in patch_sizes
            if is_divisible(s, d, p)
        )
        done = 0
        for variant in variants:
            for input_size, depth_size, label in pixel_configs:
                for patch_size in patch_sizes:
                    if not is_divisible(input_size, depth_size, patch_size):
                        continue
                    done += 1
                    tokens = compute_tokens(input_size, depth_size, patch_size)
                    print(f"\r  [{done}/{total}] DiT-{variant} {input_size}x{input_size}x{depth_size} "
                          f"patch={patch_size} ({tokens:,} tokens)...", end="", flush=True)
                    r = profile_config(
                        variant, input_size, depth_size, patch_size,
                        args.channels, args.grad_ckpt, args.batch_size,
                    )
                    results_pixel.append(r)
                    status = f"{r.peak_vram_gb:.1f}GB" if r.error is None else r.error
                    print(f"\r  [{done}/{total}] DiT-{variant} {input_size}x{input_size}x{depth_size} "
                          f"patch={patch_size} ({tokens:,} tokens) → {status}    ")
        print_results(results_pixel, "PIXEL SPACE (no compression)", args.gpu_gb)

    # ── Summary ───────────────────────────────────────────────────────────
    all_results = []
    if latent_configs:
        all_results.extend(results)
    if pixel_configs:
        all_results.extend(results_pixel)

    print_summary(all_results, args.gpu_gb)


if __name__ == "__main__":
    main()
