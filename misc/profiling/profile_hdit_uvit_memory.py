#!/usr/bin/env python3
"""Memory profiler for 3D HDiT and U-ViT architectures.

Tests HDiT and U-ViT variants across different patch sizes and configurations
at 256x256x160 (full pixel-space resolution). Reports peak VRAM for training
(forward + backward + optimizer) to find what fits on 80GB A100.

OOM errors are caught gracefully — failed configs are marked in the table.

Usage:
    python misc/profiling/profile_hdit_uvit_memory.py                     # All tests
    python misc/profiling/profile_hdit_uvit_memory.py --grad-ckpt         # With gradient checkpointing
    python misc/profiling/profile_hdit_uvit_memory.py --channels 1        # Single channel (seg mode)
    python misc/profiling/profile_hdit_uvit_memory.py --variants S B      # Only S and B
    python misc/profiling/profile_hdit_uvit_memory.py --patch-sizes 2 4   # Only patch 2 and 4
    python misc/profiling/profile_hdit_uvit_memory.py --arch hdit         # Only HDiT
"""

import argparse
import gc
import os
import signal
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

sys.path.insert(0, "src")

from medgen.models.dit import DIT_VARIANTS
from medgen.models.hdit import HDiT
from medgen.models.uvit import UViT, UVIT_VARIANTS


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ProfileResult:
    arch: str                   # "hdit" or "uvit"
    variant: str
    input_size: int
    depth_size: int
    patch_size: int
    in_channels: int
    label: str
    tokens_l0: int              # tokens at finest level (or total for uvit)
    tokens_bottleneck: int      # tokens at bottleneck (same as l0 for uvit)
    params_m: float
    peak_vram_gb: float         # -1 = OOM
    optimizer_vram_gb: float
    error: str | None = None


# ── Resolution ────────────────────────────────────────────────────────────────

H, W, D = 256, 256, 160


# ── HDiT configs ──────────────────────────────────────────────────────────────
# (patch_size, level_depths, label)
#
# Grid tokens at each patch size:
#   patch=1: 256*256*160 = 10,485,760  (need 3+ merge levels)
#   patch=2: 128*128*80  = 1,310,720   (need 2-3 merge levels)
#   patch=4: 64*64*40    = 163,840     (2 merge levels = sweet spot)

HDIT_CONFIGS: list[tuple[int, list[int], str]] = [
    # --- patch=1: grid 256x256x160 ---
    # 3 merge levels: L0=10.5M -> L1=1.3M -> L2=163K -> bottleneck=20K
    (1, [1, 1, 1, 4, 1, 1, 1],     "p1 [1,1,1,4,1,1,1]"),
    (1, [1, 1, 2, 4, 2, 1, 1],     "p1 [1,1,2,4,2,1,1]"),
    (1, [1, 2, 2, 4, 2, 2, 1],     "p1 [1,2,2,4,2,2,1]"),

    # --- patch=2: grid 128x128x80 ---
    # 2 merge levels: L0=1.3M -> L1=163K -> bottleneck=20K
    (2, [1, 2, 6, 2, 1],           "p2 [1,2,6,2,1]"),
    (2, [2, 4, 6, 4, 2],           "p2 [2,4,6,4,2]"),
    (2, [2, 4, 8, 4, 2],           "p2 [2,4,8,4,2]"),
    # 3 merge levels: L0=1.3M -> L1=163K -> L2=20K -> bottleneck=2.5K
    (2, [1, 1, 2, 6, 2, 1, 1],     "p2 [1,1,2,6,2,1,1]"),
    (2, [1, 2, 4, 6, 4, 2, 1],     "p2 [1,2,4,6,4,2,1]"),

    # --- patch=4: grid 64x64x40 ---
    # 2 merge levels: L0=163K -> L1=20K -> bottleneck=2.5K
    (4, [1, 2, 6, 2, 1],           "p4 [1,2,6,2,1]"),
    (4, [2, 4, 6, 4, 2],           "p4 [2,4,6,4,2]"),
    (4, [2, 6, 8, 6, 2],           "p4 [2,6,8,6,2]"),
    (4, [4, 6, 8, 6, 4],           "p4 [4,6,8,6,4]"),
    # 3 merge levels: L0=163K -> L1=20K -> L2=2.5K -> bottleneck=320
    (4, [1, 2, 4, 6, 4, 2, 1],     "p4 [1,2,4,6,4,2,1]"),
    (4, [2, 4, 4, 6, 4, 4, 2],     "p4 [2,4,4,6,4,4,2]"),
]


# ── U-ViT configs ────────────────────────────────────────────────────────────
# (patch_size, label) — depth comes from UVIT_VARIANTS

UVIT_CONFIGS: list[tuple[int, str]] = [
    (1,  "p1 (10.5M tok)"),
    (2,  "p2 (1.3M tok)"),
    (4,  "p4 (163K tok)"),
    (8,  "p8 (20K tok)"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_hdit_tokens(
    input_size: int, depth_size: int, patch_size: int, level_depths: list[int],
) -> tuple[int, int]:
    """Return (level0_tokens, bottleneck_tokens)."""
    num_down = len(level_depths) // 2
    d = depth_size // patch_size
    h = input_size // patch_size
    w = input_size // patch_size
    l0 = d * h * w
    for _ in range(num_down):
        d, h, w = d // 2, h // 2, w // 2
    return l0, d * h * w


def compute_uvit_tokens(input_size: int, depth_size: int, patch_size: int) -> int:
    return (depth_size // patch_size) * (input_size // patch_size) ** 2


def hdit_valid(
    input_size: int, depth_size: int, patch_size: int, level_depths: list[int],
) -> tuple[bool, str]:
    num_down = len(level_depths) // 2
    req = patch_size * (2 ** num_down)
    if input_size % req != 0:
        return False, f"H/W%{req}!=0"
    if depth_size % req != 0:
        return False, f"D%{req}!=0"
    return True, ""


def uvit_valid(input_size: int, depth_size: int, patch_size: int) -> tuple[bool, str]:
    if input_size % patch_size != 0:
        return False, f"H/W%{patch_size}!=0"
    if depth_size % patch_size != 0:
        return False, f"D%{patch_size}!=0"
    return True, ""


# ── Profiling (OOM-safe) ─────────────────────────────────────────────────────

def _run_training_step(model, in_channels, depth_size, input_size, batch_size, device):
    """Run one forward+backward+optimizer step. Raises on OOM."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    clean_gpu()
    torch.cuda.reset_peak_memory_stats()

    optimizer_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)

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
    return peak_vram, optimizer_vram


def profile_hdit(
    variant: str,
    patch_size: int,
    level_depths: list[int],
    label: str,
    in_channels: int,
    grad_ckpt: bool,
    batch_size: int,
) -> ProfileResult:
    tokens_l0, tokens_bn = compute_hdit_tokens(H, D, patch_size, level_depths)
    device = torch.device("cuda")

    # Skip obviously impossible (level 0 > 2M tokens without grad ckpt)
    skip_threshold = 2_000_000
    if tokens_l0 > skip_threshold:
        return ProfileResult(
            arch="hdit", variant=variant, input_size=H, depth_size=D,
            patch_size=patch_size, in_channels=in_channels, label=label,
            tokens_l0=tokens_l0, tokens_bottleneck=tokens_bn,
            params_m=0, peak_vram_gb=-1, optimizer_vram_gb=-1,
            error=f"skip (L0={tokens_l0/1e6:.1f}M tok)",
        )

    clean_gpu()
    model = None
    try:
        dit_cfg = DIT_VARIANTS[variant]
        model = HDiT(
            spatial_dims=3,
            input_size=H,
            depth_size=D,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=dit_cfg['hidden_size'],
            num_heads=dit_cfg['num_heads'],
            level_depths=level_depths,
        ).to(device)

        if grad_ckpt:
            model.enable_gradient_checkpointing()

        params_m = count_params(model) / 1e6
        peak, opt = _run_training_step(model, in_channels, D, H, batch_size, device)

        return ProfileResult(
            arch="hdit", variant=variant, input_size=H, depth_size=D,
            patch_size=patch_size, in_channels=in_channels, label=label,
            tokens_l0=tokens_l0, tokens_bottleneck=tokens_bn,
            params_m=params_m, peak_vram_gb=peak, optimizer_vram_gb=opt,
        )

    except torch.cuda.OutOfMemoryError:
        return ProfileResult(
            arch="hdit", variant=variant, input_size=H, depth_size=D,
            patch_size=patch_size, in_channels=in_channels, label=label,
            tokens_l0=tokens_l0, tokens_bottleneck=tokens_bn,
            params_m=count_params(model) / 1e6 if model else 0,
            peak_vram_gb=-1, optimizer_vram_gb=-1, error="OOM",
        )
    except Exception as e:
        return ProfileResult(
            arch="hdit", variant=variant, input_size=H, depth_size=D,
            patch_size=patch_size, in_channels=in_channels, label=label,
            tokens_l0=tokens_l0, tokens_bottleneck=tokens_bn,
            params_m=count_params(model) / 1e6 if model else 0,
            peak_vram_gb=-1, optimizer_vram_gb=-1, error=str(e)[:60],
        )
    finally:
        del model
        clean_gpu()


def profile_uvit(
    variant: str,
    patch_size: int,
    label: str,
    in_channels: int,
    grad_ckpt: bool,
    batch_size: int,
) -> ProfileResult:
    valid, reason = uvit_valid(H, D, patch_size)
    tokens = compute_uvit_tokens(H, D, patch_size) if valid else 0

    if not valid:
        return ProfileResult(
            arch="uvit", variant=variant, input_size=H, depth_size=D,
            patch_size=patch_size, in_channels=in_channels, label=label,
            tokens_l0=tokens, tokens_bottleneck=tokens,
            params_m=0, peak_vram_gb=-1, optimizer_vram_gb=-1,
            error=f"invalid: {reason}",
        )

    # Skip obviously impossible (quadratic attention over millions of tokens)
    skip_threshold = 5_000_000
    if tokens > skip_threshold:
        return ProfileResult(
            arch="uvit", variant=variant, input_size=H, depth_size=D,
            patch_size=patch_size, in_channels=in_channels, label=label,
            tokens_l0=tokens, tokens_bottleneck=tokens,
            params_m=0, peak_vram_gb=-1, optimizer_vram_gb=-1,
            error=f"skip ({tokens/1e6:.1f}M tok, flat attn)",
        )

    device = torch.device("cuda")
    clean_gpu()
    model = None
    try:
        cfg = UVIT_VARIANTS[variant]
        model = UViT(
            spatial_dims=3,
            input_size=H,
            depth_size=D,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=cfg['hidden_size'],
            depth=cfg['depth'],
            num_heads=cfg['num_heads'],
        ).to(device)

        if grad_ckpt:
            model.enable_gradient_checkpointing()

        params_m = count_params(model) / 1e6
        peak, opt = _run_training_step(model, in_channels, D, H, batch_size, device)

        return ProfileResult(
            arch="uvit", variant=variant, input_size=H, depth_size=D,
            patch_size=patch_size, in_channels=in_channels, label=label,
            tokens_l0=tokens, tokens_bottleneck=tokens,
            params_m=params_m, peak_vram_gb=peak, optimizer_vram_gb=opt,
        )

    except torch.cuda.OutOfMemoryError:
        return ProfileResult(
            arch="uvit", variant=variant, input_size=H, depth_size=D,
            patch_size=patch_size, in_channels=in_channels, label=label,
            tokens_l0=tokens, tokens_bottleneck=tokens,
            params_m=count_params(model) / 1e6 if model else 0,
            peak_vram_gb=-1, optimizer_vram_gb=-1, error="OOM",
        )
    except Exception as e:
        return ProfileResult(
            arch="uvit", variant=variant, input_size=H, depth_size=D,
            patch_size=patch_size, in_channels=in_channels, label=label,
            tokens_l0=tokens, tokens_bottleneck=tokens,
            params_m=count_params(model) / 1e6 if model else 0,
            peak_vram_gb=-1, optimizer_vram_gb=-1, error=str(e)[:60],
        )
    finally:
        del model
        clean_gpu()


# ── Display ───────────────────────────────────────────────────────────────────

def vram_bar(gb: float, max_gb: float = 80.0, width: int = 20) -> str:
    if gb < 0:
        return "X" * width
    filled = min(int((gb / max_gb) * width), width)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    if gb > 75:
        return f"\033[91m{bar}\033[0m"
    elif gb > 60:
        return f"\033[93m{bar}\033[0m"
    elif gb > 40:
        return f"\033[92m{bar}\033[0m"
    return bar


def fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    if n >= 1_000:
        return f"{n/1e3:.0f}K"
    return str(n)


def print_section(results: list[ProfileResult], title: str, gpu_gb: float):
    print(f"\n{'=' * 135}")
    print(f"  {title}")
    print(f"  Resolution: {H}x{W}x{D} | batch=1 | bf16")
    print(f"{'=' * 135}")

    hdr = (
        f"{'Arch':>5} {'Var':>4} | {'Patch':>5} | {'Config':>28} | "
        f"{'Tok L0':>8} {'Tok BN':>8} | {'Params':>8} | {'Mod+Opt':>8} | "
        f"{'Peak':>9} | {'':20} | Status"
    )
    print(hdr)
    print("-" * 135)

    for r in results:
        if r.error:
            status = f"\033[91m{r.error}\033[0m"
            peak_s = "---"
            opt_s = "---"
            bar = vram_bar(-1, gpu_gb)
        else:
            free = gpu_gb - r.peak_vram_gb
            if r.peak_vram_gb > gpu_gb:
                status = f"\033[91mover {gpu_gb:.0f}GB\033[0m"
            elif r.peak_vram_gb > gpu_gb * 0.9:
                status = f"\033[93mtight ({free:.1f}GB free)\033[0m"
            else:
                status = f"\033[92mOK ({free:.1f}GB free)\033[0m"
            peak_s = f"{r.peak_vram_gb:.1f} GB"
            opt_s = f"{r.optimizer_vram_gb:.1f} GB"
            bar = vram_bar(r.peak_vram_gb, gpu_gb)

        print(
            f"{r.arch:>5} {r.variant:>4} | {r.patch_size:>5} | {r.label:>28} | "
            f"{fmt_tok(r.tokens_l0):>8} {fmt_tok(r.tokens_bottleneck):>8} | "
            f"{r.params_m:>7.1f}M | {opt_s:>8} | "
            f"{peak_s:>9} | {bar} | {status}"
        )


def print_summary(results: list[ProfileResult], gpu_gb: float):
    viable = [r for r in results if r.error is None and r.peak_vram_gb <= gpu_gb]
    failed = len(results) - len(viable)

    print(f"\n{'=' * 100}")
    print(f"  SUMMARY — {H}x{W}x{D}, {gpu_gb:.0f}GB GPU")
    print(f"{'=' * 100}")
    print(f"  Total tested: {len(results)}  |  Viable: {len(viable)}  |  Failed/OOM: {failed}")

    if not viable:
        print("  No viable configurations found.")
        return

    print(f"\n  Best per (arch, variant) — most total depth that fits:")
    print(f"  {'Arch':>5} {'Var':>4} | {'Patch':>5} | {'Config':>28} | {'Tok L0':>8} | {'Params':>8} | {'Peak':>8} | {'Free':>6}")
    print(f"  {'─'*5} {'─'*4}─┼─{'─'*5}─┼─{'─'*28}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*6}")

    for arch in ["uvit", "hdit"]:
        for variant in (['S', 'S-Deep', 'M', 'L'] if arch == "uvit"
                        else ['S', 'B', 'L', 'XL']):
            av = [r for r in viable if r.arch == arch and r.variant == variant]
            if not av:
                continue
            best = max(av, key=lambda r: r.params_m)  # biggest model that fits
            free = gpu_gb - best.peak_vram_gb
            print(
                f"  {arch:>5} {variant:>4} | {best.patch_size:>5} | "
                f"{best.label:>28} | {fmt_tok(best.tokens_l0):>8} | "
                f"{best.params_m:>7.1f}M | {best.peak_vram_gb:>6.1f}GB | {free:>4.1f}GB"
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profile 3D HDiT/U-ViT memory at 256x256x160")
    parser.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--channels", type=int, default=2, help="Input channels (2=bravo, 1=seg)")
    parser.add_argument("--gpu-gb", type=float, default=80.0, help="GPU memory in GB")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--arch", choices=["hdit", "uvit", "both"], default="both",
                        help="Which architecture(s) to profile")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Variants to test (default: all for each arch)")
    parser.add_argument("--patch-sizes", nargs="+", type=int, default=None,
                        help="Only test these patch sizes")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Config: channels={args.channels}, batch={args.batch_size}, "
          f"grad_ckpt={args.grad_ckpt}, bf16=True")
    print(f"Resolution: {H}x{W}x{D}")

    all_results: list[ProfileResult] = []

    # ── U-ViT ─────────────────────────────────────────────────────────────
    if args.arch in ("uvit", "both"):
        uvit_variants = args.variants or list(UVIT_VARIANTS.keys())
        uvit_variants = [v for v in uvit_variants if v in UVIT_VARIANTS]
        uvit_cfgs = UVIT_CONFIGS
        if args.patch_sizes:
            uvit_cfgs = [(p, l) for p, l in uvit_cfgs if p in args.patch_sizes]

        for variant in uvit_variants:
            cfg = UVIT_VARIANTS[variant]
            print(f"\n{'─' * 80}")
            print(f"  U-ViT-{variant} (hidden={cfg['hidden_size']}, depth={cfg['depth']}, heads={cfg['num_heads']})")
            print(f"{'─' * 80}")

            results = []
            for i, (ps, label) in enumerate(uvit_cfgs):
                tokens = compute_uvit_tokens(H, D, ps) if H % ps == 0 and D % ps == 0 else 0
                tok_str = fmt_tok(tokens) if tokens else "?"
                print(f"  [{i+1}/{len(uvit_cfgs)}] U-ViT-{variant} {label} ({tok_str} tokens)...",
                      end="", flush=True)

                r = profile_uvit(variant, ps, label, args.channels, args.grad_ckpt, args.batch_size)
                results.append(r)

                status = f"{r.peak_vram_gb:.1f}GB" if r.error is None else r.error
                print(f" -> {status}")

            all_results.extend(results)
            print_section(results, f"U-ViT-{variant}", args.gpu_gb)

    # ── HDiT ──────────────────────────────────────────────────────────────
    if args.arch in ("hdit", "both"):
        hdit_variants = args.variants or list(DIT_VARIANTS.keys())
        hdit_variants = [v for v in hdit_variants if v in DIT_VARIANTS]
        hdit_cfgs = HDIT_CONFIGS
        if args.patch_sizes:
            hdit_cfgs = [(p, ld, l) for p, ld, l in hdit_cfgs if p in args.patch_sizes]

        for variant in hdit_variants:
            dit_cfg = DIT_VARIANTS[variant]
            print(f"\n{'─' * 80}")
            print(f"  HDiT-{variant} (hidden={dit_cfg['hidden_size']}, heads={dit_cfg['num_heads']})")
            print(f"{'─' * 80}")

            results = []
            for i, (ps, ld, label) in enumerate(hdit_cfgs):
                valid, reason = hdit_valid(H, D, ps, ld)
                if not valid:
                    print(f"  [{i+1}/{len(hdit_cfgs)}] HDiT-{variant} {label} -> invalid: {reason}")
                    results.append(ProfileResult(
                        arch="hdit", variant=variant, input_size=H, depth_size=D,
                        patch_size=ps, in_channels=args.channels, label=label,
                        tokens_l0=0, tokens_bottleneck=0,
                        params_m=0, peak_vram_gb=-1, optimizer_vram_gb=-1,
                        error=f"invalid: {reason}",
                    ))
                    continue

                tok_l0, tok_bn = compute_hdit_tokens(H, D, ps, ld)
                print(f"  [{i+1}/{len(hdit_cfgs)}] HDiT-{variant} {label} "
                      f"(L0={fmt_tok(tok_l0)}, BN={fmt_tok(tok_bn)})...",
                      end="", flush=True)

                r = profile_hdit(
                    variant, ps, ld, label, args.channels, args.grad_ckpt, args.batch_size,
                )
                results.append(r)

                status = f"{r.peak_vram_gb:.1f}GB" if r.error is None else r.error
                print(f" -> {status}")

            all_results.extend(results)
            print_section(results, f"HDiT-{variant}", args.gpu_gb)

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(all_results, args.gpu_gb)


if __name__ == "__main__":
    main()
