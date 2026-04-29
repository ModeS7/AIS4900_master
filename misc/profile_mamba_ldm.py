#!/usr/bin/env python3
"""Profile Mamba LDM at S/B/L/XL across multiple latent configurations.

Latent configs (channels × H × W × depth):
  WDM       :    8 × 128 × 128 × 80      (wavelet decomposition)
  VQ-VAE    :    4 ×  64 ×  64 × 40      (4× spatial, 4× depth)
  DC-AE 32× :  128 ×   8 ×   8 × 160     (deep spatial compression, full depth)
  DC-AE 64× :  512 ×   4 ×   4 × 160
  DC-AE 128×: 2048 ×   2 ×   2 × 160

For each (variant, latent) we report params, peak VRAM (forward+backward),
and per-step latency. OOM is caught and reported gracefully.

Usage:
    python misc/profile_mamba_ldm.py [--grad-ckpt] [--batch-size N]
"""
import argparse
import gc
import sys
import time
from dataclasses import dataclass

import torch
from torch.amp import autocast

sys.path.insert(0, "src")

from medgen.models.mamba_diff import MAMBA_VARIANTS, MambaDiff


@dataclass
class Result:
    variant: str
    config: str
    channels: int
    h: int
    w: int
    d: int
    patch: int
    window: int
    grid: tuple
    tokens: int
    params_m: float
    peak_vram_gb: float
    fwd_ms: float
    bwd_ms: float
    error: str | None = None


# (label, in_channels, H, W, D)
LATENT_CONFIGS = [
    ('WDM_8x128x128x80',   8, 128, 128, 80),
    ('VQVAE_4x64x64x40',   4,  64,  64, 40),
    ('DCAE_128x8x8x160', 128,   8,   8, 160),
    ('DCAE_512x4x4x160', 512,   4,   4, 160),
    ('DCAE_2048x2x2x160', 2048, 2,   2, 160),
]
VARIANTS = ['S', 'B', 'L', 'XL']


def pick_patch_window(h: int, w: int, d: int) -> tuple[int, int]:
    """Pick patch_size and window_size that work for this input."""
    smallest_spatial = min(h, w)
    # patch=2 unless smallest spatial dim is < 4 (would over-compress).
    patch = 2 if smallest_spatial >= 4 else 1
    # window must fit the smallest grid dim after patch embedding.
    grid_min = min(h // patch, w // patch, d // patch)
    window = min(8, max(1, grid_min))
    return patch, window


def profile_one(variant: str, cfg: tuple, batch_size: int, grad_ckpt: bool,
                device: torch.device) -> Result:
    label, ch, h, w, d = cfg
    patch, window = pick_patch_window(h, w, d)
    grid = (h // patch, w // patch, d // patch)
    tokens = grid[0] * grid[1] * grid[2]

    var_cfg = MAMBA_VARIANTS[variant]
    embed_dim = var_cfg['embed_dim']
    num_heads = var_cfg['num_heads']

    res = Result(
        variant=variant, config=label,
        channels=ch, h=h, w=w, d=d,
        patch=patch, window=window, grid=grid, tokens=tokens,
        params_m=0.0, peak_vram_gb=0.0, fwd_ms=0.0, bwd_ms=0.0,
    )

    try:
        model = MambaDiff(
            spatial_dims=3,
            input_size=h,
            depth_size=d,
            patch_size=patch,
            in_channels=ch,
            out_channels=ch,
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window,
        ).to(device)
        if grad_ckpt and hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
        elif grad_ckpt:
            for m in model.modules():
                if hasattr(m, '_use_checkpoint'):
                    m._use_checkpoint = True
        model.train()

        res.params_m = sum(p.numel() for p in model.parameters()) / 1e6

        # Inputs.
        x = torch.randn(batch_size, ch, d, h, w, device=device, dtype=torch.float32)
        t = torch.randint(0, 1000, (batch_size,), device=device)
        target = torch.randn_like(x)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        # Warm-up.
        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            _ = model(x, t)
        torch.cuda.synchronize()

        # Timed forward.
        t0 = time.perf_counter()
        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            out = model(x, t)
        torch.cuda.synchronize()
        res.fwd_ms = (time.perf_counter() - t0) * 1000

        # Timed backward.
        loss = ((out.float() - target) ** 2).mean()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        res.bwd_ms = (time.perf_counter() - t0) * 1000

        res.peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        del model, x, t, out, loss, target
    except torch.cuda.OutOfMemoryError as e:
        res.error = f"OOM ({str(e)[:80]})"
    except Exception as e:
        res.error = f"{type(e).__name__}: {str(e)[:120]}"

    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grad-ckpt', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda')
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    print(f"Batch size: {args.batch_size} | grad-ckpt: {args.grad_ckpt}")
    print()

    results: list[Result] = []
    for cfg in LATENT_CONFIGS:
        for variant in VARIANTS:
            label = cfg[0]
            print(f"  {variant:<3}  {label:<22} ", end='', flush=True)
            r = profile_one(variant, cfg, args.batch_size, args.grad_ckpt, device)
            results.append(r)
            if r.error:
                print(f"  ✗ {r.error}")
            else:
                print(f"  params={r.params_m:6.1f}M  vram={r.peak_vram_gb:5.2f}GB  "
                      f"fwd={r.fwd_ms:6.1f}ms  bwd={r.bwd_ms:6.1f}ms  "
                      f"tokens={r.tokens}  patch={r.patch} win={r.window}")

    # Summary table.
    print()
    print("=" * 130)
    print(f"{'Variant':<7} {'Config':<22} {'tokens':>8} {'patch':>5} {'win':>4} "
          f"{'params':>8} {'VRAM':>8} {'fwd':>8} {'bwd':>8} {'total':>8}")
    print("-" * 130)
    for r in results:
        if r.error:
            print(f"{r.variant:<7} {r.config:<22} {'-':>8} {'-':>5} {'-':>4} "
                  f"{'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>8}    {r.error}")
        else:
            total_ms = r.fwd_ms + r.bwd_ms
            print(f"{r.variant:<7} {r.config:<22} {r.tokens:>8} {r.patch:>5} {r.window:>4} "
                  f"{r.params_m:>7.1f}M {r.peak_vram_gb:>6.2f}GB {r.fwd_ms:>6.1f}ms "
                  f"{r.bwd_ms:>6.1f}ms {total_ms:>6.1f}ms")
    print("=" * 130)


if __name__ == '__main__':
    main()
