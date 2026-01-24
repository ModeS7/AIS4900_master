#!/usr/bin/env python3
"""
Comprehensive 3D UNet memory profiling.

Measures memory usage for each training phase:
1. Model loading
2. Forward pass
3. Backward pass
4. Optimizer step
5. Full training iteration

Usage:
    python scripts/profile_3d_memory.py
"""

import gc
import torch
import torch.nn.functional as F
from monai.networks.nets import DiffusionModelUNet


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024


def get_peak_memory_mb():
    """Get peak GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset_peak():
    """Reset peak memory tracker."""
    torch.cuda.reset_peak_memory_stats()


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def profile_config(channels, num_res_blocks, attention_levels, volume_size, name=""):
    """Profile memory for a specific configuration."""
    H, W, D = volume_size
    print(f"\n{'='*70}")
    print(f"Config: {name}")
    print(f"Channels: {channels}")
    print(f"Res blocks: {num_res_blocks}")
    print(f"Attention: {attention_levels}")
    print(f"Volume: {H}x{W}x{D}")
    print(f"{'='*70}")

    clear_memory()
    baseline = get_memory_mb()

    # 1. Model creation
    reset_peak()
    model = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=2,  # bravo mode: image + seg
        out_channels=1,
        channels=channels,
        attention_levels=attention_levels,
        num_res_blocks=num_res_blocks,
        num_head_channels=16,
        norm_num_groups=16,
    ).cuda()

    model_params = sum(p.numel() for p in model.parameters())
    model_memory = get_memory_mb() - baseline
    print(f"\n[1] MODEL: {model_params:,} params ({model_params/1e6:.0f}M)")
    print(f"    Memory: {model_memory:.0f} MB ({model_memory/1024:.2f} GB)")

    # 2. Create inputs
    x = torch.randn(1, 2, D, H, W, device="cuda", dtype=torch.bfloat16)
    t = torch.randint(0, 1000, (1,), device="cuda")

    # 3. Forward pass only
    model.train()
    reset_peak()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(x, t)
    forward_peak = get_peak_memory_mb() - baseline
    print("\n[2] FORWARD PASS")
    print(f"    Peak: {forward_peak:.0f} MB ({forward_peak/1024:.2f} GB)")

    # 4. Backward pass
    target = torch.randn_like(output)
    loss = F.mse_loss(output.float(), target.float())
    reset_peak()
    loss.backward()
    backward_peak = get_peak_memory_mb() - baseline
    after_backward = get_memory_mb() - baseline
    print("\n[3] BACKWARD PASS")
    print(f"    Peak: {backward_peak:.0f} MB ({backward_peak/1024:.2f} GB)")
    print(f"    After: {after_backward:.0f} MB ({after_backward/1024:.2f} GB)")

    # 5. Optimizer creation
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    after_opt_create = get_memory_mb() - baseline

    # First optimizer step (creates momentum/variance buffers)
    reset_peak()
    optimizer.step()
    optimizer.zero_grad()
    after_opt_step = get_memory_mb() - baseline
    opt_step_peak = get_peak_memory_mb() - baseline
    print("\n[4] OPTIMIZER (AdamW)")
    print(f"    After creation: {after_opt_create:.0f} MB")
    print(f"    After first step: {after_opt_step:.0f} MB ({after_opt_step/1024:.2f} GB)")
    print(f"    Optimizer states: {after_opt_step - after_backward:.0f} MB")

    # 6. Full training iteration (steady state)
    clear_memory()
    reset_peak()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(x, t)
    loss = F.mse_loss(output.float(), target.float())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    full_iter_peak = get_peak_memory_mb() - baseline
    steady_state = get_memory_mb() - baseline
    print("\n[5] FULL TRAINING ITERATION")
    print(f"    Peak: {full_iter_peak:.0f} MB ({full_iter_peak/1024:.2f} GB)")
    print(f"    Steady state: {steady_state:.0f} MB ({steady_state/1024:.2f} GB)")

    # Summary
    print(f"\n{'─'*70}")
    print(f"SUMMARY: {name} @ {H}x{W}x{D}")
    print(f"{'─'*70}")
    print(f"  Parameters:        {model_params/1e6:.0f}M")
    print(f"  Model weights:     {model_memory/1024:.2f} GB")
    print(f"  Forward peak:      {forward_peak/1024:.2f} GB")
    print(f"  Backward peak:     {backward_peak/1024:.2f} GB")
    print(f"  + Optimizer:       {after_opt_step/1024:.2f} GB")
    print("  ════════════════════════════════════")
    print(f"  FULL TRAINING:     {full_iter_peak/1024:.2f} GB")

    result = {
        "name": name,
        "params_m": model_params / 1e6,
        "model_gb": model_memory / 1024,
        "forward_gb": forward_peak / 1024,
        "backward_gb": backward_peak / 1024,
        "with_opt_gb": after_opt_step / 1024,
        "training_gb": full_iter_peak / 1024,
    }

    # Cleanup
    del model, optimizer, x, t, output, target, loss
    clear_memory()

    return result


def main():
    print("=" * 70)
    print("3D UNet Memory Profiling - Full Training Memory")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Test configurations
    base_channels = [16, 32, 64, 256, 512, 1024]
    base_attention = [False, False, False, False, True, True]

    configs = [
        ("47M fast", [16, 32, 64, 128, 256, 256], [1, 1, 1, 1, 1, 1]),
        ("655M safe", base_channels, [1, 1, 1, 2, 2, 2]),
        ("845M high", base_channels, [1, 1, 1, 2, 3, 3]),
        ("1.04B max", base_channels, [1, 1, 1, 2, 4, 4]),
    ]

    volume = (128, 128, 160)
    results = []

    for name, channels, res_blocks in configs:
        try:
            result = profile_config(
                channels=channels,
                num_res_blocks=res_blocks,
                attention_levels=base_attention,
                volume_size=volume,
                name=name,
            )
            results.append(result)
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n!!! OOM for {name}: {e}")
            clear_memory()

    # Final summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - 128x128x160 Volume")
    print("=" * 70)
    print(f"{'Config':<12} {'Params':<8} {'Model':<8} {'Fwd':<8} {'Bwd':<8} {'+Opt':<8} {'TRAIN':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<12} {r['params_m']:.0f}M     "
              f"{r['model_gb']:.2f}GB  "
              f"{r['forward_gb']:.2f}GB  "
              f"{r['backward_gb']:.2f}GB  "
              f"{r['with_opt_gb']:.2f}GB  "
              f"{r['training_gb']:.2f}GB")


if __name__ == "__main__":
    main()
