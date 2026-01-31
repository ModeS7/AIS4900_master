#!/usr/bin/env python3
"""Profile VRAM usage for compression models (VQ-VAE 3D, DC-AE 2D).

Measures decoder inference VRAM for latent diffusion validation.

Usage:
    python misc/profiling/profile_compression_vram.py
"""

import gc
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


def get_vram_mb() -> float:
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_peak_vram_mb() -> float:
    """Get peak VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_vram():
    """Reset VRAM tracking."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def count_parameters(model: nn.Module) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# VQ-VAE 3D Profiling
# =============================================================================

def profile_vqvae_3d(
    name: str,
    config: Dict[str, Any],
    device: torch.device,
) -> Optional[dict]:
    """Profile VQ-VAE 3D encoder/decoder VRAM."""
    from monai.networks.nets import VQVAE

    reset_vram()

    # Input shape: [B, C, D, H, W] = [1, 3, 160, 256, 256] for multi_modality
    in_channels = 3
    input_shape = (1, in_channels, 160, 256, 256)

    try:
        model = VQVAE(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=in_channels,
            channels=config['channels'],
            num_res_layers=config.get('num_res_layers', 2),
            num_res_channels=config.get('num_res_channels', config['channels']),
            downsample_parameters=config['downsample_parameters'],
            upsample_parameters=config['upsample_parameters'],
            num_embeddings=config.get('num_embeddings', 512),
            embedding_dim=config['embedding_dim'],
            commitment_cost=0.25,
            decay=0.99,
        ).to(device)

        model.eval()
        num_params = count_parameters(model)
        model_vram = get_vram_mb()

        # Create input
        x = torch.randn(*input_shape, device=device)

        # Encoder forward (MONAI VQVAE.encode returns just the quantized tensor)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            quantized = model.encode(x)
            encoder_vram = get_peak_vram_mb()
            latent_shape = quantized.shape

            # Decoder forward
            reset_vram()
            model_vram_2 = get_vram_mb()
            recon = model.decode(quantized)
            decoder_vram = get_peak_vram_mb()

        # Cleanup
        del model, x, quantized, recon
        reset_vram()

        return {
            'name': name,
            'type': 'VQ-VAE 3D',
            'params_m': num_params / 1e6,
            'model_vram_mb': model_vram,
            'encoder_vram_mb': encoder_vram,
            'decoder_vram_mb': decoder_vram,
            'latent_shape': latent_shape,
            'input_shape': input_shape,
        }

    except torch.cuda.OutOfMemoryError:
        reset_vram()
        return {'name': name, 'error': 'OOM'}
    except Exception as e:
        reset_vram()
        return {'name': name, 'error': str(e)}


# =============================================================================
# DC-AE 2D Profiling
# =============================================================================

def profile_dcae_2d(
    name: str,
    latent_channels: int,
    compression_ratio: int,
    device: torch.device,
) -> Optional[dict]:
    """Profile DC-AE 2D encoder/decoder VRAM."""
    from diffusers import AutoencoderDC

    reset_vram()

    # Input shape: [B, C, H, W] = [1, 1, 256, 256] for single slice
    in_channels = 1
    input_shape = (1, in_channels, 256, 256)

    try:
        # Create DC-AE model using diffusers
        model = AutoencoderDC(
            in_channels=in_channels,
            latent_channels=latent_channels,
            encoder_block_out_channels=[128, 256, 512, 512, 1024, 1024],
            decoder_block_out_channels=[128, 256, 512, 512, 1024, 1024],
            encoder_layers_per_block=[2, 2, 2, 3, 3, 3],
            decoder_layers_per_block=[3, 3, 3, 3, 3, 3],
            encoder_qkv_multiscales=[[], [], [], [5], [5], [5]],
            decoder_qkv_multiscales=[[], [], [], [5], [5], [5]],
            downsample_block_type='pixel_unshuffle',
            upsample_block_type='pixel_shuffle',
            encoder_out_shortcut=True,
            decoder_in_shortcut=True,
        ).to(device)

        model.eval()
        num_params = count_parameters(model)
        model_vram = get_vram_mb()

        # Create input
        x = torch.randn(*input_shape, device=device)

        # Encoder forward (diffusers AutoencoderDC returns latent directly)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            latent = model.encode(x).latent
            encoder_vram = get_peak_vram_mb()
            latent_shape = latent.shape

            # Decoder forward
            reset_vram()
            model_vram_2 = get_vram_mb()
            recon = model.decode(latent).sample
            decoder_vram = get_peak_vram_mb()

        # Cleanup
        del model, x, latent, recon
        reset_vram()

        return {
            'name': name,
            'type': 'DC-AE 2D',
            'params_m': num_params / 1e6,
            'model_vram_mb': model_vram,
            'encoder_vram_mb': encoder_vram,
            'decoder_vram_mb': decoder_vram,
            'latent_shape': latent_shape,
            'input_shape': input_shape,
        }

    except torch.cuda.OutOfMemoryError:
        reset_vram()
        return {'name': name, 'error': 'OOM'}
    except Exception as e:
        reset_vram()
        return {'name': name, 'error': str(e)}


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print("=" * 80)
    print("Compression Model VRAM Profiling")
    print("=" * 80)
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {gpu_total_mem:.1f} GB")
    print("=" * 80)

    results = []

    # =========================================================================
    # VQ-VAE 3D Experiments
    # =========================================================================
    print("\n" + "=" * 80)
    print("VQ-VAE 3D Models (input: 256x256x160)")
    print("=" * 80)

    vqvae_configs = {
        'exp8_1 (default 4x)': {
            'channels': [64, 128],
            'num_res_channels': [64, 128],
            'embedding_dim': 4,
            'num_embeddings': 512,
            'downsample_parameters': [[2, 4, 1, 1], [2, 4, 1, 1]],
            'upsample_parameters': [[2, 4, 1, 1, 0], [2, 4, 1, 1, 0]],
        },
        'exp8_14 (8x combined)': {
            'channels': [128, 256, 256],
            'num_res_channels': [128, 256, 256],
            'embedding_dim': 4,
            'num_embeddings': 1024,
            'downsample_parameters': [[2, 4, 1, 1], [2, 4, 1, 1], [2, 4, 1, 1]],
            'upsample_parameters': [[2, 4, 1, 1, 0], [2, 4, 1, 1, 0], [2, 4, 1, 1, 0]],
        },
    }

    for name, config in vqvae_configs.items():
        print(f"\nProfiling {name}...")
        result = profile_vqvae_3d(name, config, device)
        results.append(result)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Params: {result['params_m']:.1f}M")
            print(f"  Model VRAM: {result['model_vram_mb']:.0f} MB")
            print(f"  Encoder VRAM: {result['encoder_vram_mb']:.0f} MB")
            print(f"  Decoder VRAM: {result['decoder_vram_mb']:.0f} MB")
            print(f"  Latent shape: {result['latent_shape']}")

    # =========================================================================
    # DC-AE 2D Experiments
    # =========================================================================
    print("\n" + "=" * 80)
    print("DC-AE 2D Models (input: 256x256)")
    print("=" * 80)

    dcae_2d_configs = {
        'exp9_1 (f32)': {'latent_channels': 32, 'compression_ratio': 32},
        'exp9_2 (f64)': {'latent_channels': 128, 'compression_ratio': 64},
        'exp9_3 (f128)': {'latent_channels': 512, 'compression_ratio': 128},
    }

    for name, config in dcae_2d_configs.items():
        print(f"\nProfiling {name}...")
        result = profile_dcae_2d(name, config['latent_channels'], config['compression_ratio'], device)
        results.append(result)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Params: {result['params_m']:.1f}M")
            print(f"  Model VRAM: {result['model_vram_mb']:.0f} MB")
            print(f"  Encoder VRAM: {result['encoder_vram_mb']:.0f} MB")
            print(f"  Decoder VRAM: {result['decoder_vram_mb']:.0f} MB")
            print(f"  Latent shape: {result['latent_shape']}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary: Decoder VRAM for Validation")
    print("=" * 80)
    print(f"{'Model':<25} {'Params':>10} {'Decoder':>12} {'Latent Shape':<25}")
    print("-" * 80)

    for r in results:
        if 'error' in r:
            print(f"{r['name']:<25} {'ERROR':>10} {r['error']}")
        else:
            latent_str = 'x'.join(str(x) for x in r['latent_shape'][1:])
            print(f"{r['name']:<25} {r['params_m']:>9.1f}M {r['decoder_vram_mb']:>10.0f}MB  {latent_str:<25}")


if __name__ == "__main__":
    main()
