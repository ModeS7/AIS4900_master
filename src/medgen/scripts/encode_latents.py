#!/usr/bin/env python3
"""
Pre-encode images to latent space using a trained VAE.

This avoids re-encoding on-the-fly during diffusion training.

Usage:
    # Encode training data
    python misc/encode_latents.py \
        --vae_checkpoint runs/vae_2d/multi_modality/exp4_256/checkpoint_best.pt \
        --data_dir /path/to/brainmetshare-3/train \
        --output_dir /path/to/brainmetshare-3-latents/train \
        --mode multi_modality

    # Encode validation data
    python misc/encode_latents.py \
        --vae_checkpoint runs/vae_2d/multi_modality/exp4_256/checkpoint_best.pt \
        --data_dir /path/to/brainmetshare-3/val \
        --output_dir /path/to/brainmetshare-3-latents/val \
        --mode multi_modality
"""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from medgen.pipeline.spaces import load_vae_for_latent_space


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-encode images to latent space")
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        required=True,
        help="Path to trained VAE checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Input data directory (e.g., brainmetshare-3/train)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for latents",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multi_modality",
        choices=["bravo", "flair", "t1_pre", "t1_gd", "dual", "multi_modality"],
        help="Which modalities to encode",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size (default: 256)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    return parser.parse_args()


def load_nifti_slice(nifti_path: str, slice_idx: int, image_size: int) -> torch.Tensor:
    """Load a single slice from a NIfTI file."""
    img = nib.load(nifti_path)
    data = img.get_fdata()

    # Get slice
    slice_data = data[:, :, slice_idx].astype(np.float32)

    # Normalize to [0, 1]
    if slice_data.max() > slice_data.min():
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())

    # Resize if needed (simple center crop/pad for now)
    h, w = slice_data.shape
    if h != image_size or w != image_size:
        # Center crop or pad
        result = np.zeros((image_size, image_size), dtype=np.float32)
        start_h = (image_size - h) // 2
        start_w = (image_size - w) // 2
        end_h = start_h + h
        end_w = start_w + w

        # Clamp to valid range
        src_start_h = max(0, -start_h)
        src_start_w = max(0, -start_w)
        src_end_h = h - max(0, end_h - image_size)
        src_end_w = w - max(0, end_w - image_size)

        dst_start_h = max(0, start_h)
        dst_start_w = max(0, start_w)
        dst_end_h = min(image_size, end_h)
        dst_end_w = min(image_size, end_w)

        result[dst_start_h:dst_end_h, dst_start_w:dst_end_w] = \
            slice_data[src_start_h:src_end_h, src_start_w:src_end_w]
        slice_data = result

    return torch.from_numpy(slice_data).unsqueeze(0)  # [1, H, W]


def get_modalities(mode: str) -> list:
    """Get list of modalities for the mode."""
    if mode == "multi_modality":
        return ["bravo", "flair", "t1_pre", "t1_gd"]
    elif mode == "dual":
        return ["t1_pre", "t1_gd"]
    else:
        return [mode]


def main():
    args = parse_args()

    device = torch.device(args.device)

    # Load VAE
    print(f"Loading VAE from: {args.vae_checkpoint}")
    space = load_vae_for_latent_space(args.vae_checkpoint, device)
    space.vae.eval()

    print(f"VAE scale factor: {space.scale_factor}x")
    print(f"Latent channels: {space.latent_channels}")

    # Get modalities
    modalities = get_modalities(args.mode)
    print(f"Modalities: {modalities}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all patients
    data_dir = Path(args.data_dir)
    patients = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(patients)} patients")

    # Process each patient
    for patient in tqdm(patients, desc="Encoding patients"):
        patient_dir = data_dir / patient
        output_patient_dir = Path(args.output_dir) / patient
        os.makedirs(output_patient_dir, exist_ok=True)

        for modality in modalities:
            nifti_path = patient_dir / f"{modality}.nii.gz"
            if not nifti_path.exists():
                print(f"  Warning: {nifti_path} not found, skipping")
                continue

            # Load NIfTI to get number of slices
            img = nib.load(str(nifti_path))
            n_slices = img.shape[2]

            # Encode each slice
            latents = []
            for slice_idx in range(n_slices):
                # Load slice
                slice_tensor = load_nifti_slice(str(nifti_path), slice_idx, args.image_size)
                slice_tensor = slice_tensor.unsqueeze(0).to(device)  # [1, 1, H, W]

                # Encode to latent
                with torch.no_grad():
                    z = space.encode(slice_tensor)  # [1, C, H/8, W/8]

                latents.append(z.cpu())

            # Stack all slices: [N_slices, C, H/8, W/8]
            latents = torch.cat(latents, dim=0)

            # Save as .pt file
            output_path = output_patient_dir / f"{modality}_latent.pt"
            torch.save(latents, output_path)

    print(f"\nDone! Latents saved to: {args.output_dir}")
    print(f"Latent shape per slice: [1, {space.latent_channels}, {args.image_size // space.scale_factor}, {args.image_size // space.scale_factor}]")


if __name__ == "__main__":
    main()
