#!/usr/bin/env python3
"""
Single image diffusion visualization script.

Saves 11 individual images showing noise progression at different timesteps.
Includes comprehensive noise metrics saved to CSV.
"""
import os
import sys
from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.networks.schedulers import DDPMScheduler, RFlowScheduler
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    ScaleIntensity,
    SpatialCrop,
    ToTensor,
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import PathConfig

# CONFIGURATION - Using PathConfig for data path
path_config = PathConfig()
DATA_DIR = str(path_config.brainmet_train_dir)
MODE = "bravo"  # Change to "bravo" or "seg"
PATIENT_INDEX = 13  # Which patient folder to use
SLICE_INDEX = 76  # Which slice from the 3D volume
DEVICE = torch.device("cpu")  # Use CPU - much simpler for this task!
SCHEDULER_TYPE = "ddpm"  # "ddpm", "rflow", or "both"


def calculate_noise_metrics(
    original_img: Union[torch.Tensor, np.ndarray],
    noisy_img: Union[torch.Tensor, np.ndarray],
    timestep: int
) -> Dict[str, float]:
    """Calculate objective noise level measurements.

    Args:
        original_img: Original clean image.
        noisy_img: Noisy version of the image.
        timestep: Current diffusion timestep.

    Returns:
        Dictionary containing noise metrics (SNR, PSNR, MSE, MAE, etc.).
    """
    # Convert to numpy if needed
    if torch.is_tensor(original_img):
        original_img = original_img.numpy()
    if torch.is_tensor(noisy_img):
        noisy_img = noisy_img.numpy()

    # Basic statistics
    mean_original = np.mean(original_img)
    std_original = np.std(original_img)
    mean_noisy = np.mean(noisy_img)
    std_noisy = np.std(noisy_img)

    # Signal-to-Noise Ratio (SNR)
    if std_noisy > 0:
        snr = 20 * np.log10(mean_noisy / std_noisy) if mean_noisy > 0 else 0
    else:
        snr = float('inf')

    # Mean Squared Error (MSE)
    mse = np.mean((original_img - noisy_img) ** 2)

    # Peak Signal-to-Noise Ratio (PSNR)
    if mse > 0:
        max_pixel = np.max(original_img)
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    else:
        psnr = float('inf')

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(original_img - noisy_img))

    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(mse)

    # Structural metrics
    variance_original = np.var(original_img)
    variance_noisy = np.var(noisy_img)
    variance_difference = np.abs(variance_noisy - variance_original)

    # Range of pixel values
    range_original = np.max(original_img) - np.min(original_img)
    range_noisy = np.max(noisy_img) - np.min(noisy_img)

    # Noise power estimation
    noise_estimate = noisy_img - original_img
    noise_power = np.mean(noise_estimate ** 2)
    noise_std = np.std(noise_estimate)

    return {
        'timestep': timestep,
        'mean_original': mean_original,
        'std_original': std_original,
        'mean_noisy': mean_noisy,
        'std_noisy': std_noisy,
        'snr_db': snr,
        'mse': mse,
        'psnr_db': psnr,
        'mae': mae,
        'rmse': rmse,
        'variance_original': variance_original,
        'variance_noisy': variance_noisy,
        'variance_difference': variance_difference,
        'range_original': range_original,
        'range_noisy': range_noisy,
        'noise_power': noise_power,
        'noise_std': noise_std,
        'min_pixel_original': np.min(original_img),
        'max_pixel_original': np.max(original_img),
        'min_pixel_noisy': np.min(noisy_img),
        'max_pixel_noisy': np.max(noisy_img)
    }


def load_single_image(data_dir, mode, patient_index, slice_index):
    """Load a single 2D slice from specified patient and slice index."""

    # Get patient folders
    patient_folders = sorted(os.listdir(data_dir))
    if patient_index >= len(patient_folders):
        raise ValueError(f"Patient index {patient_index} >= number of patients {len(patient_folders)}")

    patient_folder = patient_folders[patient_index]
    nifti_path = os.path.join(data_dir, patient_folder, f"{mode}.nii.gz")

    # Setup transform
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        # Resize(spatial_size=(128, 128, -1)),
        SpatialCrop(roi_start=[44, 28], roi_end=[211, 227])
    ])

    # Load 3D volume
    volume = transform(nifti_path)
    print(f"Loaded volume shape: {volume.shape}")

    # Extract single slice
    if slice_index >= volume.shape[-1]:
        print(f"Warning: slice_index {slice_index} >= volume depth {volume.shape[-1]}")
        slice_index = volume.shape[-1] // 2  # Use middle slice
        print(f"Using middle slice: {slice_index}")

    image_slice = volume[0, :, :, slice_index]  # Remove channel and take slice

    # For segmentation masks, make binary
    if mode == "seg":
        image_slice = (image_slice > 0.01).float()

    print(f"Selected slice shape: {image_slice.shape}")
    print(f"Image range: [{image_slice.min():.3f}, {image_slice.max():.3f}]")
    print(f"Non-zero pixels: {torch.sum(image_slice > 0).item()}")

    return image_slice, patient_folder


def save_individual_diffusion_images(image, patient_name, mode, patient_index, slice_index):
    """Save 11 individual images showing diffusion process at different timesteps."""

    timesteps = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]

    # Add batch and channel dimensions if needed
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dim

    print(f"Image tensor shape: {image.shape}")

    # Store original image for noise metrics calculation
    original_img = image[0, 0].numpy()

    schedulers_to_test = []

    if SCHEDULER_TYPE in ["ddpm", "both"]:
        ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000)
        schedulers_to_test.append(("ddpm", ddpm_scheduler))

    if SCHEDULER_TYPE in ["rflow", "both"]:
        rflow_scheduler = RFlowScheduler(
            num_train_timesteps=1000,
            use_discrete_timesteps=True,
            sample_method='uniform',
            use_timestep_transform=True,
            base_img_size_numel=image.shape[-1] * image.shape[-2],
            spatial_dim=2
        )
        schedulers_to_test.append(("rflow", rflow_scheduler))

    for scheduler_name, scheduler in schedulers_to_test:
        print(f"\nProcessing {scheduler_name.upper()} scheduler...")
        print(f"Scheduler timesteps: 0 to {scheduler.num_train_timesteps - 1}")

        # Create output folder
        folder_name = f"diffusion_{scheduler_name}_{mode}_patient{patient_index}_slice{slice_index}_{patient_name}"
        output_dir = os.path.join(os.getcwd(), folder_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

        print(f"Saving {len(timesteps)} individual images...")

        # List to store noise metrics
        noise_metrics_list = []

        for i, t in enumerate(timesteps):
            if t == 0:
                # Original image
                noisy_image = image
            else:
                # Add noise at timestep t
                noise = torch.randn_like(image)
                timestep_tensor = torch.tensor([t], dtype=torch.long)
                noisy_image = scheduler.add_noise(
                    original_samples=image,
                    noise=noise,
                    timesteps=timestep_tensor
                )

            # Convert to numpy for visualization
            img_np = noisy_image[0, 0].numpy()

            # Calculate noise metrics
            metrics = calculate_noise_metrics(original_img, img_np, t)

            # Add metadata to metrics
            metrics.update({
                'scheduler': scheduler_name,
                'patient_name': patient_name,
                'patient_index': patient_index,
                'slice_index': slice_index,
                'mode': mode,
                'filename': f"{scheduler_name}_{mode}_patient{patient_index}_slice{slice_index}_noise{t:03d}_{patient_name}.png"
            })

            noise_metrics_list.append(metrics)

            # Save individual image without any borders or text
            filename = f"{scheduler_name}_{mode}_patient{patient_index}_slice{slice_index}_noise{t:03d}_{patient_name}.png"
            filepath = os.path.join(output_dir, filename)

            # Save directly as image data without matplotlib borders/text
            plt.imsave(filepath, img_np, cmap='gray', format='png')

            print(f"  Saved: {filename}")

            # Print statistics for first few timesteps
            if i < 4:
                print(f"    Timestep {t}: min={img_np.min():.3f}, max={img_np.max():.3f}, std={img_np.std():.3f}")
                print(
                    f"    SNR: {metrics['snr_db']:.2f} dB, PSNR: {metrics['psnr_db']:.2f} dB, MSE: {metrics['mse']:.6f}")

        # Save noise metrics to CSV
        csv_filename = f"noise_metrics_{scheduler_name}_{mode}_patient{patient_index}_slice{slice_index}_{patient_name}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)

        df = pd.DataFrame(noise_metrics_list)
        df.to_csv(csv_filepath, index=False, float_format='%.6f')

        print(f"\nNoise metrics saved to: {csv_filename}")
        print(f"All images saved to: {output_dir}")

        # Print summary of noise progression
        print(f"\n{scheduler_name.upper()} Noise Progression Summary:")
        print("Timestep | SNR (dB) | PSNR (dB) |   MSE   | Noise Std")
        print("-" * 50)
        for metrics in noise_metrics_list[::2]:  # Print every other timestep
            print(
                f"   {metrics['timestep']:3d}   | {metrics['snr_db']:7.2f} | {metrics['psnr_db']:8.2f} | {metrics['mse']:7.4f} | {metrics['noise_std']:8.4f}")


def main():
    """Main execution function."""

    print(f"Configuration:")
    print(f"- Data directory: {DATA_DIR}")
    print(f"- Mode: {MODE}")
    print(f"- Patient index: {PATIENT_INDEX}")
    print(f"- Slice index: {SLICE_INDEX}")
    print(f"- Device: {DEVICE}")
    print(f"- Scheduler type: {SCHEDULER_TYPE}")
    print("-" * 70)

    try:
        # Load single image
        image, patient_name = load_single_image(
            DATA_DIR, MODE, PATIENT_INDEX, SLICE_INDEX
        )

        print(f"Processing patient: {patient_name}")
        print("-" * 70)

        # Save individual diffusion images
        save_individual_diffusion_images(
            image, patient_name, MODE, PATIENT_INDEX, SLICE_INDEX
        )

        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    main()