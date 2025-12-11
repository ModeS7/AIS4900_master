"""
Latent diffusion model inference module.

This module generates synthetic brain MRI images using a trained latent
diffusion model and AutoencoderKL decoder. Generated images are saved as NIfTI files.

Usage:
    python LDM_inference.py --compute local --num_images 1000 \
        --diffusion_model /path/to/diffusion --autoencoder_model /path/to/autoencoder
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from monai.inferers import DiffusionInferer
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from torch.amp import autocast
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config import PathConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Generate images using trained latent diffusion model'
    )
    parser.add_argument(
        '--compute', type=str, default='local',
        choices=['local', 'cluster', 'windows'],
        help='Compute environment'
    )
    parser.add_argument(
        '--diffusion_model', type=str, required=True,
        help='Path to trained latent diffusion model'
    )
    parser.add_argument(
        '--autoencoder_model', type=str, required=True,
        help='Path to trained autoencoder model'
    )
    parser.add_argument(
        '--num_images', type=int, default=15000,
        help='Number of images to generate'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size for generation'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for generated images'
    )
    return parser.parse_args()


def create_latent_diffusion_model(
    model_path: str, device: torch.device
) -> torch.nn.Module:
    """Create and load trained latent diffusion model.

    Args:
        model_path: Path to saved model weights.
        device: Target device for the model.

    Returns:
        Loaded DiffusionModelUNet in evaluation mode.
    """
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=8,
        out_channels=8,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def create_autoencoder(model_path: str, device: torch.device) -> torch.nn.Module:
    """Create and load trained AutoencoderKL model.

    Args:
        model_path: Path to saved model weights.
        device: Target device for the model.

    Returns:
        Loaded AutoencoderKL model in evaluation mode.
    """
    autoencoder = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(128, 256, 384, 512),
        latent_channels=8,
        num_res_blocks=1,
        norm_num_groups=32,
        attention_levels=(False, False, False, True),
    )
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder


def main() -> None:
    """Main entry point for image generation."""
    args = parse_args()

    # Setup path configuration
    path_config = PathConfig(compute=args.compute)

    device = torch.device("cuda")

    # Load trained models
    print("Loading latent diffusion model...")
    latent_diffusion_model = create_latent_diffusion_model(
        args.diffusion_model, device
    )

    print("Loading autoencoder for decoding...")
    autoencoder = create_autoencoder(args.autoencoder_model, device)

    # Setup scheduler and inferer
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='cosine')
    inferer = DiffusionInferer(scheduler)

    # Generation parameters
    num_images: int = args.num_images
    batch_size: int = args.batch_size

    # Setup output directory
    if args.output_dir is None:
        output_dir = path_config.data_dir / "LDM_gen"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_images} images...")

    image_counter: int = 0
    num_batches = (num_images + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            current_batch_size = min(batch_size, num_images - batch_idx * batch_size)

            # Generate noise in latent space (8 channels, 16x16)
            noise = torch.randn((current_batch_size, 8, 16, 16), device=device)

            with autocast(device_type="cuda", enabled=False):
                # Generate latent samples using diffusion model
                latent_samples = inferer.sample(
                    input_noise=noise,
                    diffusion_model=latent_diffusion_model,
                    scheduler=scheduler
                )

                # Decode latent samples to pixel space
                decoded_images = autoencoder.decode_stage_2_outputs(latent_samples)

            # Convert to numpy and save
            decoded_images_cpu = decoded_images.cpu().numpy()

            # Save individual images as NIfTI files
            for i in range(current_batch_size):
                image = decoded_images_cpu[i, 0]  # Remove channel dimension
                image_normalized = np.clip(image, 0, 1).astype(np.float32)
                image_3d = np.expand_dims(image_normalized, axis=-1)

                nifti_image = nib.Nifti1Image(image_3d, np.eye(4))
                output_path = output_dir / f"generated_image_{image_counter:04d}.nii.gz"
                nib.save(nifti_image, output_path)

                image_counter += 1

            # Clear GPU memory
            del noise, latent_samples, decoded_images, decoded_images_cpu
            torch.cuda.empty_cache()

    print(f"Generated {image_counter} images saved as NIfTI files to {output_dir}")


if __name__ == "__main__":
    main()
