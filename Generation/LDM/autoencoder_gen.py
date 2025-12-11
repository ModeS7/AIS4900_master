"""
Latent representation generation from trained AutoencoderKL.

This module generates latent space representations from a trained AutoencoderKL
model for use in latent diffusion training.

Usage:
    python autoencoder_gen.py --compute local --model_path /path/to/model
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from monai.data import DataLoader
from monai.networks.nets import AutoencoderKL
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    ScaleIntensity,
    ToTensor,
)
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import PathConfig
from Generation.TrainGen.core.data import NiFTIDataset, extract_slices_single


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Generate latent representations from AutoencoderKL'
    )
    parser.add_argument(
        '--compute', type=str, default='local',
        choices=['local', 'cluster', 'windows'],
        help='Compute environment'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to trained autoencoder model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--output_name', type=str, default='bravo_latents.npy',
        help='Output filename for latent representations'
    )
    return parser.parse_args()


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
    """Main entry point for latent generation."""
    args = parse_args()

    # Setup path configuration
    path_config = PathConfig(compute=args.compute)

    batch_size: int = args.batch_size
    device = torch.device("cuda")

    # Load trained autoencoder
    print("Loading autoencoder model...")
    autoencoder = create_autoencoder(args.model_path, device)

    # Data setup
    data_dir = str(path_config.brainmet_train_dir)
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(spatial_size=(128, 128, -1)),
    ])

    dataset = NiFTIDataset(data_dir=data_dir, mr_sequence="bravo", transform=transform)
    train_dataset = extract_slices_single(dataset)
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # Generate latent representations
    latent_representations: List[np.ndarray] = []
    print("Generating latent space representations...")

    with torch.no_grad():
        for batch in tqdm(train_data_loader):
            images = batch.to(device)
            z_mu, z_sigma = autoencoder.encode(images)
            latent_z = z_mu
            latent_representations.append(latent_z.cpu().numpy())

    # Concatenate all latent representations
    all_latents = np.concatenate(latent_representations, axis=0)
    print(
        f"Generated {all_latents.shape[0]} latent representations "
        f"with shape {all_latents.shape[1:]}"
    )

    # Save latent representations
    output_dir = path_config.latent_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    np.save(output_path, all_latents)
    print(f"Saved latent representations to {output_path}")


if __name__ == "__main__":
    main()
