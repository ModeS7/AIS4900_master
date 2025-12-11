"""
AutoencoderKL training module for latent diffusion models.

This module provides training functionality for the AutoencoderKL model
used in latent diffusion pipelines for brain MRI image generation.

Usage:
    python autoencoder.py --compute local --epochs 100 --batch_size 16
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.layers import Act
from monai.networks.nets import AutoencoderKL, PatchDiscriminator
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    ScaleIntensity,
    ToTensor,
)
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import PathConfig
from Generation.TrainGen.core.data import NiFTIDataset, extract_slices_single

# CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description='Train AutoencoderKL model')
    parser.add_argument(
        '--compute', type=str, default='local',
        choices=['local', 'cluster', 'windows'],
        help='Compute environment'
    )
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
    return parser.parse_args()


def create_autoencoder(device: torch.device) -> torch.nn.Module:
    """Create and configure the AutoencoderKL model.

    Args:
        device: Target device for the model.

    Returns:
        Compiled AutoencoderKL model.
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
    autoencoder.to(device)
    autoencoder = torch.compile(autoencoder, mode="reduce-overhead")
    return autoencoder


def create_discriminator(device: torch.device) -> torch.nn.Module:
    """Create and configure the PatchDiscriminator model.

    Args:
        device: Target device for the model.

    Returns:
        Compiled PatchDiscriminator model.
    """
    discriminator = PatchDiscriminator(
        spatial_dims=2,
        num_layers_d=4,
        channels=64,
        in_channels=1,
        out_channels=1,
        kernel_size=4,
        activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm="BATCH",
        bias=False,
        padding=1,
    )
    discriminator.to(device)
    discriminator = torch.compile(discriminator, mode="reduce-overhead")
    return discriminator


def generate_validation_samples(
    autoencoder: torch.nn.Module,
    train_data_loader: DataLoader,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    num_samples: int = 4
) -> None:
    """Generate and log validation reconstruction samples.

    Args:
        autoencoder: Trained autoencoder model.
        train_data_loader: DataLoader for sampling images.
        writer: TensorBoard SummaryWriter.
        epoch: Current training epoch.
        device: Computation device.
        num_samples: Number of samples to generate.
    """
    autoencoder.eval()
    try:
        with torch.no_grad():
            batch = next(iter(train_data_loader))
            if len(batch) > num_samples:
                batch = batch[:num_samples]

            images = batch.to(device)
            with autocast(device_type="cuda", enabled=True):
                reconstruction, z_mu, z_sigma = autoencoder(images)

            writer.add_images('Original_Images', torch.clamp(images, 0, 1), epoch)
            writer.add_images('Reconstructed_Images', torch.clamp(reconstruction, 0, 1), epoch)
            del images, reconstruction, z_mu, z_sigma
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Warning: Failed to generate validation samples at epoch {epoch}: {e}")
        torch.cuda.empty_cache()
    autoencoder.train()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Setup path configuration
    path_config = PathConfig(compute=args.compute)

    # Training configuration
    n_epochs: int = args.epochs
    val_interval: int = args.val_interval
    batch_size: int = args.batch_size

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # TensorBoard setup
    log_dir = path_config.log_dir / f"autoencoder_bravo_{timestamp}"
    writer = SummaryWriter(str(log_dir))

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
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    device = torch.device("cuda")

    # Model setup
    autoencoder = create_autoencoder(device)
    discriminator = create_discriminator(device)

    # Loss functions
    perceptual_loss = PerceptualLoss(
        spatial_dims=2,
        network_type="radimagenet_resnet50",
        cache_dir=str(path_config.cache_dir),
    )
    perceptual_loss.to(device)
    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    # Optimizers
    optimizer_ae = torch.optim.AdamW(autoencoder.parameters(), lr=5e-5)
    optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr=2e-4)
    scaler = GradScaler('cuda')

    # Loss weights
    kl_weight: float = 1e-6
    adv_weight: float = 0.01
    perceptual_weight: float = 0.001

    # Training loop
    total_start = time.time()

    for epoch in range(n_epochs):
        autoencoder.train()
        discriminator.train()
        epoch_loss: float = 0.0
        gen_epoch_loss: float = 0.0
        disc_epoch_loss: float = 0.0
        epoch_recon_loss: float = 0.0
        epoch_kl_loss: float = 0.0
        epoch_perceptual_loss: float = 0.0

        progress_bar = tqdm(
            enumerate(train_data_loader), total=len(train_data_loader), ncols=70
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            images = batch.to(device)

            # Train Autoencoder
            optimizer_ae.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=True):
                reconstruction, z_mu, z_sigma = autoencoder(images)
                recons_loss = l1_loss(reconstruction.float(), images.float())
                kl_loss = 0.5 * torch.sum(
                    z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
                    dim=[1, 2, 3]
                )
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                p_loss = perceptual_loss(reconstruction.float(), images.float())
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(
                    logits_fake, target_is_real=True, for_discriminator=False
                )
                loss_g = (
                    recons_loss
                    + kl_weight * kl_loss
                    + perceptual_weight * p_loss
                    + adv_weight * generator_loss
                )

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_ae)
            scaler.update()

            # Train Discriminator
            optimizer_disc.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=True):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(
                    logits_fake, target_is_real=False, for_discriminator=True
                )
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(
                    logits_real, target_is_real=True, for_discriminator=True
                )
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_disc)
            scaler.update()

            epoch_loss += recons_loss.item()
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()
            epoch_recon_loss += recons_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_perceptual_loss += p_loss.item()

            progress_bar.set_postfix(loss=f"{epoch_loss / (step + 1):.4f}")

        # Calculate and log average losses
        num_batches = len(train_data_loader)
        epoch_recon_loss_avg = epoch_loss / num_batches
        epoch_gen_loss_avg = gen_epoch_loss / num_batches
        epoch_disc_loss_avg = disc_epoch_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_perceptual_loss = epoch_perceptual_loss / num_batches

        writer.add_scalar('Loss/Reconstruction', epoch_recon_loss_avg, epoch)
        writer.add_scalar('Loss/Generator', epoch_gen_loss_avg, epoch)
        writer.add_scalar('Loss/Discriminator', epoch_disc_loss_avg, epoch)
        writer.add_scalar('Loss/KL_Divergence', avg_kl_loss, epoch)
        writer.add_scalar('Loss/Perceptual', avg_perceptual_loss, epoch)

        # Validation and model saving
        if (epoch + 1) % val_interval == 0:
            save_dir = path_config.model_dir / f"autoencoder_{timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)

            msgs = [
                f"epoch {epoch:d}/{n_epochs:d}: ",
                f"recons loss: {epoch_recon_loss_avg:.4f}, "
                f"gen_loss: {epoch_gen_loss_avg:.4f}, "
                f"disc_loss: {epoch_disc_loss_avg:.4f}"
            ]
            print("".join(msgs))
            print("Generating validation samples...")
            generate_validation_samples(
                autoencoder, train_data_loader, writer, epoch, device
            )

            autoencoder.eval()
            ae_path = save_dir / f"autoencoder_bravo_{batch_size}_Epoch{epoch}_of_{n_epochs}"
            disc_path = save_dir / f"discriminator_bravo_{batch_size}_Epoch{epoch}_of_{n_epochs}"

            torch.save(autoencoder.state_dict(), ae_path)
            torch.save(discriminator.state_dict(), disc_path)
            print(f"Saved models at epoch {epoch}")

    total_time = time.time() - total_start
    print(f"Training completed in {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")
    writer.close()
    print("Training completed. Check TensorBoard for metrics.")


if __name__ == "__main__":
    main()
