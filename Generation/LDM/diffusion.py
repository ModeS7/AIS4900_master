"""
Latent diffusion model training module.

This module provides training functionality for diffusion models operating
in the latent space of a trained AutoencoderKL for brain MRI generation.

Usage:
    python diffusion.py --compute local --epochs 100 --latent --latent_path /path/to/latents.npy
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from monai.inferers import DiffusionInferer
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    ScaleIntensity,
    ToTensor,
)
from torch.amp import GradScaler, autocast
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


class LatentDataset(torch.utils.data.Dataset):
    """Dataset for loading pre-computed latent representations.

    Args:
        latent_path: Path to the .npy file containing latent vectors.
    """

    def __init__(self, latent_path: str) -> None:
        self.latents = np.load(latent_path)
        print(
            f"Loaded {len(self.latents)} latent representations "
            f"with shape {self.latents.shape[1:]}"
        )

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> torch.Tensor:
        latent = torch.from_numpy(self.latents[idx]).float()
        return latent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description='Train latent diffusion model')
    parser.add_argument(
        '--compute', type=str, default='local',
        choices=['local', 'cluster', 'windows'],
        help='Compute environment'
    )
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
    parser.add_argument(
        '--latent', action='store_true',
        help='Train on latent space (requires --latent_path)'
    )
    parser.add_argument(
        '--latent_path', type=str, default=None,
        help='Path to pre-computed latent representations'
    )
    parser.add_argument(
        '--autoencoder_path', type=str, default=None,
        help='Path to trained autoencoder (for latent mode validation)'
    )
    return parser.parse_args()


def generate_validation_samples(
    model: torch.nn.Module,
    inferer: DiffusionInferer,
    scheduler: DDPMScheduler,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
    latent: bool = False,
    autoencoder: Optional[torch.nn.Module] = None,
    num_samples: int = 4
) -> None:
    """Generate and log validation samples.

    Args:
        model: Trained diffusion model.
        inferer: DiffusionInferer for sampling.
        scheduler: Noise scheduler.
        writer: TensorBoard SummaryWriter.
        epoch: Current training epoch.
        device: Computation device.
        latent: Whether operating in latent space.
        autoencoder: Autoencoder for decoding latents (required if latent=True).
        num_samples: Number of samples to generate.
    """
    model.eval()

    try:
        with torch.no_grad():
            if latent:
                noise_shape = (num_samples, 8, 16, 16)
                log_prefix = "Latent"
            else:
                noise_shape = (num_samples, 1, 64, 64)
                log_prefix = "Pixel"

            noise = torch.randn(noise_shape, device=device)

            with autocast(device_type="cuda", enabled=True):
                samples = inferer.sample(
                    input_noise=noise,
                    diffusion_model=model,
                    scheduler=scheduler
                )

                if latent and autoencoder is not None:
                    decoded_images = autoencoder.decode_stage_2_outputs(samples)
                    decoded_images_normalized = torch.clamp(decoded_images, 0, 1)
                    writer.add_images(
                        f'Generated_{log_prefix}_Ch0',
                        torch.clamp(samples[:, 0:1], -3, 3), epoch
                    )
                    writer.add_images(
                        'Generated_Decoded_Images',
                        decoded_images_normalized, epoch
                    )
                else:
                    samples_normalized = torch.clamp(samples, 0, 1)
                    writer.add_images(
                        f'Generated_{log_prefix}_Images',
                        samples_normalized, epoch
                    )

            del noise, samples
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Warning: Failed to generate validation samples at epoch {epoch}: {e}")
        torch.cuda.empty_cache()

    finally:
        model.train()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Setup path configuration
    path_config = PathConfig(compute=args.compute)

    # Training configuration
    n_epochs: int = args.epochs
    val_interval: int = args.val_interval
    batch_size: int = args.batch_size
    latent: bool = args.latent

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # TensorBoard setup
    mode_suffix = "latent" if latent else "pixel"
    log_dir = path_config.log_dir / f"diffusion_{mode_suffix}_bravo_{timestamp}"
    writer = SummaryWriter(str(log_dir))

    # Data setup
    if latent:
        if args.latent_path is None:
            latent_path = str(path_config.latent_data_dir / "bravo_latents.npy")
        else:
            latent_path = args.latent_path
        train_dataset = LatentDataset(latent_path)
        n_channels = 8
    else:
        data_dir = str(path_config.brainmet_train_dir)
        transform = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ToTensor(),
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=(64, 64, -1)),
        ])
        dataset = NiFTIDataset(data_dir=data_dir, mr_sequence="bravo", transform=transform)
        train_dataset = extract_slices_single(dataset)
        n_channels = 1

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    device = torch.device("cuda")

    # Model setup
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=n_channels,
        out_channels=n_channels,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = DDPMScheduler(num_train_timesteps=1000, schedule='cosine')
    inferer = DiffusionInferer(scheduler)
    scaler = GradScaler('cuda')

    # Load autoencoder for latent mode validation
    autoencoder: Optional[torch.nn.Module] = None
    if latent and args.autoencoder_path:
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
        autoencoder.load_state_dict(torch.load(args.autoencoder_path))
        autoencoder.to(device)
        autoencoder.eval()

    # Training loop
    total_start = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss: float = 0.0
        progress_bar = tqdm(
            enumerate(train_data_loader), total=len(train_data_loader), ncols=70
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            images = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=True):
                noise = torch.randn_like(images).to(device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps,
                    (images.shape[0],), device=images.device
                ).long()
                noise_pred = inferer(
                    inputs=images, diffusion_model=model,
                    noise=noise, timesteps=timesteps
                )
                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

        # Calculate and log average loss
        avg_epoch_loss = epoch_loss / len(train_data_loader)
        writer.add_scalar('Loss/Training', avg_epoch_loss, epoch)
        print(f"Epoch {epoch}: Average Loss = {avg_epoch_loss:.6f}")

        # Validation and model saving
        if (epoch + 1) % val_interval == 0:
            save_dir = path_config.model_dir / f"diffusion_{mode_suffix}_{timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)

            print("Generating validation samples...")
            generate_validation_samples(
                model, inferer, scheduler, writer, epoch, device,
                latent=latent, autoencoder=autoencoder
            )

            model.eval()
            model_path = save_dir / f"diffusion_{mode_suffix}_bravo_{batch_size}_Epoch{epoch}_of_{n_epochs}"
            torch.save(model.state_dict(), model_path)
            print(f"Saved model at epoch {epoch}")

    total_time = time.time() - total_start
    print(f"Training completed in {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")
    writer.close()
    print("Training completed. Check TensorBoard for metrics.")


if __name__ == "__main__":
    main()
