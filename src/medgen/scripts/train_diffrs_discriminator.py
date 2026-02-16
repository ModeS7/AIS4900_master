"""Train DiffRS discriminator head on a trained diffusion model.

Trains a tiny classification head (~500 params) on top of the frozen UNet
encoder to distinguish real vs generated samples at various noise levels.

The trained head is used during generation to reject bad intermediate samples
(Diffusion Rejection Sampling).

Usage:
    # Train head for a specific diffusion model
    python -m medgen.scripts.train_diffrs_discriminator --config-name=diffrs \
        diffusion_checkpoint=runs/bravo/checkpoint_best.pt \
        mode=bravo

    # Quick test (local)
    python -m medgen.scripts.train_diffrs_discriminator --config-name=diffrs \
        diffusion_checkpoint=runs/bravo/checkpoint_best.pt \
        mode=bravo num_generated_samples=100 num_epochs=5
"""
import logging
import os
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import autocast
from tqdm import tqdm

from medgen.core import setup_cuda_optimizations
from medgen.diffusion import RFlowStrategy, load_diffusion_model_with_metadata
from medgen.diffusion.diffrs import (
    DiffRSHead,
    extract_encoder_features,
    get_bottleneck_channels,
)

setup_cuda_optimizations()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
)
logger = logging.getLogger(__name__)


def generate_samples(
    model: nn.Module,
    strategy: RFlowStrategy,
    num_samples: int,
    sample_shape: tuple[int, ...],
    num_steps: int,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """Generate samples using the trained diffusion model.

    Args:
        model: Trained diffusion model.
        strategy: RFlowStrategy with scheduler set up.
        num_samples: Number of samples to generate.
        sample_shape: Shape of a single sample [C, H, W] or [C, D, H, W].
        num_steps: Number of denoising steps.
        device: Computation device.
        batch_size: Generation batch size.

    Returns:
        Generated samples tensor [N, C, ...].
    """
    all_samples = []
    remaining = num_samples

    logger.info("Generating %d samples for DiffRS training...", num_samples)

    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)
            noise = torch.randn(bs, *sample_shape, device=device)

            with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                samples = strategy.generate(
                    model, noise, num_steps, device,
                )

            all_samples.append(samples.cpu())
            remaining -= bs

            if len(all_samples) % 10 == 0:
                generated = num_samples - remaining
                logger.info("Generated %d/%d samples", generated, num_samples)

    result = torch.cat(all_samples, dim=0)[:num_samples]
    logger.info("Generation complete: %s", result.shape)
    return result


def load_real_samples(
    data_dir: str,
    image_type: str,
    image_size: int,
    num_samples: int,
    spatial_dims: int = 2,
) -> torch.Tensor:
    """Load real training samples from the dataset.

    Args:
        data_dir: Path to data directory.
        image_type: Modality to load ('seg', 'bravo', etc.).
        image_size: Target image size.
        num_samples: Number of samples to load.
        spatial_dims: 2 for 2D slices, 3 for 3D volumes.

    Returns:
        Real samples tensor [N, 1, H, W] or [N, 1, D, H, W].
    """
    from medgen.data.dataset import NiFTIDataset, build_standard_transform

    train_dir = os.path.join(data_dir, "train")
    transform = build_standard_transform(image_size)

    dataset = NiFTIDataset(
        root_dir=train_dir,
        modality=image_type,
        transform=transform,
    )

    all_samples = []
    if spatial_dims == 2:
        from medgen.data.utils import extract_slices_single
        slices = extract_slices_single(dataset)
        # Shuffle and take num_samples
        indices = torch.randperm(len(slices))[:num_samples]
        for idx in indices:
            all_samples.append(slices[idx.item()])
    else:
        # 3D: load volumes directly
        indices = torch.randperm(len(dataset))[:num_samples]
        for idx in indices:
            vol = dataset[idx.item()]
            if isinstance(vol, dict):
                vol = vol[image_type]
            all_samples.append(vol)

    result = torch.stack(all_samples, dim=0)
    if result.dim() == 3:
        result = result.unsqueeze(1)  # Add channel dim
    logger.info("Loaded %d real samples: %s", len(result), result.shape)
    return result


def train_head(
    model: nn.Module,
    strategy: RFlowStrategy,
    head: DiffRSHead,
    real_samples: torch.Tensor,
    generated_samples: torch.Tensor,
    device: torch.device,
    num_epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-7,
) -> DiffRSHead:
    """Train the DiffRS discriminator head.

    Args:
        model: Frozen diffusion model (for feature extraction).
        strategy: RFlowStrategy (for timestep sampling).
        head: DiffRSHead to train.
        real_samples: Real training samples [N, C, ...].
        generated_samples: Generated samples [N, C, ...].
        device: Computation device.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.

    Returns:
        Trained DiffRSHead.
    """
    optimizer = torch.optim.Adam(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    num_real = len(real_samples)
    num_gen = len(generated_samples)
    num_total = num_real + num_gen

    # Combine and create labels
    all_data = torch.cat([real_samples, generated_samples], dim=0)
    labels = torch.cat([
        torch.ones(num_real),
        torch.zeros(num_gen),
    ])

    head.train()
    model.eval()

    logger.info(
        "Training DiffRS head: %d real + %d generated, %d epochs, lr=%.1e",
        num_real, num_gen, num_epochs, learning_rate,
    )

    for epoch in range(num_epochs):
        # Shuffle
        perm = torch.randperm(num_total)
        all_data_shuffled = all_data[perm]
        labels_shuffled = labels[perm]

        epoch_loss = 0.0
        epoch_correct = 0
        num_batches = 0

        for i in range(0, num_total, batch_size):
            batch_x = all_data_shuffled[i:i + batch_size].to(device)
            batch_y = labels_shuffled[i:i + batch_size].to(device)

            # Sample random timesteps
            timesteps = strategy.sample_timesteps(batch_x)

            # Forward-diffuse to timestep t
            noise = torch.randn_like(batch_x)
            noisy = strategy.scheduler.add_noise(batch_x, noise, timesteps)

            # Extract features from frozen encoder
            with torch.no_grad():
                features = extract_encoder_features(model, noisy, timesteps)

            # Head prediction
            logits = head(features.float())
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += ((logits > 0).float() == batch_y).sum().item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        accuracy = epoch_correct / num_total * 100

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d: loss=%.4f, accuracy=%.1f%%",
                epoch + 1, num_epochs, avg_loss, accuracy,
            )

    head.eval()
    return head


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="diffrs",
)
def main(cfg: DictConfig) -> None:
    """Train DiffRS discriminator head."""
    if cfg.diffusion_checkpoint is None:
        raise ValueError(
            "diffusion_checkpoint is required. "
            "Point it to a trained diffusion model checkpoint."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load diffusion model
    logger.info("Loading diffusion model from %s", cfg.diffusion_checkpoint)
    result = load_diffusion_model_with_metadata(
        checkpoint_path=cfg.diffusion_checkpoint,
        device=device,
        spatial_dims=cfg.get('spatial_dims', 2),
    )
    model = result.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Set up strategy
    spatial_dims = cfg.get('spatial_dims', 2)
    image_size = cfg.get('image_size', 128)
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000,
        image_size=image_size,
        depth_size=cfg.get('depth', 160) if spatial_dims == 3 else None,
        spatial_dims=spatial_dims,
    )

    # Determine sample shape from model
    out_channels = result.config.get('model', {}).get('out_channels', 1)
    if spatial_dims == 3:
        depth = cfg.get('depth', 160)
        sample_shape = (out_channels, depth, image_size, image_size)
    else:
        sample_shape = (out_channels, image_size, image_size)

    # Generate samples
    generated = generate_samples(
        model=model,
        strategy=strategy,
        num_samples=cfg.num_generated_samples,
        sample_shape=sample_shape,
        num_steps=cfg.generation_num_steps,
        device=device,
        batch_size=cfg.batch_size,
    )

    # Load real samples
    mode = cfg.get('mode', 'bravo')
    # Map mode to image type for loading
    mode_to_type = {
        'bravo': 'bravo',
        'seg': 'seg',
        'seg_conditioned': 'seg',
        'dual': 'bravo',
    }
    image_type = mode_to_type.get(mode, 'bravo')

    real = load_real_samples(
        data_dir=cfg.paths.data_dir,
        image_type=image_type,
        image_size=image_size,
        num_samples=cfg.num_generated_samples,
        spatial_dims=spatial_dims,
    )

    # Create head
    bottleneck_channels = get_bottleneck_channels(model)
    head = DiffRSHead(
        in_channels=bottleneck_channels,
        spatial_dims=spatial_dims,
    ).to(device)

    num_params = sum(p.numel() for p in head.parameters())
    logger.info(
        "DiffRS head created: in_channels=%d, spatial_dims=%d, params=%d",
        bottleneck_channels, spatial_dims, num_params,
    )

    # Train
    head = train_head(
        model=model,
        strategy=strategy,
        head=head,
        real_samples=real,
        generated_samples=generated,
        device=device,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Save checkpoint
    output_dir = Path(cfg.get('output_dir', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "diffrs_head.pt"

    torch.save({
        'head_state_dict': head.state_dict(),
        'in_channels': bottleneck_channels,
        'spatial_dims': spatial_dims,
        'diffusion_checkpoint': cfg.diffusion_checkpoint,
        'num_epochs': cfg.num_epochs,
        'num_generated_samples': cfg.num_generated_samples,
    }, save_path)

    logger.info("DiffRS head saved to %s", save_path)


if __name__ == "__main__":
    main()
