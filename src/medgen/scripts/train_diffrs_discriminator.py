"""Train DiffRS discriminator head on a trained diffusion model.

Trains a convolutional classification head (~0.9M params for 3D) on top of the
frozen UNet encoder to distinguish real vs generated samples at various noise levels.

The trained head is used during generation to reject bad intermediate samples
(Diffusion Rejection Sampling).

Handles both conditioned models (bravo: in_channels=2 for noise+seg) and
unconditioned models (seg: in_channels=1).

Usage:
    # Train head for a conditioned 3D bravo model
    python -m medgen.scripts.train_diffrs_discriminator --config-name=diffrs \
        diffusion_checkpoint=runs/bravo/checkpoint_best.pt \
        mode=bravo spatial_dims=3 num_generated_samples=50 batch_size=1

    # Train head for a 2D seg model (unconditioned)
    python -m medgen.scripts.train_diffrs_discriminator --config-name=diffrs \
        diffusion_checkpoint=runs/seg/checkpoint_best.pt \
        mode=seg num_generated_samples=5000 num_epochs=60

    # Quick test
    python -m medgen.scripts.train_diffrs_discriminator --config-name=diffrs \
        diffusion_checkpoint=runs/bravo/checkpoint_best.pt \
        mode=bravo num_generated_samples=5 num_epochs=5
"""
import logging
from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import autocast

from medgen.core import setup_cuda_optimizations
from medgen.diffusion import RFlowStrategy, load_diffusion_model_with_metadata
from medgen.diffusion.diffrs import (
    DiffRSHead,
    _unwrap_to_unet,
    extract_encoder_features,
    get_bottleneck_channels,
)

setup_cuda_optimizations()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_paired_volumes_3d(
    data_dir: str,
    image_type: str,
    num_volumes: int,
    depth: int,
    split: str = "train",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load paired (image, seg) 3D volumes from NIfTI files.

    Args:
        data_dir: Root data directory containing split subdirectories.
        image_type: Image modality to load ('bravo', 'seg', etc.).
        num_volumes: Maximum volumes to load.
        depth: Target depth (pad/crop if needed).
        split: Data split to load from ('train' or 'test').

    Returns:
        (images [N, 1, D, H, W], segs [N, 1, D, H, W]) both in [0, 1].
    """
    train_dir = Path(data_dir) / split
    images, segs = [], []

    for patient_dir in sorted(train_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        img_path = patient_dir / f"{image_type}.nii.gz"
        seg_path = patient_dir / "seg.nii.gz"
        if not img_path.exists() or not seg_path.exists():
            continue
        if len(images) >= num_volumes:
            break

        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        seg = nib.load(str(seg_path)).get_fdata().astype(np.float32)

        # Normalize image to [0, 1]
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        seg = (seg > 0.5).astype(np.float32)

        # [H, W, D] -> [D, H, W]
        img = np.transpose(img, (2, 0, 1))
        seg = np.transpose(seg, (2, 0, 1))

        # Pad/crop to target depth
        d = img.shape[0]
        if d < depth:
            pad_shape = (depth - d, img.shape[1], img.shape[2])
            img = np.concatenate([img, np.zeros(pad_shape, dtype=np.float32)], axis=0)
            seg = np.concatenate([seg, np.zeros(pad_shape, dtype=np.float32)], axis=0)
        elif d > depth:
            img = img[:depth]
            seg = seg[:depth]

        images.append(torch.from_numpy(img).unsqueeze(0))  # [1, D, H, W]
        segs.append(torch.from_numpy(seg).unsqueeze(0))

    logger.info("Loaded %d paired 3D volumes from %s [%s]", len(images), train_dir, split)
    return torch.stack(images), torch.stack(segs)


def load_paired_slices_2d(
    data_dir: str,
    image_type: str,
    num_samples: int,
    split: str = "train",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load paired (image, seg) 2D slices from NIfTI volumes.

    Only keeps slices that have tumor (seg > 0).

    Args:
        data_dir: Root data directory containing split subdirectories.
        image_type: Image modality to load.
        num_samples: Maximum slices to return.
        split: Data split to load from ('train' or 'test').

    Returns:
        (images [N, 1, H, W], segs [N, 1, H, W]) both in [0, 1].
    """
    train_dir = Path(data_dir) / split
    all_images, all_segs = [], []

    for patient_dir in sorted(train_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        img_path = patient_dir / f"{image_type}.nii.gz"
        seg_path = patient_dir / "seg.nii.gz"
        if not img_path.exists() or not seg_path.exists():
            continue

        img_vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
        seg_vol = nib.load(str(seg_path)).get_fdata().astype(np.float32)

        vmin, vmax = img_vol.min(), img_vol.max()
        if vmax > vmin:
            img_vol = (img_vol - vmin) / (vmax - vmin)
        seg_vol = (seg_vol > 0.5).astype(np.float32)

        # Extract paired slices along depth (axis 2)
        for d in range(img_vol.shape[2]):
            seg_slice = seg_vol[:, :, d]
            if seg_slice.sum() > 0:
                img_slice = img_vol[:, :, d]
                all_images.append(torch.from_numpy(img_slice).unsqueeze(0))
                all_segs.append(torch.from_numpy(seg_slice).unsqueeze(0))

    # Shuffle and take num_samples
    n = len(all_images)
    indices = torch.randperm(n)[:min(num_samples, n)]
    images = torch.stack([all_images[i] for i in indices])
    segs = torch.stack([all_segs[i] for i in indices])
    logger.info("Loaded %d paired 2D slices from %s [%s]", len(images), train_dir, split)
    return images, segs


def load_unconditioned_samples(
    data_dir: str,
    image_type: str,
    image_size: int,
    num_samples: int,
    spatial_dims: int = 2,
    split: str = "train",
) -> torch.Tensor:
    """Load real samples (no conditioning needed) for unconditioned models.

    Args:
        data_dir: Path to data directory.
        image_type: Modality to load.
        image_size: Target image size.
        num_samples: Number of samples to load.
        spatial_dims: 2 for 2D slices, 3 for 3D volumes.
        split: Data split to load from ('train' or 'test').

    Returns:
        Tensor [N, 1, H, W] or [N, 1, D, H, W].
    """
    import os

    from medgen.data.dataset import NiFTIDataset, build_standard_transform

    train_dir = os.path.join(data_dir, split)
    transform = build_standard_transform(image_size)
    dataset = NiFTIDataset(root_dir=train_dir, modality=image_type, transform=transform)

    all_samples = []
    if spatial_dims == 2:
        from medgen.data.utils import extract_slices_single
        slices = extract_slices_single(dataset)
        indices = torch.randperm(len(slices))[:num_samples]
        for idx in indices:
            all_samples.append(slices[idx.item()])
    else:
        indices = torch.randperm(len(dataset))[:num_samples]
        for idx in indices:
            vol = dataset[idx.item()]
            if isinstance(vol, dict):
                vol = vol[image_type]
            all_samples.append(vol)

    result = torch.stack(all_samples, dim=0)
    if result.dim() == 3:
        result = result.unsqueeze(1)
    logger.info("Loaded %d unconditioned samples from %s [%s]: %s", len(result), train_dir, split, result.shape)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Sample generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_conditioned_samples(
    model: nn.Module,
    strategy: RFlowStrategy,
    seg_masks: torch.Tensor,
    out_channels: int,
    num_steps: int,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """Generate samples conditioned on seg masks.

    Args:
        model: Trained diffusion model.
        strategy: RFlowStrategy with scheduler.
        seg_masks: Conditioning masks [N, 1, ...].
        out_channels: Output channels of the model.
        num_steps: Denoising steps.
        device: Computation device.
        batch_size: Generation batch size.

    Returns:
        Generated samples [N, out_channels, ...].
    """
    all_samples = []

    logger.info("Generating %d conditioned samples...", len(seg_masks))

    for i in range(0, len(seg_masks), batch_size):
        batch_seg = seg_masks[i:i + batch_size].to(device)
        bs = batch_seg.shape[0]
        spatial_shape = batch_seg.shape[2:]
        noise = torch.randn(bs, out_channels, *spatial_shape, device=device)
        model_input = torch.cat([noise, batch_seg], dim=1)

        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            with torch.no_grad():
                samples = strategy.generate(model, model_input, num_steps, device)

        all_samples.append(samples.cpu())

        done = min(i + batch_size, len(seg_masks))
        if done % max(batch_size * 5, 1) == 0 or done == len(seg_masks):
            logger.info("Generated %d/%d samples", done, len(seg_masks))

    result = torch.cat(all_samples, dim=0)
    logger.info("Generation complete: %s", result.shape)
    return result


def generate_unconditioned_samples(
    model: nn.Module,
    strategy: RFlowStrategy,
    num_samples: int,
    sample_shape: tuple[int, ...],
    num_steps: int,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """Generate samples without conditioning.

    Args:
        model: Trained diffusion model.
        strategy: RFlowStrategy with scheduler.
        num_samples: Number to generate.
        sample_shape: Shape of a single sample [C, H, W] or [C, D, H, W].
        num_steps: Denoising steps.
        device: Computation device.
        batch_size: Generation batch size.

    Returns:
        Generated samples [N, C, ...].
    """
    all_samples = []
    remaining = num_samples

    logger.info("Generating %d unconditioned samples...", num_samples)

    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)
            noise = torch.randn(bs, *sample_shape, device=device)

            with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                samples = strategy.generate(model, noise, num_steps, device)

            all_samples.append(samples.cpu())
            remaining -= bs

            if len(all_samples) % 10 == 0:
                logger.info("Generated %d/%d samples", num_samples - remaining, num_samples)

    result = torch.cat(all_samples, dim=0)[:num_samples]
    logger.info("Generation complete: %s", result.shape)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Head training
# ═══════════════════════════════════════════════════════════════════════════════

def _save_head_checkpoint(
    head: DiffRSHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    save_path: Path,
    metadata: dict,
) -> None:
    """Save a DiffRS head checkpoint."""
    torch.save({
        'head_state_dict': head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
        **metadata,
    }, save_path)


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
    real_conditioning: torch.Tensor | None = None,
    generated_conditioning: torch.Tensor | None = None,
    val_real_samples: torch.Tensor | None = None,
    val_generated_samples: torch.Tensor | None = None,
    val_real_conditioning: torch.Tensor | None = None,
    val_generated_conditioning: torch.Tensor | None = None,
    output_dir: Path | None = None,
    checkpoint_metadata: dict | None = None,
) -> DiffRSHead:
    """Train the DiffRS discriminator head.

    For conditioned models, real_conditioning and generated_conditioning
    provide the seg masks that are concatenated with noisy images before
    extracting encoder features (matching the model's expected input).

    Saves checkpoints and logs to TensorBoard throughout training.

    Args:
        model: Frozen diffusion model (for feature extraction).
        strategy: RFlowStrategy (for timestep sampling).
        head: DiffRSHead to train.
        real_samples: Real training samples [N, C, ...].
        generated_samples: Generated samples [N, C, ...].
        device: Computation device.
        num_epochs: Training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        real_conditioning: Conditioning for real samples [N, C_cond, ...] or None.
        generated_conditioning: Conditioning for generated samples [N, C_cond, ...] or None.
        val_real_samples: Validation real samples (optional).
        val_generated_samples: Validation generated samples (optional).
        val_real_conditioning: Conditioning for val real samples (optional).
        val_generated_conditioning: Conditioning for val generated samples (optional).
        output_dir: Directory for checkpoints and TensorBoard logs.
        checkpoint_metadata: Extra metadata to include in checkpoint files.

    Returns:
        Trained DiffRSHead.
    """
    from torch.utils.tensorboard import SummaryWriter

    optimizer = torch.optim.Adam(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    num_real = len(real_samples)
    num_gen = len(generated_samples)
    num_total = num_real + num_gen
    metadata = checkpoint_metadata or {}

    # Combine data and free originals to reduce peak RAM
    # For 3D: 105 volumes × 42MB = 4.4GB per tensor, so freeing matters
    all_data = torch.cat([real_samples, generated_samples], dim=0)
    labels = torch.cat([torch.ones(num_real), torch.zeros(num_gen)])
    del real_samples, generated_samples

    # Combine conditioning if present
    has_cond = (real_conditioning is not None and generated_conditioning is not None)
    if has_cond:
        all_cond = torch.cat([real_conditioning, generated_conditioning], dim=0)
        del real_conditioning, generated_conditioning
    else:
        all_cond = None

    # Validation data (optional)
    has_val = (val_real_samples is not None and val_generated_samples is not None)
    if has_val:
        num_val_real = len(val_real_samples)
        num_val_gen = len(val_generated_samples)
        num_val_total = num_val_real + num_val_gen
        val_data = torch.cat([val_real_samples, val_generated_samples], dim=0)
        val_labels = torch.cat([torch.ones(num_val_real), torch.zeros(num_val_gen)])
        del val_real_samples, val_generated_samples
        has_val_cond = (val_real_conditioning is not None and val_generated_conditioning is not None)
        if has_val_cond:
            val_cond = torch.cat([val_real_conditioning, val_generated_conditioning], dim=0)
            del val_real_conditioning, val_generated_conditioning
        else:
            val_cond = None
    else:
        val_data = val_labels = val_cond = None
        num_val_total = 0

    import gc
    gc.collect()

    # Set up output directory, TensorBoard, and resume support
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    else:
        writer = None

    # Resume from checkpoint if it exists
    start_epoch = 0
    best_loss = float('inf')
    if output_dir is not None:
        latest_path = output_dir / "checkpoint_latest.pt"
        if latest_path.exists():
            ckpt = torch.load(latest_path, map_location=device, weights_only=True)
            head.load_state_dict(ckpt['head_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt.get('best_loss', float('inf'))
            logger.info("Resumed from epoch %d (best_loss=%.4f)", start_epoch, best_loss)

    head.train()
    model.eval()

    logger.info(
        "Training DiffRS head: %d real + %d generated, epochs %d-%d, lr=%.1e, conditioning=%s",
        num_real, num_gen, start_epoch + 1, num_epochs, learning_rate, has_cond,
    )
    if has_val:
        logger.info("Validation: %d real + %d generated = %d total", num_val_real, num_val_gen, num_val_total)

    for epoch in range(start_epoch, num_epochs):
        # Shuffle indices only — avoids copying the entire dataset
        perm = torch.randperm(num_total)

        epoch_loss = 0.0
        epoch_correct = 0
        num_batches = 0

        for i in range(0, num_total, batch_size):
            batch_idx = perm[i:i + batch_size]
            batch_x = all_data[batch_idx].to(device)
            batch_y = labels[batch_idx].to(device)

            # Sample random timesteps
            timesteps = strategy.sample_timesteps(batch_x)

            # Forward-diffuse image to timestep t
            noise = torch.randn_like(batch_x)
            noisy = strategy.scheduler.add_noise(batch_x, noise, timesteps)

            # Build encoder input: [noisy_image, conditioning] for conditioned models
            if all_cond is not None:
                batch_cond = all_cond[batch_idx].to(device)
                encoder_input = torch.cat([noisy, batch_cond], dim=1)
            else:
                encoder_input = noisy

            # Extract features from frozen encoder
            with torch.no_grad():
                features = extract_encoder_features(model, encoder_input, timesteps)

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

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('train/loss', avg_loss, epoch)
            writer.add_scalar('train/accuracy', accuracy, epoch)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation loop
        val_loss = None
        val_accuracy = None
        if has_val:
            head.eval()
            val_epoch_loss = 0.0
            val_epoch_correct = 0
            val_num_batches = 0

            with torch.no_grad():
                for i in range(0, num_val_total, batch_size):
                    batch_x = val_data[i:i + batch_size].to(device)
                    batch_y = val_labels[i:i + batch_size].to(device)

                    timesteps = strategy.sample_timesteps(batch_x)
                    noise = torch.randn_like(batch_x)
                    noisy = strategy.scheduler.add_noise(batch_x, noise, timesteps)

                    if val_cond is not None:
                        batch_cond = val_cond[i:i + batch_size].to(device)
                        encoder_input = torch.cat([noisy, batch_cond], dim=1)
                    else:
                        encoder_input = noisy

                    features = extract_encoder_features(model, encoder_input, timesteps)
                    logits = head(features.float())
                    loss = criterion(logits, batch_y)

                    val_epoch_loss += loss.item()
                    val_epoch_correct += ((logits > 0).float() == batch_y).sum().item()
                    val_num_batches += 1

            val_loss = val_epoch_loss / max(val_num_batches, 1)
            val_accuracy = val_epoch_correct / num_val_total * 100
            head.train()

            if writer is not None:
                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar('val/accuracy', val_accuracy, epoch)

        # Use val_loss for best model selection when available, else train loss
        selection_loss = val_loss if val_loss is not None else avg_loss

        # Console logging
        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            msg = f"Epoch {epoch + 1}/{num_epochs}: loss={avg_loss:.4f}, acc={accuracy:.1f}%"
            if val_loss is not None:
                msg += f" | val_loss={val_loss:.4f}, val_acc={val_accuracy:.1f}%"
            logger.info(msg)

        # Checkpointing
        if output_dir is not None:
            # Save latest every epoch
            _save_head_checkpoint(
                head, optimizer, epoch, best_loss,
                output_dir / "checkpoint_latest.pt", metadata,
            )

            # Save best if improved
            if selection_loss < best_loss:
                best_loss = selection_loss
                _save_head_checkpoint(
                    head, optimizer, epoch, best_loss,
                    output_dir / "checkpoint_best.pt", metadata,
                )
                loss_type = "val_loss" if val_loss is not None else "train_loss"
                logger.info("  New best %s: %.4f (epoch %d)", loss_type, best_loss, epoch + 1)

            # Periodic checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                _save_head_checkpoint(
                    head, optimizer, epoch, best_loss,
                    output_dir / f"checkpoint_epoch{epoch + 1:03d}.pt", metadata,
                )

    if writer is not None:
        writer.close()

    head.eval()
    logger.info("Training complete. Best loss: %.4f", best_loss)
    return head


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

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

    # Load diffusion model (channels inferred from checkpoint weight shapes)
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

    # Detect model type
    unet = _unwrap_to_unet(model)
    in_channels = unet.in_channels
    out_channels = unet.out_channels
    has_conditioning = (in_channels > out_channels)

    spatial_dims = cfg.get('spatial_dims', 2)
    image_size = cfg.get('image_size', 128)
    depth = cfg.get('depth', 160)

    # Set up strategy
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000,
        image_size=image_size,
        depth_size=depth if spatial_dims == 3 else None,
        spatial_dims=spatial_dims,
    )

    # Map mode to image type
    mode = cfg.get('data_mode', 'bravo')
    mode_to_type = {
        'bravo': 'bravo', 'seg': 'seg',
        'seg_conditioned': 'seg', 'dual': 'bravo',
    }
    image_type = mode_to_type.get(mode, 'bravo')

    num_samples = cfg.num_generated_samples
    val_split = cfg.get('val_split', 'test')

    if has_conditioning:
        logger.info(
            "Conditioned model (in=%d, out=%d): using paired data + conditioned generation",
            in_channels, out_channels,
        )

        # --- Training split ---
        if spatial_dims == 3:
            real_images, real_segs = load_paired_volumes_3d(
                data_dir=cfg.paths.data_dir,
                image_type=image_type,
                num_volumes=num_samples,
                depth=depth,
                split="train",
            )
        else:
            real_images, real_segs = load_paired_slices_2d(
                data_dir=cfg.paths.data_dir,
                image_type=image_type,
                num_samples=num_samples,
                split="train",
            )

        num_train = len(real_images)
        logger.info("Training paired data: %d samples", num_train)

        generated = generate_conditioned_samples(
            model=model,
            strategy=strategy,
            seg_masks=real_segs,
            out_channels=out_channels,
            num_steps=cfg.generation_num_steps,
            device=device,
            batch_size=cfg.batch_size,
        )

        real_conditioning = real_segs
        generated_conditioning = real_segs

        # --- Validation split (optional) ---
        val_real_images = val_generated = val_real_conditioning = val_generated_conditioning = None
        if val_split:
            if spatial_dims == 3:
                val_real_images, val_real_segs = load_paired_volumes_3d(
                    data_dir=cfg.paths.data_dir,
                    image_type=image_type,
                    num_volumes=num_samples,
                    depth=depth,
                    split=val_split,
                )
            else:
                val_real_images, val_real_segs = load_paired_slices_2d(
                    data_dir=cfg.paths.data_dir,
                    image_type=image_type,
                    num_samples=num_samples,
                    split=val_split,
                )

            logger.info("Validation paired data: %d samples", len(val_real_images))

            val_generated = generate_conditioned_samples(
                model=model,
                strategy=strategy,
                seg_masks=val_real_segs,
                out_channels=out_channels,
                num_steps=cfg.generation_num_steps,
                device=device,
                batch_size=cfg.batch_size,
            )

            val_real_conditioning = val_real_segs
            val_generated_conditioning = val_real_segs
    else:
        logger.info("Unconditioned model (in=%d, out=%d)", in_channels, out_channels)

        # Determine sample shape
        if spatial_dims == 3:
            sample_shape = (out_channels, depth, image_size, image_size)
        else:
            sample_shape = (out_channels, image_size, image_size)

        # --- Training split ---
        real_images = load_unconditioned_samples(
            data_dir=cfg.paths.data_dir,
            image_type=image_type,
            image_size=image_size,
            num_samples=num_samples,
            spatial_dims=spatial_dims,
            split="train",
        )

        num_train = len(real_images)

        generated = generate_unconditioned_samples(
            model=model,
            strategy=strategy,
            num_samples=num_train,
            sample_shape=sample_shape,
            num_steps=cfg.generation_num_steps,
            device=device,
            batch_size=cfg.batch_size,
        )

        real_conditioning = None
        generated_conditioning = None

        # --- Validation split (optional) ---
        val_real_images = val_generated = val_real_conditioning = val_generated_conditioning = None
        if val_split:
            val_real_images = load_unconditioned_samples(
                data_dir=cfg.paths.data_dir,
                image_type=image_type,
                image_size=image_size,
                num_samples=num_samples,
                spatial_dims=spatial_dims,
                split=val_split,
            )

            val_generated = generate_unconditioned_samples(
                model=model,
                strategy=strategy,
                num_samples=len(val_real_images),
                sample_shape=sample_shape,
                num_steps=cfg.generation_num_steps,
                device=device,
                batch_size=cfg.batch_size,
            )

            val_real_conditioning = None
            val_generated_conditioning = None

    # Create head
    bottleneck_channels = get_bottleneck_channels(model)
    mid_channels = cfg.get('mid_channels', 128)
    head = DiffRSHead(
        in_channels=bottleneck_channels,
        spatial_dims=spatial_dims,
        mid_channels=mid_channels,
    ).to(device)

    num_params = sum(p.numel() for p in head.parameters())
    logger.info(
        "DiffRS head: in_channels=%d, mid_channels=%d, spatial_dims=%d, params=%d",
        bottleneck_channels, mid_channels, spatial_dims, num_params,
    )

    # Train
    output_dir = Path(cfg.get('output_dir', 'runs/diffrs'))
    checkpoint_metadata = {
        'in_channels': bottleneck_channels,
        'mid_channels': mid_channels,
        'spatial_dims': spatial_dims,
        'model_in_channels': in_channels,
        'model_out_channels': out_channels,
        'diffusion_checkpoint': cfg.diffusion_checkpoint,
        'num_epochs': cfg.num_epochs,
        'num_samples': num_samples,
        'has_conditioning': has_conditioning,
    }

    # Free the diffusion model before training — only the encoder features are
    # needed, and extract_encoder_features accesses it via the model reference.
    # But we DO still need the model for feature extraction, so don't delete it.
    # Instead, free data that gets duplicated inside train_head.
    import gc
    gc.collect()

    head = train_head(
        model=model,
        strategy=strategy,
        head=head,
        real_samples=real_images,
        generated_samples=generated,
        device=device,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        real_conditioning=real_conditioning,
        generated_conditioning=generated_conditioning,
        val_real_samples=val_real_images,
        val_generated_samples=val_generated,
        val_real_conditioning=val_real_conditioning,
        val_generated_conditioning=val_generated_conditioning,
        output_dir=output_dir,
        checkpoint_metadata=checkpoint_metadata,
    )

    # Save convenience alias (used by eval_diffrs.py)
    save_path = output_dir / "diffrs_head.pt"
    best_path = output_dir / "checkpoint_best.pt"
    if best_path.exists():
        import shutil
        shutil.copy2(best_path, save_path)
    else:
        torch.save({
            'head_state_dict': head.state_dict(),
            **checkpoint_metadata,
        }, save_path)

    logger.info("DiffRS head saved to %s", save_path)


if __name__ == "__main__":
    main()
