"""
Learning Rate Finder for diffusion models and VAE.

Runs a learning rate range test to find optimal learning rates
by sweeping through a range and plotting loss vs learning rate.

Usage:
    # Diffusion model
    python -m medgen.scripts.lr_finder mode=dual strategy=rflow

    # VAE
    python -m medgen.scripts.lr_finder mode=dual model_type=vae model.image_size=128

    # Custom LR range
    python -m medgen.scripts.lr_finder min_lr=1e-8 max_lr=1e-2 num_steps=300
"""
import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast
from torch.optim import AdamW
from tqdm import tqdm

from monai.losses import PerceptualLoss
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet

from medgen.core import DEFAULT_DUAL_IMAGE_KEYS, ModeType, setup_cuda_optimizations
from medgen.data import create_dataloader, create_dual_image_dataloader, create_vae_dataloader
from medgen.diffusion.modes import ConditionalDualMode, ConditionalSingleMode, SegmentationMode
from medgen.diffusion.strategies import DDPMStrategy, RFlowStrategy

setup_cuda_optimizations()

log = logging.getLogger(__name__)


def find_lr_diffusion(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    strategy,
    mode,
    perceptual_loss_fn: nn.Module,
    device: torch.device,
    min_lr: float = 1e-7,
    max_lr: float = 1e-1,
    num_steps: int = 200,
    perceptual_weight: float = 0.001,
    smoothing: float = 0.05,
    diverge_th: float = 5.0,
) -> Tuple[List[float], List[float]]:
    """Run LR range test for diffusion model.

    Args:
        model: Diffusion model.
        dataloader: Training dataloader.
        strategy: Diffusion strategy (DDPM or RFlow).
        mode: Training mode (seg, bravo, dual).
        perceptual_loss_fn: Perceptual loss function.
        device: Device to run on.
        min_lr: Minimum learning rate.
        max_lr: Maximum learning rate.
        num_steps: Number of LR steps.
        perceptual_weight: Weight for perceptual loss.
        smoothing: Exponential smoothing factor.
        diverge_th: Stop if loss > diverge_th * best_loss.

    Returns:
        (learning_rates, losses)
    """
    optimizer = AdamW(model.parameters(), lr=min_lr)
    lr_mult = (max_lr / min_lr) ** (1.0 / num_steps)

    learning_rates: List[float] = []
    losses: List[float] = []
    smoothed_loss: Optional[float] = None
    best_loss = float('inf')

    model.train()
    data_iter = iter(dataloader)

    pbar = tqdm(range(num_steps), desc="LR Finder (Diffusion)")

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            prepared = mode.prepare_batch(batch, device)
            images = prepared['images']
            labels_dict = {'labels': prepared.get('labels')}

            if isinstance(images, dict):
                noise = {key: torch.randn_like(img).to(device) for key, img in images.items()}
            else:
                noise = torch.randn_like(images).to(device)

            timesteps = strategy.sample_timesteps(images)
            noisy_images = strategy.add_noise(images, noise, timesteps)
            model_input = mode.format_model_input(noisy_images, labels_dict)
            prediction = strategy.predict_noise_or_velocity(model, model_input, timesteps)
            mse_loss, predicted_clean = strategy.compute_loss(prediction, images, noise, noisy_images, timesteps)

            if isinstance(predicted_clean, dict):
                p_loss = sum(
                    perceptual_loss_fn(pred.float(), images[key].float())
                    for key, pred in predicted_clean.items()
                ) / len(predicted_clean)
            else:
                p_loss = perceptual_loss_fn(predicted_clean.float(), images.float())

            total_loss = mse_loss + perceptual_weight * p_loss

        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()

        if smoothed_loss is None:
            smoothed_loss = loss_val
        else:
            smoothed_loss = smoothing * loss_val + (1 - smoothing) * smoothed_loss

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        if step > 10 and smoothed_loss > diverge_th * best_loss:
            log.info(f"Stopping early: loss diverged at LR={optimizer.param_groups[0]['lr']:.2e}")
            break

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        losses.append(smoothed_loss)

        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult

        pbar.set_postfix(lr=f"{current_lr:.2e}", loss=f"{smoothed_loss:.4f}")

    return learning_rates, losses


def find_lr_vae(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    perceptual_loss_fn: nn.Module,
    device: torch.device,
    min_lr: float = 1e-7,
    max_lr: float = 1e-1,
    num_steps: int = 200,
    kl_weight: float = 1e-8,
    perceptual_weight: float = 0.002,
    smoothing: float = 0.05,
    diverge_th: float = 5.0,
) -> Tuple[List[float], List[float]]:
    """Run LR range test for VAE (without GAN for stability).

    Uses only reconstruction + perceptual + KL loss to find optimal LR.
    GAN component is excluded to avoid instability during LR search.

    Args:
        model: AutoencoderKL model.
        dataloader: Training dataloader.
        perceptual_loss_fn: Perceptual loss function.
        device: Device to run on.
        min_lr: Minimum learning rate.
        max_lr: Maximum learning rate.
        num_steps: Number of LR steps.
        kl_weight: KL divergence weight.
        perceptual_weight: Perceptual loss weight.
        smoothing: Exponential smoothing factor.
        diverge_th: Stop if loss > diverge_th * best_loss.

    Returns:
        (learning_rates, losses)
    """
    optimizer = AdamW(model.parameters(), lr=min_lr)
    lr_mult = (max_lr / min_lr) ** (1.0 / num_steps)

    learning_rates: List[float] = []
    losses: List[float] = []
    smoothed_loss: Optional[float] = None
    best_loss = float('inf')

    model.train()
    data_iter = iter(dataloader)

    def compute_kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=[1, 2, 3])
        return kl.mean()

    pbar = tqdm(range(num_steps), desc="LR Finder (VAE)")

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            reconstruction, mean, logvar = model(images)
            l1_loss = torch.abs(reconstruction - images).mean()

            # Perceptual loss requires 3 channels - adapt if needed
            n_ch = images.shape[1]
            if n_ch == 3:
                p_loss = perceptual_loss_fn(reconstruction.float(), images.float())
            elif n_ch < 3:
                # Repeat channels to get 3
                rec_3ch = reconstruction.float().repeat(1, 3 // n_ch + 1, 1, 1)[:, :3]
                img_3ch = images.float().repeat(1, 3 // n_ch + 1, 1, 1)[:, :3]
                p_loss = perceptual_loss_fn(rec_3ch, img_3ch)
            else:
                # More than 3 channels - use first 3
                p_loss = perceptual_loss_fn(reconstruction[:, :3].float(), images[:, :3].float())

            kl_loss = compute_kl_loss(mean, logvar)
            total_loss = l1_loss + perceptual_weight * p_loss + kl_weight * kl_loss

        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()

        if smoothed_loss is None:
            smoothed_loss = loss_val
        else:
            smoothed_loss = smoothing * loss_val + (1 - smoothing) * smoothed_loss

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        if step > 10 and smoothed_loss > diverge_th * best_loss:
            log.info(f"Stopping early: loss diverged at LR={optimizer.param_groups[0]['lr']:.2e}")
            break

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        losses.append(smoothed_loss)

        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult

        pbar.set_postfix(lr=f"{current_lr:.2e}", loss=f"{smoothed_loss:.4f}")

    return learning_rates, losses


def suggest_lr(losses: List[float], lrs: List[float], div_factor: float = 10.0) -> float:
    """Suggest optimal LR using 10x before divergence rule.

    Finds where loss starts diverging (loss > 2x minimum) and suggests
    an LR that is div_factor times smaller than the divergence point.

    Args:
        losses: Smoothed loss values.
        lrs: Corresponding learning rates.
        div_factor: How much smaller than divergence point (default 10x).

    Returns:
        Suggested learning rate.
    """
    if len(losses) < 10:
        return lrs[len(lrs) // 2]

    losses_arr = np.array(losses)
    lrs_arr = np.array(lrs)

    # Find minimum loss and its index
    min_loss = losses_arr.min()
    min_idx = int(np.argmin(losses_arr))

    # Find divergence point: where loss > 2x minimum (after the minimum)
    diverge_threshold = min_loss * 2.0
    diverge_idx = None
    for i in range(min_idx, len(losses_arr)):
        if losses_arr[i] > diverge_threshold:
            diverge_idx = i
            break

    if diverge_idx is None:
        # No divergence found, use minimum point
        diverge_idx = len(losses_arr) - 1

    # Get LR at divergence point and divide by div_factor
    diverge_lr = lrs_arr[diverge_idx]
    suggested_lr = diverge_lr / div_factor

    # Clamp to valid range
    suggested_lr = max(suggested_lr, lrs_arr[0])
    suggested_lr = min(suggested_lr, lrs_arr[-1])

    return float(suggested_lr)


def plot_lr_finder_diffusion(
    lrs: List[float],
    losses: List[float],
    save_path: str,
) -> float:
    """Plot LR finder results for diffusion model."""
    suggested_lr = suggest_lr(losses, lrs)

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses, 'b-', linewidth=2)
    plt.axvline(x=suggested_lr, color='r', linestyle='--', linewidth=2,
                label=f'Suggested: {suggested_lr:.2e}')
    plt.xscale('log')
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Loss (smoothed)', fontsize=12)
    plt.title('LR Finder - Diffusion Model', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return suggested_lr


def plot_lr_finder_vae(
    lrs: List[float],
    losses: List[float],
    save_path: str,
) -> float:
    """Plot LR finder results for VAE."""
    suggested_lr = suggest_lr(losses, lrs)

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses, 'b-', linewidth=2)
    plt.axvline(x=suggested_lr, color='r', linestyle='--', linewidth=2,
                label=f'Suggested: {suggested_lr:.2e}')
    plt.xscale('log')
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Loss (smoothed)', fontsize=12)
    plt.title('LR Finder - VAE (L1 + Perceptual + KL)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return suggested_lr


@hydra.main(version_base=None, config_path="../../../configs", config_name="lr_finder")
def main(cfg: DictConfig) -> None:
    """Run LR finder."""
    device = torch.device('cuda')

    # LR finder config
    model_type = cfg.model_type
    min_lr = cfg.min_lr
    max_lr = cfg.max_lr
    num_steps = cfg.num_steps

    mode_name = cfg.mode.name
    strategy_name = cfg.get('strategy', {}).get('name', 'ddpm')
    log.info(f"LR Finder Configuration:")
    log.info(f"  Model type: {model_type}")
    log.info(f"  Mode: {mode_name}")
    log.info(f"  LR range: {min_lr:.2e} - {max_lr:.2e}")
    log.info(f"  Steps: {num_steps}")

    # Create dataloader and determine channels based on model type
    if model_type == 'vae':
        # VAE uses create_vae_dataloader (no seg concatenation)
        # Override in_channels like train_vae.py does
        if mode_name == ModeType.DUAL:
            n_channels = 2  # t1_pre + t1_gd, NO seg
        else:
            n_channels = 1  # bravo, seg, t1_pre, t1_gd individually
        out_channels = n_channels

        dataloader, _ = create_vae_dataloader(
            cfg=cfg, modality=mode_name,
            use_distributed=False, rank=0, world_size=1
        )
    else:
        # Diffusion uses regular dataloaders with seg conditioning
        n_channels = cfg.mode.get('in_channels', 1)
        out_channels = cfg.mode.get('out_channels', 1)

        if mode_name == ModeType.DUAL:
            image_keys = list(cfg.mode.image_keys) if 'image_keys' in cfg.mode else DEFAULT_DUAL_IMAGE_KEYS
            dataloader, _ = create_dual_image_dataloader(
                cfg=cfg, image_keys=image_keys, conditioning='seg',
                use_distributed=False, rank=0, world_size=1
            )
        else:
            image_type = 'seg' if mode_name == ModeType.SEG else 'bravo'
            dataloader, _ = create_dataloader(
                cfg=cfg, image_type=image_type,
                use_distributed=False, rank=0, world_size=1
            )

    if model_type == 'vae':
        # VAE LR finder (without GAN for stability)
        vae_cfg = cfg.get('vae', {})

        model = AutoencoderKL(
            spatial_dims=2,
            in_channels=n_channels,
            out_channels=n_channels,
            channels=tuple(vae_cfg.get('channels', [64, 128, 256, 512])),
            attention_levels=tuple(vae_cfg.get('attention_levels', [False, False, False, True])),
            latent_channels=vae_cfg.get('latent_channels', 3),
            num_res_blocks=vae_cfg.get('num_res_blocks', 2),
            norm_num_groups=32,
        ).to(device)

        perceptual_loss_fn = PerceptualLoss(
            spatial_dims=2, network_type="squeeze", is_fake_3d=False
        ).to(device)

        print(f"\n{'=' * 60}")
        print(f"LR Finder - VAE ({mode_name} mode)")
        print(f"Image size: {cfg.model.image_size} | Channels: {n_channels}")
        print(f"VAE params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Note: GAN disabled for stable LR finding")
        print(f"{'=' * 60}\n")

        lrs, losses = find_lr_vae(
            model=model,
            dataloader=dataloader,
            perceptual_loss_fn=perceptual_loss_fn,
            device=device,
            min_lr=min_lr,
            max_lr=max_lr,
            num_steps=num_steps,
            kl_weight=vae_cfg.get('kl_weight', 1e-8),
            perceptual_weight=vae_cfg.get('perceptual_weight', 0.002),
        )

        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        plot_path = os.path.join(output_dir, 'lr_finder.png')
        suggested_lr = plot_lr_finder_vae(lrs, losses, plot_path)

        # Discriminator LR is typically 5x generator LR
        suggested_disc_lr = suggested_lr * 5

        print(f"\n{'=' * 60}")
        print(f"Results:")
        print(f"  VAE LR:           {suggested_lr:.2e}")
        print(f"  Discriminator LR: {suggested_disc_lr:.2e} (5x VAE)")
        print(f"{'=' * 60}")
        print(f"\nPlot saved to: {plot_path}")
        print(f"\nSuggested command:")
        print(f"  python -m medgen.scripts.train_vae mode={mode_name} \\")
        print(f"    training.learning_rate={suggested_lr:.2e} \\")
        print(f"    vae.disc_lr={suggested_disc_lr:.2e}")

    else:
        # Diffusion LR finder
        modes = {
            'seg': SegmentationMode,
            'bravo': ConditionalSingleMode,
            'dual': ConditionalDualMode
        }
        mode = modes[mode_name]()

        if strategy_name == 'rflow':
            strategy = RFlowStrategy()
        else:
            strategy = DDPMStrategy()

        num_timesteps = cfg.get('strategy', {}).get('num_train_timesteps', 1000)
        strategy.setup_scheduler(num_timesteps, cfg.model.image_size)

        model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=n_channels,
            out_channels=out_channels,
            channels=tuple(cfg.model.get('channels', [128, 256, 256])),
            attention_levels=tuple(cfg.model.get('attention_levels', [False, True, True])),
            num_res_blocks=cfg.model.get('num_res_blocks', 1),
            num_head_channels=cfg.model.get('num_head_channels', 256),
        ).to(device)

        perceptual_loss_fn = PerceptualLoss(
            spatial_dims=2, network_type="squeeze", is_fake_3d=False
        ).to(device)

        print(f"\n{'=' * 60}")
        print(f"LR Finder - Diffusion ({mode_name} mode, {strategy_name})")
        print(f"Image size: {cfg.model.image_size}")
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"{'=' * 60}\n")

        lrs, losses = find_lr_diffusion(
            model=model,
            dataloader=dataloader,
            strategy=strategy,
            mode=mode,
            perceptual_loss_fn=perceptual_loss_fn,
            device=device,
            min_lr=min_lr,
            max_lr=max_lr,
            num_steps=num_steps,
            perceptual_weight=cfg.training.get('perceptual_weight', 0.001),
        )

        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        plot_path = os.path.join(output_dir, 'lr_finder.png')
        suggested_lr = plot_lr_finder_diffusion(lrs, losses, plot_path)

        print(f"\n{'=' * 60}")
        print(f"Results:")
        print(f"  Suggested LR: {suggested_lr:.2e}")
        print(f"{'=' * 60}")
        print(f"\nPlot saved to: {plot_path}")
        print(f"\nSuggested command:")
        print(f"  python -m medgen.scripts.train mode={mode_name} strategy={strategy_name} \\")
        print(f"    training.learning_rate={suggested_lr:.2e}")


if __name__ == "__main__":
    main()
