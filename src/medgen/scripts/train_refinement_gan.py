#!/usr/bin/env python3
"""Pix2Pix refinement GAN training (exp42).

Trains a dedicated post-hoc refinement generator that maps synthetic diffusion
outputs (exp1_1_1000 / exp37_3) toward real brain MRI. NOT a diffusion model —
one forward pass per volume, trained with L1 + 2.5D LPIPS + 3D PatchGAN.

Usage:
    python -m medgen.scripts.train_refinement_gan \\
        --data-root /path/to/restoration_pairs_real \\
        --output-dir /path/to/runs/exp42_... \\
        --epochs 100 --batch-size 1

Inputs expected from exp41b_pregen:
    <data_root>/train/<subj>/
        clean_bravo.nii.gz
        seg.nii.gz
        degraded_000.nii.gz
        degraded_001.nii.gz
        degraded_002.nii.gz
    <data_root>/val/<subj>/... (same structure)

At each step the dataset randomly picks one of the precomputed degraded
variants, so the network sees all 3 variants per subject across epochs.
"""
import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ema_pytorch import EMA
from monai.networks.nets import DiffusionModelUNet
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from medgen.data.loaders.restoration_3d import Restoration3DDataset
from medgen.losses.perceptual_manager import PerceptualLossManager
from medgen.pipeline.discriminator_manager import DiscriminatorManager
from medgen.pipeline.utils import save_full_checkpoint

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp42 Pix2Pix refinement GAN training")
    # Data
    parser.add_argument('--data-root', required=True,
                        help='Root of paired data (contains train/ and val/ subdirs)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--augmentation-level', default='heavy',
                        choices=['basic', 'medium', 'heavy'])
    parser.add_argument('--num-workers', type=int, default=2)
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr-g', type=float, default=1.0e-4)
    parser.add_argument('--lr-d', type=float, default=2.0e-4,  # TTUR
                        help='Discriminator LR (2× G LR is a common GAN heuristic)')
    parser.add_argument('--warmup-epochs', type=int, default=3)
    parser.add_argument('--gradient-clip-norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    # Loss weights
    parser.add_argument('--lambda-l1', type=float, default=100.0)
    parser.add_argument('--lambda-perc', type=float, default=10.0)
    parser.add_argument('--adv-weight', type=float, default=1.0)
    # GAN stability
    parser.add_argument('--d-warmup-steps', type=int, default=500,
                        help='Steps of D-alone training before G sees adversarial gradient')
    parser.add_argument('--adv-ramp-steps', type=int, default=2000,
                        help='Steps over which adversarial weight ramps 0→1')
    parser.add_argument('--disc-num-layers', type=int, default=3)
    parser.add_argument('--disc-num-channels', type=int, default=32)
    parser.add_argument('--r1-weight', type=float, default=0.0,
                        help='R1 gradient penalty weight on D (0 = disabled). '
                             'Penalizes |∇D(real)|² to prevent D saturation. '
                             '~0.1 is a typical starting value.')
    parser.add_argument('--r1-every', type=int, default=4,
                        help='Apply R1 penalty every N D steps (lazy R1, common '
                             'pattern that halves R1 cost without hurting effect).')
    parser.add_argument('--use-residual-learning', action='store_true',
                        help='Predict residual (clean - degraded) instead of full clean. '
                             'DnCNN-style. Effective for paired deblur because the '
                             'network only needs to learn the small delta. At inference: '
                             'refined = degraded + G(degraded).')
    parser.add_argument('--use-feature-matching', action='store_true',
                        help='Add Pix2PixHD-style feature-matching loss: L1 between '
                             'intermediate D features (multi-scale) on real vs fake. '
                             'Stabilizes adversarial training.')
    parser.add_argument('--lambda-fm', type=float, default=10.0,
                        help='Feature-matching loss weight. Only used if '
                             '--use-feature-matching is set.')
    parser.add_argument('--extra-blur-sigma-max', type=float, default=0.0,
                        help='If >0, apply random Gaussian blur to the degraded '
                             'input only (training only, not val). σ uniform in '
                             '[0, this]. 0 = disabled. Use ~0.4 to approximate '
                             'diffusion-MSE blur on top of VQ-VAE compression.')
    parser.add_argument('--extra-noise-std-max', type=float, default=0.0,
                        help='If >0, add Gaussian noise (training only) with std '
                             'uniform in [0, this]. 0 = disabled. Use ~0.015 to '
                             'simulate residual diffusion-sampler noise.')
    parser.add_argument('--extra-aug-prob', type=float, default=0.5,
                        help='Per-step probability of applying each extra aug.')
    parser.add_argument('--ema-beta', type=float, default=0.9999)
    parser.add_argument('--use-compile', action='store_true')
    # Logging / checkpointing
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--val-every-epoch', type=int, default=1)
    parser.add_argument('--save-every-epoch', type=int, default=5)
    # Resume
    parser.add_argument('--resume-from', default=None,
                        help='Checkpoint to resume training from')
    # Quick dry-run flags
    parser.add_argument('--max-steps-per-epoch', type=int, default=0,
                        help='0 = all batches; otherwise cap for debugging')
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────
# Model construction
# ────────────────────────────────────────────────────────────────
def build_generator(
    spatial_dims: int,
    device: torch.device,
) -> nn.Module:
    """Instantiate the refinement generator.

    Architecture: matches `configs/model/default_3d.yaml` (~270M params), the
    same backbone used by the exp1_1_1000 bravo model. 6-level U-Net with
    attention at the two deepest levels (L4, L5) and narrow first level (16ch)
    to keep memory manageable at full 256×256×160.

    We pass a constant t=0 tensor at forward time; the FiLM time embedding
    becomes a constant bias that the network absorbs as a no-op over training.
    """
    model = DiffusionModelUNet(
        spatial_dims=spatial_dims,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 256, 512, 512),
        attention_levels=(False, False, False, False, True, True),
        num_res_blocks=(1, 1, 1, 2, 2, 2),
        num_head_channels=256,
        norm_num_groups=16,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Generator params: {n_params:,}")
    return model


# ────────────────────────────────────────────────────────────────
# Extra training-time augmentation on degraded input
# ────────────────────────────────────────────────────────────────
def _gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D Gaussian blur on a 5D tensor [B, C, D, H, W]."""
    if sigma <= 0:
        return x
    import math
    radius = max(1, math.ceil(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()

    def _conv_axis(vol: torch.Tensor, axis: int) -> torch.Tensor:
        shape = [1, 1, 1, 1, 1]
        shape[axis] = kernel.numel()
        k = kernel.view(*shape)
        pad = [0, 0, 0, 0, 0, 0]  # W_left, W_right, H_left, H_right, D_left, D_right
        idx = (4 - axis) * 2  # axis=2 (D) → pad[4:6]; axis=3 (H) → pad[2:4]; axis=4 (W) → pad[0:2]
        pad[idx] = radius
        pad[idx + 1] = radius
        vol = torch.nn.functional.pad(vol, pad, mode='reflect')
        return torch.nn.functional.conv3d(vol, k)

    out = _conv_axis(x, 2)
    out = _conv_axis(out, 3)
    out = _conv_axis(out, 4)
    return out


def maybe_apply_extra_aug(
    degraded: torch.Tensor,
    blur_sigma_max: float,
    noise_std_max: float,
    prob: float,
) -> torch.Tensor:
    """Random blur + noise on the degraded input at training time only.
    Applied to nudge the model's training distribution toward the actual
    inference phenotype (e.g., diffusion-MSE blur + residual noise on top of
    a deterministic VQ-VAE roundtrip).

    Each augmentation is independently sampled with probability `prob`. Sigma
    and std are then sampled uniformly in [0, max]. If sampled magnitude is
    too small (< 0.05 sigma or < 0.002 std), the aug is skipped to avoid no-op
    convolutions.
    """
    if blur_sigma_max > 0 and torch.rand(1).item() < prob:
        sigma = torch.rand(1).item() * blur_sigma_max
        if sigma > 0.05:
            with torch.no_grad():
                degraded = _gaussian_blur_3d(degraded, sigma)
    if noise_std_max > 0 and torch.rand(1).item() < prob:
        noise_std = torch.rand(1).item() * noise_std_max
        if noise_std > 0.002:
            degraded = (degraded + torch.randn_like(degraded) * noise_std).clamp(0, 1)
    return degraded


# ────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_validation(
    g_ema: nn.Module,
    val_loader: DataLoader,
    perceptual_manager: PerceptualLossManager,
    device: torch.device,
    max_batches: int = 0,
    use_residual_learning: bool = False,
) -> dict[str, float]:
    """Evaluate the EMA generator on the val split.

    Returns dict with val_l1 and val_lpips averaged across batches.
    """
    g_ema.eval()
    l1_sum = 0.0
    perc_sum = 0.0
    n = 0
    t_zero_cache = {}
    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break
        clean = batch['image'].float().to(device)
        degraded = batch['degraded'].float().to(device)
        bs = degraded.shape[0]
        t_zero = t_zero_cache.setdefault(
            bs, torch.zeros(bs, dtype=torch.long, device=device),
        )
        with autocast('cuda', dtype=torch.bfloat16):
            g_out = g_ema(x=degraded, timesteps=t_zero)
            refined = (g_out + degraded).clamp(0, 1) if use_residual_learning else g_out
        refined = refined.float()

        l1 = torch.mean(torch.abs(refined - clean)).item()
        l1_sum += l1
        if perceptual_manager.is_enabled and perceptual_manager.loss_fn is not None:
            perc_sum += perceptual_manager.compute(refined, clean).item()
        n += 1
    g_ema.train()
    return {
        'val_l1': l1_sum / max(1, n),
        'val_lpips': perc_sum / max(1, n),
        'val_batches': float(n),
    }


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir
    log_dir = output_dir / 'tb'
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Repro
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ─── Datasets ───────────────────────────────────────────────
    train_root = Path(args.data_root) / 'train'
    val_root = Path(args.data_root) / 'val'
    if not train_root.is_dir():
        raise SystemExit(f"Missing train dir: {train_root}")
    if not val_root.is_dir():
        logger.warning(f"Missing val dir {val_root} — will skip validation.")

    logger.info("Building training dataset (precomputed pairs, heavy augmentation)...")
    train_ds = Restoration3DDataset(
        data_dir=str(train_root),
        height=args.image_size,
        width=args.image_size,
        pad_depth_to=args.depth,
        augment=True,
        patch_size=None,          # full-volume training
        slice_2d=False,
        augmentation_level=args.augmentation_level,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    logger.info(f"Train: {len(train_ds)} subjects, {len(train_loader)} batches/epoch")

    val_loader: DataLoader | None = None
    if val_root.is_dir():
        val_ds = Restoration3DDataset(
            data_dir=str(val_root),
            height=args.image_size,
            width=args.image_size,
            pad_depth_to=args.depth,
            augment=False,
            patch_size=None,
            slice_2d=False,
            augmentation_level='basic',
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True,
        )
        logger.info(f"Val: {len(val_ds)} subjects, {len(val_loader)} batches")

    # ─── Generator ──────────────────────────────────────────────
    generator = build_generator(spatial_dims=3, device=device)
    ema = EMA(generator, beta=args.ema_beta, update_after_step=0, update_every=1)

    # ─── Discriminator ──────────────────────────────────────────
    disc_manager = DiscriminatorManager(
        spatial_dims=3,
        in_channels=1,
        num_layers=args.disc_num_layers,
        num_channels=args.disc_num_channels,
        learning_rate=args.lr_d,
        optimizer_betas=(0.5, 0.999),
        warmup_epochs=max(1, args.warmup_epochs),
        total_epochs=args.epochs,
        device=device,
        enabled=True,
        gradient_clip_norm=args.gradient_clip_norm,
        is_main_process=True,
    )
    disc_manager.create()
    disc_manager.create_loss_fn()
    disc_manager.setup_optimizer(use_constant_lr=False)
    disc_manager.wrap_model(
        use_multi_gpu=False, local_rank=0, use_compile=args.use_compile,
        compile_mode='default', weight_dtype=torch.float32, pure_weights=False,
    )

    # ─── Losses ─────────────────────────────────────────────────
    l1_fn = nn.L1Loss()
    perceptual_manager = PerceptualLossManager(
        spatial_dims=3,
        weight=args.lambda_perc,
        loss_type='lpips',           # with use_2_5d=True, manager builds 2D LPIPS internally
        device=device,
        use_2_5d=True,
        slice_fraction=0.25,
    )
    perceptual_manager.create()
    logger.info(
        f"Perceptual: {perceptual_manager.loss_type}, "
        f"use_2_5d={perceptual_manager.use_2_5d}, "
        f"slice_fraction={perceptual_manager.slice_fraction}"
    )

    # ─── Optimizer + scheduler for G ────────────────────────────
    optimizer_g = AdamW(
        generator.parameters(),
        lr=args.lr_g,
        betas=(0.5, 0.999),
        weight_decay=0.0,
    )
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=args.epochs, eta_min=args.lr_g * 0.01)

    # ─── Resume ─────────────────────────────────────────────────
    start_epoch = 0
    best_val_lpips = float('inf')
    if args.resume_from and Path(args.resume_from).exists():
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer_g.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
            scheduler_g.load_state_dict(ckpt['scheduler_state_dict'])
        if 'ema_state_dict' in ckpt and ckpt['ema_state_dict'] is not None:
            ema.load_state_dict(ckpt['ema_state_dict'])
        disc_state = ckpt.get('discriminator')
        if disc_state is not None:
            disc_manager.load_state_dict(disc_state, load_optimizer=True)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_lpips = ckpt.get('best_val_lpips', float('inf'))
        logger.info(f"Resumed from {args.resume_from} at epoch {start_epoch}")

    # ─── Training loop ──────────────────────────────────────────
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        generator.train()
        epoch_losses = {'total': 0.0, 'l1': 0.0, 'perc': 0.0, 'adv_g': 0.0,
                        'd': 0.0, 'r1': 0.0, 'fm': 0.0}
        n_batches = 0
        for step, batch in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break

            clean = batch['image'].float().to(device, non_blocking=True)       # [B, 1, D, H, W]
            degraded = batch['degraded'].float().to(device, non_blocking=True)
            # Extra augmentation on the degraded input only (training only).
            # Nudges the training input distribution toward the inference phenotype.
            if args.extra_blur_sigma_max > 0 or args.extra_noise_std_max > 0:
                degraded = maybe_apply_extra_aug(
                    degraded,
                    blur_sigma_max=args.extra_blur_sigma_max,
                    noise_std_max=args.extra_noise_std_max,
                    prob=args.extra_aug_prob,
                )
            bs = degraded.shape[0]
            t_zero = torch.zeros(bs, dtype=torch.long, device=device)

            # ───── Generator step ─────
            optimizer_g.zero_grad(set_to_none=True)
            with autocast('cuda', dtype=torch.bfloat16):
                g_out = generator(x=degraded, timesteps=t_zero)

            # Residual learning: G predicts (clean - degraded). Add input back to
            # reconstruct the full clean estimate. Network only learns the small
            # delta the degradation removed (DnCNN style).
            if args.use_residual_learning:
                refined = (g_out + degraded).clamp(0, 1)
            else:
                refined = g_out

            refined_fp32 = refined.float()
            l1 = l1_fn(refined_fp32, clean)
            perc = (perceptual_manager.compute(refined_fp32, clean)
                    if perceptual_manager.is_enabled and perceptual_manager.loss_fn is not None
                    else torch.zeros((), device=device))

            adv_g = torch.zeros((), device=device)
            fm_loss = torch.zeros((), device=device)
            if global_step >= args.d_warmup_steps:
                ramp = min(1.0, (global_step - args.d_warmup_steps)
                           / max(1, args.adv_ramp_steps))
                adv_raw = disc_manager.compute_generator_loss(refined_fp32)
                adv_g = ramp * args.adv_weight * adv_raw

                # Pix2PixHD feature-matching: L1 between intermediate D features
                # on real vs fake. MONAI PatchDiscriminator returns a list of
                # multi-scale feature maps; use all but the last (final logits).
                if (args.use_feature_matching
                        and disc_manager.discriminator is not None):
                    feats_real = disc_manager.discriminator(clean.detach().contiguous())
                    feats_fake = disc_manager.discriminator(refined_fp32.contiguous())
                    if isinstance(feats_real, (list, tuple)) and len(feats_real) > 1:
                        # Skip final element (final logits map) — supervise intermediate features.
                        fm_terms = [
                            torch.nn.functional.l1_loss(ff.float(), fr.float().detach())
                            for ff, fr in zip(feats_fake[:-1], feats_real[:-1])
                        ]
                        if fm_terms:
                            fm_loss = ramp * args.lambda_fm * torch.stack(fm_terms).mean()

            total_g = args.lambda_l1 * l1 + args.lambda_perc * perc + adv_g + fm_loss
            total_g.backward()
            torch.nn.utils.clip_grad_norm_(
                generator.parameters(), max_norm=args.gradient_clip_norm,
            )
            optimizer_g.step()
            ema.update()

            # ───── Discriminator step ─────
            with torch.no_grad():
                fake_for_d = refined_fp32.detach()
            d_loss = disc_manager.train_step(
                real=clean, fake=fake_for_d,
                weight_dtype=torch.bfloat16,
            )

            # ───── R1 gradient penalty (lazy: every r1_every steps) ─────
            # R1 = |∇_x D(x)|² evaluated at real x. Penalizing this prevents D
            # from saturating (gradients vanishing as D approaches certainty),
            # which keeps adversarial signal flowing to G. See Mescheder 2018.
            r1_value = 0.0
            if (args.r1_weight > 0
                    and disc_manager.discriminator is not None
                    and disc_manager.optimizer is not None
                    and global_step % args.r1_every == 0):
                real_for_r1 = clean.detach().clone().requires_grad_(True)
                # fp32 to keep gradient stable
                logits_real = disc_manager.discriminator(real_for_r1.contiguous())
                # MONAI's PatchDiscriminator returns a list of multi-scale
                # feature maps; sum all of them into a single scalar.
                if isinstance(logits_real, (list, tuple)):
                    logits_scalar = sum(t.sum() for t in logits_real)
                else:
                    logits_scalar = logits_real.sum()
                grad_real = torch.autograd.grad(
                    outputs=logits_scalar, inputs=real_for_r1,
                    create_graph=True,  # retain_graph defaults to True when create_graph=True
                )[0]
                r1 = grad_real.pow(2).flatten(1).sum(1).mean()
                # Scale by r1_every since we apply only every N steps (lazy R1).
                r1_loss = (args.r1_weight * args.r1_every * 0.5) * r1
                disc_manager.optimizer.zero_grad(set_to_none=True)
                r1_loss.backward()
                disc_manager.optimizer.step()
                r1_value = float(r1.detach())

            # Stats
            epoch_losses['total'] += total_g.item()
            epoch_losses['l1'] += l1.item()
            epoch_losses['perc'] += (perc.item() if perceptual_manager.is_enabled else 0.0)
            epoch_losses['adv_g'] += adv_g.item() if isinstance(adv_g, torch.Tensor) else float(adv_g)
            epoch_losses['d'] += d_loss
            epoch_losses['r1'] += r1_value
            epoch_losses['fm'] += fm_loss.item() if isinstance(fm_loss, torch.Tensor) else float(fm_loss)
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                writer.add_scalar('train/total_g', total_g.item(), global_step)
                writer.add_scalar('train/l1', l1.item(), global_step)
                if perceptual_manager.is_enabled:
                    writer.add_scalar('train/perceptual', perc.item(), global_step)
                writer.add_scalar('train/adv_g', adv_g.item() if isinstance(adv_g, torch.Tensor) else adv_g, global_step)
                writer.add_scalar('train/d_loss', d_loss, global_step)
                if args.r1_weight > 0:
                    writer.add_scalar('train/r1', r1_value, global_step)
                if args.use_feature_matching:
                    writer.add_scalar('train/fm', fm_loss.item() if isinstance(fm_loss, torch.Tensor) else fm_loss, global_step)
                writer.add_scalar('train/lr_g', optimizer_g.param_groups[0]['lr'], global_step)
                logger.info(
                    f"ep{epoch} step{step} global{global_step} | "
                    f"L1={l1.item():.4f} Perc={perc.item():.4f} "
                    f"AdvG={adv_g.item() if isinstance(adv_g, torch.Tensor) else adv_g:.4f} "
                    f"D={d_loss:.4f}"
                )

        # End of epoch
        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)
        logger.info(
            f"[EPOCH {epoch + 1}/{args.epochs} ({time.time() - epoch_start:.0f}s)] "
            f"total={epoch_losses['total']:.4f}  L1={epoch_losses['l1']:.4f}  "
            f"Perc={epoch_losses['perc']:.4f}  AdvG={epoch_losses['adv_g']:.4f}  "
            f"D={epoch_losses['d']:.4f}"
        )
        scheduler_g.step()
        disc_manager.on_epoch_end()

        # ─── Validation ───
        if val_loader is not None and (epoch + 1) % args.val_every_epoch == 0:
            val_metrics = run_validation(
                ema.ema_model, val_loader, perceptual_manager, device,
                max_batches=(3 if args.max_steps_per_epoch else 0),
                use_residual_learning=args.use_residual_learning,
            )
            for k, v in val_metrics.items():
                writer.add_scalar(f'val/{k}', v, epoch)
            logger.info(
                f"[VAL ep{epoch + 1}] L1={val_metrics['val_l1']:.4f}  "
                f"LPIPS={val_metrics['val_lpips']:.4f}  "
                f"(n={int(val_metrics['val_batches'])})"
            )
            if val_metrics['val_lpips'] < best_val_lpips and val_metrics['val_lpips'] > 0:
                best_val_lpips = val_metrics['val_lpips']
                save_full_checkpoint(
                    model=ema.ema_model,
                    optimizer=optimizer_g,
                    epoch=epoch,
                    save_dir=str(ckpt_dir),
                    filename='checkpoint_best',
                    scheduler=scheduler_g,
                    ema=ema,
                    extra_state={
                        'discriminator': disc_manager.state_dict(),
                        'best_val_lpips': best_val_lpips,
                        'args': vars(args),
                    },
                )
                logger.info(f"New best val LPIPS={best_val_lpips:.4f} — saved checkpoint_best")

        # ─── Periodic checkpoint ───
        if (epoch + 1) % args.save_every_epoch == 0 or epoch + 1 == args.epochs:
            save_full_checkpoint(
                model=ema.ema_model,
                optimizer=optimizer_g,
                epoch=epoch,
                save_dir=str(ckpt_dir),
                filename='checkpoint_latest',
                scheduler=scheduler_g,
                ema=ema,
                extra_state={
                    'discriminator': disc_manager.state_dict(),
                    'best_val_lpips': best_val_lpips,
                    'args': vars(args),
                },
            )
            logger.info(f"Saved checkpoint_latest at epoch {epoch + 1}")

    writer.close()
    logger.info("✓ Training complete.")


if __name__ == '__main__':
    main()
