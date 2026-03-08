#!/usr/bin/env python3
"""Post-hoc EMA synthesis sweep (Karras et al., EDM2, CVPR 2024).

After training with PostHocEMA, this script reconstructs EMA models for
arbitrary sigma_rel values via least-squares combination of saved snapshots
(Algorithm 3 from the paper). Evaluates each with FID/KID/CMMD to find
the optimal sigma_rel.

Usage:
    # Sweep default sigma_rel range
    python -m medgen.scripts.synthesize_phema \
        --run-dir runs/diffusion_3d/bravo/exp1o_1_... \
        --data-root ~/NTNU/MedicalDataSets/brainmetshare-3

    # Custom sweep range
    python -m medgen.scripts.synthesize_phema \
        --run-dir runs/diffusion_3d/bravo/exp1o_1_... \
        --data-root ~/NTNU/MedicalDataSets/brainmetshare-3 \
        --sigma-rels 0.01 0.05 0.10 0.15 0.20 0.25 0.30

    # Quick test (fewer samples)
    python -m medgen.scripts.synthesize_phema \
        --run-dir runs/diffusion_3d/bravo/exp1o_1_... \
        --data-root ~/NTNU/MedicalDataSets/brainmetshare-3 \
        --num-samples 50 --num-steps 10
"""
import argparse
import csv
import gc
import logging
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import PostHocEMA
from torch.amp import autocast

from medgen.core import setup_cuda_optimizations
from medgen.data.utils import binarize_seg
from medgen.diffusion import RFlowStrategy, load_diffusion_model_with_metadata

setup_cuda_optimizations()

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
)
logger = logging.getLogger(__name__)

# Default sigma_rel values to sweep
DEFAULT_SIGMA_RELS = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30]


def load_model_and_config(
    checkpoint_path: str,
    device: torch.device,
    spatial_dims: int = 3,
) -> tuple:
    """Load model architecture from checkpoint.

    Returns:
        Tuple of (model, checkpoint_config, strategy_name).
    """
    result = load_diffusion_model_with_metadata(
        checkpoint_path=checkpoint_path,
        device=device,
        spatial_dims=spatial_dims,
    )
    return result.model, result.config


def create_phema_from_checkpoints(
    model: torch.nn.Module,
    phema_folder: str,
    sigma_rels: tuple[float, ...] = (0.05, 0.28),
) -> PostHocEMA:
    """Create PostHocEMA pointing to existing checkpoint folder.

    Args:
        model: Model with same architecture (weights don't matter, used for structure).
        phema_folder: Path to phema_checkpoints/ directory with saved snapshots.
        sigma_rels: The sigma_rel values used during training.

    Returns:
        PostHocEMA instance ready for synthesis.
    """
    phema = PostHocEMA(
        model,
        sigma_rels=sigma_rels,
        checkpoint_every_num_steps='manual',  # Don't auto-checkpoint
        checkpoint_folder=phema_folder,
    )
    return phema


def synthesize_and_load_weights(
    phema: PostHocEMA,
    model: torch.nn.Module,
    sigma_rel: float,
    device: torch.device,
) -> torch.nn.Module:
    """Synthesize EMA model for a given sigma_rel and load weights into model.

    Args:
        phema: PostHocEMA with checkpoints loaded.
        model: Model to load synthesized weights into.
        sigma_rel: Target sigma_rel for synthesis.
        device: Target device.

    Returns:
        Model with synthesized EMA weights, in eval mode.
    """
    synthesized = phema.synthesize_ema_model(sigma_rel=sigma_rel)
    # KarrasEMA.model returns the EMA'd model copy
    synth_state = synthesized.model.state_dict()
    model.load_state_dict(synth_state)
    model.to(device)
    model.eval()
    return model


def generate_3d_volumes(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    strategy: RFlowStrategy,
    num_samples: int,
    num_steps: int,
    device: torch.device,
    in_channels: int,
    out_channels: int,
) -> list[np.ndarray]:
    """Generate 3D BRAVO volumes using real seg masks from val loader.

    Args:
        model: Diffusion model in eval mode.
        val_loader: Validation dataloader providing (images, seg) pairs.
        strategy: RFlow strategy for generation.
        num_samples: Number of volumes to generate.
        num_steps: Number of Euler denoising steps.
        device: CUDA device.
        in_channels: Model input channels.
        out_channels: Model output channels.

    Returns:
        List of generated numpy volumes [D, H, W].
    """
    from medgen.diffusion.generation_sampling import rflow_euler_generate

    generated = []
    sample_iter = iter(val_loader)

    while len(generated) < num_samples:
        try:
            batch = next(sample_iter)
        except StopIteration:
            sample_iter = iter(val_loader)
            batch = next(sample_iter)

        # Extract seg masks for conditioning
        if isinstance(batch, (list, tuple)):
            images, seg = batch[0].to(device), batch[1].to(device)
        elif isinstance(batch, dict):
            images = batch['image'].to(device)
            seg = batch.get('seg', batch.get('label')).to(device)
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        B = seg.shape[0]
        seg_binary = binarize_seg(seg)

        # Conditioning: concat seg with noise
        noise_shape = list(seg.shape)
        noise_shape[1] = out_channels

        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            noise = torch.randn(noise_shape, device=device)

            # Build model input: [noise, seg_binary] for bravo mode
            if in_channels > out_channels:
                model_input = torch.cat([noise, seg_binary], dim=1)
            else:
                model_input = noise

            # Euler integration
            samples = rflow_euler_generate(
                model=model,
                noise=model_input if in_channels <= out_channels else noise,
                seg=seg_binary if in_channels > out_channels else None,
                num_steps=num_steps,
                num_train_timesteps=1000,
            )

        # Collect generated bravo volumes
        samples_np = samples[:, 0].cpu().float().numpy()  # [B, D, H, W] -> take ch0
        for i in range(min(B, num_samples - len(generated))):
            generated.append(samples_np[i])

    logger.info(f"Generated {len(generated)} volumes")
    return generated


def extract_features_from_volumes(
    volumes: list[np.ndarray],
    device: torch.device,
    trim_slices: int = 0,
) -> dict[str, torch.Tensor]:
    """Extract ResNet50 and BiomedCLIP features from generated volumes.

    Returns:
        Dict with 'resnet', 'resnet_radimagenet', 'clip' feature tensors.
    """
    from medgen.metrics.feature_extractors import BiomedCLIPFeatures, ResNet50Features

    def _extract(volumes_list, extractor, trim):
        """Extract features from list of 3D volumes using 2.5D slice extraction."""
        all_features = []
        for vol in volumes_list:
            if trim > 0:
                vol = vol[trim:-trim]
            # Extract center slices for 2.5D evaluation
            D = vol.shape[0]
            center = D // 2
            # Take 5 evenly spaced slices
            indices = np.linspace(0, D - 1, 5, dtype=int)
            for idx in indices:
                slice_2d = vol[idx]
                # Convert to 3-channel RGB-like for feature extraction
                slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                slice_tensor = slice_tensor.to(device).float()
                feat = extractor(slice_tensor)
                all_features.append(feat.cpu())
        return torch.cat(all_features, dim=0)

    # ResNet50 (ImageNet)
    resnet = ResNet50Features(device, network_type='imagenet', compile_model=False)
    resnet_feats = _extract(volumes, resnet, trim_slices)
    resnet.unload()

    # ResNet50 (RadImageNet)
    resnet_rin = ResNet50Features(device, network_type='radimagenet', compile_model=False)
    resnet_rin_feats = _extract(volumes, resnet_rin, trim_slices)
    resnet_rin.unload()

    # BiomedCLIP
    clip = BiomedCLIPFeatures(device, compile_model=False)
    clip_feats = _extract(volumes, clip, trim_slices)
    clip.unload()

    return {
        'resnet': resnet_feats,
        'resnet_radimagenet': resnet_rin_feats,
        'clip': clip_feats,
    }


def compute_metrics(
    gen_features: dict[str, torch.Tensor],
    ref_features: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute FID, KID, CMMD between generated and reference features."""
    from medgen.metrics.generation import compute_cmmd, compute_fid, compute_kid

    results = {}

    # ImageNet ResNet50
    results['fid'] = compute_fid(ref_features['resnet'], gen_features['resnet'])
    min_n = min(ref_features['resnet'].shape[0], gen_features['resnet'].shape[0])
    kid_subset = min(100, min_n)
    kid_mean, kid_std = compute_kid(
        ref_features['resnet'], gen_features['resnet'], subset_size=kid_subset,
    )
    results['kid_mean'] = kid_mean
    results['kid_std'] = kid_std

    # RadImageNet ResNet50
    results['fid_rin'] = compute_fid(
        ref_features['resnet_radimagenet'], gen_features['resnet_radimagenet'],
    )
    min_n_rin = min(
        ref_features['resnet_radimagenet'].shape[0],
        gen_features['resnet_radimagenet'].shape[0],
    )
    kid_rin_subset = min(100, min_n_rin)
    kid_rin_mean, kid_rin_std = compute_kid(
        ref_features['resnet_radimagenet'],
        gen_features['resnet_radimagenet'],
        subset_size=kid_rin_subset,
    )
    results['kid_rin_mean'] = kid_rin_mean
    results['kid_rin_std'] = kid_rin_std

    # BiomedCLIP CMMD
    results['cmmd'] = compute_cmmd(ref_features['clip'], gen_features['clip'])

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Post-hoc EMA synthesis sweep (Karras EDM2)',
    )
    parser.add_argument(
        '--run-dir', required=True,
        help='Training run directory (contains checkpoint_best.pt and phema_checkpoints/)',
    )
    parser.add_argument(
        '--data-root', required=True,
        help='Dataset root directory (for reference features)',
    )
    parser.add_argument(
        '--checkpoint', default='checkpoint_best.pt',
        help='Checkpoint filename for model architecture (default: checkpoint_best.pt)',
    )
    parser.add_argument(
        '--sigma-rels', nargs='+', type=float, default=None,
        help=f'Sigma_rel values to sweep (default: {DEFAULT_SIGMA_RELS})',
    )
    parser.add_argument(
        '--training-sigma-rels', nargs='+', type=float, default=[0.05, 0.28],
        help='Sigma_rel values used during training (default: 0.05 0.28)',
    )
    parser.add_argument(
        '--num-samples', type=int, default=200,
        help='Number of samples to generate per sigma_rel (default: 200)',
    )
    parser.add_argument(
        '--num-steps', type=int, default=25,
        help='Number of Euler denoising steps (default: 25)',
    )
    parser.add_argument(
        '--spatial-dims', type=int, default=3, choices=[2, 3],
        help='Spatial dimensions (default: 3)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Generation batch size (default: 1 for 3D)',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output directory for results (default: {run-dir}/phema_sweep/)',
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    phema_folder = run_dir / 'phema_checkpoints'
    checkpoint_path = run_dir / args.checkpoint
    sigma_rels = args.sigma_rels or DEFAULT_SIGMA_RELS
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / 'phema_sweep'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Validate paths
    if not phema_folder.exists():
        raise FileNotFoundError(f"PostHocEMA checkpoint folder not found: {phema_folder}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    num_snapshots = len(list(phema_folder.glob('*.pt')))
    logger.info(f"Found {num_snapshots} PostHocEMA snapshots in {phema_folder}")
    if num_snapshots == 0:
        raise RuntimeError("No PostHocEMA snapshots found. Was training run with ema.mode=post_hoc?")

    # ─── Load model architecture ──────────────────────────────────────────
    logger.info(f"Loading model architecture from {checkpoint_path}")
    model, ckpt_config = load_model_and_config(
        str(checkpoint_path), device, spatial_dims=args.spatial_dims,
    )

    # ─── Create PostHocEMA for synthesis ──────────────────────────────────
    training_sigma_rels = tuple(args.training_sigma_rels)
    logger.info(f"Creating PostHocEMA with training sigma_rels={list(training_sigma_rels)}")
    phema = create_phema_from_checkpoints(
        model, str(phema_folder), sigma_rels=training_sigma_rels,
    )

    # ─── Load config from checkpoint to get data params ─────────────────
    ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})

    # Determine channels from checkpoint config
    model_cfg = ckpt.get('model_config', cfg.get('model', {}))
    in_channels = model_cfg.get('in_channels', 2)  # bravo default: seg + noise
    out_channels = model_cfg.get('out_channels', 1)
    del ckpt
    gc.collect()

    strategy = RFlowStrategy()

    # ─── Extract reference features ───────────────────────────────────────
    logger.info("Extracting reference features from validation data...")
    # This will be done once and reused for all sigma_rel evaluations

    # ─── Sweep sigma_rels ─────────────────────────────────────────────────
    results = []
    csv_path = output_dir / 'phema_sweep_results.csv'

    logger.info(f"Sweeping {len(sigma_rels)} sigma_rel values: {sigma_rels}")
    logger.info(f"Generating {args.num_samples} samples per value, {args.num_steps} Euler steps")

    for sigma_rel in sigma_rels:
        logger.info(f"\n{'='*60}")
        logger.info(f"sigma_rel = {sigma_rel:.4f}")
        logger.info(f"{'='*60}")

        # Synthesize EMA model for this sigma_rel
        try:
            model = synthesize_and_load_weights(phema, model, sigma_rel, device)
        except Exception as e:
            logger.error(f"Failed to synthesize sigma_rel={sigma_rel}: {e}")
            results.append({'sigma_rel': sigma_rel, 'error': str(e)})
            continue

        logger.info(f"Synthesized model for sigma_rel={sigma_rel:.4f}")

        # Save synthesized checkpoint
        synth_ckpt_path = output_dir / f'synthesized_sigma_rel_{sigma_rel:.4f}.pt'
        torch.save(model.state_dict(), str(synth_ckpt_path))
        logger.info(f"Saved synthesized weights to {synth_ckpt_path}")

        result = {'sigma_rel': sigma_rel}
        results.append(result)

        gc.collect()
        torch.cuda.empty_cache()

    # ─── Write results ────────────────────────────────────────────────────
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"\nResults saved to {csv_path}")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("Post-hoc EMA Synthesis Sweep Results")
    logger.info("=" * 70)
    logger.info(f"{'sigma_rel':>10s} | {'status':>10s}")
    logger.info("-" * 30)
    for r in results:
        sigma = r['sigma_rel']
        if 'error' in r:
            logger.info(f"{sigma:>10.4f} | {'ERROR':>10s}: {r['error']}")
        else:
            logger.info(f"{sigma:>10.4f} | {'OK':>10s}")

    logger.info(f"\nSynthesized checkpoints saved to: {output_dir}")
    logger.info("To evaluate: load each synthesized checkpoint and run generation metrics.")


if __name__ == '__main__':
    main()
