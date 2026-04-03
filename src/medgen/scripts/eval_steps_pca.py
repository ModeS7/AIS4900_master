#!/usr/bin/env python3
"""Evaluate how generation step count affects brain shape quality (PCA error).

Generates volumes at different step counts using the same noise/conditioning,
then measures PCA reconstruction error to find the step count that produces
the most brain-like shapes.

Usage:
    python -m medgen.scripts.eval_steps_pca \
        --checkpoint /path/to/bravo_checkpoint.pt \
        --data-root /path/to/brainmetshare-3 \
        --pca-model data/brain_pca_256x256x160.npz \
        --num-volumes 10 --lo 10 --hi 100
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from medgen.core import setup_cuda_optimizations
from medgen.diffusion.loading import load_diffusion_model
from medgen.diffusion.strategy_rflow import RFlowStrategy
from medgen.metrics.brain_mask import BrainPCAModel, create_brain_mask

setup_cuda_optimizations()
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_conditioning_masks(
    data_root: Path, num_volumes: int, depth: int, image_size: int,
) -> list[torch.Tensor]:
    """Load seg masks from val split as conditioning."""
    import nibabel as nib

    val_dir = data_root / 'val'
    seg_files = sorted(val_dir.glob("*/seg.nii.gz"))[:num_volumes]
    if len(seg_files) < num_volumes:
        raise ValueError(f"Need {num_volumes} masks, found {len(seg_files)} in {val_dir}")

    masks = []
    for path in seg_files:
        vol = nib.load(str(path)).get_fdata().astype(np.float32)
        vol = np.transpose(vol, (2, 0, 1))  # [H,W,D] -> [D,H,W]
        d = vol.shape[0]
        if d < depth:
            vol = np.concatenate([vol, np.zeros((depth - d, *vol.shape[1:]), dtype=np.float32)], axis=0)
        elif d > depth:
            vol = vol[:depth]
        # Resize if needed
        if vol.shape[1] != image_size or vol.shape[2] != image_size:
            t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=(depth, image_size, image_size),
                                                 mode='trilinear', align_corners=False)
            vol = t.squeeze().numpy()
        mask = (vol > 0.5).astype(np.float32)
        masks.append(torch.from_numpy(mask).unsqueeze(0).unsqueeze(0))  # [1,1,D,H,W]

    return masks


def generate_and_evaluate(
    model: torch.nn.Module,
    strategy: RFlowStrategy,
    cond_masks: list[torch.Tensor],
    noise_list: list[torch.Tensor],
    num_steps: int,
    device: torch.device,
    pca: BrainPCAModel,
    brain_threshold: float = 0.05,
    decoder: object | None = None,
    latent_channels: int = 1,
    ref_features: dict | None = None,
    out_channels: int = 1,
) -> dict:
    """Generate volumes at given step count and measure PCA errors + FID/KID/CMMD."""
    errors = []
    cpu_volumes = []  # Keep for FID extraction

    for cond, noise in zip(cond_masks, noise_list):
        cond = cond.to(device)
        noise = noise.to(device)

        model_input = torch.cat([noise, cond], dim=1)

        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            with torch.no_grad():
                output = strategy.generate(
                    model, model_input, num_steps, device,
                    latent_channels=latent_channels,
                )

        # Decode latent to pixel space if needed
        if decoder is not None:
            with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                with torch.no_grad():
                    output = decoder(output)

        output = torch.clamp(output, 0, 1)

        # PCA on first channel (brain shape)
        bravo_np = output[0, 0].cpu().numpy()
        brain_mask = create_brain_mask(bravo_np, threshold=brain_threshold, dilate_pixels=0)
        _, error = pca.is_valid(brain_mask)
        errors.append(error)

        # Keep first channel for FID (all modalities show the brain)
        cpu_volumes.append(output[:, 0:1].cpu())

        del output, cond, noise, model_input
        torch.cuda.empty_cache()

    errors = np.array(errors)
    result: dict = {
        'steps': num_steps,
        'mean_error': float(errors.mean()),
        'std_error': float(errors.std()),
        'min_error': float(errors.min()),
        'max_error': float(errors.max()),
        'median_error': float(np.median(errors)),
        'pass_rate': float((errors <= pca.error_threshold).mean()),
        'errors': errors.tolist(),
    }

    # Compute FID/KID/CMMD if reference features provided
    if ref_features is not None and cpu_volumes:
        result.update(_compute_fid_metrics(cpu_volumes, ref_features, device))

    return result


def _compute_fid_metrics(
    cpu_volumes: list[torch.Tensor],
    ref_features: dict,
    device: torch.device,
) -> dict:
    """Extract slice features from generated volumes and compute FID/KID/CMMD."""
    from medgen.metrics.generation import compute_cmmd, compute_fid, compute_kid
    from medgen.metrics.generation_3d import volumes_to_slices

    # Stack volumes [N, 1, D, H, W] -> slices [N*D, 1, H, W]
    all_vols = torch.cat(cpu_volumes, dim=0)
    slices = volumes_to_slices(all_vols)
    # Repeat to 3ch for feature extractors
    slices_3ch = slices.repeat(1, 3, 1, 1) if slices.shape[1] == 1 else slices

    metrics = {}
    try:
        from medgen.metrics.feature_extractors import BiomedCLIPFeatures, ResNet50Features

        # ResNet50 (ImageNet) -> FID + KID
        if 'resnet' in ref_features:
            extractor = ResNet50Features(device, network_type='imagenet', compile_model=False)
            gen_feats = extractor.extract_features(slices_3ch.to(device))
            ref = ref_features['resnet'].to(device)
            metrics['FID'] = float(compute_fid(gen_feats, ref))
            kid_mean, kid_std = compute_kid(gen_feats, ref)
            metrics['KID_mean'] = float(kid_mean)
            metrics['KID_std'] = float(kid_std)
            extractor.unload()
            del gen_feats
            torch.cuda.empty_cache()

        # ResNet50 (RadImageNet) -> FID_RIN + KID_RIN
        if 'resnet_rin' in ref_features:
            extractor = ResNet50Features(device, network_type='radimagenet', compile_model=False)
            gen_feats = extractor.extract_features(slices_3ch.to(device))
            ref = ref_features['resnet_rin'].to(device)
            metrics['FID_RIN'] = float(compute_fid(gen_feats, ref))
            kid_mean, kid_std = compute_kid(gen_feats, ref)
            metrics['KID_RIN_mean'] = float(kid_mean)
            extractor.unload()
            del gen_feats
            torch.cuda.empty_cache()

        # BiomedCLIP -> CMMD
        if 'biomed' in ref_features:
            extractor = BiomedCLIPFeatures(device, compile_model=False)
            gen_feats = extractor.extract_features(slices_3ch.to(device))
            ref = ref_features['biomed'].to(device)
            metrics['CMMD'] = float(compute_cmmd(gen_feats, ref))
            extractor.unload()
            del gen_feats
            torch.cuda.empty_cache()

    except Exception as e:
        logger.warning(f"FID/KID/CMMD computation failed: {e}")

    return metrics


def _extract_reference_features(
    data_root: Path, split: str, depth: int, image_size: int, device: torch.device,
) -> dict:
    """Extract slice features from real volumes for FID/KID/CMMD reference."""
    import nibabel as nib

    from medgen.metrics.feature_extractors import BiomedCLIPFeatures, ResNet50Features
    from medgen.metrics.generation_3d import volumes_to_slices

    split_dir = data_root / split
    files = sorted(split_dir.glob("*/bravo.nii.gz"))
    logger.info(f"Extracting reference features from {len(files)} volumes ({split})...")

    # Load and stack volumes
    vols = []
    for path in files:
        vol = nib.load(str(path)).get_fdata().astype(np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        vol = np.transpose(vol, (2, 0, 1))  # [H,W,D] -> [D,H,W]
        d = vol.shape[0]
        if d < depth:
            vol = np.concatenate([vol, np.zeros((depth - d, *vol.shape[1:]), dtype=np.float32)], axis=0)
        elif d > depth:
            vol = vol[:depth]
        if vol.shape[1] != image_size or vol.shape[2] != image_size:
            t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=(depth, image_size, image_size),
                                                 mode='trilinear', align_corners=False)
            vol = t.squeeze().numpy()
        vols.append(torch.from_numpy(vol).unsqueeze(0).unsqueeze(0))  # [1,1,D,H,W]

    all_vols = torch.cat(vols, dim=0)  # [N,1,D,H,W]
    slices = volumes_to_slices(all_vols)  # [N*D,1,H,W]
    slices_3ch = slices.repeat(1, 3, 1, 1)

    ref = {}
    chunk = 64

    # ResNet50 (ImageNet)
    ext = ResNet50Features(device, network_type='imagenet', compile_model=False)
    feats = []
    for i in range(0, len(slices_3ch), chunk):
        feats.append(ext.extract_features(slices_3ch[i:i+chunk].to(device)).cpu())
    ref['resnet'] = torch.cat(feats, dim=0)
    ext.unload()
    torch.cuda.empty_cache()

    # ResNet50 (RadImageNet)
    ext = ResNet50Features(device, network_type='radimagenet', compile_model=False)
    feats = []
    for i in range(0, len(slices_3ch), chunk):
        feats.append(ext.extract_features(slices_3ch[i:i+chunk].to(device)).cpu())
    ref['resnet_rin'] = torch.cat(feats, dim=0)
    ext.unload()
    torch.cuda.empty_cache()

    # BiomedCLIP
    ext = BiomedCLIPFeatures(device, compile_model=False)
    feats = []
    for i in range(0, len(slices_3ch), chunk):
        feats.append(ext.extract_features(slices_3ch[i:i+chunk].to(device)).cpu())
    ref['biomed'] = torch.cat(feats, dim=0)
    ext.unload()
    torch.cuda.empty_cache()

    logger.info(f"Reference features: resnet={ref['resnet'].shape}, biomed={ref['biomed'].shape}")
    return ref


def main():
    parser = argparse.ArgumentParser(description="Evaluate step count vs brain shape quality")
    parser.add_argument('--checkpoint', required=True, help='Bravo model checkpoint')
    parser.add_argument('--data-root', required=True, help='Dataset root')
    parser.add_argument('--pca-model', required=True, help='Brain PCA model (.npz)')
    parser.add_argument('--output-dir', default=None, help='Output directory for results')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--num-volumes', type=int, default=10, help='Volumes per step count')
    parser.add_argument('--lo', type=int, default=10, help='Min steps')
    parser.add_argument('--hi', type=int, default=100, help='Max steps')
    parser.add_argument('--step-list', type=str, default=None,
                        help='Comma-separated step counts (overrides lo/hi)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shift-ratio', type=float, default=1.0, help='Time-shift ratio')
    parser.add_argument('--compute-fid', action='store_true', help='Also compute FID/KID/CMMD')
    parser.add_argument('--ref-split', type=str, default='test1', help='Reference split for FID')
    args = parser.parse_args()

    device = torch.device("cuda")
    data_root = Path(args.data_root)
    pca = BrainPCAModel(args.pca_model)
    logger.info(f"PCA model: {args.pca_model} (threshold={pca.error_threshold:.6f})")

    # Load checkpoint config to detect model type
    # Load checkpoint config (same approach as find_optimal_steps.py)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_cfg = ckpt.get('config', {})
    latent_cfg = ckpt_cfg.get('latent', {})
    is_latent = latent_cfg.get('enabled', False)
    compression_ckpt = latent_cfg.get('compression_checkpoint', None)
    compression_type = latent_cfg.get('compression_type', 'vqvae')
    wavelet_cfg = ckpt_cfg.get('wavelet', {})
    is_wavelet = wavelet_cfg.get('enabled', False)
    strategy_type = ckpt_cfg.get('strategy', 'rflow')

    # Base channels from mode config (e.g., bravo: in=2, out=1)
    base_in_ch = ckpt_cfg.get('in_channels', 2)
    base_out_ch = ckpt_cfg.get('out_channels', 1)
    del ckpt

    # Compute actual model channels based on space (same as find_optimal_steps.py)
    decoder = None
    latent_channels = 1
    comp_model = None

    if is_latent and compression_ckpt:
        logger.info(f"Loading VQ-VAE decoder: {compression_ckpt}")
        from medgen.data.loaders.compression_detection import load_compression_model
        comp_model = load_compression_model(
            compression_ckpt, compression_type=compression_type, device=device, spatial_dims=3,
        )
        decoder = comp_model.decode
        latent_channels = comp_model.latent_channels if hasattr(comp_model, 'latent_channels') else 4
        in_ch = base_in_ch * latent_channels
        out_ch = base_out_ch * latent_channels
        sf = comp_model.spatial_scale_factor if hasattr(comp_model, 'spatial_scale_factor') else 4
        depth_sf = comp_model.depth_scale_factor if hasattr(comp_model, 'depth_scale_factor') else sf
        logger.info(f"Latent space: {latent_channels}ch, {sf}x spatial, {depth_sf}x depth")
    elif is_wavelet:
        logger.info("Loading wavelet encoder/decoder")
        from medgen.models.haar_wavelet_3d import HaarWavelet3D, InverseHaarWavelet3D
        wavelet_encoder = HaarWavelet3D().to(device)
        decoder = InverseHaarWavelet3D().to(device)
        wav_ch = 8  # 8 wavelet subbands
        in_ch = base_in_ch * wav_ch
        out_ch = base_out_ch * wav_ch
        sf = 2
        depth_sf = 2
    else:
        in_ch = base_in_ch
        out_ch = base_out_ch
        sf = 1
        depth_sf = 1

    space_name = 'latent' if is_latent else 'wavelet' if is_wavelet else 'pixel'
    logger.info(f"Model type: {space_name}")
    logger.info(f"Strategy: {strategy_type}, base_ch=({base_in_ch},{base_out_ch}), model_ch=({in_ch},{out_ch})")

    # Load model with correct channels
    logger.info(f"Loading model: {args.checkpoint}")
    model = load_diffusion_model(
        args.checkpoint, device=device,
        in_channels=in_ch, out_channels=out_ch,
        compile_model=False, spatial_dims=3,
    )

    # Setup strategy
    if strategy_type == 'ddpm':
        from medgen.diffusion.strategy_ddpm import DDPMStrategy
        strategy = DDPMStrategy()
    else:
        strategy = RFlowStrategy()
    strategy.setup_scheduler(num_timesteps=1000, image_size=args.image_size,
                             depth_size=args.depth, spatial_dims=3)

    # Apply time-shift
    if args.shift_ratio != 1.0:
        input_numel = args.image_size * args.image_size * args.depth
        base_numel = max(1, int(input_numel / (args.shift_ratio ** 3)))
        if hasattr(strategy, 'scheduler') and hasattr(strategy.scheduler, 'base_img_size_numel'):
            strategy.scheduler.base_img_size_numel = base_numel
        logger.info(f"Time-shift: ratio={args.shift_ratio}")

    # Load conditioning masks
    logger.info(f"Loading {args.num_volumes} conditioning masks from val split...")
    cond_masks = load_conditioning_masks(data_root, args.num_volumes, args.depth, args.image_size)

    # Encode conditioning for latent/wavelet models
    if is_latent and comp_model is not None:
        logger.info("Encoding conditioning masks to latent space...")
        encoded_masks = []
        for m in cond_masks:
            with torch.no_grad():
                enc = comp_model.encode(m.to(device))
            encoded_masks.append(enc.cpu())
        cond_masks = encoded_masks
        noise_depth = cond_masks[0].shape[2]
        noise_h = cond_masks[0].shape[3]
        noise_w = cond_masks[0].shape[4]
        noise_ch = base_out_ch * latent_channels
    elif is_wavelet:
        logger.info("Encoding conditioning masks to wavelet space...")
        encoded_masks = []
        for m in cond_masks:
            with torch.no_grad():
                enc = wavelet_encoder(m.to(device))
            encoded_masks.append(enc.cpu())
        cond_masks = encoded_masks
        noise_depth = args.depth // depth_sf
        noise_h = args.image_size // sf
        noise_w = args.image_size // sf
        noise_ch = base_out_ch * wav_ch
    else:
        noise_depth = args.depth
        noise_h = args.image_size
        noise_w = args.image_size
        noise_ch = out_ch

    # Pre-generate noise (same for all step counts)
    logger.info(f"Pre-generating {args.num_volumes} noise tensors (seed={args.seed})...")
    logger.info(f"Noise shape: [1, {noise_ch}, {noise_depth}, {noise_h}, {noise_w}]")
    gen = torch.Generator(device='cpu').manual_seed(args.seed)
    noise_list = [
        torch.randn(1, noise_ch, noise_depth, noise_h, noise_w, generator=gen)
        for _ in range(args.num_volumes)
    ]

    # Extract reference features for FID/KID/CMMD
    ref_features = None
    if args.compute_fid:
        ref_features = _extract_reference_features(
            data_root, args.ref_split, args.depth, args.image_size, device,
        )

    # Cache for evaluated step counts (avoid re-evaluating)
    eval_cache: dict[int, dict] = {}

    def evaluate_steps(steps: int) -> dict:
        """Evaluate a step count, using cache to avoid re-computation."""
        if steps in eval_cache:
            return eval_cache[steps]
        logger.info(f"\n--- Evaluating: {steps} steps ---")
        t0 = time.time()
        result = generate_and_evaluate(
            model, strategy, cond_masks, noise_list, steps, device, pca,
            decoder=decoder, latent_channels=latent_channels,
            ref_features=ref_features, out_channels=out_ch,
        )
        result['wall_time_s'] = time.time() - t0
        logger.info(f"  Mean PCA error: {result['mean_error']:.6f} | "
                     f"Pass rate: {result['pass_rate']:.0%} | "
                     f"Time: {result['wall_time_s']:.1f}s")
        eval_cache[steps] = result
        return result

    if args.step_list:
        # Fixed step list — evaluate each
        step_counts = [int(x) for x in args.step_list.split(',')]
        logger.info(f"\nEvaluating fixed steps: {step_counts}")
        for steps in step_counts:
            evaluate_steps(steps)
    else:
        # Golden section search for optimal step count (minimize mean PCA error)
        logger.info(f"\nGolden section search: [{args.lo}, {args.hi}]")
        gr = (np.sqrt(5) + 1) / 2  # golden ratio
        a, b = args.lo, args.hi

        c = round(b - (b - a) / gr)
        d = round(a + (b - a) / gr)

        while b - a > 2:
            c = max(a, min(b, c))
            d = max(a, min(b, d))
            if c == d:
                break

            fc = evaluate_steps(c)['mean_error']
            fd = evaluate_steps(d)['mean_error']

            if fc < fd:
                b = d
            else:
                a = c

            c = round(b - (b - a) / gr)
            d = round(a + (b - a) / gr)

        # Evaluate remaining candidates in final bracket
        for s in range(a, b + 1):
            evaluate_steps(s)

    # Collect all results sorted by steps
    results = sorted(eval_cache.values(), key=lambda r: r['steps'])

    # Summary
    has_fid = any('FID' in r for r in results)
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Steps vs PCA Error" + (" + FID/KID/CMMD" if has_fid else ""))
    logger.info("=" * 80)
    if has_fid:
        logger.info(f"{'Steps':>6} {'PCA_err':>10} {'Pass%':>7} {'FID':>8} {'KID':>8} {'CMMD':>8} {'FID_RIN':>8} {'Time':>8}")
        logger.info("-" * 75)
        for r in results:
            fid = f"{r['FID']:.2f}" if 'FID' in r else "—"
            kid = f"{r['KID_mean']:.4f}" if 'KID_mean' in r else "—"
            cmmd = f"{r['CMMD']:.4f}" if 'CMMD' in r else "—"
            fid_rin = f"{r['FID_RIN']:.2f}" if 'FID_RIN' in r else "—"
            logger.info(f"{r['steps']:>6} {r['mean_error']:>10.6f} {r['pass_rate']:>6.0%} "
                         f"{fid:>8} {kid:>8} {cmmd:>8} {fid_rin:>8} {r['wall_time_s']:>7.1f}s")
    else:
        logger.info(f"{'Steps':>6} {'Mean':>10} {'Median':>10} {'Pass%':>8} {'Time':>8}")
        logger.info("-" * 48)
        for r in results:
            logger.info(f"{r['steps']:>6} {r['mean_error']:>10.6f} {r['median_error']:>10.6f} "
                         f"{r['pass_rate']:>7.0%} {r['wall_time_s']:>7.1f}s")

    best = min(results, key=lambda r: r['mean_error'])
    logger.info(f"\nBest PCA: {best['steps']} steps (mean error={best['mean_error']:.6f}, "
                f"pass rate={best['pass_rate']:.0%})")
    if has_fid:
        best_fid = min((r for r in results if 'FID' in r), key=lambda r: r['FID'])
        logger.info(f"Best FID: {best_fid['steps']} steps (FID={best_fid['FID']:.2f})")

    # Save results
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output = {
            'checkpoint': args.checkpoint,
            'pca_model': args.pca_model,
            'pca_threshold': pca.error_threshold,
            'num_volumes': args.num_volumes,
            'shift_ratio': args.shift_ratio,
            'seed': args.seed,
            'best_steps': best['steps'],
            'results': results,
        }
        with open(out_dir / 'pca_step_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved to {out_dir / 'pca_step_results.json'}")


if __name__ == '__main__':
    main()
