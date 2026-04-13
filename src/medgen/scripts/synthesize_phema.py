#!/usr/bin/env python3
"""Post-hoc EMA synthesis sweep with full evaluation.

After training with PostHocEMA, this script:
1. Reconstructs EMA models at multiple sigma_rel values
2. Generates volumes with each synthesized model
3. Computes FID, KID, CMMD, FWD, PCA metrics against real data
4. Also evaluates the raw (non-EMA) checkpoint as baseline

Uses the same generation and metrics infrastructure as eval_ode_solvers.py
for consistent, comparable results.

Usage:
    python -m medgen.scripts.synthesize_phema \
        --run-dir runs/diffusion_3d/bravo/exp1o_1_... \
        --data-root ~/NTNU/MedicalDataSets/brainmetshare-3

    # Custom sigma_rel range
    python -m medgen.scripts.synthesize_phema \
        --run-dir runs/diffusion_3d/bravo/exp1o_1_... \
        --data-root ~/NTNU/MedicalDataSets/brainmetshare-3 \
        --sigma-rels 0.01 0.05 0.10 0.15 0.20 0.25 0.30

    # Quick test
    python -m medgen.scripts.synthesize_phema \
        --run-dir runs/diffusion_3d/bravo/exp1o_1_... \
        --data-root ~/NTNU/MedicalDataSets/brainmetshare-3 \
        --num-volumes 10 --num-steps 10
"""
import argparse
import gc
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import PostHocEMA

from medgen.diffusion import RFlowStrategy, load_diffusion_model
from medgen.scripts.eval_ode_solvers import (
    SolverConfig,
    compute_all_metrics,
    discover_splits,
    generate_noise_tensors,
    generate_volumes,
    get_or_cache_reference_features,
    load_conditioning,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

DEFAULT_SIGMA_RELS = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.28, 0.35, 0.50]


def create_phema(
    model: torch.nn.Module,
    phema_folder: str,
    sigma_rels: tuple[float, ...] = (0.05, 0.28),
) -> PostHocEMA:
    """Create PostHocEMA pointing to existing checkpoint folder."""
    return PostHocEMA(
        model,
        sigma_rels=sigma_rels,
        checkpoint_every_num_steps='manual',
        checkpoint_folder=phema_folder,
    )


def synthesize_model(
    phema: PostHocEMA,
    model: torch.nn.Module,
    sigma_rel: float,
    device: torch.device,
) -> torch.nn.Module:
    """Synthesize EMA model at given sigma_rel and load weights.

    Note: KarrasEMA.model returns the online (original) model.
    KarrasEMA.ema_model holds the actual EMA weights after synthesis.
    """
    synthesized = phema.synthesize_ema_model(sigma_rel=sigma_rel)
    model.load_state_dict(synthesized.ema_model.state_dict())
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    model: torch.nn.Module,
    strategy: RFlowStrategy,
    noise_list: list[torch.Tensor],
    cond_list: list[tuple[str, torch.Tensor]],
    solver_cfg: SolverConfig,
    device: torch.device,
    ref_features: dict,
    real_volumes: list[np.ndarray],
    brain_pca,
    trim_slices: int,
    is_seg: bool = False,
) -> dict:
    """Generate volumes and compute all metrics for one model configuration.

    Returns dict with fid, kid, cmmd, fwd, pca metrics.
    """
    from medgen.metrics.fwd import compute_fwd_3d

    # Generate (seg is unconditional — no conditioning)
    t0 = time.time()
    volumes, nfe, wall_time = generate_volumes(
        model, strategy, noise_list,
        None if is_seg else cond_list,
        solver_cfg, device,
        is_seg=is_seg,
    )
    gen_time = time.time() - t0
    logger.info(f"  Generated {len(volumes)} volumes in {gen_time:.0f}s")

    # FID / KID / CMMD
    split_metrics = compute_all_metrics(volumes, ref_features, device, trim_slices)
    # Use the first available split for metrics
    split_name = next(iter(split_metrics))
    m = asdict(split_metrics[split_name])

    result = {
        'fid': m['fid'],
        'kid_mean': m['kid_mean'],
        'kid_std': m['kid_std'],
        'cmmd': m['cmmd'],
        'fid_radimagenet': m['fid_radimagenet'],
        'kid_radimagenet_mean': m['kid_radimagenet_mean'],
        'kid_radimagenet_std': m['kid_radimagenet_std'],
        'generation_time_s': gen_time,
    }

    # FWD
    if real_volumes:
        fwd_score, fwd_bands = compute_fwd_3d(
            real_volumes, volumes, trim_slices=trim_slices, max_level=4,
        )
        n_bands = len(fwd_bands)
        quarter = n_bands // 4
        vals = list(fwd_bands.values())
        result['fwd'] = fwd_score
        result['fwd_low'] = float(np.mean(vals[:quarter]))
        result['fwd_mid'] = float(np.mean(vals[quarter:3 * quarter]))
        result['fwd_high'] = float(np.mean(vals[3 * quarter:]))

    # PCA (bravo only)
    if brain_pca is not None and not is_seg:
        from medgen.metrics.brain_mask import create_brain_mask
        pca_errors = []
        for vol in volumes:
            mask = create_brain_mask(vol, threshold=0.05, fill_holes=True, dilate_pixels=0)
            _, err = brain_pca.is_valid(mask)
            pca_errors.append(err)
        result['pca_mean'] = float(np.mean(pca_errors))
        result['pca_pass_rate'] = float(np.mean([e <= brain_pca.error_threshold for e in pca_errors]))

    # Morphological score (seg only)
    if is_seg and real_volumes:
        from medgen.metrics.morphological import compute_morphological_score
        # Binarize generated volumes
        gen_masks = [(vol > 0.5).astype(np.float32) for vol in volumes]
        real_masks = [(vol > 0.5).astype(np.float32) for vol in real_volumes]
        morph = compute_morphological_score(real_masks, gen_masks)
        result['morph_total'] = morph.total
        result['morph_volume'] = morph.volume_dist
        result['morph_count'] = morph.count_dist
        result['n_real_tumors'] = morph.n_real_tumors
        result['n_gen_tumors'] = morph.n_gen_tumors

    return result


def main():
    parser = argparse.ArgumentParser(description='Post-hoc EMA sweep with full evaluation')
    parser.add_argument('--run-dir', required=True,
                        help='Training run directory (contains checkpoint and phema_checkpoints/)')
    parser.add_argument('--data-root', required=True,
                        help='Dataset root (brainmetshare-3)')
    parser.add_argument('--checkpoint', default='checkpoint_latest.pt',
                        help='Checkpoint filename (default: checkpoint_latest.pt)')
    parser.add_argument('--sigma-rels', nargs='+', type=float, default=None,
                        help=f'Sigma_rel values to sweep (default: {DEFAULT_SIGMA_RELS})')
    parser.add_argument('--training-sigma-rels', nargs='+', type=float, default=[0.05, 0.28],
                        help='Sigma_rel values used during training (default: 0.05 0.28)')
    parser.add_argument('--num-volumes', type=int, default=25,
                        help='Number of volumes to generate per config (default: 25)')
    parser.add_argument('--num-steps', type=int, default=32,
                        help='Euler denoising steps (default: 32)')
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--trim-slices', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: {run-dir}/phema_sweep/)')
    parser.add_argument('--include-raw', action='store_true', default=True,
                        help='Also evaluate raw (non-EMA) model as baseline')
    parser.add_argument('--is-seg', action='store_true', default=False,
                        help='Set for segmentation models (binarize output)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = Path(args.run_dir)
    phema_folder = run_dir / 'phema_checkpoints'
    checkpoint_path = run_dir / args.checkpoint
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / 'phema_sweep'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    has_phema_folder = phema_folder.exists() and len(list(phema_folder.glob('*.pt'))) > 0
    if has_phema_folder:
        num_snapshots = len(list(phema_folder.glob('*.pt')))

        # Auto-detect training sigma_rels from checkpoint config
        ckpt_data = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        ckpt_cfg = ckpt_data.get('config', {})
        ema_cfg = ckpt_cfg.get('training', {}).get('ema', {})
        detected_sigma_rels = ema_cfg.get('sigma_rels', None)
        del ckpt_data

        if detected_sigma_rels is not None:
            training_sigma_rels = tuple(float(s) for s in detected_sigma_rels)
            logger.info(f"Auto-detected training sigma_rels from checkpoint: {list(training_sigma_rels)}")
        else:
            training_sigma_rels = tuple(args.training_sigma_rels)
            logger.info(f"Using provided training sigma_rels: {list(training_sigma_rels)}")

        # Evaluate at ALL training sigma_rels (the only stable synthesis points)
        sigma_rels = args.sigma_rels or list(training_sigma_rels)
        logger.info(f"Found {num_snapshots} PostHocEMA snapshots, evaluating at sigma_rels={sigma_rels}")
    else:
        logger.warning(
            f"No phema_checkpoints/ in {run_dir}. "
            f"Falling back to live EMA models from checkpoint (training sigma_rels only)."
        )
        # Override sigma_rels to only use training values
        sigma_rels = list(args.training_sigma_rels)
        logger.info(f"Will evaluate: raw + live EMA at sigma_rels={sigma_rels}")

    # ── Load model ───────────────────────────────────────────────────
    logger.info(f"Loading model from {checkpoint_path}")
    model = load_diffusion_model(
        str(checkpoint_path), device=device,
        compile_model=False, spatial_dims=3,
    )

    # ── Setup strategy ───────────────────────────────────────────────
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000, image_size=args.image_size,
        depth_size=args.depth, spatial_dims=3,
    )

    # ── Load conditioning + noise (shared across all evaluations) ────
    data_root = Path(args.data_root)
    splits = discover_splits(data_root, modality='bravo' if not args.is_seg else 'seg')

    logger.info(f"Loading {args.num_volumes} conditioning masks...")
    cond_list = load_conditioning(splits.get('val', splits.get('test', next(iter(splits.values())))),
                                  args.num_volumes, args.depth)

    logger.info(f"Generating {args.num_volumes} noise tensors (seed={args.seed})...")
    out_channels = 1
    noise_list = generate_noise_tensors(
        args.num_volumes, args.depth, args.image_size, device, args.seed,
        out_channels=out_channels,
    )

    # ── Reference features ───────────────────────────────────────────
    logger.info("Loading reference features...")
    cache_dir = output_dir / 'reference_features'
    needed_splits = {'test': splits['test']} if 'test' in splits else {'val': splits['val']}
    ref_features = get_or_cache_reference_features(
        needed_splits, cache_dir, device, args.depth, args.trim_slices, args.image_size,
        modality='bravo' if not args.is_seg else 'seg', build_all=False,
    )

    # ── Real volumes for FWD ─────────────────────────────────────────
    import nibabel as nib
    real_volumes = []
    modality = 'bravo' if not args.is_seg else 'seg'
    for split_name in ['test', 'val']:
        if split_name not in splits:
            continue
        for fp in sorted(splits[split_name].glob(f'*/{modality}.nii.gz')):
            vol = nib.load(str(fp)).get_fdata().astype(np.float32)
            if vol.max() > 0:
                vol = vol / vol.max()
            vol = np.transpose(vol, (2, 0, 1))
            if vol.shape[0] < args.depth:
                vol = np.pad(vol, ((0, args.depth - vol.shape[0]), (0, 0), (0, 0)))
            elif vol.shape[0] > args.depth:
                vol = vol[:args.depth]
            real_volumes.append(vol)
        break  # Use first available split
    logger.info(f"Loaded {len(real_volumes)} real volumes for FWD")

    # ── PCA model ────────────────────────────────────────────────────
    brain_pca = None
    if not args.is_seg:
        pca_path = Path(args.data_root).parent / f'brain_pca_{args.image_size}x{args.image_size}x{args.depth}.npz'
        if not pca_path.exists():
            # Try repo root
            pca_path = run_dir.parents[3] / 'data' / f'brain_pca_{args.image_size}x{args.image_size}x{args.depth}.npz'
        if pca_path.exists():
            from medgen.metrics.brain_mask import BrainPCAModel
            brain_pca = BrainPCAModel(pca_path)
            logger.info(f"Brain PCA loaded (threshold={brain_pca.error_threshold:.6f})")

    # ── Create PostHocEMA ────────────────────────────────────────────
    phema = None
    ema_state = None

    if has_phema_folder:
        logger.info(f"Creating PostHocEMA (training sigma_rels={list(training_sigma_rels)})")
        phema = create_phema(model, str(phema_folder), sigma_rels=training_sigma_rels)
    else:
        # Load live EMA state from checkpoint for fallback
        ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        ema_state = ckpt.get('ema_state_dict')
        if ema_state is None:
            logger.warning("No EMA state in checkpoint either — only raw model available")
            sigma_rels = []
        del ckpt

    solver_cfg = SolverConfig(solver='euler', steps=args.num_steps)

    # ── Evaluate raw model (baseline) ────────────────────────────────
    all_results = {}

    if args.include_raw:
        logger.info(f"\n{'=' * 60}")
        logger.info("  Evaluating: RAW model (no EMA)")
        logger.info(f"{'=' * 60}")

        # Reload raw weights (phema creation may have modified model)
        raw_model = load_diffusion_model(
            str(checkpoint_path), device=device,
            compile_model=False, spatial_dims=3,
        )
        result = evaluate_model(
            raw_model, strategy, noise_list, cond_list, solver_cfg, device,
            ref_features, real_volumes, brain_pca, args.trim_slices, args.is_seg,
        )
        result['sigma_rel'] = 'raw'
        all_results['raw'] = result

        del raw_model
        torch.cuda.empty_cache()
        gc.collect()

        _log_result('raw', result)

    # ── Sweep sigma_rels ─────────────────────────────────────────────
    for sigma_rel in sigma_rels:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Evaluating: sigma_rel = {sigma_rel:.4f}")
        logger.info(f"{'=' * 60}")

        try:
            if phema is not None:
                # Full synthesis from snapshots
                model = synthesize_model(phema, model, sigma_rel, device)
            elif ema_state is not None:
                # Fallback: load live EMA model from checkpoint
                # PostHocEMA stores multiple KarrasEMA models; index by sigma_rel
                idx = list(training_sigma_rels).index(sigma_rel)
                # Reconstruct PostHocEMA to access the live model
                phema_live = PostHocEMA(
                    model, sigma_rels=training_sigma_rels,
                    checkpoint_every_num_steps='manual',
                    checkpoint_folder=str(run_dir / '_dummy'),
                )
                phema_live.load_state_dict(ema_state)
                # Get the live EMA model at this index
                model.load_state_dict(phema_live.ema_models[idx].ema_model.state_dict())
                model.to(device)
                model.eval()
                del phema_live
            else:
                logger.error(f"No EMA data available for sigma_rel={sigma_rel}")
                continue
        except Exception as e:
            logger.error(f"Failed to synthesize sigma_rel={sigma_rel}: {e}")
            all_results[f'sigma_{sigma_rel:.4f}'] = {'sigma_rel': sigma_rel, 'error': str(e)}
            continue

        result = evaluate_model(
            model, strategy, noise_list, cond_list, solver_cfg, device,
            ref_features, real_volumes, brain_pca, args.trim_slices, args.is_seg,
        )
        result['sigma_rel'] = sigma_rel
        all_results[f'sigma_{sigma_rel:.4f}'] = result

        _log_result(f'sigma_rel={sigma_rel:.4f}', result)

        gc.collect()
        torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────────
    results_path = output_dir / 'phema_sweep_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")

    # ── Summary table ────────────────────────────────────────────────
    logger.info("\n" + "=" * 110)
    logger.info("POST-HOC EMA SWEEP RESULTS")
    logger.info("=" * 110)
    header = (f"{'Config':>15}  {'FID':>8}  {'FID_RIN':>8}  {'KID':>10}  "
              f"{'CMMD':>8}  {'FWD':>8}  {'FWD_hi':>8}  {'PCA_err':>8}  {'PCA%':>6}")
    logger.info(header)
    logger.info("-" * 120)

    for name, r in all_results.items():
        if 'error' in r:
            logger.info(f"{name:>15}  {'ERROR':>8}  {r['error']}")
            continue
        pca_pct = f"{r.get('pca_pass_rate', 0) * 100:.0f}%" if 'pca_pass_rate' in r else 'N/A'
        pca_err = f"{r.get('pca_mean', 0):.6f}" if 'pca_mean' in r else 'N/A'
        fwd_str = f"{r.get('fwd', 0):.4f}" if 'fwd' in r else 'N/A'
        fwd_hi = f"{r.get('fwd_high', 0):.4f}" if 'fwd_high' in r else 'N/A'
        logger.info(
            f"{name:>15}  {r['fid']:>8.2f}  {r['fid_radimagenet']:>8.2f}  "
            f"{r['kid_mean']:>10.6f}  {r['cmmd']:>8.4f}  "
            f"{fwd_str:>8}  {fwd_hi:>8}  {pca_err:>8}  {pca_pct:>6}"
        )

    # Find best sigma_rel by FID
    ema_results = {k: v for k, v in all_results.items() if k != 'raw' and 'error' not in v}
    if ema_results:
        best_name = min(ema_results, key=lambda k: ema_results[k]['fid'])
        best = ema_results[best_name]
        logger.info(f"\nBest by FID: {best_name} (FID={best['fid']:.2f})")

    logger.info("=" * 110)


def _log_result(name: str, r: dict) -> None:
    """Log a single result line."""
    fwd = f"FWD={r.get('fwd', 0):.4f}" if 'fwd' in r else ""
    pca = f"PCA={r.get('pca_pass_rate', 0) * 100:.0f}%" if 'pca_pass_rate' in r else ""
    logger.info(
        f"  FID={r['fid']:.2f}  FID_RIN={r['fid_radimagenet']:.2f}  "
        f"KID={r['kid_mean']:.6f}  CMMD={r['cmmd']:.4f}  {fwd}  {pca}"
    )


if __name__ == '__main__':
    main()
