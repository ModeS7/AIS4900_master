#!/usr/bin/env python3
"""FID test: does VQ-VAE compression equalize synth and real distributions?

Hypothesis: synthetic volumes have a different spectral phenotype than real,
but both become band-limited when passed through a VQ-VAE. Comparing the
*reconstructed* distributions may show smaller FID than the raw comparison.
If so, we can train a refinement model on (real_reconstructed → real) pairs
(unlimited paired data!) and apply it to synth_reconstructed at inference.

Computes four FIDs against real:
    1. FID(real, synth)         — raw baseline
    2. FID(real, synth_rt)      — synth through VQ-VAE, real unchanged
    3. FID(real_rt, synth_rt)   — both through VQ-VAE  ← EQUALIZATION TEST
    4. FID(real, real_rt)       — reconstruction ceiling

Outputs:
    <output-dir>/
        fid_results.json
        volumes/synth_rt/<subj>/bravo.nii.gz
        volumes/real_rt/<subj>/bravo.nii.gz

Usage:
    python -m medgen.scripts.fid_compression_equalize \\
        --compression-checkpoint /path/to/vqvae/ckpt.pt \\
        --input-dir  /path/to/generated/exp1_1_bravo_imagenet_525 \\
        --real-dir   /path/to/brainmetshare-3/test_new \\
        --output-dir runs/eval/fid_eq_$(date +%Y%m%d-%H%M%S)
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from medgen.data.loaders.compression_detection import load_compression_model
from medgen.data.utils import save_nifti
from medgen.diffusion.spaces import LatentSpace
from medgen.metrics.feature_extractors import ResNet50Features
from medgen.metrics.generation_3d import compute_fid_3d
from medgen.scripts.analyze_generation_spectrum import load_volume

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def find_volumes(root: Path, n: int | None = None) -> list[Path]:
    files = sorted(root.glob("*/bravo.nii.gz"))
    if not files:
        files = sorted(root.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No NIfTI files under {root}")
    return files[:n] if n else files


def roundtrip_set(
    files: list[Path], space: LatentSpace, device: torch.device,
    depth: int, save_to: Path | None = None,
) -> torch.Tensor:
    """Roundtrip a list of volume files through the compression model.

    Returns a stacked [B, 1, D, H, W] float32 CPU tensor of reconstructions.
    Optionally saves each reconstruction as bravo.nii.gz under save_to/<subj>/.
    """
    recons: list[np.ndarray] = []
    t0 = time.time()
    for i, f in enumerate(files):
        vol = load_volume(f, depth)  # [D,H,W] in [0,1]
        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device).float()
        with torch.no_grad():
            z = space.encode(x)
            x_hat = space.decode(z).clamp(0, 1)
        arr = x_hat.squeeze().cpu().numpy()
        recons.append(arr)
        if save_to is not None:
            subj = f.parent.name if f.parent != f.parents[-1] else f.stem
            out_dir = save_to / subj
            out_dir.mkdir(parents=True, exist_ok=True)
            save_nifti(np.transpose(arr, (1, 2, 0)), str(out_dir / 'bravo.nii.gz'))
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            log.info(f"  {i + 1}/{len(files)} roundtripped ({time.time() - t0:.1f}s)")
    return torch.from_numpy(np.stack(recons, axis=0)).unsqueeze(1)  # [B,1,D,H,W]


def load_set(files: list[Path], depth: int) -> torch.Tensor:
    """Load volumes as [B,1,D,H,W] CPU tensor."""
    arrs = [load_volume(f, depth) for f in files]
    return torch.from_numpy(np.stack(arrs, axis=0)).unsqueeze(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--compression-checkpoint', required=True)
    parser.add_argument('--compression-type', default='vqvae',
                        choices=['auto', 'vae', 'vqvae', 'dcae'])
    parser.add_argument('--input-dir', required=True,
                        help='Synth volumes dir (e.g. exp1_1_bravo_imagenet_525)')
    parser.add_argument('--real-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-synth', type=int, default=0,
                        help='Cap synth volumes (0 = all)')
    parser.add_argument('--num-real', type=int, default=0,
                        help='Cap real volumes (0 = all)')
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--fid-network', default='radimagenet',
                        choices=['imagenet', 'radimagenet'])
    parser.add_argument('--fid-chunk', type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vols_dir = out_dir / 'volumes'
    vols_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Load compression model ─────────────────────────────────────
    comp_model, ctype, comp_sd, sf, latent_ch = load_compression_model(
        args.compression_checkpoint, args.compression_type, device, spatial_dims=3,
    )
    log.info(f"Compression: {ctype} sf={sf} latent_ch={latent_ch}")
    space = LatentSpace(
        compression_model=comp_model, device=device,
        deterministic=True, spatial_dims=comp_sd,
        compression_type=ctype, scale_factor=sf, latent_channels=latent_ch,
        slicewise_encoding=False,
        latent_shift=None, latent_scale=None,
    )

    # ── Load volume lists ──────────────────────────────────────────
    synth_files = find_volumes(Path(args.input_dir),
                               args.num_synth if args.num_synth else None)
    real_files = find_volumes(Path(args.real_dir),
                              args.num_real if args.num_real else None)
    log.info(f"Synth: {len(synth_files)} volumes from {args.input_dir}")
    log.info(f"Real:  {len(real_files)} volumes from {args.real_dir}")

    # ── Originals (CPU tensors) ────────────────────────────────────
    log.info("Loading originals...")
    synth_t = load_set(synth_files, args.depth)
    real_t = load_set(real_files, args.depth)
    log.info(f"synth_t: {tuple(synth_t.shape)} | real_t: {tuple(real_t.shape)}")

    # ── Roundtrip both sets (saves volumes) ────────────────────────
    log.info("Roundtripping synth...")
    synth_rt_t = roundtrip_set(synth_files, space, device, args.depth,
                               save_to=vols_dir / 'synth_rt')
    log.info("Roundtripping real...")
    real_rt_t = roundtrip_set(real_files, space, device, args.depth,
                              save_to=vols_dir / 'real_rt')

    # ── Feature extractor (single load) ────────────────────────────
    log.info(f"Loading feature extractor: ResNet50 / {args.fid_network}")
    extractor = ResNet50Features(
        device=device, network_type=args.fid_network, compile_model=False,
    )

    # ── Compute the four FIDs ──────────────────────────────────────
    log.info("\n=== Computing FIDs ===")

    log.info("(1) FID(real, synth)         — raw baseline")
    fid_raw = compute_fid_3d(real_t, synth_t, extractor, chunk_size=args.fid_chunk)

    log.info("(2) FID(real, synth_rt)      — synth only through VQ-VAE")
    fid_synth_only = compute_fid_3d(real_t, synth_rt_t, extractor, chunk_size=args.fid_chunk)

    log.info("(3) FID(real_rt, synth_rt)   — both through VQ-VAE (EQUALIZATION)")
    fid_both = compute_fid_3d(real_rt_t, synth_rt_t, extractor, chunk_size=args.fid_chunk)

    log.info("(4) FID(real, real_rt)       — reconstruction ceiling")
    fid_ceiling = compute_fid_3d(real_t, real_rt_t, extractor, chunk_size=args.fid_chunk)

    # ── Summary ────────────────────────────────────────────────────
    results = {
        'compression_checkpoint': args.compression_checkpoint,
        'input_dir': str(Path(args.input_dir).resolve()),
        'real_dir': str(Path(args.real_dir).resolve()),
        'fid_network': args.fid_network,
        'n_synth': len(synth_files),
        'n_real': len(real_files),
        'fid': {
            'real_vs_synth':       fid_raw,
            'real_vs_synth_rt':    fid_synth_only,
            'real_rt_vs_synth_rt': fid_both,       # equalization test
            'real_vs_real_rt':     fid_ceiling,    # reconstruction floor
        },
        'deltas': {
            'equalize_gain':
                fid_raw - fid_both,                # positive = hypothesis works
            'synth_rt_vs_raw_gain':
                fid_raw - fid_synth_only,          # positive = synth roundtrip helped on its own
            'headroom_above_ceiling':
                fid_both - fid_ceiling,            # residual gap in equalized space
        },
    }

    log.info("\n" + "=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    log.info(f"  (1) FID(real,    synth)    = {fid_raw:.4f}")
    log.info(f"  (2) FID(real,    synth_rt) = {fid_synth_only:.4f}")
    log.info(f"  (3) FID(real_rt, synth_rt) = {fid_both:.4f}   ← EQUALIZATION")
    log.info(f"  (4) FID(real,    real_rt)  = {fid_ceiling:.4f}   (ceiling)")
    log.info("")
    log.info(f"  Δ equalize_gain = (1) − (3) = {fid_raw - fid_both:+.4f}")
    log.info(f"    {'→ compression equalizes (hypothesis holds)' if fid_raw - fid_both > 0 else '→ compression does not equalize'}")
    log.info(f"  Δ headroom above ceiling    = {fid_both - fid_ceiling:+.4f}")
    log.info("    (how far synth_rt is from real_rt given VQ-VAE floor)")

    with open(out_dir / 'fid_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved JSON: {out_dir / 'fid_results.json'}")
    log.info(f"Saved reconstructions: {vols_dir}/synth_rt/, {vols_dir}/real_rt/")


if __name__ == '__main__':
    main()
