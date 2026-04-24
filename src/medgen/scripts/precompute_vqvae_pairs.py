#!/usr/bin/env python3
"""Precompute (real, real_rt) pair directory for exp43 VQ-VAE deblur training.

For each subject in brainmetshare-3 {train, val, test, test_new}:
    - Symlink `bravo.nii.gz` → `<out>/<split>/<subj>/clean_bravo.nii.gz` (target)
    - Symlink `seg.nii.gz`   → `<out>/<split>/<subj>/seg.nii.gz`         (loader req.)
    - Roundtrip `bravo.nii.gz` through the VQ-VAE and save as
      `<out>/<split>/<subj>/degraded_000.nii.gz` (training input)

Output structure is compatible with Restoration3DDataset's 'precomputed' mode
so `train_refinement_gan.py --data-root <out>` works unchanged.

Usage:
    python -m medgen.scripts.precompute_vqvae_pairs \\
        --compression-checkpoint /path/to/vqvae/ckpt.pt \\
        --data-root /path/to/brainmetshare-3 \\
        --output-dir /path/to/vqvae_pairs
"""
import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from medgen.data.loaders.compression_detection import load_compression_model
from medgen.data.utils import save_nifti
from medgen.diffusion.spaces import LatentSpace
from medgen.scripts.analyze_generation_spectrum import load_volume

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def symlink_if_missing(src: Path, dst: Path) -> None:
    """Create a relative symlink src → dst, unless dst already exists."""
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(os.path.relpath(src, dst.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--compression-checkpoint', required=True)
    parser.add_argument('--compression-type', default='vqvae',
                        choices=['auto', 'vae', 'vqvae', 'dcae'])
    parser.add_argument('--data-root', required=True,
                        help='Root of brainmetshare-3 (contains train/val/test/test_new)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'val', 'test', 'test_new'])
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip subjects that already have degraded_000.nii.gz')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Load compression ──
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

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {}
    t0 = time.time()
    for split in args.splits:
        split_in = data_root / split
        if not split_in.is_dir():
            log.warning(f"Missing split dir: {split_in} — skipping")
            continue
        split_out = out_root / split
        subjects = sorted(d for d in split_in.iterdir() if d.is_dir())
        log.info(f"\n[{split}] {len(subjects)} subjects")
        done = 0
        for subj_in in subjects:
            bravo_src = subj_in / 'bravo.nii.gz'
            seg_src = subj_in / 'seg.nii.gz'
            if not bravo_src.exists() or not seg_src.exists():
                log.warning(f"  skip {subj_in.name}: missing bravo or seg")
                continue

            subj_out = split_out / subj_in.name
            subj_out.mkdir(parents=True, exist_ok=True)
            clean_link = subj_out / 'clean_bravo.nii.gz'
            seg_link = subj_out / 'seg.nii.gz'
            degraded_path = subj_out / 'degraded_000.nii.gz'

            symlink_if_missing(bravo_src, clean_link)
            symlink_if_missing(seg_src, seg_link)

            if args.skip_existing and degraded_path.exists():
                done += 1
                continue

            # Roundtrip
            vol = load_volume(bravo_src, args.depth)  # [D,H,W] in [0,1]
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device).float()
            with torch.no_grad():
                z = space.encode(x)
                x_hat = space.decode(z).clamp(0, 1)
            arr = x_hat.squeeze().cpu().numpy()
            save_nifti(np.transpose(arr, (1, 2, 0)), str(degraded_path))

            done += 1
            if done % 20 == 0 or done == len(subjects):
                log.info(f"  {done}/{len(subjects)} ({time.time() - t0:.1f}s total)")
        totals[split] = done

    log.info("\n=== Summary ===")
    for split, n in totals.items():
        log.info(f"  {split}: {n} subjects  →  {out_root / split}")
    log.info(f"Total: {sum(totals.values())} pairs in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
