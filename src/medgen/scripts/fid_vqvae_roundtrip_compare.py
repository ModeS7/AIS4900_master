#!/usr/bin/env python3
"""FID comparison: original synth vs VQ-VAE-roundtripped synth vs real.

For one or more input directories of generated bravo volumes:
  1. Roundtrip the first N volumes through a trained 3D VQ-VAE.
  2. Compute FID(real, original_M) — established baseline using M originals.
  3. Compute FID(real, roundtrip_N) — does the perceptual quality
     improvement we measured with LPIPS show up in FID too?
  4. Save roundtripped volumes + a JSON summary.

Usage:
    python -m medgen.scripts.fid_vqvae_roundtrip_compare \\
        --compression-checkpoint /path/to/vqvae_3d/checkpoint_latest.pt \\
        --input-dirs <dir1> <dir2> \\
        --real-dir /path/to/test_new \\
        --output-dir runs/eval/fid_vqvae_compare_$(date +%Y%m%d-%H%M%S) \\
        --num-roundtrip 20 \\
        --num-baseline 100
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


def find_volumes(root: Path, n: int) -> list[Path]:
    files = sorted(root.glob("*/bravo.nii.gz"))
    if not files:
        files = sorted(root.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No NIfTI files under {root}")
    return files[:n]


def stack_volumes(files: list[Path], depth: int) -> torch.Tensor:
    """Load volumes and stack to [B, 1, D, H, W] on CPU."""
    arrs = [load_volume(f, depth) for f in files]
    return torch.from_numpy(np.stack(arrs, axis=0)).unsqueeze(1)  # [B,1,D,H,W]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--compression-checkpoint', required=True)
    parser.add_argument('--compression-type', default='vqvae',
                        choices=['auto', 'vae', 'vqvae', 'dcae'])
    parser.add_argument('--input-dirs', nargs='+', required=True,
                        help='One or more dirs of generated bravo volumes')
    parser.add_argument('--real-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-roundtrip', type=int, default=20,
                        help='Volumes per dir to roundtrip + FID-evaluate')
    parser.add_argument('--num-baseline', type=int, default=100,
                        help='Volumes per dir for baseline FID')
    parser.add_argument('--num-real', type=int, default=0,
                        help='Real volumes for FID reference (0 = all available)')
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--fid-network', default='radimagenet',
                        choices=['imagenet', 'radimagenet'])
    parser.add_argument('--fid-chunk', type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Load VQ-VAE ──
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

    # ── Load real reference ──
    real_files = sorted(Path(args.real_dir).glob("*/bravo.nii.gz"))
    if args.num_real:
        real_files = real_files[:args.num_real]
    log.info(f"Real reference: {len(real_files)} volumes from {args.real_dir}")
    real_t = stack_volumes(real_files, args.depth)  # [N_real,1,D,H,W]

    # ── Feature extractor (loaded once) ──
    log.info(f"Loading FID feature extractor: ResNet50 / {args.fid_network}")
    extractor = ResNet50Features(
        device=device, network_type=args.fid_network, compile_model=False,
    )

    # ── Per input dir ──
    results = {
        'compression_checkpoint': args.compression_checkpoint,
        'fid_network': args.fid_network,
        'num_real': len(real_files),
        'num_baseline': args.num_baseline,
        'num_roundtrip': args.num_roundtrip,
        'per_dir': {},
    }

    for in_dir_str in args.input_dirs:
        in_dir = Path(in_dir_str)
        tag = in_dir.name
        log.info(f"\n========== {tag} ==========")
        recon_dir = out_dir / 'volumes' / tag
        recon_dir.mkdir(parents=True, exist_ok=True)

        # Originals: load up to num_baseline (we'll use the same volumes for both
        # baseline FID and the roundtrip subset — first num_roundtrip get encoded).
        baseline_files = find_volumes(in_dir, args.num_baseline)
        log.info(f"Baseline originals: {len(baseline_files)} volumes")
        orig_baseline_t = stack_volumes(baseline_files, args.depth)

        # ── Roundtrip first N ──
        n_rt = min(args.num_roundtrip, len(baseline_files))
        log.info(f"Roundtripping first {n_rt} volumes...")
        rt_arrs: list[np.ndarray] = []
        t0 = time.time()
        for i in range(n_rt):
            x = orig_baseline_t[i:i + 1].to(device).float()  # [1,1,D,H,W]
            with torch.no_grad():
                z = space.encode(x)
                x_hat = space.decode(z).clamp(0, 1)
            arr = x_hat.squeeze().cpu().numpy()
            rt_arrs.append(arr)
            subj = baseline_files[i].parent.name
            save_nifti(np.transpose(arr, (1, 2, 0)),
                       str(recon_dir / f"{subj}_roundtrip.nii.gz"))
            if (i + 1) % 5 == 0:
                log.info(f"  {i + 1}/{n_rt} done ({time.time() - t0:.1f}s)")
        rt_t = torch.from_numpy(np.stack(rt_arrs, axis=0)).unsqueeze(1)  # [N,1,D,H,W]
        log.info(f"Roundtrip wall time: {time.time() - t0:.1f}s")

        # ── FID computations ──
        log.info("Computing FID(real, original baseline)...")
        fid_baseline = compute_fid_3d(real_t, orig_baseline_t, extractor,
                                      chunk_size=args.fid_chunk)

        # FID on roundtripped subset
        log.info("Computing FID(real, roundtrip subset)...")
        fid_rt = compute_fid_3d(real_t, rt_t, extractor, chunk_size=args.fid_chunk)

        # Apples-to-apples: also FID(real, original first-N) using the same N as roundtrip
        log.info(f"Computing FID(real, original first-{n_rt})...")
        orig_subset_t = orig_baseline_t[:n_rt]
        fid_orig_subset = compute_fid_3d(real_t, orig_subset_t, extractor,
                                         chunk_size=args.fid_chunk)

        delta_subset = fid_rt - fid_orig_subset
        delta_baseline = fid_rt - fid_baseline
        log.info(
            f"\n{tag} FID summary:\n"
            f"  baseline (orig N={len(baseline_files)}):   {fid_baseline:.3f}\n"
            f"  orig subset (N={n_rt}):                     {fid_orig_subset:.3f}\n"
            f"  roundtrip (N={n_rt}):                       {fid_rt:.3f}\n"
            f"  Δ vs orig subset (apples-to-apples):        {delta_subset:+.3f}  "
            f"({'better' if delta_subset < 0 else 'worse'})\n"
            f"  Δ vs baseline (subset effect confounded):   {delta_baseline:+.3f}"
        )

        results['per_dir'][tag] = {
            'input_dir': str(in_dir),
            'n_baseline': len(baseline_files),
            'n_roundtrip': n_rt,
            'fid_baseline': fid_baseline,
            'fid_orig_subset': fid_orig_subset,
            'fid_roundtrip': fid_rt,
            'delta_subset': delta_subset,
            'delta_baseline': delta_baseline,
        }

        # Free memory before next dir
        del orig_baseline_t, rt_t, orig_subset_t

    with open(out_dir / 'fid_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {out_dir / 'fid_results.json'}")
    log.info(f"Roundtripped volumes under: {out_dir / 'volumes'}/<tag>/")


if __name__ == '__main__':
    main()
