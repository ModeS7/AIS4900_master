#!/usr/bin/env python3
"""VQ-VAE roundtrip probe on synthetic diffusion outputs.

Hypothesis: vector quantization snaps OOD synthetic latents to the nearest
codebook entries (trained on real data), acting as a manifold projection
that might partially recover real-data high-frequency content.

Cheap test (≤3 volumes): encode through trained 3D VQ-VAE, decode, compute
radial 3D power spectrum. Compare (synthetic, roundtripped, real). If the
mid/high band doesn't move toward real, hypothesis is dead.

Usage:
    python -m medgen.scripts.probe_vqvae_roundtrip \\
        --compression-checkpoint /path/to/vqvae_3d/checkpoint_best.pt \\
        --input-dir  /path/to/compare_.../exp1_1_1000 \\
        --real-dir   /path/to/brainmetshare-3/test_new \\
        --output-dir runs/eval/vqvae_roundtrip_probe \\
        --num-volumes 3
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from medgen.data.loaders.compression_detection import load_compression_model
from medgen.data.utils import save_nifti
from medgen.diffusion.spaces import LatentSpace
from medgen.scripts.analyze_generation_spectrum import (
    BANDS,
    compute_radial_power_spectrum_3d,
    load_volume,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def find_volumes(root: Path, n: int) -> list[Path]:
    files = sorted(root.glob("*/bravo.nii.gz"))
    if not files:
        files = sorted(root.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No NIfTI files under {root}")
    return files[:n]


def band_energy(bins: np.ndarray, power: np.ndarray) -> dict[str, float]:
    """Integrate power over named frequency bands."""
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (bins >= lo) & (bins < hi)
        out[name] = float(power[mask].sum()) if mask.any() else 0.0
    return out


def average_spectrum(volumes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    bins_ref = None
    accum = None
    for v in volumes:
        bins, power = compute_radial_power_spectrum_3d(v)
        if accum is None:
            bins_ref = bins
            accum = np.zeros_like(power)
        accum = accum + power
    return bins_ref, accum / max(1, len(volumes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--compression-checkpoint', required=True)
    parser.add_argument('--compression-type', default='auto',
                        choices=['auto', 'vae', 'vqvae', 'dcae'])
    parser.add_argument('--input-dir', required=True,
                        help='Synthetic volumes dir (exp1_1_1000 outputs)')
    parser.add_argument('--real-dir', required=True,
                        help='Real held-out volumes for reference spectrum')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-volumes', type=int, default=3)
    parser.add_argument('--num-real', type=int, default=10,
                        help='Real volumes averaged for reference spectrum')
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--image-size', type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = out_dir / 'volumes'
    recon_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # ── Load VQ-VAE (wrapped in LatentSpace for clean encode/decode) ──
    comp_model, ctype, comp_sd, sf, latent_ch = load_compression_model(
        args.compression_checkpoint, args.compression_type, device, spatial_dims=3,
    )
    logger.info(f"Compression: {ctype} | sf={sf} | latent_ch={latent_ch} | comp_sd={comp_sd}")
    if ctype != 'vqvae':
        logger.warning(f"Loaded {ctype}, not vqvae — roundtrip will skip quantization snap.")

    slicewise = (comp_sd == 2 and 3 == 3)
    space = LatentSpace(
        compression_model=comp_model, device=device,
        deterministic=True, spatial_dims=comp_sd,
        compression_type=ctype, scale_factor=sf, latent_channels=latent_ch,
        slicewise_encoding=slicewise,
        latent_shift=None, latent_scale=None,  # no normalization — raw roundtrip
    )

    def roundtrip_volume(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (reconstruction, latent) for one [D,H,W] volume."""
        x = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            z = space.encode(x)
            x_hat = space.decode(z)
        return (
            x_hat.squeeze().cpu().numpy().clip(0, 1),
            z.squeeze(0).cpu().numpy(),  # [C, D/sf, H/sf, W/sf]
        )

    # ── Synthetic: load, encode, decode ──
    synth_files = find_volumes(Path(args.input_dir), args.num_volumes)
    logger.info(f"{len(synth_files)} synthetic volumes")

    synth_np: list[np.ndarray] = []
    synth_recon_np: list[np.ndarray] = []
    synth_latents: list[np.ndarray] = []
    for idx, fpath in enumerate(synth_files):
        logger.info(f"[synth {idx + 1}/{len(synth_files)}] {fpath.name}")
        vol = load_volume(fpath, args.depth)
        synth_np.append(vol)
        recon, z = roundtrip_volume(vol)
        synth_recon_np.append(recon)
        synth_latents.append(z)
        subj = fpath.parent.name if fpath.parent != Path(args.input_dir) else fpath.stem
        save_nifti(np.transpose(recon, (1, 2, 0)),
                   str(recon_dir / f"synth_{subj}_roundtrip.nii.gz"))

    # ── Real: load, and also roundtrip a subset to measure decoder ceiling ──
    real_files = find_volumes(Path(args.real_dir), args.num_real)
    logger.info(f"{len(real_files)} real volumes for reference")
    real_np = [load_volume(f, args.depth) for f in real_files]

    # Roundtrip the first `num_volumes` real volumes to measure the ceiling.
    real_recon_np: list[np.ndarray] = []
    real_latents: list[np.ndarray] = []
    for idx in range(min(args.num_volumes, len(real_np))):
        logger.info(f"[real  {idx + 1}/{args.num_volumes}] {real_files[idx].name}")
        recon, z = roundtrip_volume(real_np[idx])
        real_recon_np.append(recon)
        real_latents.append(z)
        subj = real_files[idx].parent.name
        save_nifti(np.transpose(recon, (1, 2, 0)),
                   str(recon_dir / f"real_{subj}_roundtrip.nii.gz"))

    # ── Spectra ──
    bins, synth_spec = average_spectrum(synth_np)
    _, recon_spec = average_spectrum(synth_recon_np)
    _, real_spec = average_spectrum(real_np)
    _, real_recon_spec = average_spectrum(real_recon_np)

    # ── Latent stats (per-channel mean/std/min/max across volumes) ──
    def latent_stats(latents: list[np.ndarray]) -> dict[str, list[float]]:
        arr = np.stack(latents, axis=0)  # [N, C, ...]
        C = arr.shape[1]
        return {
            'mean_per_channel': [float(arr[:, c].mean()) for c in range(C)],
            'std_per_channel':  [float(arr[:, c].std())  for c in range(C)],
            'min_per_channel':  [float(arr[:, c].min())  for c in range(C)],
            'max_per_channel':  [float(arr[:, c].max())  for c in range(C)],
            'global_mean': float(arr.mean()),
            'global_std':  float(arr.std()),
        }
    stats_synth = latent_stats(synth_latents)
    stats_real = latent_stats(real_latents)
    logger.info(
        f"Latent global mean  — real={stats_real['global_mean']:.4f}  "
        f"synth={stats_synth['global_mean']:.4f}  "
        f"Δ={stats_synth['global_mean'] - stats_real['global_mean']:+.4f}"
    )
    logger.info(
        f"Latent global std   — real={stats_real['global_std']:.4f}  "
        f"synth={stats_synth['global_std']:.4f}  "
        f"Δ={stats_synth['global_std'] - stats_real['global_std']:+.4f}"
    )
    logger.info("Per-channel mean (real | synth):")
    for c, (rm, sm) in enumerate(zip(stats_real['mean_per_channel'],
                                     stats_synth['mean_per_channel'])):
        logger.info(f"  ch{c}: {rm:+.4f} | {sm:+.4f}   Δ={sm - rm:+.4f}")
    logger.info("Per-channel std  (real | synth):")
    for c, (rs, ss) in enumerate(zip(stats_real['std_per_channel'],
                                     stats_synth['std_per_channel'])):
        logger.info(f"  ch{c}: {rs:.4f} | {ss:.4f}   ratio={ss / rs if rs else float('nan'):.3f}")

    # ── Plots ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, spec, color, style in [
        ('real', real_spec, 'black', '-'),
        ('real→VQ-VAE (ceiling)', real_recon_spec, 'tab:green', '--'),
        ('exp1_1_1000 (synthetic)', synth_spec, 'gray', '-'),
        ('synth→VQ-VAE', recon_spec, 'tab:red', '-'),
    ]:
        ax1.loglog(bins[1:], spec[1:], label=name, color=color,
                   linewidth=2, linestyle=style)
    ax1.set_xlabel('Radial frequency (Nyquist=0.5)')
    ax1.set_ylabel('Power')
    ax1.set_title('Radial power spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(bins[1:], real_recon_spec[1:] / real_spec[1:],
                 label='real roundtrip / real (ceiling)',
                 color='tab:green', linestyle='--', linewidth=2)
    ax2.semilogx(bins[1:], synth_spec[1:] / real_spec[1:],
                 label='synthetic / real', color='gray', linewidth=2)
    ax2.semilogx(bins[1:], recon_spec[1:] / real_spec[1:],
                 label='synth roundtrip / real', color='tab:red', linewidth=2)
    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Radial frequency (Nyquist=0.5)')
    ax2.set_ylabel('Power ratio to real')
    ax2.set_title('Ratio to real (1.0 = matched)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / 'spectrum.png', dpi=120)
    fig.savefig(out_dir / 'spectrum.pdf')
    plt.close(fig)

    # ── Numeric summary ──
    results = {
        'compression_type': ctype,
        'n_synthetic': len(synth_np),
        'n_real': len(real_np),
        'n_real_roundtrip': len(real_recon_np),
        'bins': bins.tolist(),
        'spectrum': {
            'real': real_spec.tolist(),
            'real_roundtrip': real_recon_spec.tolist(),
            'synthetic': synth_spec.tolist(),
            'synth_roundtrip': recon_spec.tolist(),
        },
        'band_energy': {
            'real': band_energy(bins, real_spec),
            'real_roundtrip': band_energy(bins, real_recon_spec),
            'synthetic': band_energy(bins, synth_spec),
            'synth_roundtrip': band_energy(bins, recon_spec),
        },
        'latent_stats': {
            'real': stats_real,
            'synthetic': stats_synth,
        },
    }

    # Quick verdict: ceiling (real→rt/real) and synth→rt vs synth
    real_bands = results['band_energy']['real']
    real_rt_bands = results['band_energy']['real_roundtrip']
    synth_bands = results['band_energy']['synthetic']
    synth_rt_bands = results['band_energy']['synth_roundtrip']
    verdict_rows = []
    for band in BANDS:
        r = real_bands[band]
        rrt = real_rt_bands[band] / r if r else float('nan')   # decoder ceiling
        s = synth_bands[band] / r if r else float('nan')
        srt = synth_rt_bands[band] / r if r else float('nan')
        verdict_rows.append(
            f"  {band:10s}  ceiling={rrt:6.3f}  synth={s:6.3f}  "
            f"synth_rt={srt:6.3f}  headroom(synth_rt→ceiling)={rrt - srt:+.3f}"
        )
    logger.info("Band energy ratios to real:\n" + "\n".join(verdict_rows))
    results['band_verdict'] = verdict_rows

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_dir / 'spectrum.png'}")
    logger.info(f"Saved: {out_dir / 'results.json'}")


if __name__ == '__main__':
    main()
