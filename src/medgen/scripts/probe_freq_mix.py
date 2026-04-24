#!/usr/bin/env python3
"""Frequency-mixing probe: HP real + LP synth.

Supervisor's idea: combine low-frequency structure from synthetic volumes
with high-frequency texture from real volumes, using Gaussian blur as the
frequency filter. Output is visibly sharp + real-textured but anatomically
incoherent (real vessels end up in wrong places).

Pipeline per synth volume:
    synth_LP = GaussianBlur(synth, σ)          # smooth structure
    real_HP  = real - GaussianBlur(real, σ)    # texture residual
    mixed    = synth_LP + real_HP

Outputs saved per σ:
    <output>/
        sigma_X.X/
            mixed/<subj>/bravo.nii.gz
            spectrum.png  (synth vs mixed vs real spectrum overlay)
        lpips_fid.json    (LPIPS and band-energy vs real)

Usage (local):
    python -m medgen.scripts.probe_freq_mix \\
        --synth-dir /home/mode/NTNU/MedicalDataSets/generated/exp1_1_bravo_imagenet_525 \\
        --real-dir  /home/mode/NTNU/MedicalDataSets/brainmetshare-3/test_new \\
        --output-dir runs/eval/freq_mix_probe_$(date +%Y%m%d-%H%M%S) \\
        --sigmas 2 4 6 \\
        --num-volumes 5
"""
import argparse
import json
import logging
import random
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from medgen.data.utils import save_nifti
from medgen.metrics.quality import compute_lpips_3d
from medgen.scripts.analyze_generation_spectrum import (
    BANDS,
    compute_radial_power_spectrum_3d,
    load_volume,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def find_volumes(root: Path, n: int | None = None) -> list[Path]:
    files = sorted(root.glob("*/bravo.nii.gz"))
    if not files:
        files = sorted(root.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No NIfTI files under {root}")
    return files[:n] if n else files


def gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D Gaussian blur via three 1D convolutions. x: [1,1,D,H,W]."""
    if sigma <= 0:
        return x
    radius = max(1, int(np.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Apply along each spatial axis separately (axes: D=2, H=3, W=4 for 5D tensor).
    def _conv_axis(vol: torch.Tensor, axis: int) -> torch.Tensor:
        shape = [1, 1, 1, 1, 1]
        shape[axis] = kernel_1d.numel()
        k = kernel_1d.view(*shape)
        pad = [0, 0, 0, 0, 0, 0]  # W_left, W_right, H_left, H_right, D_left, D_right
        idx = (4 - axis) * 2  # D→4, H→2, W→0  (F.pad axis ordering)
        pad[idx] = radius
        pad[idx + 1] = radius
        vol = F.pad(vol, pad, mode='reflect')
        return F.conv3d(vol, k)

    out = _conv_axis(x, 2)
    out = _conv_axis(out, 3)
    out = _conv_axis(out, 4)
    return out


def band_energy(bins: np.ndarray, power: np.ndarray) -> dict[str, float]:
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (bins >= lo) & (bins < hi)
        out[name] = float(power[mask].sum()) if mask.any() else 0.0
    return out


def radial_average(volumes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
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
    parser.add_argument('--synth-dir', required=True)
    parser.add_argument('--real-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sigmas', type=float, nargs='+', default=[2.0, 4.0, 6.0],
                        help='Gaussian σ (voxels) for LP/HP split. Smaller σ → '
                             'narrower LP, more HF replaced from real.')
    parser.add_argument('--hf-smooth-sigma', type=float, default=0.0,
                        help='Extra Gaussian σ applied to real HF before adding. '
                             '0=no smoothing (pure transplant, anatomically noisy); '
                             '>0=spatially diffuse real HF to destroy specific '
                             'anatomy while keeping statistical texture. Converts '
                             'HP(real) → bandpass(real).')
    parser.add_argument('--hf-scale', type=float, default=1.0,
                        help='Amplitude scale for HF contribution (default 1.0). '
                             'Try 0.5 or 0.3 to dilute real HF contribution.')
    parser.add_argument('--num-volumes', type=int, default=5)
    parser.add_argument('--num-real-ref', type=int, default=10,
                        help='Real volumes averaged into reference spectrum')
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────
    synth_files = find_volumes(Path(args.synth_dir), args.num_volumes)
    real_files_all = find_volumes(Path(args.real_dir), None)
    real_files_ref = real_files_all[:args.num_real_ref]
    log.info(f"Synth: {len(synth_files)}  |  Real reference (for spectrum): {len(real_files_ref)}")

    synth_np = [load_volume(f, args.depth) for f in synth_files]  # [D,H,W]
    real_np_ref = [load_volume(f, args.depth) for f in real_files_ref]

    # Pair each synth with a specific real volume for HF donation
    real_pair_indices = [random.randrange(len(real_files_all)) for _ in synth_files]
    real_pair_files = [real_files_all[i] for i in real_pair_indices]
    real_np_pairs = [load_volume(f, args.depth) for f in real_pair_files]
    log.info("Pairings (synth → real HF donor):")
    for sf, rf in zip(synth_files, real_pair_files):
        log.info(f"  {sf.parent.name}  ←  {rf.parent.name}")

    # ── Reference spectra ──────────────────────────────────────────
    bins, real_spec = radial_average(real_np_ref)
    _, synth_spec = radial_average(synth_np)

    # LPIPS tensors on GPU (once, reuse)
    def _to5d(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device).float()

    real_stack_t = torch.cat([_to5d(a) for a in real_np_ref], dim=0)      # [N_ref,1,D,H,W]
    synth_stack_t = torch.cat([_to5d(a) for a in synth_np], dim=0)        # [N,1,D,H,W]

    # Baseline LPIPS(real_ref, synth): average across cross-pairs
    log.info("\nComputing baseline LPIPS(real, synth)...")
    baseline_lpips_vals = []
    for i in range(real_stack_t.shape[0]):
        for j in range(synth_stack_t.shape[0]):
            val = compute_lpips_3d(real_stack_t[i:i + 1], synth_stack_t[j:j + 1],
                                   device=device, chunk_size=32)
            baseline_lpips_vals.append(float(val))
    baseline_lpips = float(np.mean(baseline_lpips_vals))
    log.info(f"Baseline LPIPS(real, synth) = {baseline_lpips:.4f} "
             f"(cross-pairs, mean over {len(baseline_lpips_vals)} comparisons)")

    # ── Per-sigma loop ─────────────────────────────────────────────
    all_results = {
        'baseline_lpips_real_vs_synth': baseline_lpips,
        'baseline_band_energy': {
            'real': band_energy(bins, real_spec),
            'synth': band_energy(bins, synth_spec),
        },
        'per_sigma': {},
    }

    for sigma in args.sigmas:
        log.info(f"\n========== σ = {sigma}  "
                 f"(hf_smooth={args.hf_smooth_sigma}, hf_scale={args.hf_scale}) ==========")
        tag = f"sigma_{sigma:.1f}"
        if args.hf_smooth_sigma > 0:
            tag += f"_smooth{args.hf_smooth_sigma:.1f}"
        if args.hf_scale != 1.0:
            tag += f"_scale{args.hf_scale:.2f}"
        sig_dir = out_dir / tag
        mixed_vol_dir = sig_dir / 'mixed'
        mixed_vol_dir.mkdir(parents=True, exist_ok=True)

        mixed_np: list[np.ndarray] = []
        for synth_arr, real_arr, sf, rf in zip(
            synth_np, real_np_pairs, synth_files, real_pair_files,
        ):
            synth_t = _to5d(synth_arr)
            real_t = _to5d(real_arr)
            synth_lp = gaussian_blur_3d(synth_t, sigma)
            real_hp = real_t - gaussian_blur_3d(real_t, sigma)
            # Spatially diffuse HF to destroy specific anatomy while keeping texture stats.
            if args.hf_smooth_sigma > 0:
                real_hp = gaussian_blur_3d(real_hp, args.hf_smooth_sigma)
            real_hp = real_hp * args.hf_scale
            mixed_t = (synth_lp + real_hp).clamp(0, 1)
            mixed_arr = mixed_t.squeeze().cpu().numpy()
            mixed_np.append(mixed_arr)

            subj = sf.parent.name
            subj_dir = mixed_vol_dir / subj
            subj_dir.mkdir(parents=True, exist_ok=True)
            save_nifti(np.transpose(mixed_arr, (1, 2, 0)),
                       str(subj_dir / 'bravo.nii.gz'))
            # Also stash the donor id as a breadcrumb
            (subj_dir / 'hf_donor.txt').write_text(rf.parent.name + "\n")

        # Spectrum of mixed set (averaged)
        _, mixed_spec = radial_average(mixed_np)

        # LPIPS(real_ref, mixed) — cross pairs
        mixed_stack_t = torch.cat([_to5d(a) for a in mixed_np], dim=0)
        lpips_vals = []
        for i in range(real_stack_t.shape[0]):
            for j in range(mixed_stack_t.shape[0]):
                val = compute_lpips_3d(real_stack_t[i:i + 1], mixed_stack_t[j:j + 1],
                                       device=device, chunk_size=32)
                lpips_vals.append(float(val))
        mixed_lpips = float(np.mean(lpips_vals))
        lpips_delta_pct = 100 * (mixed_lpips - baseline_lpips) / baseline_lpips

        # Band ratios
        bands_mixed = band_energy(bins, mixed_spec)
        bands_real = band_energy(bins, real_spec)
        bands_synth = band_energy(bins, synth_spec)
        verdict_rows = []
        for b in BANDS:
            s = bands_synth[b] / bands_real[b] if bands_real[b] else float('nan')
            m = bands_mixed[b] / bands_real[b] if bands_real[b] else float('nan')
            verdict_rows.append(
                f"  {b:10s}  synth/real={s:6.3f}  mixed/real={m:6.3f}  Δ={m - s:+.3f}"
            )
        log.info("LPIPS: baseline=%.4f  mixed=%.4f  Δ=%+.2f%%",
                 baseline_lpips, mixed_lpips, lpips_delta_pct)
        log.info("Band ratios:\n" + "\n".join(verdict_rows))

        # Spectrum plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for name, spec, color in [
            ('real', real_spec, 'black'),
            ('synth', synth_spec, 'gray'),
            (f'mixed (σ={sigma})', mixed_spec, 'tab:red'),
        ]:
            ax1.loglog(bins[1:], spec[1:], label=name, color=color, linewidth=2)
        ax1.set_xlabel('Radial frequency (Nyquist=0.5)')
        ax1.set_ylabel('Power')
        ax1.set_title(f'Spectrum overlay (σ={sigma})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.semilogx(bins[1:], synth_spec[1:] / real_spec[1:],
                     label='synth / real', color='gray', linewidth=2)
        ax2.semilogx(bins[1:], mixed_spec[1:] / real_spec[1:],
                     label='mixed / real', color='tab:red', linewidth=2)
        ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Radial frequency (Nyquist=0.5)')
        ax2.set_ylabel('Power ratio to real')
        ax2.set_title(f'Ratio to real (σ={sigma})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(sig_dir / 'spectrum.png', dpi=120)
        plt.close(fig)

        all_results['per_sigma'][f'{sigma}'] = {
            'sigma': sigma,
            'lpips_mean': mixed_lpips,
            'lpips_delta_vs_baseline_pct': lpips_delta_pct,
            'band_energy_mixed': bands_mixed,
            'band_verdict': verdict_rows,
        }

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSaved: {out_dir / 'results.json'}")
    log.info(f"Per-sigma dirs with mixed volumes: {out_dir}/sigma_*.*/mixed/<subj>/bravo.nii.gz")


if __name__ == '__main__':
    main()
