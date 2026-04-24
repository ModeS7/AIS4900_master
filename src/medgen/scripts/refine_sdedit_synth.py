#!/usr/bin/env python3
"""SDEdit-style refinement of synthetic volumes — T1A of the blur-attack plan.

Uses an already-trained pixel-space diffusion model (e.g., exp1_1_1000) to
refine a synth volume *without* any new training:

    noised = scheduler.add_noise(synth, noise, t=t₀·T)
    refined = denoise(noised, t₀ → 0)  # uniform Euler steps

Reuses the `sdedit_denoise()` pattern from `eval_light_sdedit.py`. Sweeps t₀
to find the sweet spot — too low = no change, too high = anatomy destroyed.

Each synth subject must have a matching `seg.nii.gz` (used as conditioning).

Usage:
    python -m medgen.scripts.refine_sdedit_synth \\
        --checkpoint /path/to/exp1_1_1000/checkpoint_latest.pt \\
        --synth-dirs <dir1> <dir2> \\
        --real-dir   /path/to/test_new \\
        --output-dir runs/eval/sdedit_$(date +%Y%m%d-%H%M%S) \\
        --t0-values 0.02 0.05 0.10 0.20 0.30 \\
        --num-steps 24 \\
        --num-volumes 5
"""
import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

from medgen.data.utils import save_nifti
from medgen.diffusion import RFlowStrategy, load_diffusion_model
from medgen.metrics.quality import compute_lpips_3d
from medgen.scripts.analyze_generation_spectrum import (
    BANDS,
    compute_radial_power_spectrum_3d,
    load_volume,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def find_pairs(root: Path, n: int | None = None) -> list[tuple[Path, Path]]:
    """Return (bravo, seg) paths per subject."""
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        b, s = d / 'bravo.nii.gz', d / 'seg.nii.gz'
        if b.exists() and s.exists():
            out.append((b, s))
            if n and len(out) >= n:
                break
    if not out:
        raise SystemExit(f"No (bravo, seg) pairs under {root}")
    return out


def find_real_volumes(root: Path, n: int) -> list[Path]:
    files = sorted(root.glob("*/bravo.nii.gz"))
    return files[:n]


def sdedit_denoise(
    model: torch.nn.Module,
    strategy: RFlowStrategy,
    clean: torch.Tensor,
    seg: torch.Tensor,
    t0: float,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """SDEdit at noise level t0. Mirrors eval_light_sdedit.py:89-132."""
    T = strategy.scheduler.num_train_timesteps
    noise = torch.randn_like(clean)
    t_scaled = torch.tensor([int(t0 * T)], device=device)
    noisy = strategy.scheduler.add_noise(clean, noise, t_scaled)

    d, h, w = clean.shape[2], clean.shape[3], clean.shape[4]
    strategy.scheduler.set_timesteps(
        num_inference_steps=T, device=device,
        input_img_size_numel=d * h * w,
    )

    start_t = t0 * T
    uniform_ts = torch.linspace(start_t, 0.0, num_steps + 1, device=device)
    all_timesteps = uniform_ts[:-1]
    all_next = uniform_ts[1:]

    x = noisy
    for t, next_t in zip(all_timesteps, all_next):
        timesteps_batch = t.unsqueeze(0).to(device)
        model_input = torch.cat([x, seg], dim=1)
        with autocast('cuda', dtype=torch.bfloat16):
            velocity = model(model_input, timesteps_batch)
        x, _ = strategy.scheduler.step(velocity.float(), t, x, next_t)
    return x.clamp(0, 1)


def average_spectrum(volumes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    bins_ref, accum = None, None
    for v in volumes:
        bins, power = compute_radial_power_spectrum_3d(v)
        if accum is None:
            bins_ref, accum = bins, np.zeros_like(power)
        accum = accum + power
    return bins_ref, accum / max(1, len(volumes))


def band_energy(bins: np.ndarray, power: np.ndarray) -> dict[str, float]:
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (bins >= lo) & (bins < hi)
        out[name] = float(power[mask].sum()) if mask.any() else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Trained pixel-space diffusion model (e.g., exp1_1_1000)')
    parser.add_argument('--synth-dirs', nargs='+', required=True,
                        help='One or more synth dirs (each subj needs bravo + seg)')
    parser.add_argument('--real-dir', required=True,
                        help='Real volumes for LPIPS / spectrum reference')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--t0-values', type=float, nargs='+',
                        default=[0.02, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument('--num-steps', type=int, default=24,
                        help='Uniform Euler steps from t₀ down to 0')
    parser.add_argument('--num-volumes', type=int, default=5,
                        help='Synth volumes to refine per dir')
    parser.add_argument('--num-real', type=int, default=10,
                        help='Real volumes for spectrum reference')
    parser.add_argument('--num-real-lpips', type=int, default=3,
                        help='Real volumes used in cross-pair LPIPS evaluation')
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Load diffusion model ──────────────────────────────────────
    log.info(f"Loading checkpoint: {args.checkpoint}")
    model = load_diffusion_model(
        args.checkpoint, device=device, compile_model=False, spatial_dims=3,
    )
    model.eval()

    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000, image_size=args.image_size,
        depth_size=args.depth, spatial_dims=3,
    )

    # ── Real reference (spectrum + LPIPS) ─────────────────────────
    real_files = find_real_volumes(Path(args.real_dir), args.num_real)
    log.info(f"Real reference: {len(real_files)} volumes")
    real_np = [load_volume(f, args.depth) for f in real_files]
    bins, real_spec = average_spectrum(real_np)

    real_lpips_np = real_np[:args.num_real_lpips]
    real_lpips_t = torch.cat(
        [torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device).float() for a in real_lpips_np],
        dim=0,
    )

    # ── Per-dir loop ──────────────────────────────────────────────
    results = {
        'checkpoint': args.checkpoint,
        't0_values': args.t0_values,
        'num_steps': args.num_steps,
        'real_dir': str(Path(args.real_dir).resolve()),
        'per_dir': {},
    }

    for synth_dir_str in args.synth_dirs:
        synth_dir = Path(synth_dir_str)
        tag = synth_dir.name
        log.info(f"\n========== {tag} ==========")
        pairs = find_pairs(synth_dir, args.num_volumes)
        log.info(f"Synth: {len(pairs)} (bravo, seg) pairs")

        synth_np = [load_volume(b, args.depth) for b, _ in pairs]
        seg_np = [load_volume(s, args.depth) for _, s in pairs]
        _, synth_spec = average_spectrum(synth_np)

        # Baseline LPIPS(real, synth)
        synth_t = torch.cat(
            [torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device).float() for a in synth_np],
            dim=0,
        )
        baseline_vals = []
        for i in range(real_lpips_t.shape[0]):
            for j in range(synth_t.shape[0]):
                baseline_vals.append(float(compute_lpips_3d(
                    real_lpips_t[i:i + 1], synth_t[j:j + 1], device=device, chunk_size=32,
                )))
        baseline_lpips = float(np.mean(baseline_vals))
        log.info(f"Baseline LPIPS(real, synth) = {baseline_lpips:.4f}")

        per_t0 = {}
        for t0 in args.t0_values:
            log.info(f"\n  --- t₀ = {t0} ---")
            t0_dir = out_dir / tag / f"t0_{t0:.2f}"
            vol_dir = t0_dir / 'volumes'
            vol_dir.mkdir(parents=True, exist_ok=True)

            mixed_np: list[np.ndarray] = []
            t_start = time.time()
            for (b_path, _), s_arr in zip(pairs, seg_np):
                synth_arr = load_volume(b_path, args.depth)
                synth_t1 = torch.from_numpy(synth_arr).unsqueeze(0).unsqueeze(0).to(device).float()
                seg_t1 = torch.from_numpy(s_arr).unsqueeze(0).unsqueeze(0).to(device).float()
                # Binarize seg (clamp + threshold, project convention)
                seg_t1 = (seg_t1.clamp(0, 1) > 0.5).float()

                refined = sdedit_denoise(
                    model, strategy, synth_t1, seg_t1, t0, args.num_steps, device,
                )
                arr = refined.squeeze().cpu().numpy()
                mixed_np.append(arr)
                subj = b_path.parent.name
                subj_dir = vol_dir / subj
                subj_dir.mkdir(parents=True, exist_ok=True)
                save_nifti(np.transpose(arr, (1, 2, 0)), str(subj_dir / 'bravo.nii.gz'))
            log.info(f"    refined {len(pairs)} volumes in {time.time() - t_start:.1f}s")

            _, mixed_spec = average_spectrum(mixed_np)
            mixed_t = torch.cat(
                [torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device).float() for a in mixed_np],
                dim=0,
            )
            lpips_vals = []
            for i in range(real_lpips_t.shape[0]):
                for j in range(mixed_t.shape[0]):
                    lpips_vals.append(float(compute_lpips_3d(
                        real_lpips_t[i:i + 1], mixed_t[j:j + 1], device=device, chunk_size=32,
                    )))
            mixed_lpips = float(np.mean(lpips_vals))
            delta_pct = 100 * (mixed_lpips - baseline_lpips) / baseline_lpips

            bands_real = band_energy(bins, real_spec)
            bands_synth = band_energy(bins, synth_spec)
            bands_mixed = band_energy(bins, mixed_spec)
            verdict_rows = []
            for b in BANDS:
                s = bands_synth[b] / bands_real[b] if bands_real[b] else float('nan')
                m = bands_mixed[b] / bands_real[b] if bands_real[b] else float('nan')
                verdict_rows.append(f"  {b:10s}  synth/real={s:6.3f}  sde/real={m:6.3f}  Δ={m - s:+.3f}")
            log.info(f"    LPIPS  baseline={baseline_lpips:.4f}  sde={mixed_lpips:.4f}  Δ={delta_pct:+.2f}%")
            log.info("    " + "\n    ".join(verdict_rows))

            # Spectrum overlay
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            for name, spec, color in [
                ('real', real_spec, 'black'),
                ('synth', synth_spec, 'gray'),
                (f'sdedit (t₀={t0})', mixed_spec, 'tab:red'),
            ]:
                ax1.loglog(bins[1:], spec[1:], label=name, color=color, linewidth=2)
            ax1.set_xlabel('Radial frequency (Nyquist=0.5)')
            ax1.set_ylabel('Power')
            ax1.set_title(f'Spectrum (t₀={t0})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.semilogx(bins[1:], synth_spec[1:] / real_spec[1:],
                         label='synth / real', color='gray', linewidth=2)
            ax2.semilogx(bins[1:], mixed_spec[1:] / real_spec[1:],
                         label=f'sde / real (t₀={t0})', color='tab:red', linewidth=2)
            ax2.axhline(1.0, color='black', linestyle=':', alpha=0.5)
            ax2.set_xlabel('Radial frequency (Nyquist=0.5)')
            ax2.set_ylabel('Ratio to real')
            ax2.set_title(f'Ratio to real (t₀={t0})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(t0_dir / 'spectrum.png', dpi=120)
            plt.close(fig)

            per_t0[f'{t0}'] = {
                't0': t0,
                'lpips_mean': mixed_lpips,
                'lpips_baseline': baseline_lpips,
                'lpips_delta_pct': delta_pct,
                'band_ratio_synth_to_real': {b: bands_synth[b] / bands_real[b] if bands_real[b] else None for b in BANDS},
                'band_ratio_sde_to_real': {b: bands_mixed[b] / bands_real[b] if bands_real[b] else None for b in BANDS},
                'band_verdict': verdict_rows,
            }

        results['per_dir'][tag] = {
            'synth_dir': str(synth_dir.resolve()),
            'n_synth': len(pairs),
            'baseline_lpips': baseline_lpips,
            'per_t0': per_t0,
        }

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {out_dir / 'results.json'}")
    log.info(f"Refined volumes: {out_dir}/<dir>/t0_<X.XX>/volumes/<subj>/bravo.nii.gz")


if __name__ == '__main__':
    main()
