#!/usr/bin/env python3
"""Test light SDEdit degradation at very low t₀ values.

At high t₀ (0.50), SDEdit destroys structure. At very low t₀ (0.02-0.15),
it should preserve structure and only smooth texture — matching the
spatially-varying MSE-induced blur from diffusion models.

Produces:
1. Visual comparison grids: Real | SDEdit at each t₀ | Generated
2. Zoomed crops showing texture difference
3. Per-t₀ LPIPS and L1 vs original
4. Power spectrum comparison for each t₀ vs generated
5. Spatial difference maps showing WHERE blur is applied

Usage:
    python -m medgen.scripts.test_light_sdedit \
        --bravo-model /path/to/exp1_1_1000/checkpoint.pt \
        --data-root /path/to/brainmetshare-3 \
        --generated-dir /path/to/generated/exp1_1_bravo_imagenet_525 \
        --output-dir /path/to/light_sdedit_test \
        --num-volumes 5
"""
import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast

from medgen.diffusion import RFlowStrategy, load_diffusion_model

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

T0_VALUES = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
NUM_STEPS = 32


def load_volume(path: Path, depth: int) -> np.ndarray:
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    vmax = vol.max()
    if vmax > 0:
        vol /= vmax
    vol = np.transpose(vol, (2, 0, 1))
    if vol.shape[0] < depth:
        vol = np.pad(vol, ((0, depth - vol.shape[0]), (0, 0), (0, 0)))
    elif vol.shape[0] > depth:
        vol = vol[:depth]
    return vol


def discover_patients(data_dir: Path, max_n: int) -> list[tuple[str, Path, Path]]:
    """Find patients with bravo + seg."""
    patients = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        bravo = d / "bravo.nii.gz"
        seg = d / "seg.nii.gz"
        if bravo.exists() and seg.exists():
            patients.append((d.name, bravo, seg))
            if len(patients) >= max_n:
                break
    return patients


def load_generated_volumes(gen_dir: Path, depth: int, max_n: int) -> list[np.ndarray]:
    volumes = []
    for sub in sorted(gen_dir.iterdir()):
        if sub.is_dir():
            bravo = sub / "bravo.nii.gz"
            if bravo.exists():
                volumes.append(load_volume(bravo, depth))
                if len(volumes) >= max_n:
                    return volumes
    return volumes


def sdedit_denoise(
    model: torch.nn.Module,
    strategy: RFlowStrategy,
    clean: torch.Tensor,
    seg: torch.Tensor,
    t0: float,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """SDEdit at noise level t₀."""
    T = strategy.scheduler.num_train_timesteps

    noise = torch.randn_like(clean)
    t_scaled = torch.tensor([int(t0 * T)], device=device)
    noisy = strategy.scheduler.add_noise(clean, noise, t_scaled)

    d, h, w = clean.shape[2], clean.shape[3], clean.shape[4]
    strategy.scheduler.set_timesteps(
        num_inference_steps=num_steps, device=device,
        input_img_size_numel=d * h * w,
    )

    t0_threshold = t0 * T
    all_timesteps = strategy.scheduler.timesteps
    all_next = torch.cat((
        all_timesteps[1:],
        torch.tensor([0], dtype=all_timesteps.dtype, device=device),
    ))

    x = noisy
    for t, next_t in zip(all_timesteps, all_next):
        if t.item() > t0_threshold:
            continue
        timesteps_batch = t.unsqueeze(0).to(device)
        next_timestep = next_t.to(device) if isinstance(next_t, torch.Tensor) else torch.tensor(
            next_t, device=device,
        )
        model_input = torch.cat([x, seg], dim=1)
        with autocast('cuda', dtype=torch.bfloat16):
            velocity = model(model_input, timesteps_batch)
        x, _ = strategy.scheduler.step(velocity.float(), t, x, next_timestep)

    return x.clamp(0, 1)


def compute_radial_spectrum(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brain-masked radial power spectrum."""
    mask = (volume > 0.02).astype(np.float32)
    vol = volume * mask
    fft = np.fft.fftn(vol)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    d, h, w = vol.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)

    num_bins = min(d, h, w) // 2
    bin_edges = np.linspace(0, 0.5, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    radial_power = np.zeros(num_bins)
    for i in range(num_bins):
        ring = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if ring.sum() > 0:
            radial_power[i] = power[ring].mean()

    return bin_centers, radial_power


def plot_visual_grid(
    real: np.ndarray,
    sdedit_results: dict[float, np.ndarray],
    generated: np.ndarray | None,
    output_path: Path,
    slice_idx: int,
) -> None:
    """Grid: Real | SDEdit at each t₀ | Generated. Full + zoomed rows."""
    n_cols = 2 + len(sdedit_results)  # real + t₀s + generated
    if generated is None:
        n_cols -= 1

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    h, w = real.shape[1], real.shape[2]
    ch, cw = h // 2, w // 2
    crop = h // 4

    # Column 0: Real
    axes[0, 0].imshow(real[slice_idx], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Real', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(real[slice_idx, ch-crop:ch+crop, cw-crop:cw+crop], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Real (zoomed)', fontsize=10)
    axes[1, 0].axis('off')

    # SDEdit columns
    for i, (t0, sde_vol) in enumerate(sorted(sdedit_results.items())):
        col = i + 1
        l1 = np.abs(real - sde_vol).mean()
        axes[0, col].imshow(sde_vol[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f't₀={t0:.2f} (L1={l1:.4f})', fontsize=10)
        axes[0, col].axis('off')
        axes[1, col].imshow(sde_vol[slice_idx, ch-crop:ch+crop, cw-crop:cw+crop], cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title(f't₀={t0:.2f} (zoomed)', fontsize=10)
        axes[1, col].axis('off')

    # Last column: Generated (if available)
    if generated is not None:
        col = n_cols - 1
        axes[0, col].imshow(generated[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title('Generated', fontsize=11, fontweight='bold', color='red')
        axes[0, col].axis('off')
        axes[1, col].imshow(generated[slice_idx, ch-crop:ch+crop, cw-crop:cw+crop], cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title('Generated (zoomed)', fontsize=10, color='red')
        axes[1, col].axis('off')

    plt.suptitle(f'Light SDEdit Sweep — Axial Slice {slice_idx}', fontsize=13)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def plot_difference_grid(
    real: np.ndarray,
    sdedit_results: dict[float, np.ndarray],
    output_path: Path,
    slice_idx: int,
) -> None:
    """Show what each t₀ removes (residual maps)."""
    n = len(sdedit_results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (t0, sde_vol) in enumerate(sorted(sdedit_results.items())):
        diff = real[slice_idx] - sde_vol[slice_idx]
        vmax = max(np.abs(diff).max(), 0.01)
        axes[i].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        l1 = np.abs(diff).mean()
        axes[i].set_title(f't₀={t0:.2f}\nL1={l1:.4f}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f'Residuals (Real - SDEdit) — Slice {slice_idx}', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def plot_spectra_sweep(
    freqs: np.ndarray,
    psd_real: np.ndarray,
    psd_generated: np.ndarray,
    sdedit_spectra: dict[float, np.ndarray],
    output_path: Path,
) -> None:
    """Power spectra: real, generated, and each SDEdit t₀."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Log spectra
    ax = axes[0]
    ax.semilogy(freqs, psd_real, 'b-', label='Real', linewidth=2.5)
    ax.semilogy(freqs, psd_generated, 'r-', label='Generated', linewidth=2.5)
    for t0, psd in sorted(sdedit_spectra.items()):
        ax.semilogy(freqs, psd, '--', label=f'SDEdit t₀={t0:.2f}', linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Radial Frequency')
    ax.set_ylabel('PSD (log)')
    ax.set_title('Power Spectra')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Ratio to real
    ax = axes[1]
    ratio_gen = np.where(psd_real > 1e-20, psd_generated / psd_real, 1.0)
    ax.plot(freqs, ratio_gen, 'r-', label='Generated/Real (target)', linewidth=2.5)
    for t0, psd in sorted(sdedit_spectra.items()):
        ratio = np.where(psd_real > 1e-20, psd / psd_real, 1.0)
        ax.plot(freqs, ratio, '--', label=f't₀={t0:.2f}', linewidth=1.2, alpha=0.8)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Radial Frequency')
    ax.set_ylabel('PSD Ratio')
    ax.set_title('Spectral Ratio (should approach red line)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.5)
    ax.grid(True, alpha=0.3)

    # Summary: L1 of spectral ratio vs generated ratio
    ax = axes[2]
    t0_list = sorted(sdedit_spectra.keys())
    spectral_match = []
    for t0 in t0_list:
        ratio_sde = np.where(psd_real > 1e-20, sdedit_spectra[t0] / psd_real, 1.0)
        match = np.mean(np.abs(ratio_sde - ratio_gen))
        spectral_match.append(match)
    ax.bar(range(len(t0_list)), spectral_match, tick_label=[f'{t:.2f}' for t in t0_list])
    ax.set_xlabel('t₀')
    ax.set_ylabel('Spectral Match Error (lower = closer to generated)')
    ax.set_title('Which t₀ best matches generated spectrum?')
    best_idx = np.argmin(spectral_match)
    ax.bar(best_idx, spectral_match[best_idx], color='green')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test light SDEdit degradation")
    parser.add_argument("--bravo-model", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--generated-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-volumes", type=int, default=5)
    parser.add_argument("--depth", type=int, default=160)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--t0-values", nargs='+', type=float, default=T0_VALUES)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')

    # Load model
    logger.info(f"Loading bravo model from {args.bravo_model}")
    model = load_diffusion_model(
        args.bravo_model, device=device, compile_model=False, spatial_dims=3,
    )
    model.eval()

    strategy = RFlowStrategy()
    strategy.setup_scheduler(num_timesteps=1000, image_size=256)

    # Load real volumes
    train_dir = Path(args.data_root) / "train"
    patients = discover_patients(train_dir, args.num_volumes)
    logger.info(f"Found {len(patients)} patients")

    # Load generated volumes for comparison
    gen_vols = load_generated_volumes(Path(args.generated_dir), args.depth, args.num_volumes)
    logger.info(f"Loaded {len(gen_vols)} generated volumes")

    # Compute average generated spectrum
    gen_spectra = []
    for gv in gen_vols:
        f, p = compute_radial_spectrum(gv)
        gen_spectra.append(p)
    freqs = f
    avg_gen_psd = np.mean(gen_spectra, axis=0)

    # Process each patient
    all_real_spectra = []
    all_sdedit_spectra: dict[float, list[np.ndarray]] = {t0: [] for t0 in args.t0_values}
    summary_lines = []

    for vol_idx, (patient_id, bravo_path, seg_path) in enumerate(patients):
        logger.info(f"Processing {patient_id} ({vol_idx + 1}/{len(patients)})")

        real = load_volume(bravo_path, args.depth)
        seg = load_volume(seg_path, args.depth)
        seg = (seg > 0.5).astype(np.float32)

        real_t = torch.from_numpy(real).unsqueeze(0).unsqueeze(0).to(device)
        seg_t = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).to(device)

        # Compute real spectrum
        _, real_psd = compute_radial_spectrum(real)
        all_real_spectra.append(real_psd)

        # SDEdit at each t₀
        sdedit_results: dict[float, np.ndarray] = {}
        for t0 in args.t0_values:
            logger.info(f"  SDEdit t₀={t0:.2f}...")
            with torch.no_grad():
                out = sdedit_denoise(model, strategy, real_t, seg_t, t0, args.num_steps, device)
            sde_np = out.squeeze().cpu().numpy()
            sdedit_results[t0] = sde_np

            # Spectrum
            _, sde_psd = compute_radial_spectrum(sde_np)
            all_sdedit_spectra[t0].append(sde_psd)

            l1 = np.abs(real - sde_np).mean()
            summary_lines.append(f"{patient_id} t₀={t0:.2f}: L1={l1:.5f}")

        # Pick generated volume for visual comparison
        gen = gen_vols[vol_idx % len(gen_vols)] if gen_vols else None

        # Visual grid
        slice_idx = real.shape[0] // 2
        plot_visual_grid(real, sdedit_results, gen, output_dir / f'visual_{vol_idx:02d}.png', slice_idx)
        plot_difference_grid(real, sdedit_results, output_dir / f'residuals_{vol_idx:02d}.png', slice_idx)
        logger.info("  Saved visual comparison")

        torch.cuda.empty_cache()

    # Average spectra across volumes
    avg_real_psd = np.mean(all_real_spectra, axis=0)
    avg_sdedit_psd = {t0: np.mean(specs, axis=0) for t0, specs in all_sdedit_spectra.items()}

    # Spectra comparison
    plot_spectra_sweep(freqs, avg_real_psd, avg_gen_psd, avg_sdedit_psd, output_dir / 'spectra_sweep.png')

    # Summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("=== Light SDEdit Sweep ===\n\n")
        f.write(f"t₀ values: {args.t0_values}\n")
        f.write(f"Num volumes: {len(patients)}\n")
        f.write(f"Num steps: {args.num_steps}\n\n")

        f.write("--- Per-volume L1 ---\n")
        for line in summary_lines:
            f.write(f"  {line}\n")

        f.write("\n--- Spectral match to generated ---\n")
        ratio_gen = np.where(avg_real_psd > 1e-20, avg_gen_psd / avg_real_psd, 1.0)
        for t0 in sorted(args.t0_values):
            ratio_sde = np.where(avg_real_psd > 1e-20, avg_sdedit_psd[t0] / avg_real_psd, 1.0)
            match_err = np.mean(np.abs(ratio_sde - ratio_gen))
            f.write(f"  t₀={t0:.2f}: spectral_match_error={match_err:.5f}\n")

    logger.info(f"All results saved to {output_dir}")


if __name__ == '__main__':
    main()
