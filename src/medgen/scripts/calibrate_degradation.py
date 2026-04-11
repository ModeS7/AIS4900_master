#!/usr/bin/env python3
"""Calibrate SDEdit degradation strength for restoration training.

Compares radially averaged 3D power spectra of:
1. Real sharp volumes (ground truth)
2. Generated volumes from exp1_1_1000 (MSE-smoothed)
3. SDEdit outputs at various t₀ strengths (degradation candidates)

Finds the t₀ range where SDEdit degradation best matches the actual
MSE-smoothing from the diffusion model. This informs the training
range for the restoration network.

Usage:
    python -m medgen.scripts.calibrate_degradation \
        --bravo-model /path/to/exp1_1_1000/checkpoint.pt \
        --data-root /path/to/brainmetshare-3 \
        --generated-dir /path/to/generated_imagenet \
        --generated-dir-rin /path/to/generated_radimagenet \
        --output-dir /path/to/calibration_output \
        --num-volumes 5 --num-steps 32
"""
import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast

from medgen.data.utils import binarize_seg
from medgen.diffusion import RFlowStrategy, load_diffusion_model

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def compute_radial_power_spectrum_3d(
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radially averaged 3D power spectrum.

    Args:
        volume: 3D array [D, H, W] with values in [0, 1].

    Returns:
        (frequencies, power): 1D arrays of radial frequency bins and
        corresponding mean power spectral density.
    """
    # 3D FFT
    fft = np.fft.fftn(volume)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    # Build radial distance grid
    d, h, w = volume.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]

    # Normalize spatial frequencies by voxel size ratios
    # D has 1.0mm spacing, H/W have 0.9375mm -> higher spatial freq
    radius = np.sqrt(
        (dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2
    )

    # Bin by radial distance
    max_radius = 0.5  # Nyquist
    num_bins = min(d, h, w) // 2
    bin_edges = np.linspace(0, max_radius, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    radial_power = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if mask.sum() > 0:
            radial_power[i] = power[mask].mean()

    return bin_centers, radial_power


def load_volumes(
    directory: Path, modality: str, max_volumes: int, depth: int,
) -> list[np.ndarray]:
    """Load NIfTI volumes from a directory.

    Args:
        directory: Path containing patient_id/modality.nii.gz files.
        modality: Filename without extension (e.g. 'bravo').
        max_volumes: Maximum number to load.
        depth: Target depth (pad/crop).

    Returns:
        List of [D, H, W] numpy arrays normalized to [0, 1].
    """
    pattern = f"*/{modality}.nii.gz"
    files = sorted(directory.glob(pattern))
    if not files:
        # Maybe it's a flat directory of .nii.gz files
        files = sorted(directory.glob("*.nii.gz"))
    files = files[:max_volumes]

    volumes = []
    for fp in files:
        vol = nib.load(str(fp)).get_fdata().astype(np.float32)
        # Normalize
        vmax = vol.max()
        if vmax > 0:
            vol = vol / vmax
        # Transpose [H, W, D] -> [D, H, W]
        vol = np.transpose(vol, (2, 0, 1))
        # Pad/crop depth
        if vol.shape[0] < depth:
            vol = np.pad(vol, ((0, depth - vol.shape[0]), (0, 0), (0, 0)))
        elif vol.shape[0] > depth:
            vol = vol[:depth]
        volumes.append(vol)

    logger.info(f"Loaded {len(volumes)} volumes from {directory}")
    return volumes


def sdedit_denoise(
    model: torch.nn.Module,
    strategy: RFlowStrategy,
    clean_volume: torch.Tensor,
    seg_mask: torch.Tensor,
    t0: float,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Apply SDEdit: add noise at level t₀, then denoise from t₀ to 0.

    For RFlow, t₀ is in [0, 1] where 0=clean and 1=noise.

    Args:
        model: Trained bravo diffusion model.
        strategy: RFlowStrategy instance.
        clean_volume: [1, 1, D, H, W] clean volume.
        seg_mask: [1, 1, D, H, W] binary segmentation mask.
        t0: Noise level in [0, 1]. Higher = more degradation.
        num_steps: Total Euler steps (we only run from t₀ to 0).
        device: CUDA device.

    Returns:
        [1, 1, D, H, W] denoised (SDEdit) volume.
    """
    T = strategy.scheduler.num_train_timesteps  # 1000

    # 1. Add noise at level t₀
    noise = torch.randn_like(clean_volume)
    t_scaled = torch.tensor([int(t0 * T)], device=device)
    noisy = strategy.scheduler.add_noise(clean_volume, noise, t_scaled)

    # 2. Setup scheduler for inference
    d, h, w = clean_volume.shape[2], clean_volume.shape[3], clean_volume.shape[4]
    numel = d * h * w
    strategy.scheduler.set_timesteps(
        num_inference_steps=num_steps, device=device,
        input_img_size_numel=numel,
    )

    # 3. Filter timesteps: only keep those <= t₀ * T
    t0_threshold = t0 * T
    all_timesteps = strategy.scheduler.timesteps
    all_next = torch.cat((
        all_timesteps[1:],
        torch.tensor([0], dtype=all_timesteps.dtype, device=device),
    ))

    # 4. Run Euler from t₀ down to 0
    x = noisy
    for t, next_t in zip(all_timesteps, all_next):
        if t.item() > t0_threshold:
            continue  # Skip timesteps above our starting point

        timesteps_batch = t.unsqueeze(0).to(device)
        next_timestep = next_t.to(device) if isinstance(next_t, torch.Tensor) else torch.tensor(
            next_t, device=device
        )

        # Model input: [noisy, seg_conditioning]
        model_input = torch.cat([x, seg_mask], dim=1)

        with autocast('cuda', dtype=torch.bfloat16):
            velocity = model(model_input, timesteps_batch)

        x, _ = strategy.scheduler.step(velocity.float(), t, x, next_timestep)

    return x.clamp(0, 1)


def compute_mean_spectrum(volumes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean radial power spectrum across multiple volumes."""
    spectra = []
    freqs = None
    for vol in volumes:
        f, p = compute_radial_power_spectrum_3d(vol)
        spectra.append(p)
        if freqs is None:
            freqs = f
    assert freqs is not None
    mean_spectrum = np.mean(spectra, axis=0)
    return freqs, mean_spectrum


def spectrum_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Log-space L2 distance between two power spectra."""
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    log_s1 = np.log10(s1 + eps)
    log_s2 = np.log10(s2 + eps)
    return float(np.sqrt(np.mean((log_s1 - log_s2) ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Calibrate SDEdit degradation strength")
    parser.add_argument("--bravo-model", type=str, required=True,
                        help="Path to bravo model checkpoint (exp1_1_1000)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to dataset root (brainmetshare-3)")
    parser.add_argument("--generated-dir", type=str, default=None,
                        help="Path to ImageNet-optimized generated volumes")
    parser.add_argument("--generated-dir-rin", type=str, default=None,
                        help="Path to RadImageNet-optimized generated volumes")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for plots and results")
    parser.add_argument("--num-volumes", type=int, default=5,
                        help="Number of volumes per set (default: 5)")
    parser.add_argument("--num-steps", type=int, default=32,
                        help="Euler steps for SDEdit denoising (default: 32)")
    parser.add_argument("--depth", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)

    # ── Load real volumes ────────────────────────────────────────────
    logger.info("Loading real volumes from train split...")
    train_dir = data_root / "train"
    real_volumes = load_volumes(train_dir, "bravo", args.num_volumes, args.depth)

    # ── Load generated volumes ───────────────────────────────────────
    gen_volumes_imagenet = []
    gen_volumes_rin = []

    if args.generated_dir and Path(args.generated_dir).exists():
        logger.info(f"Loading ImageNet-opt generated volumes from {args.generated_dir}...")
        gen_volumes_imagenet = load_volumes(
            Path(args.generated_dir), "bravo", args.num_volumes, args.depth,
        )

    if args.generated_dir_rin and Path(args.generated_dir_rin).exists():
        logger.info(f"Loading RadImageNet-opt generated volumes from {args.generated_dir_rin}...")
        gen_volumes_rin = load_volumes(
            Path(args.generated_dir_rin), "bravo", args.num_volumes, args.depth,
        )

    # ── Compute reference spectra ────────────────────────────────────
    logger.info("Computing reference power spectra...")
    freqs, real_spectrum = compute_mean_spectrum(real_volumes)

    gen_spectra = {}
    if gen_volumes_imagenet:
        _, gen_spectrum_in = compute_mean_spectrum(gen_volumes_imagenet)
        gen_spectra["imagenet"] = gen_spectrum_in
    if gen_volumes_rin:
        _, gen_spectrum_rin = compute_mean_spectrum(gen_volumes_rin)
        gen_spectra["radimagenet"] = gen_spectrum_rin

    # ── Load model for SDEdit ────────────────────────────────────────
    logger.info(f"Loading bravo model: {args.bravo_model}")
    model = load_diffusion_model(
        args.bravo_model, device=device,
        in_channels=2, out_channels=1,
        compile_model=False, spatial_dims=3,
    )

    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000,
        image_size=args.image_size,
        depth_size=args.depth,
        spatial_dims=3,
    )

    # ── Load seg masks for conditioning ──────────────────────────────
    logger.info("Loading seg masks for SDEdit conditioning...")
    seg_volumes = []
    seg_files = sorted(train_dir.glob("*/seg.nii.gz"))[:args.num_volumes]
    for seg_path in seg_files:
        seg_np = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        seg_np = (seg_np > 0.5).astype(np.float32)
        seg_np = np.transpose(seg_np, (2, 0, 1))  # [H,W,D] -> [D,H,W]
        if seg_np.shape[0] < args.depth:
            seg_np = np.pad(seg_np, ((0, args.depth - seg_np.shape[0]), (0, 0), (0, 0)))
        seg_volumes.append(seg_np)

    # ── SDEdit sweep ─────────────────────────────────────────────────
    t0_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]

    sdedit_spectra = {}
    sdedit_distances = {}

    for t0 in t0_values:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"SDEdit at t₀ = {t0:.2f}")
        logger.info(f"{'=' * 50}")

        sdedit_volumes = []
        for i in range(min(len(real_volumes), len(seg_volumes))):
            real_vol = real_volumes[i]
            seg_vol = seg_volumes[i]

            # Convert to tensors [1, 1, D, H, W]
            clean_t = torch.from_numpy(real_vol).unsqueeze(0).unsqueeze(0).to(device)
            seg_t = torch.from_numpy(seg_vol).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                degraded = sdedit_denoise(
                    model, strategy, clean_t, seg_t,
                    t0=t0, num_steps=args.num_steps, device=device,
                )

            sdedit_vol = degraded.squeeze().cpu().numpy()
            sdedit_volumes.append(sdedit_vol)
            logger.info(f"  Volume {i + 1}/{len(real_volumes)} done")

        # Compute spectrum
        _, sdedit_spectrum = compute_mean_spectrum(sdedit_volumes)
        sdedit_spectra[t0] = sdedit_spectrum

        # Compute distances to generated spectra
        distances = {}
        for gen_name, gen_spec in gen_spectra.items():
            dist = spectrum_distance(sdedit_spectrum, gen_spec)
            distances[gen_name] = dist
            logger.info(f"  Distance to {gen_name}: {dist:.4f}")

        sdedit_distances[t0] = distances

    # ── Free GPU memory ──────────────────────────────────────────────
    del model
    torch.cuda.empty_cache()

    # ── Find optimal t₀ ──────────────────────────────────────────────
    results = {
        "real_spectrum": real_spectrum.tolist(),
        "frequencies": freqs.tolist(),
        "generated_spectra": {k: v.tolist() for k, v in gen_spectra.items()},
        "sdedit_spectra": {str(t0): s.tolist() for t0, s in sdedit_spectra.items()},
        "sdedit_distances": {str(t0): d for t0, d in sdedit_distances.items()},
    }

    best_t0 = {}
    for gen_name in gen_spectra:
        dists = [(t0, sdedit_distances[t0][gen_name]) for t0 in t0_values if gen_name in sdedit_distances[t0]]
        if dists:
            best = min(dists, key=lambda x: x[1])
            best_t0[gen_name] = {"best_t0": best[0], "distance": best[1]}
            logger.info(f"\nBest t₀ for {gen_name}: {best[0]:.2f} (distance: {best[1]:.4f})")

    results["best_t0"] = best_t0

    # Recommend training range: ±0.15 around best t₀
    if best_t0:
        all_best = [v["best_t0"] for v in best_t0.values()]
        center = np.mean(all_best)
        recommended_range = [max(0.1, center - 0.15), min(0.8, center + 0.15)]
        results["recommended_range"] = recommended_range
        logger.info(f"\nRecommended training range: [{recommended_range[0]:.2f}, {recommended_range[1]:.2f}]")

    # ── Save results ─────────────────────────────────────────────────
    results_path = output_dir / "calibration_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")

    # ── Generate matplotlib plot ─────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Power spectra comparison
        ax = axes[0]
        ax.semilogy(freqs, real_spectrum, 'k-', linewidth=2, label='Real')
        for gen_name, gen_spec in gen_spectra.items():
            ax.semilogy(freqs, gen_spec, '--', linewidth=2, label=f'Generated ({gen_name})')
        # Plot SDEdit curves (subset for readability)
        cmap = plt.cm.viridis
        for idx, t0 in enumerate(t0_values):
            if t0 in sdedit_spectra:
                color = cmap(idx / len(t0_values))
                ax.semilogy(freqs, sdedit_spectra[t0], '-', color=color,
                            alpha=0.6, linewidth=1, label=f'SDEdit t₀={t0:.2f}')
        ax.set_xlabel('Radial frequency')
        ax.set_ylabel('Power spectral density')
        ax.set_title('Radially averaged 3D power spectra')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Plot 2: Spectrum distance vs t₀
        ax = axes[1]
        for gen_name in gen_spectra:
            dists = [sdedit_distances[t0].get(gen_name, float('nan')) for t0 in t0_values]
            ax.plot(t0_values, dists, 'o-', linewidth=2, markersize=6, label=gen_name)
            # Mark minimum
            min_idx = np.nanargmin(dists)
            ax.plot(t0_values[min_idx], dists[min_idx], '*', markersize=15, color='red')
        ax.set_xlabel('SDEdit strength (t₀)')
        ax.set_ylabel('Log-space L2 distance to generated spectrum')
        ax.set_title('Spectrum distance: SDEdit vs Generated')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "calibration_spectra.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Plot saved to: {plot_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plot generation")

    # ── Print summary ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Real volumes: {len(real_volumes)}")
    for gen_name in gen_spectra:
        logger.info(f"Generated ({gen_name}): {len(gen_volumes_imagenet if gen_name == 'imagenet' else gen_volumes_rin)}")
    logger.info(f"SDEdit strengths tested: {t0_values}")
    for gen_name, bt0 in best_t0.items():
        logger.info(f"Best t₀ for {gen_name}: {bt0['best_t0']:.2f} (distance: {bt0['distance']:.4f})")
    if "recommended_range" in results:
        logger.info(f"Recommended training range: {results['recommended_range']}")


if __name__ == "__main__":
    main()
