"""Measure latent space statistics of a trained compression checkpoint.

Reports per-channel mean/std and LDM normalization factors.
For VAE: also reports KL-specific stats (mu, logvar, posterior_std).

Usage:
    # Local (small dataset)
    python -m medgen.scripts.measure_latent_std \
        checkpoint=runs/compression_3d/.../checkpoint_best.pt

    # Cluster (full dataset)
    python -m medgen.scripts.measure_latent_std \
        checkpoint=runs/compression_3d/.../checkpoint_best.pt \
        paths=cluster

    # With explicit spatial dims and max samples
    python -m medgen.scripts.measure_latent_std \
        checkpoint=runs/compression_3d/.../checkpoint_best.pt \
        spatial_dims=3 max_samples=50
"""
import logging
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def measure_latent_stats(cfg: DictConfig) -> None:
    checkpoint_path = cfg.get('checkpoint', None)
    if checkpoint_path is None:
        logger.error("checkpoint= is required")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_samples = cfg.get('max_samples', 100)

    # Load model via auto-detection (infers architecture from state_dict if needed)
    from medgen.data.loaders.compression_detection import load_compression_model
    model, comp_type, spatial_dims, scale_factor, latent_channels = load_compression_model(
        checkpoint_path, compression_type='auto', device=device,
    )
    model.eval()

    logger.info(f"Loaded {comp_type} model: spatial_dims={spatial_dims}, "
                f"scale_factor={scale_factor}, latent_channels={latent_channels}")

    # Create dataloader
    if spatial_dims == 3:
        from medgen.data.loaders.volume_3d import create_vae_3d_multi_modality_dataloader
        loader, dataset = create_vae_3d_multi_modality_dataloader(cfg)
        logger.info(f"Loaded 3D dataset: {len(dataset)} volumes")
    else:
        from medgen.data.loaders import create_vae_dataloader
        loader, dataset = create_vae_dataloader(
            cfg, image_type=cfg.mode.get('image_keys', ['bravo'])[0]
        )
        logger.info(f"Loaded 2D dataset: {len(dataset)} images")

    is_vae = comp_type == 'vae'

    # Per-channel accumulators (built after first batch)
    ch_latent_mean: list[list[float]] = []
    ch_latent_std: list[list[float]] = []

    # VAE-specific accumulators
    all_z_std: list[float] = []
    all_z_mean: list[float] = []
    all_mu_std: list[float] = []
    all_mu_mean: list[float] = []
    all_logvar_mean: list[float] = []
    all_posterior_std: list[float] = []
    ch_mu_mean: list[list[float]] = []
    ch_mu_std: list[list[float]] = []
    ch_logvar_mean: list[list[float]] = []
    ch_posterior_std: list[list[float]] = []

    n_processed = 0

    logger.info(f"Encoding up to {max_samples} samples...")

    with torch.no_grad():
        for batch in loader:
            if n_processed >= max_samples:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch.get('image', batch.get('images'))
            elif isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            if images is None:
                continue

            images = images.to(device).float()

            # Get latents from encoder
            z_mu = None
            z_logvar = None
            z_sampled = None
            posterior_std = None
            if is_vae:
                # VAE encode() returns (z_mu, z_logvar)
                h = model.encoder(images)
                z_mu = model.quant_conv_mu(h)
                z_logvar = model.quant_conv_log_sigma(h)
                z_sampled = model.sampling(z_mu, z_logvar)
                posterior_std = torch.exp(0.5 * z_logvar)
                latents = z_sampled
            else:
                # VQ-VAE / DC-AE: encode returns a tensor directly
                latents = model.encode(images)

            # Initialize per-channel lists on first batch
            if not ch_latent_mean:
                n_ch = latents.shape[1]
                ch_latent_mean = [[] for _ in range(n_ch)]
                ch_latent_std = [[] for _ in range(n_ch)]
                if is_vae:
                    ch_mu_mean = [[] for _ in range(n_ch)]
                    ch_mu_std = [[] for _ in range(n_ch)]
                    ch_logvar_mean = [[] for _ in range(n_ch)]
                    ch_posterior_std = [[] for _ in range(n_ch)]

            # Per-sample stats
            for i in range(images.shape[0]):
                if n_processed >= max_samples:
                    break

                li = latents[i]  # [C, ...]

                # Per-channel stats
                for c in range(li.shape[0]):
                    ch_latent_mean[c].append(li[c].mean().item())
                    ch_latent_std[c].append(li[c].std().item())

                # VAE-specific per-sample stats
                if is_vae:
                    zi = z_sampled[i]
                    mui = z_mu[i]
                    logvari = z_logvar[i]
                    pstdi = posterior_std[i]

                    all_z_std.append(zi.std().item())
                    all_z_mean.append(zi.mean().item())
                    all_mu_std.append(mui.std().item())
                    all_mu_mean.append(mui.mean().item())
                    all_logvar_mean.append(logvari.mean().item())
                    all_posterior_std.append(pstdi.mean().item())

                    for c in range(zi.shape[0]):
                        ch_mu_mean[c].append(mui[c].mean().item())
                        ch_mu_std[c].append(mui[c].std().item())
                        ch_logvar_mean[c].append(logvari[c].mean().item())
                        ch_posterior_std[c].append(pstdi[c].mean().item())

                n_processed += 1

            if n_processed % 10 == 0:
                logger.info(f"  Processed {n_processed}/{max_samples}")

    if n_processed == 0:
        logger.error("No samples processed!")
        sys.exit(1)

    # Build per-channel stats
    n_ch = len(ch_latent_mean)
    ch_stats = []
    for c in range(n_ch):
        entry: dict[str, float] = {
            'latent_mean': np.mean(ch_latent_mean[c]),
            'latent_std': np.mean(ch_latent_std[c]),
        }
        if is_vae:
            entry.update({
                'mu_mean': np.mean(ch_mu_mean[c]),
                'mu_std': np.mean(ch_mu_std[c]),
                'logvar': np.mean(ch_logvar_mean[c]),
                'post_std': np.mean(ch_posterior_std[c]),
            })
        ch_stats.append(entry)

    # Print results
    print("\n" + "=" * 60)
    print(f"LATENT SPACE STATISTICS ({n_processed} samples)")
    print("=" * 60)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Model: {comp_type.upper()}, {spatial_dims}D, scale={scale_factor}x, latent_ch={latent_channels}")

    # Generic latent stats (all model types)
    latent_means = [s['latent_mean'] for s in ch_stats]
    latent_stds = [s['latent_std'] for s in ch_stats]
    print("\n--- Encoder features (model.encode) ---")
    print(f"  Overall mean = {np.mean(latent_means):.4f}")
    print(f"  Overall std  = {np.mean(latent_stds):.4f}")

    # VAE-specific stats
    if is_vae:
        z_stds = np.array(all_z_std)
        z_means = np.array(all_z_mean)
        mu_stds = np.array(all_mu_std)
        mu_means = np.array(all_mu_mean)
        logvar_means = np.array(all_logvar_mean)
        posterior_stds = np.array(all_posterior_std)

        print("\n--- Sampled z = mu + std*eps ---")
        print(f"  z.std()   = {z_stds.mean():.4f}  (target: 0.9-1.1)")
        print(f"  z.mean()  = {z_means.mean():.4f}  (target: ~0.0)")

        print("\n--- Encoder posterior q(z|x) ---")
        print(f"  mu.std()              = {mu_stds.mean():.4f}  (spread of means)")
        print(f"  mu.mean()             = {mu_means.mean():.4f}  (center of means)")
        print(f"  logvar.mean()         = {logvar_means.mean():.4f}  (target: ~0.0 for std~1)")
        print(f"  posterior_std.mean()  = {posterior_stds.mean():.4f}  (exp(0.5*logvar), target: ~1.0)")

    # Per-channel table
    if is_vae:
        print(f"\n--- Per-channel statistics ({n_ch} channels) ---")
        print(f"  {'Ch':>3}  {'z.mean':>8}  {'z.std':>8}  {'mu.mean':>8}  {'mu.std':>8}  {'logvar':>8}  {'post_std':>8}")
        print(f"  {'-' * 59}")
        for c, s in enumerate(ch_stats):
            print(f"  {c:>3}  {s['latent_mean']:>8.4f}  {s['latent_std']:>8.4f}  "
                  f"{s['mu_mean']:>8.4f}  {s['mu_std']:>8.4f}  "
                  f"{s['logvar']:>8.4f}  {s['post_std']:>8.4f}")
    else:
        print(f"\n--- Per-channel statistics ({n_ch} channels) ---")
        print(f"  {'Ch':>3}  {'mean':>8}  {'std':>8}")
        print(f"  {'-' * 23}")
        for c, s in enumerate(ch_stats):
            print(f"  {c:>3}  {s['latent_mean']:>8.4f}  {s['latent_std']:>8.4f}")

    # LDM normalization table
    print("\n--- LDM normalization (per-channel scale factors) ---")
    print("  To normalize latents to ~N(0,1) for diffusion training:")
    print("  z_norm = (z - shift) / scale")
    print(f"  {'Ch':>3}  {'shift':>8}  {'scale':>8}")
    print(f"  {'-' * 23}")
    for c, s in enumerate(ch_stats):
        print(f"  {c:>3}  {s['latent_mean']:>8.4f}  {s['latent_std']:>8.4f}")

    # Interpretation
    print("\n--- Interpretation ---")

    if is_vae:
        z_std_avg = z_stds.mean()
        p_std_avg = posterior_stds.mean()
        if z_std_avg < 0.5:
            print(f"  z.std={z_std_avg:.3f} < 0.5: KL too strong, posterior collapsed. Lower kl_weight.")
        elif z_std_avg < 0.9:
            print(f"  z.std={z_std_avg:.3f} < 0.9: KL slightly strong. Consider lowering kl_weight.")
        elif z_std_avg <= 1.1:
            print(f"  z.std={z_std_avg:.3f} in [0.9, 1.1]: Good balance (MAISI target range).")
        elif z_std_avg <= 2.0:
            print(f"  z.std={z_std_avg:.3f} > 1.1: KL slightly weak. Consider raising kl_weight.")
        else:
            print(f"  z.std={z_std_avg:.3f} >> 1: KL very weak, latents not regularized. Raise kl_weight.")

        if p_std_avg < 0.01:
            print(f"  posterior_std={p_std_avg:.4f} << 1: Near-deterministic encoder (KL ~ 0).")
            print("    -> Sampling z adds almost no noise. VAE behaves like a regular AE.")

    # Check per-channel variance spread (all model types)
    ch_std_vals = [s['latent_std'] for s in ch_stats]
    ch_std_ratio = max(ch_std_vals) / max(min(ch_std_vals), 1e-8)
    if ch_std_ratio > 3.0:
        print(f"  Channel std ratio = {ch_std_ratio:.1f}x: Channels have very different scales.")
        print("    -> Per-channel normalization recommended for LDM.")
    elif ch_std_ratio > 1.5:
        print(f"  Channel std ratio = {ch_std_ratio:.1f}x: Moderate channel imbalance.")
        print("    -> Per-channel normalization may help LDM quality.")
    else:
        print(f"  Channel std ratio = {ch_std_ratio:.1f}x: Channels are balanced.")

    print("=" * 60)


@hydra.main(config_path="../../../configs", config_name="vae_3d", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    measure_latent_stats(cfg)


if __name__ == "__main__":
    main()
