"""Measure latent space statistics of a trained VAE checkpoint.

Reports z.std(), z.mean(), and per-channel statistics across the dataset.
Used to tune kl_weight: MAISI recommends latent std in [0.9, 1.1].

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
    spatial_dims = cfg.get('spatial_dims', 'auto')

    # Load model via auto-detection
    from medgen.data.loaders.compression_detection import load_compression_model
    model, comp_type, spatial_dims, scale_factor, latent_channels = load_compression_model(
        checkpoint_path, compression_type='auto', device=device, spatial_dims=spatial_dims,
    )
    model.eval()

    if comp_type != 'vae':
        logger.error(f"Checkpoint is {comp_type}, not a VAE. Only VAEs have KL latent statistics.")
        sys.exit(1)

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

    # Collect latent statistics
    all_z_std = []
    all_z_mean = []
    all_mu_std = []
    all_mu_mean = []
    all_logvar_mean = []
    all_posterior_std = []  # exp(0.5 * logvar) = learned std
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

            # Encode: get mu, logvar from the VAE encoder
            h = model.encoder(images)
            z_mu = model.quant_conv_mu(h)
            z_logvar = model.quant_conv_log_sigma(h)

            # Sample z via reparameterization
            z = model.sampling(z_mu, z_logvar)

            # Posterior std = exp(0.5 * logvar)
            posterior_std = torch.exp(0.5 * z_logvar)

            # Per-sample stats (flatten spatial dims, keep batch)
            for i in range(images.shape[0]):
                if n_processed >= max_samples:
                    break

                zi = z[i]  # [C, ...]
                mui = z_mu[i]
                logvari = z_logvar[i]
                pstdi = posterior_std[i]

                all_z_std.append(zi.std().item())
                all_z_mean.append(zi.mean().item())
                all_mu_std.append(mui.std().item())
                all_mu_mean.append(mui.mean().item())
                all_logvar_mean.append(logvari.mean().item())
                all_posterior_std.append(pstdi.mean().item())
                n_processed += 1

            if n_processed % 10 == 0:
                logger.info(f"  Processed {n_processed}/{max_samples}")

    if n_processed == 0:
        logger.error("No samples processed!")
        sys.exit(1)

    # Compute aggregate statistics
    import numpy as np
    z_stds = np.array(all_z_std)
    z_means = np.array(all_z_mean)
    mu_stds = np.array(all_mu_std)
    mu_means = np.array(all_mu_mean)
    logvar_means = np.array(all_logvar_mean)
    posterior_stds = np.array(all_posterior_std)

    print("\n" + "=" * 60)
    print(f"LATENT SPACE STATISTICS ({n_processed} samples)")
    print("=" * 60)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Model: {comp_type}, {spatial_dims}D, scale={scale_factor}x, latent_ch={latent_channels}")

    print(f"\n--- Sampled z = mu + std*eps ---")
    print(f"  z.std()   = {z_stds.mean():.4f}  (target: 0.9-1.1)")
    print(f"  z.mean()  = {z_means.mean():.4f}  (target: ~0.0)")

    print(f"\n--- Encoder posterior q(z|x) ---")
    print(f"  mu.std()              = {mu_stds.mean():.4f}  (spread of means)")
    print(f"  mu.mean()             = {mu_means.mean():.4f}  (center of means)")
    print(f"  logvar.mean()         = {logvar_means.mean():.4f}  (target: ~0.0 for std~1)")
    print(f"  posterior_std.mean()  = {posterior_stds.mean():.4f}  (exp(0.5*logvar), target: ~1.0)")

    print(f"\n--- Interpretation ---")
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
        print(f"  posterior_std={p_std_avg:.4f} << 1: Near-deterministic encoder (KL ≈ 0).")
        print(f"    → Sampling z adds almost no noise. VAE behaves like a regular AE.")
    print("=" * 60)


@hydra.main(config_path="../../../../configs", config_name="vae_3d", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    measure_latent_stats(cfg)


if __name__ == "__main__":
    main()
