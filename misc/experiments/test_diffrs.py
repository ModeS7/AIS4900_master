"""Quick end-to-end test of DiffRS on the bravo model.

Tests whether DiffRS improves generation quality by:
1. Loading the trained bravo model
2. Generating samples for DiffRS head training
3. Loading real bravo volumes
4. Training the tiny discriminator head
5. Generating samples WITH and WITHOUT DiffRS
6. Comparing quality metrics

Usage:
    python misc/experiments/test_diffrs.py
"""
import logging
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from medgen.diffusion import RFlowStrategy, load_diffusion_model
from medgen.diffusion.diffrs import (
    DiffRSDiscriminator,
    DiffRSHead,
    extract_encoder_features,
    get_bottleneck_channels,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
CHECKPOINT = "runs/checkpoint_bravo.pt"
DATA_DIR = "/home/mode/NTNU/MedicalDataSets/brainmetshare-3/train"
OUTPUT_DIR = "misc/experiments/diffrs_results"

# Keep numbers small for a quick local test
NUM_GEN_SAMPLES = 6        # Samples for DiffRS head training
NUM_REAL_SAMPLES = 6       # Real samples for head training
HEAD_EPOCHS = 20           # Training epochs for head
GEN_STEPS = 10             # Denoising steps for sample generation
NUM_TEST_SAMPLES = 4       # Samples to generate for comparison
TEST_STEPS = 10            # Denoising steps for test generation

# DiffRS hyperparameters
DIFFRS_WARMUP = 3          # Warmup iterations for threshold estimation
DIFFRS_PERCENTILE = 0.75
DIFFRS_BACKSTEPS = 1

IMAGE_SIZE = 256
DEPTH = 160
BATCH_SIZE = 1  # Memory-limited
# ───────────────────────────────────────────────────────────────────────────


def load_real_bravo_volumes(data_dir: str, num_samples: int) -> torch.Tensor:
    """Load real bravo volumes from NIfTI files."""
    subjects = sorted(os.listdir(data_dir))[:num_samples]
    volumes = []
    for subj in subjects:
        path = os.path.join(data_dir, subj, "bravo.nii.gz")
        if not os.path.exists(path):
            continue
        nii = nib.load(path)
        vol = nii.get_fdata().astype(np.float32)
        # Normalize to [0, 1]
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        # Resize to target size if needed (simple center crop / pad)
        vol = _resize_volume(vol, IMAGE_SIZE, DEPTH)
        volumes.append(torch.from_numpy(vol).unsqueeze(0))  # [1, D, H, W]
        if len(volumes) >= num_samples:
            break
    result = torch.stack(volumes, dim=0)  # [N, 1, D, H, W]
    logger.info("Loaded %d real bravo volumes: %s", len(result), result.shape)
    return result


def load_real_seg_masks(data_dir: str, num_samples: int) -> torch.Tensor:
    """Load real seg masks from NIfTI files (for bravo conditioning)."""
    subjects = sorted(os.listdir(data_dir))[:num_samples]
    volumes = []
    for subj in subjects:
        path = os.path.join(data_dir, subj, "seg.nii.gz")
        if not os.path.exists(path):
            continue
        nii = nib.load(path)
        vol = nii.get_fdata().astype(np.float32)
        vol = (vol > 0.5).astype(np.float32)  # Binarize
        vol = _resize_volume(vol, IMAGE_SIZE, DEPTH)
        volumes.append(torch.from_numpy(vol).unsqueeze(0))
        if len(volumes) >= num_samples:
            break
    result = torch.stack(volumes, dim=0)
    logger.info("Loaded %d seg masks: %s", len(result), result.shape)
    return result


def _resize_volume(vol: np.ndarray, target_hw: int, target_d: int) -> np.ndarray:
    """Resize volume to [D, H, W] via center crop/pad."""
    # vol is [H, W, D] from NIfTI → transpose to [D, H, W]
    if vol.ndim == 3:
        vol = np.transpose(vol, (2, 0, 1))  # [D, H, W]

    d, h, w = vol.shape

    # Pad depth if needed
    if d < target_d:
        pad = target_d - d
        vol = np.pad(vol, ((0, pad), (0, 0), (0, 0)), mode='constant')
    elif d > target_d:
        start = (d - target_d) // 2
        vol = vol[start:start + target_d]

    # Simple resize for H, W if needed
    if h != target_hw or w != target_hw:
        from scipy.ndimage import zoom
        zoom_h = target_hw / vol.shape[1]
        zoom_w = target_hw / vol.shape[2]
        vol = zoom(vol, (1.0, zoom_h, zoom_w), order=1)

    return vol


def generate_bravo_samples(
    model: nn.Module,
    strategy: RFlowStrategy,
    seg_masks: torch.Tensor,
    device: torch.device,
    num_steps: int,
    diffrs_disc=None,
    diffrs_config=None,
) -> torch.Tensor:
    """Generate bravo samples conditioned on seg masks."""
    all_samples = []
    for i in range(len(seg_masks)):
        noise = torch.randn(1, 1, DEPTH, IMAGE_SIZE, IMAGE_SIZE, device=device)
        seg = seg_masks[i:i+1].to(device)
        model_input = torch.cat([noise, seg], dim=1)

        gen_kwargs = {}
        if diffrs_disc is not None:
            gen_kwargs['diffrs_discriminator'] = diffrs_disc
            gen_kwargs['diffrs_config'] = diffrs_config

        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            with torch.no_grad():
                sample = strategy.generate(
                    model, model_input, num_steps, device, **gen_kwargs,
                )
        all_samples.append(sample.cpu())
        torch.cuda.empty_cache()
        logger.info("  Generated sample %d/%d", i + 1, len(seg_masks))

    return torch.cat(all_samples, dim=0)


def compute_simple_metrics(
    generated: torch.Tensor, real: torch.Tensor,
) -> dict[str, float]:
    """Compute simple quality metrics between generated and real volumes."""
    metrics = {}

    # Pixel statistics
    gen_mean = generated.mean().item()
    gen_std = generated.std().item()
    real_mean = real.mean().item()
    real_std = real.std().item()
    metrics['gen_mean'] = gen_mean
    metrics['gen_std'] = gen_std
    metrics['mean_diff'] = abs(gen_mean - real_mean)
    metrics['std_diff'] = abs(gen_std - real_std)

    # Mean absolute error (lower = better, rough proxy)
    # Compare distributions via random pairing
    n = min(len(generated), len(real))
    mae = (generated[:n] - real[:n]).abs().mean().item()
    metrics['mae_paired'] = mae

    # SSIM-like: structural similarity on middle slices
    mid_d = generated.shape[2] // 2
    gen_slices = generated[:, 0, mid_d].numpy()  # [N, H, W]
    real_slices = real[:n, 0, mid_d].numpy()

    # Per-sample SSIM
    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_scores = []
        for i in range(min(n, len(gen_slices))):
            s = ssim(
                gen_slices[i], real_slices[i],
                data_range=1.0,
            )
            ssim_scores.append(s)
        metrics['ssim_mid_slice'] = float(np.mean(ssim_scores))
    except ImportError:
        logger.warning("skimage not available, skipping SSIM")

    # Value range check
    metrics['gen_min'] = generated.min().item()
    metrics['gen_max'] = generated.max().item()
    metrics['pct_in_range'] = (
        (generated >= 0) & (generated <= 1)
    ).float().mean().item() * 100

    return metrics


def save_comparison_slices(
    baseline: torch.Tensor,
    diffrs: torch.Tensor,
    real: torch.Tensor,
    output_dir: str,
) -> None:
    """Save middle-slice comparison images."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    mid_d = baseline.shape[2] // 2
    n = min(len(baseline), len(diffrs), len(real))

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        axes[i, 0].imshow(real[i, 0, mid_d].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Real')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(
            baseline[i, 0, mid_d].clamp(0, 1).numpy(), cmap='gray', vmin=0, vmax=1,
        )
        axes[i, 1].set_title('Baseline')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(
            diffrs[i, 0, mid_d].clamp(0, 1).numpy(), cmap='gray', vmin=0, vmax=1,
        )
        axes[i, 2].set_title('DiffRS')
        axes[i, 2].axis('off')

    plt.suptitle('Real vs Baseline vs DiffRS (middle slice)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
    plt.close()
    logger.info("Saved comparison image to %s/comparison.png", output_dir)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DiffRS Performance Test")
    logger.info("=" * 60)

    # ── Step 1: Load model ─────────────────────────────────────────────
    logger.info("Step 1: Loading bravo model...")
    model = load_diffusion_model(
        CHECKPOINT, device=device,
        in_channels=2, out_channels=1, spatial_dims=3,
    )
    logger.info("VRAM after model load: %.1fGB", torch.cuda.memory_allocated() / 1e9)

    strategy = RFlowStrategy()
    strategy.setup_scheduler(1000, image_size=IMAGE_SIZE, depth_size=DEPTH, spatial_dims=3)

    # ── Step 2: Load real data ─────────────────────────────────────────
    logger.info("Step 2: Loading real data...")
    # Load enough for both head training and comparison
    total_needed = max(NUM_REAL_SAMPLES, NUM_TEST_SAMPLES)
    real_bravo = load_real_bravo_volumes(DATA_DIR, total_needed)
    real_seg = load_real_seg_masks(DATA_DIR, total_needed)

    # ── Step 3: Generate samples for head training ─────────────────────
    logger.info("Step 3: Generating %d samples for DiffRS head training...", NUM_GEN_SAMPLES)
    t0 = time.time()
    # Use real seg masks as conditioning for generation
    gen_seg = real_seg[:NUM_GEN_SAMPLES]
    generated_bravo = generate_bravo_samples(
        model, strategy, gen_seg, device, num_steps=GEN_STEPS,
    )
    gen_time = time.time() - t0
    logger.info("Generation took %.1fs (%.1fs/sample)", gen_time, gen_time / NUM_GEN_SAMPLES)

    # ── Step 4: Train DiffRS head ──────────────────────────────────────
    logger.info("Step 4: Training DiffRS head...")
    bottleneck_ch = get_bottleneck_channels(model)
    head = DiffRSHead(in_channels=bottleneck_ch, spatial_dims=3).to(device)
    num_params = sum(p.numel() for p in head.parameters())
    logger.info("Head params: %d", num_params)

    # Simple training loop
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-4, weight_decay=1e-7)
    criterion = nn.BCEWithLogitsLoss()

    real_train = real_bravo[:NUM_REAL_SAMPLES]
    gen_train = generated_bravo[:NUM_GEN_SAMPLES]
    all_data = torch.cat([real_train, gen_train], dim=0)
    labels = torch.cat([
        torch.ones(len(real_train)),
        torch.zeros(len(gen_train)),
    ])

    head.train()
    for epoch in range(HEAD_EPOCHS):
        perm = torch.randperm(len(all_data))
        epoch_loss = 0
        epoch_correct = 0

        for i in range(0, len(all_data), BATCH_SIZE):
            batch_x = all_data[perm[i:i + BATCH_SIZE]].to(device)
            batch_y = labels[perm[i:i + BATCH_SIZE]].to(device)

            timesteps = strategy.sample_timesteps(batch_x)
            noise = torch.randn_like(batch_x)
            noisy = strategy.scheduler.add_noise(batch_x, noise, timesteps)

            with torch.no_grad():
                features = extract_encoder_features(model, noisy, timesteps)

            logits = head(features.float())
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += ((logits > 0).float() == batch_y).sum().item()

            torch.cuda.empty_cache()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            acc = epoch_correct / len(all_data) * 100
            logger.info(
                "  Epoch %d/%d: loss=%.4f, accuracy=%.1f%%",
                epoch + 1, HEAD_EPOCHS, epoch_loss / max(1, len(all_data)),
                acc,
            )

    head.eval()

    # Save head
    head_path = os.path.join(OUTPUT_DIR, "diffrs_head_test.pt")
    torch.save({
        'head_state_dict': head.state_dict(),
        'in_channels': bottleneck_ch,
        'spatial_dims': 3,
    }, head_path)
    logger.info("Head saved to %s", head_path)

    # ── Step 5: Generate test samples ──────────────────────────────────
    test_seg = real_seg[:NUM_TEST_SAMPLES]

    # Baseline (no DiffRS)
    logger.info("Step 5a: Generating %d baseline samples...", NUM_TEST_SAMPLES)
    t0 = time.time()
    baseline_samples = generate_bravo_samples(
        model, strategy, test_seg, device, num_steps=TEST_STEPS,
    )
    baseline_time = time.time() - t0
    logger.info("Baseline: %.1fs (%.1fs/sample)", baseline_time, baseline_time / NUM_TEST_SAMPLES)

    # DiffRS
    logger.info("Step 5b: Generating %d DiffRS samples...", NUM_TEST_SAMPLES)
    disc = DiffRSDiscriminator(model, head, device)
    diffrs_config = {
        'rej_percentile': DIFFRS_PERCENTILE,
        'backsteps': DIFFRS_BACKSTEPS,
        'max_iter': 999999,
        'iter_warmup': DIFFRS_WARMUP,
    }

    t0 = time.time()
    diffrs_samples = generate_bravo_samples(
        model, strategy, test_seg, device, num_steps=TEST_STEPS,
        diffrs_disc=disc, diffrs_config=diffrs_config,
    )
    diffrs_time = time.time() - t0
    logger.info("DiffRS: %.1fs (%.1fs/sample)", diffrs_time, diffrs_time / NUM_TEST_SAMPLES)

    # ── Step 6: Compare ────────────────────────────────────────────────
    logger.info("Step 6: Computing metrics...")
    real_test = real_bravo[:NUM_TEST_SAMPLES]

    baseline_metrics = compute_simple_metrics(baseline_samples, real_test)
    diffrs_metrics = compute_simple_metrics(diffrs_samples, real_test)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    logger.info("%-25s %12s %12s", "Metric", "Baseline", "DiffRS")
    logger.info("-" * 60)
    for key in baseline_metrics:
        b = baseline_metrics[key]
        d = diffrs_metrics[key]
        better = ""
        if key in ('mean_diff', 'std_diff', 'mae_paired'):
            better = " <--" if d < b else ""
        elif key in ('ssim_mid_slice', 'pct_in_range'):
            better = " <--" if d > b else ""
        logger.info("%-25s %12.4f %12.4f%s", key, b, d, better)
    logger.info("-" * 60)
    logger.info("Time per sample:          %10.1fs %10.1fs",
                baseline_time / NUM_TEST_SAMPLES, diffrs_time / NUM_TEST_SAMPLES)
    logger.info("")

    # Save comparison images
    save_comparison_slices(baseline_samples, diffrs_samples, real_test, OUTPUT_DIR)

    logger.info("Done! Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
