"""
Visualize augmentation pipelines for debugging and verification.

Creates side-by-side comparisons of:
- Original vs Diffusion augmentation (conservative)
- Original vs VAE augmentation (aggressive)
- Mixup blending examples
- CutMix pasting examples

Usage:
    # Visualize with default settings (uses paths from config)
    python -m medgen.scripts.visualize_augmentations

    # Override data directory
    python -m medgen.scripts.visualize_augmentations paths.data_dir=/path/to/data

    # Use cluster paths
    python -m medgen.scripts.visualize_augmentations paths=cluster

    # Show specific number of samples
    python -m medgen.scripts.visualize_augmentations n_samples=8

    # Force synthetic data
    python -m medgen.scripts.visualize_augmentations synthetic=true
"""
import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from medgen.augmentation import (
    apply_augmentation,
    build_diffusion_augmentation,
    build_vae_augmentation,
    cutmix,
    mixup,
)
from medgen.data.dataset import NiFTIDataset, build_standard_transform

logger = logging.getLogger(__name__)


def generate_synthetic_slices(
    image_size: int = 128,
    n_samples: int = 4,
) -> tuple[list[np.ndarray], list[str]]:
    """Generate synthetic brain-like slices for testing.

    Creates elliptical shapes with varying intensities to simulate
    brain MRI appearance.

    Args:
        image_size: Target image size.
        n_samples: Number of slices to generate.

    Returns:
        Tuple of (list of slices [H, W], list of slice names).
    """
    slices = []
    names = []

    for i in range(n_samples):
        # Create base image
        img = np.zeros((image_size, image_size), dtype=np.float32)

        # Add brain-like ellipse
        y, x = np.ogrid[:image_size, :image_size]
        center = image_size // 2
        # Slightly different shapes for variety
        a = image_size // 2.5 + np.random.randint(-5, 5)
        b = image_size // 3 + np.random.randint(-5, 5)

        # Brain outline
        brain_mask = ((x - center) ** 2 / a ** 2 + (y - center) ** 2 / b ** 2) <= 1
        img[brain_mask] = 0.3 + np.random.rand() * 0.2

        # Add some internal structures (ventricles, lesions)
        for _ in range(3):
            cx = center + np.random.randint(-20, 20)
            cy = center + np.random.randint(-15, 15)
            r = np.random.randint(5, 15)
            structure = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
            intensity = np.random.rand() * 0.5 + 0.3
            img[structure & brain_mask] = intensity

        # Add some noise
        img += np.random.randn(image_size, image_size) * 0.02
        img = np.clip(img, 0, 1)

        slices.append(img)
        names.append(f"synthetic_{i}")

    return slices, names


def load_sample_slices(
    data_dir: str | None,
    modality: str = "t1_pre",
    image_size: int = 128,
    n_samples: int = 4,
) -> tuple[list[np.ndarray], list[str]]:
    """Load sample 2D slices from NIfTI dataset.

    Args:
        data_dir: Path to data directory containing patient folders.
                  If None or invalid, generates synthetic data.
        modality: MR sequence to load (t1_pre, t1_gd, bravo, seg, flair).
        image_size: Target image size.
        n_samples: Number of slices to extract.

    Returns:
        Tuple of (list of slices [H, W], list of slice names).
    """
    # Try to load real data, fall back to synthetic
    if data_dir is None:
        logger.info("No data directory provided, using synthetic data")
        return generate_synthetic_slices(image_size, n_samples)

    transform = build_standard_transform(image_size)

    # Try train directory first, then root
    train_dir = os.path.join(data_dir, "train")
    if os.path.exists(train_dir):
        data_dir = train_dir

    # Check if data directory exists and has patient folders
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory not found: {data_dir}")
        logger.info("Falling back to synthetic data")
        return generate_synthetic_slices(image_size, n_samples)

    try:
        dataset = NiFTIDataset(data_dir=data_dir, mr_sequence=modality, transform=transform)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
    except Exception as e:
        logger.warning(f"Failed to load dataset: {e}")
        logger.info("Falling back to synthetic data")
        return generate_synthetic_slices(image_size, n_samples)

    slices = []
    names = []

    # Extract slices from first few volumes
    samples_per_volume = max(1, n_samples // min(len(dataset), 4))
    volumes_needed = min(len(dataset), (n_samples + samples_per_volume - 1) // samples_per_volume)

    for vol_idx in range(volumes_needed):
        if len(slices) >= n_samples:
            break

        volume, patient_name = dataset[vol_idx]  # volume: [C, H, W, D], patient_name: str
        # Remove channel dim if present: [C, H, W, D] -> [H, W, D]
        if volume.ndim == 4:
            volume = volume[0]
        # Now volume is [H, W, D], transpose to [D, H, W] for slice extraction
        volume = volume.permute(2, 0, 1)  # [D, H, W]
        num_slices = volume.shape[0]

        # Pick slices from middle region (more interesting content)
        start = num_slices // 4
        end = 3 * num_slices // 4
        indices = np.linspace(start, end - 1, samples_per_volume, dtype=int)

        for slice_idx in indices:
            if len(slices) >= n_samples:
                break
            slices.append(volume[slice_idx].numpy())
            names.append(f"{patient_name}_slice{slice_idx}")

    return slices, names


def visualize_per_sample_augmentation(
    slices: list[np.ndarray],
    names: list[str],
    output_path: str,
    n_augmented: int = 3,
) -> None:
    """Visualize per-sample augmentations (diffusion vs VAE).

    Creates a grid showing original and augmented versions.

    Args:
        slices: List of 2D slices [H, W].
        names: List of slice names.
        output_path: Path to save visualization.
        n_augmented: Number of augmented versions per original.
    """
    diffusion_aug = build_diffusion_augmentation(enabled=True)
    vae_aug = build_vae_augmentation(enabled=True)

    n_slices = len(slices)
    n_cols = 1 + n_augmented * 2  # Original + diffusion versions + VAE versions
    n_rows = n_slices

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Column headers
    col_titles = ["Original"] + [f"Diffusion {i+1}" for i in range(n_augmented)] + [f"VAE {i+1}" for i in range(n_augmented)]

    for row, (slice_2d, name) in enumerate(zip(slices, names)):
        # Original
        axes[row, 0].imshow(slice_2d, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_ylabel(name, fontsize=8)
        if row == 0:
            axes[row, 0].set_title(col_titles[0], fontsize=10)
        axes[row, 0].axis('off')

        # Prepare for augmentation: [1, H, W] -> apply_augmentation expects [C, H, W]
        slice_chw = slice_2d[np.newaxis, :, :]

        # Diffusion augmentations
        for i in range(n_augmented):
            aug_slice = apply_augmentation(slice_chw.copy(), diffusion_aug, has_mask=False)
            axes[row, 1 + i].imshow(aug_slice[0], cmap='gray', vmin=0, vmax=1)
            if row == 0:
                axes[row, 1 + i].set_title(col_titles[1 + i], fontsize=10)
            axes[row, 1 + i].axis('off')

        # VAE augmentations
        for i in range(n_augmented):
            aug_slice = apply_augmentation(slice_chw.copy(), vae_aug, has_mask=False)
            axes[row, 1 + n_augmented + i].imshow(aug_slice[0], cmap='gray', vmin=0, vmax=1)
            if row == 0:
                axes[row, 1 + n_augmented + i].set_title(col_titles[1 + n_augmented + i], fontsize=10)
            axes[row, 1 + n_augmented + i].axis('off')

    plt.suptitle("Per-Sample Augmentation: Diffusion (conservative) vs VAE (aggressive)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved per-sample augmentation visualization to {output_path}")


def visualize_batch_augmentations(
    slices: list[np.ndarray],
    output_path: str,
) -> None:
    """Visualize batch-level augmentations (mixup and cutmix).

    Args:
        slices: List of 2D slices [H, W].
        output_path: Path to save visualization.
    """
    # Need at least 4 slices for good visualization
    n_slices = min(len(slices), 4)
    slices = slices[:n_slices]

    # Stack into batch [B, 1, H, W]
    batch = np.stack([s[np.newaxis, :, :] for s in slices], axis=0)

    # Apply mixup multiple times
    n_examples = 3
    mixup_results = []
    cutmix_results = []

    for _ in range(n_examples):
        mixed, indices, lam = mixup(batch.copy(), alpha=0.4)
        mixup_results.append((mixed, indices, lam))

        cut, indices, lam = cutmix(batch.copy(), alpha=1.0)
        cutmix_results.append((cut, indices, lam))

    # Create visualization
    fig, axes = plt.subplots(3, n_slices + n_examples, figsize=((n_slices + n_examples) * 2.5, 3 * 2.5))

    # Row 0: Original images
    for i in range(n_slices):
        axes[0, i].imshow(slices[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Original {i}", fontsize=10)
        axes[0, i].axis('off')

    # Empty cells in row 0
    for i in range(n_slices, n_slices + n_examples):
        axes[0, i].axis('off')

    # Row 1: Mixup examples
    axes[1, 0].text(0.5, 0.5, "Mixup\nBlending", ha='center', va='center', fontsize=12,
                    transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')

    for i, (mixed, indices, lam) in enumerate(mixup_results):
        col = 1 + i
        if col < n_slices + n_examples:
            # Show first image in the mixed batch
            axes[1, col].imshow(mixed[0, 0], cmap='gray', vmin=0, vmax=1)
            axes[1, col].set_title(f"λ={lam:.2f}\n0+{indices[0]}", fontsize=9)
            axes[1, col].axis('off')

    # Fill remaining with more mixup examples from different batch positions
    col = 1 + n_examples
    for _, (mixed, indices, lam) in enumerate(mixup_results):
        if col < n_slices + n_examples and len(mixed) > 1:
            axes[1, col].imshow(mixed[1, 0], cmap='gray', vmin=0, vmax=1)
            axes[1, col].set_title(f"λ={lam:.2f}\n1+{indices[1]}", fontsize=9)
            axes[1, col].axis('off')
            col += 1

    # Hide any remaining cells
    for i in range(col, n_slices + n_examples):
        axes[1, i].axis('off')

    # Row 2: CutMix examples
    axes[2, 0].text(0.5, 0.5, "CutMix\nPasting", ha='center', va='center', fontsize=12,
                    transform=axes[2, 0].transAxes)
    axes[2, 0].axis('off')

    for i, (cut, indices, lam) in enumerate(cutmix_results):
        col = 1 + i
        if col < n_slices + n_examples:
            axes[2, col].imshow(cut[0, 0], cmap='gray', vmin=0, vmax=1)
            axes[2, col].set_title(f"λ={lam:.2f}\n0+{indices[0]}", fontsize=9)
            axes[2, col].axis('off')

    # Fill remaining with more cutmix examples
    col = 1 + n_examples
    for _, (cut, indices, lam) in enumerate(cutmix_results):
        if col < n_slices + n_examples and len(cut) > 1:
            axes[2, col].imshow(cut[1, 0], cmap='gray', vmin=0, vmax=1)
            axes[2, col].set_title(f"λ={lam:.2f}\n1+{indices[1]}", fontsize=9)
            axes[2, col].axis('off')
            col += 1

    for i in range(col, n_slices + n_examples):
        axes[2, i].axis('off')

    plt.suptitle("Batch-Level Augmentation: Mixup (blending) and CutMix (rectangular paste)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved batch augmentation visualization to {output_path}")


def visualize_augmentation_comparison(
    slices: list[np.ndarray],
    output_path: str,
) -> None:
    """Create a compact comparison showing all augmentation types.

    Args:
        slices: List of 2D slices [H, W].
        output_path: Path to save visualization.
    """
    diffusion_aug = build_diffusion_augmentation(enabled=True)
    vae_aug = build_vae_augmentation(enabled=True)

    # Use first 2 slices
    n_slices = min(2, len(slices))
    slices = slices[:n_slices]

    # Create batch for mixup/cutmix
    batch = np.stack([s[np.newaxis, :, :] for s in slices * 2], axis=0)  # Need at least 4

    fig, axes = plt.subplots(n_slices, 6, figsize=(15, n_slices * 2.5))
    if n_slices == 1:
        axes = axes.reshape(1, -1)

    titles = ["Original", "Diffusion Aug", "VAE Aug", "VAE + More", "Mixup", "CutMix"]

    for row, slice_2d in enumerate(slices):
        slice_chw = slice_2d[np.newaxis, :, :]

        # Original
        axes[row, 0].imshow(slice_2d, cmap='gray', vmin=0, vmax=1)

        # Diffusion augmentation
        diff_aug = apply_augmentation(slice_chw.copy(), diffusion_aug, has_mask=False)
        axes[row, 1].imshow(diff_aug[0], cmap='gray', vmin=0, vmax=1)

        # VAE augmentation (2 examples)
        vae_aug1 = apply_augmentation(slice_chw.copy(), vae_aug, has_mask=False)
        axes[row, 2].imshow(vae_aug1[0], cmap='gray', vmin=0, vmax=1)

        vae_aug2 = apply_augmentation(slice_chw.copy(), vae_aug, has_mask=False)
        axes[row, 3].imshow(vae_aug2[0], cmap='gray', vmin=0, vmax=1)

        # Mixup
        mixed, _, lam = mixup(batch.copy(), alpha=0.4)
        axes[row, 4].imshow(mixed[row, 0], cmap='gray', vmin=0, vmax=1)

        # CutMix
        cut, _, lam = cutmix(batch.copy(), alpha=1.0)
        axes[row, 5].imshow(cut[row, 0], cmap='gray', vmin=0, vmax=1)

        for col in range(6):
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=10)
            axes[row, col].axis('off')

    plt.suptitle("Augmentation Pipeline Overview", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison visualization to {output_path}")


@hydra.main(version_base=None, config_path="../../../configs", config_name="visualize_augmentations")
def main(cfg: DictConfig) -> None:
    """Main entry point for augmentation visualization.

    Args:
        cfg: Hydra configuration object.
    """
    # Get settings from config
    data_dir = cfg.paths.data_dir
    synthetic = cfg.get('synthetic', False)
    modality = cfg.get('modality', 't1_pre')
    n_samples = cfg.get('n_samples', 4)
    image_size = cfg.get('image_size', 128)
    output_dir = Path(cfg.get('output_dir', 'outputs/augmentation_viz'))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine data source
    data_dir = None if synthetic else data_dir

    if data_dir:
        logger.info(f"Loading samples from {data_dir}")
        logger.info(f"Modality: {modality}, Image size: {image_size}")
    else:
        logger.info(f"Using synthetic data, Image size: {image_size}")

    # Load sample slices
    slices, names = load_sample_slices(
        data_dir=data_dir,
        modality=modality,
        image_size=image_size,
        n_samples=n_samples,
    )
    logger.info(f"Loaded {len(slices)} slices")

    # Generate visualizations
    logger.info("\nGenerating visualizations...")

    # 1. Per-sample augmentation comparison
    visualize_per_sample_augmentation(
        slices=slices,
        names=names,
        output_path=str(output_dir / "per_sample_augmentation.png"),
        n_augmented=3,
    )

    # 2. Batch-level augmentation
    visualize_batch_augmentations(
        slices=slices,
        output_path=str(output_dir / "batch_augmentation.png"),
    )

    # 3. Compact comparison
    visualize_augmentation_comparison(
        slices=slices,
        output_path=str(output_dir / "augmentation_comparison.png"),
    )

    logger.info(f"\nAll visualizations saved to {output_dir}/")
    logger.info("Files:")
    logger.info("  - per_sample_augmentation.png: Diffusion vs VAE augmentation")
    logger.info("  - batch_augmentation.png: Mixup and CutMix examples")
    logger.info("  - augmentation_comparison.png: Compact overview")


if __name__ == "__main__":
    main()
