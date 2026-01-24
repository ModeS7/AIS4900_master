#!/usr/bin/env python3
"""
Randomly sample images from a dataset and convert to JPG for manual quality inspection.

Usage:
    python sample_dataset.py /path/to/dataset --num-samples 50 --output samples/
    python sample_dataset.py /path/to/dataset -n 100 --seed 42 --recursive
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List
import nibabel as nib
import numpy as np
from PIL import Image


def find_images(dataset_path: Path, recursive: bool = True) -> List[Path]:
    """Find all image files in dataset."""
    extensions = ['.nii.gz', '.nii', '.png', '.jpg', '.jpeg', '.dcm']

    files = []
    if recursive:
        for ext in extensions:
            if ext == '.nii.gz':
                files.extend(dataset_path.rglob('*.nii.gz'))
            else:
                files.extend(dataset_path.rglob(f'*{ext}'))
    else:
        for ext in extensions:
            if ext == '.nii.gz':
                files.extend(dataset_path.glob('*.nii.gz'))
            else:
                files.extend(dataset_path.glob(f'*{ext}'))

    # Remove duplicates (nii.gz might be found twice)
    files = list(set(files))
    return sorted(files)


def normalize_to_uint8(data: np.ndarray, use_percentile: bool = False) -> np.ndarray:
    """Normalize image data to 0-255 uint8 range."""
    data = data.astype(np.float32)

    if use_percentile:
        # Clip to 1st-99th percentile to handle outliers
        vmin, vmax = np.percentile(data, [1, 99])
        data = np.clip(data, vmin, vmax)
    else:
        # Use full min-max range (matches NIfTI viewers)
        vmin, vmax = data.min(), data.max()

    # Normalize to 0-255
    if vmax > vmin:
        data = (data - vmin) / (vmax - vmin) * 255
    else:
        data = np.zeros_like(data)

    return data.astype(np.uint8)


def convert_nifti_to_jpg(nii_path: Path, output_path: Path, slice_idx: int = None, channel: str = 'bravo'):
    """Convert NIfTI file to JPG (middle slice by default or specified channel for 2-channel data).

    Args:
        nii_path: Path to NIfTI file
        output_path: Path to output JPG
        slice_idx: Slice index for 3D volumes (None = middle)
        channel: 'bravo' (channel 0), 'seg' (channel 1), or 'both' (side-by-side)
    """
    nii = nib.load(nii_path)
    volume = nii.get_fdata()

    # Handle 2-channel case (bravo + seg)
    if len(volume.shape) == 3 and volume.shape[2] == 2:
        if channel == 'bravo':
            slice_data = volume[:, :, 0]
            slice_data = normalize_to_uint8(slice_data)
        elif channel == 'seg':
            slice_data = volume[:, :, 1]
            slice_data = normalize_to_uint8(slice_data)
        elif channel == 'both':
            # Create side-by-side image
            bravo = normalize_to_uint8(volume[:, :, 0])
            seg = normalize_to_uint8(volume[:, :, 1])
            slice_data = np.hstack([bravo, seg])
        else:
            raise ValueError(f"Invalid channel: {channel}. Use 'bravo', 'seg', or 'both'")
    else:
        # Get middle slice if not specified
        if slice_idx is None:
            slice_idx = volume.shape[2] // 2
        # Extract slice
        slice_data = volume[:, :, slice_idx]
        slice_data = normalize_to_uint8(slice_data)

    # Save as JPG
    img = Image.fromarray(slice_data, mode='L')
    img.save(output_path, 'JPEG', quality=95)

    return slice_data.shape


def convert_png_to_jpg(png_path: Path, output_path: Path):
    """Convert PNG to JPG."""
    img = Image.open(png_path)

    # Convert to grayscale if needed
    if img.mode != 'L' and img.mode != 'RGB':
        img = img.convert('L')

    img.save(output_path, 'JPEG', quality=95)
    return img.size


def copy_jpg(jpg_path: Path, output_path: Path):
    """Copy existing JPG."""
    shutil.copy2(jpg_path, output_path)
    img = Image.open(jpg_path)
    return img.size


def sample_and_convert(dataset_path: Path, output_dir: Path, num_samples: int,
                       seed: int = 42, recursive: bool = True, channel: str = 'bravo'):
    """Sample random images and convert to JPG."""

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Find all images
    print(f"Scanning dataset: {dataset_path}")
    all_images = find_images(dataset_path, recursive)
    print(f"Found {len(all_images)} images")

    if len(all_images) == 0:
        print("No images found!")
        return

    # Sample random images
    num_samples = min(num_samples, len(all_images))
    sampled_images = random.sample(all_images, num_samples)

    print(f"\nSampling {num_samples} random images (seed={seed})")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert and save
    manifest = []
    for idx, img_path in enumerate(sampled_images, 1):
        # Create output filename
        relative_path = img_path.relative_to(dataset_path)
        safe_name = str(relative_path).replace('/', '_').replace('\\', '_')
        safe_name = safe_name.replace('.nii.gz', '').replace('.nii', '')
        output_path = output_dir / f"{idx:03d}_{safe_name}.jpg"

        try:
            # Convert based on file type
            if img_path.suffix == '.gz' or img_path.name.endswith('.nii.gz'):
                shape = convert_nifti_to_jpg(img_path, output_path, channel=channel)
                manifest.append(f"{idx:03d}: {relative_path} (NIfTI, channel={channel}, shape={shape})")
            elif img_path.suffix == '.nii':
                shape = convert_nifti_to_jpg(img_path, output_path, channel=channel)
                manifest.append(f"{idx:03d}: {relative_path} (NIfTI, channel={channel}, shape={shape})")
            elif img_path.suffix.lower() in ['.png']:
                size = convert_png_to_jpg(img_path, output_path)
                manifest.append(f"{idx:03d}: {relative_path} (PNG, size={size})")
            elif img_path.suffix.lower() in ['.jpg', '.jpeg']:
                size = copy_jpg(img_path, output_path)
                manifest.append(f"{idx:03d}: {relative_path} (JPG, size={size})")
            else:
                manifest.append(f"{idx:03d}: {relative_path} (SKIPPED - unsupported format)")
                continue

            print(f"  [{idx}/{num_samples}] ✓ {relative_path}")

        except Exception as e:
            print(f"  [{idx}/{num_samples}] ✗ {relative_path} - Error: {e}")
            manifest.append(f"{idx:03d}: {relative_path} (ERROR: {e})")

    # Save manifest
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Total images in dataset: {len(all_images)}\n")
        f.write(f"Sampled: {num_samples}\n")
        f.write("\n")
        f.write("\n".join(manifest))

    print(f"\n✓ Converted {num_samples} images to {output_dir}")
    print(f"✓ Manifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Randomly sample images from dataset and convert to JPG for manual inspection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample 50 images with default seed
  python sample_dataset.py /path/to/dataset -n 50 -o samples/

  # Sample 100 images with specific seed
  python sample_dataset.py /path/to/dataset -n 100 --seed 42 -o quality_check/

  # Sample from current directory only (non-recursive)
  python sample_dataset.py /path/to/dataset -n 30 --no-recursive

  # Sample from BrainMetShare dataset
  python sample_dataset.py /home/mode/NTNU/MedicalDataSets/brainmetshare-3/train -n 50 -o samples/

  # Sample segmentation masks only
  python sample_dataset.py /path/to/dataset -n 50 --channel seg -o seg_samples/

  # Sample both bravo and seg side-by-side
  python sample_dataset.py /path/to/dataset -n 50 --channel both -o both_samples/
        """
    )

    parser.add_argument('dataset_path', type=str, help='Path to dataset directory')
    parser.add_argument('-n', '--num-samples', type=int, default=500,
                        help='Number of images to sample (default: 500)')
    parser.add_argument('-o', '--output', type=str, default='samples',
                        help='Output directory for JPG files (default: samples/)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--channel', type=str, default='bravo', choices=['bravo', 'seg', 'both'],
                        help='Channel to extract: bravo (default), seg, or both (side-by-side)')
    parser.add_argument('--no-recursive', action='store_true',
                        help='Do not search subdirectories')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    sample_and_convert(
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        recursive=not args.no_recursive,
        channel=args.channel
    )


if __name__ == '__main__':
    main()
