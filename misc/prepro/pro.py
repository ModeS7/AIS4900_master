#!/usr/bin/env python3
"""
Preprocess medical images to standardized format using MONAI transforms.

Two-step processing pipeline to preserve aspect ratio:
  1. Pad to intermediate size (240x240) - preserves aspect ratio, centered
  2. Resize to target size (256x256) - power of 2 for deep learning

Processing pipeline:
  Images (bravo, flair, t1_gd, t1_pre):
    1. Load NIfTI image
    2. Ensure channel-first format [C, H, W, D]
    3. Pad to 240x240 (centered, preserves aspect ratio)
    4. Resize to 256x256 with bilinear interpolation
    5. Save as float32 compressed NIfTI

  Segmentation masks (seg):
    1. Load NIfTI image
    2. Ensure channel-first format [C, H, W, D]
    3. Pad to 240x240 (centered, preserves aspect ratio)
    4. Resize to 256x256 with nearest-neighbor interpolation
    5. Binarize to strictly {0, 1} values (threshold at 0.5)
    6. Save as float32 compressed NIfTI

Processes all modalities (bravo, seg, flair, t1_gd, t1_pre).

Usage:
    python misc/prepro/pro.py --input /path/to/raw --output /path/to/processed
    python misc/prepro/pro.py  # Uses default paths
"""
import argparse
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    SpatialPad,
)
from tqdm import tqdm

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================
DEFAULT_INPUT_DIR = "/home/mode/NTNU/MedicalDataSets/StanfordSkullStripped_1mm"
DEFAULT_OUTPUT_DIR = "/home/mode/NTNU/MedicalDataSets/brainmetshare-3_non_public_256"
INTERMEDIATE_SIZE: Tuple[int, int] = (240, 240)  # Pad to this size first
TARGET_SIZE: Tuple[int, int] = (256, 256)        # Then resize to this final size
# ============================================================================


def create_transforms(
    is_segmentation: bool,
    intermediate_size: Tuple[int, int] = (240, 240),
    target_size: Tuple[int, int] = (256, 256)
) -> Compose:
    """Create MONAI transform pipeline for preprocessing.

    Two-step process: 1) Pad to intermediate size, 2) Resize to target size

    Args:
        is_segmentation: Whether this is a segmentation mask (uses nearest-neighbor).
        intermediate_size: Intermediate size to pad to (H, W).
        target_size: Final target spatial size (H, W).

    Returns:
        MONAI Compose transform.
    """
    if is_segmentation:
        # Segmentation: pad (centered), then nearest-neighbor resize, no normalization
        transform = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            SpatialPad(spatial_size=(intermediate_size[0], intermediate_size[1], -1), mode='constant', method='symmetric'),
            Resize(spatial_size=(target_size[0], target_size[1], -1), mode='nearest')
        ])
    else:
        # Images: pad (centered), then bilinear resize (no normalization - done during training)
        transform = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            SpatialPad(spatial_size=(intermediate_size[0], intermediate_size[1], -1), mode='constant', method='symmetric'),
            Resize(spatial_size=(target_size[0], target_size[1], -1), mode='bilinear')
        ])

    return transform


def process_patient(
    patient_dir: Path,
    output_dir: Path,
    intermediate_size: Tuple[int, int] = (240, 240),
    target_size: Tuple[int, int] = (256, 256)
) -> None:
    """Process a single patient directory using MONAI transforms.

    Args:
        patient_dir: Path to patient directory.
        output_dir: Path to output directory for this patient.
        intermediate_size: Intermediate size to pad to (H, W).
        target_size: Target spatial size (H, W).
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {patient_dir.name}:")

    # Find all .nii.gz files in patient directory
    nii_files = sorted(patient_dir.glob("*.nii.gz"))

    if not nii_files:
        print(f"  [WARNING] No .nii.gz files found in {patient_dir.name}")
        return

    for input_file in nii_files:
        modality = input_file.stem.replace('.nii', '')
        print(f"  Processing {modality}...")

        # Detect if segmentation mask
        is_segmentation = 'seg' in modality.lower()

        # Create appropriate transform pipeline
        transform = create_transforms(is_segmentation, intermediate_size, target_size)

        # Apply transforms
        processed_volume = transform(str(input_file))

        # Convert from torch tensor to numpy if needed
        if isinstance(processed_volume, torch.Tensor):
            processed_volume = processed_volume.cpu().numpy()

        # Remove channel dimension if present [1, H, W, D] -> [H, W, D]
        if processed_volume.shape[0] == 1:
            processed_volume = processed_volume[0]

        # Convert to float32 for storage efficiency
        processed_volume = processed_volume.astype(np.float32)

        # For segmentation masks, ensure binary (0 or 1) values
        if is_segmentation:
            # Threshold at 0.5 to make strictly binary
            processed_volume = np.where(processed_volume > 0.5, 1.0, 0.0).astype(np.float32)
            print(f"  Binarized: {np.unique(processed_volume)} (threshold=0.5)")

        # Load original to get affine
        original_nii = nib.load(input_file)

        # Create new NIfTI with identity affine (MONAI resets it)
        output_nii = nib.Nifti1Image(processed_volume, affine=np.eye(4))
        output_nii.header.set_data_dtype(np.float32)

        # Save compressed NIfTI
        output_file = output_dir / input_file.name
        nib.save(output_nii, output_file)

        file_size = output_file.stat().st_size / 1024 / 1024
        interp_type = "nearest-neighbor" if is_segmentation else "bilinear + normalized"
        print(f"  [SUCCESS] {processed_volume.shape} | {interp_type} | {file_size:.2f} MB")


def process_dataset(
    input_base: Path,
    output_base: Path,
    intermediate_size: Tuple[int, int] = (240, 240),
    target_size: Tuple[int, int] = (256, 256)
) -> None:
    """Process entire dataset (both test and train splits) using MONAI transforms.

    Args:
        input_base: Base directory of input dataset.
        output_base: Base directory for output dataset.
        intermediate_size: Intermediate size to pad to (H, W).
        target_size: Target spatial size (H, W).
    """
    for split in ['test', 'train']:
        split_dir = input_base / split
        if not split_dir.exists():
            print(f"[WARNING] {split} directory not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")

        # Get all patient directories
        patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        print(f"Found {len(patient_dirs)} patients in {split} split")

        # Process each patient
        for patient_dir in tqdm(patient_dirs, desc=f"Processing {split}"):
            output_patient_dir = output_base / split / patient_dir.name
            process_patient(
                patient_dir,
                output_patient_dir,
                intermediate_size,
                target_size
            )

    print(f"\n{'='*60}")
    print(f"Processing complete")
    print(f"{'='*60}")
    print(f"Input:  {input_base}")
    print(f"Output: {output_base}")


def main() -> int:
    """Main function to process dataset with command-line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Preprocess medical images to standardized format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python misc/prepro/pro.py --input /path/to/raw --output /path/to/processed
    python misc/prepro/pro.py  # Uses default paths
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"Input directory (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    input_base = args.input
    output_base = args.output

    # Validate input directory
    if not input_base.exists():
        print(f"[ERROR] Input directory does not exist: {input_base}")
        return 1

    # Show configuration
    print(f"\n{'='*60}")
    print(f"IMAGE PREPROCESSING CONFIGURATION (MONAI)")
    print(f"{'='*60}")
    print(f"Input directory:     {input_base}")
    print(f"Output directory:    {output_base}")
    print(f"Intermediate size:   {INTERMEDIATE_SIZE[0]}x{INTERMEDIATE_SIZE[1]} (padded)")
    print(f"Target size:         {TARGET_SIZE[0]}x{TARGET_SIZE[1]} (final)")
    print(f"\nTransforms:")
    print(f"  Images (bravo, flair, t1_gd, t1_pre):")
    print(f"    1. Load and ensure channel-first")
    print(f"    2. Pad to {INTERMEDIATE_SIZE[0]}x{INTERMEDIATE_SIZE[1]} (centered)")
    print(f"    3. Resize to {TARGET_SIZE[0]}x{TARGET_SIZE[1]} with bilinear interpolation")
    print(f"  Segmentation masks:")
    print(f"    1. Load and ensure channel-first")
    print(f"    2. Pad to {INTERMEDIATE_SIZE[0]}x{INTERMEDIATE_SIZE[1]} (preserves aspect ratio)")
    print(f"    3. Resize to {TARGET_SIZE[0]}x{TARGET_SIZE[1]} with nearest-neighbor")
    print(f"    4. Binarize to {{0, 1}} (threshold=0.5)")
    print(f"\nOutput format:       Float32, compressed .nii.gz")
    print(f"{'='*60}")

    # Count patients
    total_patients = 0
    for split in ['test', 'train']:
        split_dir = input_base / split
        if split_dir.exists():
            patient_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            print(f"{split.capitalize()}: {len(patient_dirs)} patients")
            total_patients += len(patient_dirs)

    print(f"Total: {total_patients} patients")
    print(f"{'='*60}")

    # Ask for confirmation if output directory doesn't exist
    if not output_base.exists():
        print(f"\n[INFO] Output directory will be created: {output_base}")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return 0
    else:
        response = input("\n[WARNING] Output directory already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return 0

    # Process dataset
    process_dataset(input_base, output_base, INTERMEDIATE_SIZE, TARGET_SIZE)

    return 0


if __name__ == "__main__":
    exit(main())
