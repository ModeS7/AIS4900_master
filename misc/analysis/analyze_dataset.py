#!/usr/bin/env python3
"""
Dataset Statistics Analysis for BrainMetShare.

Analyzes brainmetshare-3_non_public dataset to extract detailed statistics
including slice counts, brain tissue coverage, and metastasis distribution.
"""
import os
import sys
from pathlib import Path

import numpy as np
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, ToTensor

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import PathConfig


def make_binary(image: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Convert image to binary using threshold.

    Args:
        image: Input image array.
        threshold: Threshold value for binarization.

    Returns:
        Binary image array with values 0.0 or 1.0.
    """
    return np.where(image > threshold, 1.0, 0.0)


def analyze_dataset_split(data_dir: str, split_name: str) -> None:
    """Analyze one split (train or test) of the dataset.

    Args:
        data_dir: Path to train or test directory.
        split_name: 'Train' or 'Test' for printing.
    """
    print(f"\n{'='*60}")
    print(f"=== {split_name} Set Analysis ===")
    print(f"{'='*60}")

    # Setup MONAI transforms (same as training code)
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
    ])

    # Get list of patient directories
    patients = sorted(os.listdir(data_dir))
    num_patients = len(patients)
    print(f"Number of patients: {num_patients}")

    # Statistics storage
    slices_per_patient = []
    total_slices = 0
    total_brain_slices = 0
    total_positive_slices = 0
    total_negative_slices = 0

    # Process each patient
    for patient_id in patients:
        patient_path = os.path.join(data_dir, patient_id)

        # Load brain image (Bravo) and segmentation mask
        bravo_path = os.path.join(patient_path, "bravo.nii.gz")
        seg_path = os.path.join(patient_path, "seg.nii.gz")

        if not os.path.exists(bravo_path):
            print(f"Warning: No bravo image for {patient_id}")
            continue
        if not os.path.exists(seg_path):
            print(f"Warning: No seg mask for {patient_id}")
            continue

        bravo_volume = transform(bravo_path)  # Shape: [C, H, W, D]
        seg_volume = transform(seg_path)      # Shape: [C, H, W, D]

        # Get number of slices (depth dimension)
        num_slices = bravo_volume.shape[3]
        slices_per_patient.append(num_slices)
        total_slices += num_slices

        # Analyze each slice
        for slice_idx in range(num_slices):
            brain_slice = bravo_volume[0, :, :, slice_idx].numpy()
            mask_slice = seg_volume[0, :, :, slice_idx].numpy()

            # First check if slice has brain tissue
            if np.sum(brain_slice) <= 1.0:
                continue  # Skip slices with no brain tissue

            total_brain_slices += 1

            # Then check if slice has metastases
            binary_mask = make_binary(mask_slice, threshold=0.01)
            if np.sum(binary_mask) > 0:
                total_positive_slices += 1
            else:
                total_negative_slices += 1

    # Print statistics
    slices_array = np.array(slices_per_patient)
    print("\nSlice Statistics:")
    print(f"  Total slices: {total_slices}")
    print(f"  Slices per patient: min={np.min(slices_array)}, max={np.max(slices_array)}, mean={np.mean(slices_array):.1f}")

    print("\nSlice Distribution:")
    print(f"  Slices with brain tissue: {total_brain_slices} ({100*total_brain_slices/total_slices:.1f}%)")
    print(f"  Positive slices (with metastases): {total_positive_slices} ({100*total_positive_slices/total_brain_slices:.1f}% of brain slices)")
    print(f"  Negative slices (no metastases): {total_negative_slices} ({100*total_negative_slices/total_brain_slices:.1f}% of brain slices)")


def main() -> None:
    """Main entry point for dataset analysis."""
    # Set base directory using PathConfig
    path_config = PathConfig()
    base_dir = str(path_config.data_dir / "brainmetshare-3_non_public")

    print("="*60)
    print("BrainMetShare Dataset Statistics Analysis")
    print("="*60)
    print(f"Dataset directory: {base_dir}")

    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"\nERROR: Directory not found: {base_dir}")
        print("Please verify the dataset path")
        return

    # Analyze train set
    train_dir = os.path.join(base_dir, "train")
    if os.path.exists(train_dir):
        analyze_dataset_split(train_dir, "Train")
    else:
        print(f"Warning: Train directory not found at {train_dir}")

    # Analyze test set
    test_dir = os.path.join(base_dir, "test")
    if os.path.exists(test_dir):
        analyze_dataset_split(test_dir, "Test")
    else:
        print(f"Warning: Test directory not found at {test_dir}")

    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)


if __name__ == "__main__":
    main()
