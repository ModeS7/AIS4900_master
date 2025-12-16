#!/usr/bin/env python3
"""
Interactive script to trim slices from the start of NIfTI volumes.

Goes through each patient in the test directory and asks how many slices
to remove from the beginning of each volume.

Usage:
    python misc/prepro/trim_slices.py
    python misc/prepro/trim_slices.py --data_dir /path/to/data/test
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

# Default path
DEFAULT_DATA_DIR = "/home/mode/NTNU/MedicalDataSets/brainmetshare-3_non_public_256/test"


def get_volume_info(nii_path: Path) -> tuple:
    """Get shape and slice count from NIfTI file."""
    nii = nib.load(nii_path)
    shape = nii.shape
    n_slices = shape[2] if len(shape) >= 3 else shape[0]
    return shape, n_slices


def trim_volume(nii_path: Path, remove_start: int, remove_end: int = 0) -> None:
    """Trim slices from start (and optionally end) of NIfTI volume.

    Args:
        nii_path: Path to NIfTI file.
        remove_start: Number of slices to remove from start.
        remove_end: Number of slices to remove from end (default: 0).
    """
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    affine = nii.affine

    # Trim slices (assuming slices are in the 3rd dimension)
    if remove_end > 0:
        data = data[:, :, remove_start:-remove_end]
    else:
        data = data[:, :, remove_start:]

    # Save back
    new_nii = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(new_nii, nii_path)


def process_patient(patient_dir: Path, remove_start: int, remove_end: int = 0) -> None:
    """Process all modalities for a patient."""
    nii_files = sorted(patient_dir.glob("*.nii.gz"))

    for nii_file in nii_files:
        modality = nii_file.stem.replace('.nii', '')
        old_shape, _ = get_volume_info(nii_file)

        trim_volume(nii_file, remove_start, remove_end)

        new_shape, _ = get_volume_info(nii_file)
        print(f"    {modality}: {old_shape} -> {new_shape}")


def main():
    parser = argparse.ArgumentParser(description="Interactive slice trimming tool")
    parser.add_argument(
        "--data_dir", "-d",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help=f"Test data directory (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--dry_run", "-n",
        action="store_true",
        help="Show what would be done without making changes"
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return 1

    # Get all patient directories
    patients = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    print(f"\n{'='*60}")
    print(f"INTERACTIVE SLICE TRIMMING")
    print(f"{'='*60}")
    print(f"Directory: {data_dir}")
    print(f"Patients:  {len(patients)}")
    print(f"{'='*60}")
    print("\nFor each patient, enter slices to remove from START.")
    print("Press Enter for 0 (no change), or 'q' to quit.\n")

    changes = {}

    for patient_dir in patients:
        # Get info from first modality
        nii_files = sorted(patient_dir.glob("*.nii.gz"))
        if not nii_files:
            print(f"[SKIP] {patient_dir.name}: No NIfTI files found")
            continue

        sample_file = nii_files[0]
        shape, n_slices = get_volume_info(sample_file)

        # Show patient info and ask for input
        print(f"{patient_dir.name} ({n_slices} slices, shape {shape})")

        try:
            user_input = input(f"  Remove from start [0]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nAborted by user.")
            return 0

        if user_input.lower() == 'q':
            print("\nQuitting...")
            break

        remove_start = int(user_input) if user_input else 0

        if remove_start > 0:
            changes[patient_dir] = remove_start
            if not args.dry_run:
                process_patient(patient_dir, remove_start)
                print(f"  [TRIMMED] Removed first {remove_start} slices")
            else:
                print(f"  [DRY RUN] Would remove first {remove_start} slices")
        else:
            print(f"  [SKIP] No changes")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Patients modified: {len(changes)}")
    for patient_dir, remove_start in changes.items():
        print(f"  {patient_dir.name}: removed {remove_start} slices from start")

    return 0


if __name__ == "__main__":
    exit(main())
