#!/usr/bin/env python3
"""
Delete seg.nii.gz files from dataset test directory only.

Usage:
  # Dry run (show what would be deleted)
  python misc/delete_seg_files.py --dataset-path /home/mode/NTNU/MedicalDataSets/brainmetshare-3 --dry-run

  # Actually delete files
  python misc/delete_seg_files.py --dataset-path /home/mode/NTNU/MedicalDataSets/brainmetshare-3
"""

import argparse
import os
from pathlib import Path


def find_seg_files(dataset_path):
    """Find all seg.nii.gz files in test directory only."""
    dataset_path = Path(dataset_path)
    test_path = dataset_path / 'test'

    if not test_path.exists():
        print(f"Warning: Test directory not found: {test_path}")
        return []

    seg_files = list(test_path.rglob("*seg.nii.gz"))
    return sorted(seg_files)


def delete_files(files, dry_run=False):
    """Delete files with confirmation."""
    if not files:
        print("No seg.nii.gz files found.")
        return

    print(f"Found {len(files)} seg.nii.gz files:")
    for f in files:
        print(f"  {f}")

    if dry_run:
        print("\n[DRY RUN] No files were deleted.")
        return

    print(f"\nThis will delete {len(files)} files.")
    response = input("Continue? [y/N]: ").strip().lower()

    if response != 'y':
        print("Cancelled.")
        return

    deleted_count = 0
    failed_count = 0

    for f in files:
        try:
            f.unlink()
            deleted_count += 1
            print(f"  Deleted: {f}")
        except Exception as e:
            failed_count += 1
            print(f"  Failed to delete {f}: {e}")

    print(f"\nSummary: {deleted_count} deleted, {failed_count} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Delete seg.nii.gz files from dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be deleted without actually deleting')

    args = parser.parse_args()

    # Verify path exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return

    if not dataset_path.is_dir():
        print(f"Error: Dataset path is not a directory: {dataset_path}")
        return

    # Find and delete files
    seg_files = find_seg_files(dataset_path)
    delete_files(seg_files, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
