"""
Trim empty leading slices from NIfTI volumes.

Uses bravo modality as reference to find the first slice with data,
then removes all leading empty slices from all modalities for each patient.

Usage:
    # Dry run (shows what would be done)
    python misc/trim_empty_slices.py --data_dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3 --dry_run

    # Actually trim slices (overwrites files!)
    python misc/trim_empty_slices.py --data_dir /path/to/data

    # Trim trailing slices too
    python misc/trim_empty_slices.py --data_dir /path/to/data --trim_trailing

    # Custom threshold for "empty" detection
    python misc/trim_empty_slices.py --data_dir /path/to/data --threshold 0.01
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm


def find_first_data_slice(volume: np.ndarray, threshold: float = 0.01) -> int:
    """Find the index of the first slice with data.

    Args:
        volume: 3D numpy array [H, W, D] or [D, H, W]
        threshold: Minimum fraction of non-zero pixels to consider slice as having data

    Returns:
        Index of first slice with data (along last axis)
    """
    # Assume slices are along the last axis (standard NIfTI convention)
    num_slices = volume.shape[-1]

    for i in range(num_slices):
        slice_data = volume[..., i]
        # Check if slice has meaningful data
        non_zero_fraction = np.count_nonzero(slice_data) / slice_data.size
        if non_zero_fraction > threshold:
            return i

    # No data found, return 0 (don't trim anything)
    return 0


def find_last_data_slice(volume: np.ndarray, threshold: float = 0.01) -> int:
    """Find the index of the last slice with data.

    Args:
        volume: 3D numpy array
        threshold: Minimum fraction of non-zero pixels to consider slice as having data

    Returns:
        Index of last slice with data (along last axis)
    """
    num_slices = volume.shape[-1]

    for i in range(num_slices - 1, -1, -1):
        slice_data = volume[..., i]
        non_zero_fraction = np.count_nonzero(slice_data) / slice_data.size
        if non_zero_fraction > threshold:
            return i

    # No data found, return last index
    return num_slices - 1


def get_patient_dirs(data_dir: Path) -> List[Path]:
    """Get list of patient directories.

    Handles both flat structure and train/test split structure:
        data_dir/Mets_001/bravo.nii.gz  (flat)
        data_dir/train/Mets_001/bravo.nii.gz  (split)
    """
    patient_dirs = []

    for item in sorted(data_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            # Check if this is a split directory (train/test)
            if item.name in ['train', 'test', 'val', 'validation']:
                # Look for patient dirs inside
                for patient in sorted(item.iterdir()):
                    if patient.is_dir() and not patient.name.startswith('.'):
                        patient_dirs.append(patient)
            else:
                # Check if it's a patient directory (has modality files)
                if any((item / f"{mod}.nii.gz").exists() or (item / f"{mod}.nii").exists()
                       for mod in ['bravo', 't1_pre', 't1_gd', 'seg']):
                    patient_dirs.append(item)

    return patient_dirs


def get_modality_files(patient_dir: Path) -> Dict[str, Path]:
    """Get paths to all modality files for a patient.

    Expected structure:
        patient_dir/
            bravo.nii.gz
            t1_pre.nii.gz
            t1_gd.nii.gz
            flair.nii.gz
            seg.nii.gz
    """
    modalities = ['bravo', 't1_pre', 't1_gd', 'flair', 'seg']
    files = {}

    for mod in modalities:
        # Try different naming conventions
        for ext in ['.nii.gz', '.nii']:
            path = patient_dir / f"{mod}{ext}"
            if path.exists():
                files[mod] = path
                break

    return files


def trim_volume(
    nifti_path: Path,
    start_slice: int,
    end_slice: Optional[int] = None,
    dry_run: bool = True
) -> Tuple[int, int]:
    """Trim slices from a NIfTI volume.

    Args:
        nifti_path: Path to NIfTI file
        start_slice: First slice to keep
        end_slice: Last slice to keep (None = keep all remaining)
        dry_run: If True, don't actually modify files

    Returns:
        (original_slices, new_slices) tuple
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    original_slices = data.shape[-1]

    # Determine slice range
    if end_slice is None:
        end_slice = original_slices - 1

    # Trim the data
    trimmed_data = data[..., start_slice:end_slice + 1]
    new_slices = trimmed_data.shape[-1]

    if not dry_run and start_slice > 0 or (end_slice is not None and end_slice < original_slices - 1):
        # Update affine to account for removed slices
        # The origin needs to shift by start_slice * voxel_size in the slice direction
        affine = img.affine.copy()

        # Get voxel size in slice direction (usually z)
        voxel_size = np.abs(affine[2, 2])  # Assuming axial slices

        # Shift origin
        affine[2, 3] += start_slice * voxel_size

        # Create new NIfTI image
        new_img = nib.Nifti1Image(trimmed_data.astype(data.dtype), affine, img.header)

        # Save (overwrite original)
        nib.save(new_img, nifti_path)

    return original_slices, new_slices


def process_patient(
    patient_dir: Path,
    threshold: float = 0.01,
    trim_trailing: bool = False,
    dry_run: bool = True
) -> Dict:
    """Process a single patient directory.

    Args:
        patient_dir: Path to patient directory
        threshold: Threshold for empty slice detection
        trim_trailing: Also trim empty trailing slices
        dry_run: If True, don't modify files

    Returns:
        Dict with processing results
    """
    result = {
        'patient': patient_dir.name,
        'status': 'success',
        'slices_removed_start': 0,
        'slices_removed_end': 0,
        'original_slices': 0,
        'new_slices': 0,
        'modalities_processed': []
    }

    # Get modality files
    mod_files = get_modality_files(patient_dir)

    if 'bravo' not in mod_files:
        result['status'] = 'error: no bravo file found'
        return result

    # Load bravo to find slice range
    bravo_img = nib.load(mod_files['bravo'])
    bravo_data = bravo_img.get_fdata()

    result['original_slices'] = bravo_data.shape[-1]

    # Find first data slice
    start_slice = find_first_data_slice(bravo_data, threshold)
    result['slices_removed_start'] = start_slice

    # Find last data slice (optional)
    if trim_trailing:
        end_slice = find_last_data_slice(bravo_data, threshold)
        result['slices_removed_end'] = result['original_slices'] - 1 - end_slice
    else:
        end_slice = None

    # Calculate new slice count
    if end_slice is not None:
        result['new_slices'] = end_slice - start_slice + 1
    else:
        result['new_slices'] = result['original_slices'] - start_slice

    # Skip if nothing to trim
    if start_slice == 0 and (end_slice is None or end_slice == result['original_slices'] - 1):
        result['status'] = 'skipped: no empty slices'
        return result

    # Process all modalities
    for mod_name, mod_path in mod_files.items():
        try:
            orig, new = trim_volume(mod_path, start_slice, end_slice, dry_run)
            result['modalities_processed'].append(mod_name)
        except Exception as e:
            result['status'] = f'error processing {mod_name}: {str(e)}'
            return result

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Trim empty leading/trailing slices from NIfTI volumes'
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to data directory containing patient folders'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.01,
        help='Fraction of non-zero pixels to consider slice as having data (default: 0.01)'
    )
    parser.add_argument(
        '--trim_trailing', action='store_true',
        help='Also trim empty trailing slices'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Show what would be done without modifying files'
    )
    parser.add_argument(
        '--patient', type=str, default=None,
        help='Process only this patient (for testing)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    # Get patient directories
    patient_dirs = get_patient_dirs(data_dir)

    if args.patient:
        patient_dirs = [p for p in patient_dirs if p.name == args.patient]
        if not patient_dirs:
            print(f"Error: Patient not found: {args.patient}")
            return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(patient_dirs)} patients...")
    print(f"Threshold: {args.threshold}")
    print(f"Trim trailing: {args.trim_trailing}")
    print()

    # Process patients
    results = []
    total_removed = 0

    for patient_dir in tqdm(patient_dirs, desc="Processing"):
        result = process_patient(
            patient_dir,
            threshold=args.threshold,
            trim_trailing=args.trim_trailing,
            dry_run=args.dry_run
        )
        results.append(result)
        total_removed += result['slices_removed_start'] + result['slices_removed_end']

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if 'skipped' in r['status'])
    error_count = sum(1 for r in results if 'error' in r['status'])

    print(f"Patients processed: {len(results)}")
    print(f"  - Modified: {success_count}")
    print(f"  - Skipped (no empty slices): {skipped_count}")
    print(f"  - Errors: {error_count}")
    print(f"Total slices removed: {total_removed}")

    # Print details for modified patients
    if success_count > 0:
        print("\nModified patients:")
        for r in results:
            if r['status'] == 'success':
                print(f"  {r['patient']}: {r['original_slices']} -> {r['new_slices']} slices "
                      f"(removed {r['slices_removed_start']} start, {r['slices_removed_end']} end)")

    # Print errors
    if error_count > 0:
        print("\nErrors:")
        for r in results:
            if 'error' in r['status']:
                print(f"  {r['patient']}: {r['status']}")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified. Run without --dry_run to apply changes.")


if __name__ == "__main__":
    main()
