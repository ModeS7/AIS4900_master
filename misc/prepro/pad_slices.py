"""
Pad NIfTI volumes to a target number of slices.

Uses bravo modality as reference, pads all modalities equally.
Padding is added at the end of the volume (after the last slice).

Usage:
    # Dry run (shows what would be done)
    python misc/prepro/pad_slices.py --data_dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3 --target_slices 150 --dry_run

    # Actually pad slices (overwrites files!)
    python misc/prepro/pad_slices.py --data_dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3 --target_slices 150

    # Pad at the beginning instead of end
    python misc/prepro/pad_slices.py --data_dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3 --target_slices 150 --pad_start
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm


def get_patient_dirs(data_dir: Path) -> List[Path]:
    """Get list of patient directories.

    Handles both flat structure and train/test split structure.
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
    """Get paths to all modality files for a patient."""
    modalities = ['bravo', 't1_pre', 't1_gd', 'flair', 'seg']
    files = {}

    for mod in modalities:
        for ext in ['.nii.gz', '.nii']:
            path = patient_dir / f"{mod}{ext}"
            if path.exists():
                files[mod] = path
                break

    return files


def pad_volume(
    nifti_path: Path,
    target_slices: int,
    pad_start: bool = False,
    dry_run: bool = True
) -> Tuple[int, int]:
    """Pad a NIfTI volume to target number of slices.

    Args:
        nifti_path: Path to NIfTI file
        target_slices: Target number of slices
        pad_start: If True, pad at start; if False, pad at end
        dry_run: If True, don't actually modify files

    Returns:
        (original_slices, new_slices) tuple
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    original_slices = data.shape[-1]

    if original_slices >= target_slices:
        # No padding needed
        return original_slices, original_slices

    slices_to_add = target_slices - original_slices

    # Create padding array (zeros)
    pad_shape = list(data.shape)
    pad_shape[-1] = slices_to_add
    padding = np.zeros(pad_shape, dtype=data.dtype)

    # Add padding
    if pad_start:
        padded_data = np.concatenate([padding, data], axis=-1)
    else:
        padded_data = np.concatenate([data, padding], axis=-1)

    new_slices = padded_data.shape[-1]

    if not dry_run:
        # Update affine if padding at start
        affine = img.affine.copy()
        if pad_start:
            # Shift origin back by the padding amount
            voxel_size = np.abs(affine[2, 2])
            affine[2, 3] -= slices_to_add * voxel_size

        # Create new NIfTI image
        new_img = nib.Nifti1Image(padded_data.astype(data.dtype), affine, img.header)

        # Update header for new dimensions
        new_img.header.set_data_shape(padded_data.shape)

        # Save (overwrite original)
        nib.save(new_img, nifti_path)

    return original_slices, new_slices


def process_patient(
    patient_dir: Path,
    target_slices: int,
    pad_start: bool = False,
    dry_run: bool = True
) -> Dict:
    """Process a single patient directory.

    Args:
        patient_dir: Path to patient directory
        target_slices: Target number of slices
        pad_start: If True, pad at start
        dry_run: If True, don't modify files

    Returns:
        Dict with processing results
    """
    result = {
        'patient': patient_dir.name,
        'status': 'success',
        'original_slices': 0,
        'new_slices': 0,
        'slices_added': 0,
        'modalities_processed': []
    }

    # Get modality files
    mod_files = get_modality_files(patient_dir)

    if 'bravo' not in mod_files:
        result['status'] = 'error: no bravo file found'
        return result

    # Load bravo to check current slice count
    bravo_img = nib.load(mod_files['bravo'])
    bravo_data = bravo_img.get_fdata()
    original_slices = bravo_data.shape[-1]

    result['original_slices'] = original_slices

    if original_slices >= target_slices:
        result['status'] = f'skipped: already has {original_slices} slices (>= {target_slices})'
        result['new_slices'] = original_slices
        return result

    result['slices_added'] = target_slices - original_slices
    result['new_slices'] = target_slices

    # Process all modalities
    for mod_name, mod_path in mod_files.items():
        try:
            orig, new = pad_volume(mod_path, target_slices, pad_start, dry_run)
            result['modalities_processed'].append(mod_name)
        except Exception as e:
            result['status'] = f'error processing {mod_name}: {str(e)}'
            return result

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Pad NIfTI volumes to a target number of slices'
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to data directory containing patient folders'
    )
    parser.add_argument(
        '--target_slices', type=int, required=True,
        help='Target number of slices (e.g., 150)'
    )
    parser.add_argument(
        '--pad_start', action='store_true',
        help='Pad at the start instead of end'
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
    print(f"Target slices: {args.target_slices}")
    print(f"Pad position: {'start' if args.pad_start else 'end'}")
    print()

    # Process patients
    results = []
    total_added = 0

    for patient_dir in tqdm(patient_dirs, desc="Processing"):
        result = process_patient(
            patient_dir,
            target_slices=args.target_slices,
            pad_start=args.pad_start,
            dry_run=args.dry_run
        )
        results.append(result)
        total_added += result['slices_added']

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if 'skipped' in r['status'])
    error_count = sum(1 for r in results if 'error' in r['status'])

    print(f"Patients processed: {len(results)}")
    print(f"  - Modified: {success_count}")
    print(f"  - Skipped (already >= target): {skipped_count}")
    print(f"  - Errors: {error_count}")
    print(f"Total slices added: {total_added}")

    # Print details for modified patients
    if success_count > 0:
        print("\nModified patients:")
        for r in results:
            if r['status'] == 'success':
                print(f"  {r['patient']}: {r['original_slices']} -> {r['new_slices']} slices "
                      f"(added {r['slices_added']})")

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
