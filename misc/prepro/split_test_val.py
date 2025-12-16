"""
Split test dataset into validation and test sets.

Creates new val/ and test_new/ directories by copying from test/.
Original test/ directory is kept unchanged.

Usage:
    # Dry run (shows what would be done)
    python misc/prepro/split_test_val.py --data_dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3 --dry_run

    # Actually create the split
    python misc/prepro/split_test_val.py --data_dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3

    # Custom split ratio (default 0.5)
    python misc/prepro/split_test_val.py --data_dir /path/to/data --val_ratio 0.4

    # Custom random seed
    python misc/prepro/split_test_val.py --data_dir /path/to/data --seed 123
"""
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def get_patient_dirs(test_dir: Path) -> List[Path]:
    """Get list of patient directories in test folder."""
    patient_dirs = []
    for item in sorted(test_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            patient_dirs.append(item)
    return patient_dirs


def split_patients(
    patients: List[Path],
    val_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """Split patients into validation and test sets.

    Args:
        patients: List of patient directory paths
        val_ratio: Fraction of patients for validation (default 0.5)
        seed: Random seed for reproducibility

    Returns:
        (val_patients, test_patients) tuple
    """
    random.seed(seed)
    patients_shuffled = patients.copy()
    random.shuffle(patients_shuffled)

    n_val = int(len(patients_shuffled) * val_ratio)
    val_patients = patients_shuffled[:n_val]
    test_patients = patients_shuffled[n_val:]

    return sorted(val_patients), sorted(test_patients)


def copy_patient(src_dir: Path, dst_dir: Path, dry_run: bool = True) -> None:
    """Copy a patient directory to destination.

    Args:
        src_dir: Source patient directory
        dst_dir: Destination directory (will create patient folder inside)
        dry_run: If True, don't actually copy
    """
    dst_patient = dst_dir / src_dir.name

    if not dry_run:
        shutil.copytree(src_dir, dst_patient)


def main():
    parser = argparse.ArgumentParser(
        description='Split test dataset into validation and test sets'
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to data directory containing train/test folders'
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.5,
        help='Fraction of test patients to use for validation (default: 0.5)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Show what would be done without copying files'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_dir = data_dir / 'test'
    val_dir = data_dir / 'val'
    test_new_dir = data_dir / 'test_new'

    # Validate paths
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return

    if val_dir.exists():
        print(f"Error: val/ directory already exists: {val_dir}")
        print("Please remove it first if you want to re-run the split.")
        return

    if test_new_dir.exists():
        print(f"Error: test_new/ directory already exists: {test_new_dir}")
        print("Please remove it first if you want to re-run the split.")
        return

    # Get patients and split
    patients = get_patient_dirs(test_dir)
    val_patients, test_patients = split_patients(patients, args.val_ratio, args.seed)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Splitting test dataset")
    print(f"Data directory: {data_dir}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Random seed: {args.seed}")
    print()
    print(f"Original test patients: {len(patients)}")
    print(f"  -> val/: {len(val_patients)} patients")
    print(f"  -> test_new/: {len(test_patients)} patients")
    print()

    # Show which patients go where
    print("Validation patients:")
    for p in val_patients:
        print(f"  {p.name}")
    print()
    print("Test patients (test_new):")
    for p in test_patients:
        print(f"  {p.name}")
    print()

    if args.dry_run:
        print("[DRY RUN] No files were copied. Run without --dry_run to apply.")
        return

    # Create directories
    print("Creating directories...")
    val_dir.mkdir(parents=True, exist_ok=True)
    test_new_dir.mkdir(parents=True, exist_ok=True)

    # Copy validation patients
    print(f"Copying {len(val_patients)} patients to val/...")
    for patient in val_patients:
        copy_patient(patient, val_dir, dry_run=False)
        print(f"  Copied {patient.name}")

    # Copy test patients
    print(f"Copying {len(test_patients)} patients to test_new/...")
    for patient in test_patients:
        copy_patient(patient, test_new_dir, dry_run=False)
        print(f"  Copied {patient.name}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Created: {val_dir} ({len(val_patients)} patients)")
    print(f"Created: {test_new_dir} ({len(test_patients)} patients)")
    print(f"Original test/ directory unchanged")
    print()
    print("Next steps:")
    print("  1. Verify the split is correct")
    print("  2. Update your code to use val/ for validation")
    print("  3. Optionally: rm -rf test/ && mv test_new/ test/")


if __name__ == "__main__":
    main()
