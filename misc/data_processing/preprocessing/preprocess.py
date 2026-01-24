#!/usr/bin/env python3
"""
Unified preprocessing tool for NIfTI medical images.

Subcommands:
    resize      - Resize images to 256x256 (pad to 240 centered, then resize)
    align       - Align all modalities to same slice count (truncate/pad)
    pad         - Pad volumes to target slice count (add zeros)
    trim-auto   - Auto-detect and trim empty leading/trailing slices
    trim-manual - Interactive per-patient slice trimming
    split       - Split test into val and test_new

Usage:
    python misc/preprocessing/preprocess.py <command> [options]
    python misc/preprocessing/preprocess.py <command> --help

Examples:
    # Align slices (ensures all modalities per patient have same slice count)
    python misc/preprocessing/preprocess.py align --data_dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3 -t 150 --dry_run
    python misc/preprocessing/preprocess.py align --data_dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3 -t 150
    python misc/preprocessing/preprocess.py align --data_dir /path/to/data -t 150 --patient Mets_001

    # Pad volumes to target slice count (add zeros at end)
    python misc/preprocessing/preprocess.py pad --data_dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3 -t 150 --dry_run
    python misc/preprocessing/preprocess.py pad --data_dir /path/to/data -t 150 --pad_start

    # Auto-trim empty slices (auto-detect empty leading/trailing slices)
    python misc/preprocessing/preprocess.py trim-auto --data_dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3 --dry_run
    python misc/preprocessing/preprocess.py trim-auto --data_dir /path/to/data --trim_trailing --threshold 0.01

    # Manual trimming (interactively specify slices to remove per patient)
    python misc/preprocessing/preprocess.py trim-manual --data_dir /path/to/data/test
    python misc/preprocessing/preprocess.py trim-manual -d /path/to/data/val --dry_run

    # Split test set into validation and test_new
    python misc/preprocessing/preprocess.py split --data_dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3 --dry_run
    python misc/preprocessing/preprocess.py split --data_dir /path/to/data --val_ratio 0.4 --seed 123

    # Resize images (pad to 240x240 centered, resize to 256x256)
    python misc/preprocessing/preprocess.py resize -i /path/to/raw -o /path/to/processed --dry_run
    python misc/preprocessing/preprocess.py resize -i /home/mode/NTNU/MedicalDataSets/StanfordSkullStripped -o /home/mode/NTNU/MedicalDataSets/brainmetshare-3_256
"""
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm

# Optional imports for process command
try:
    import torch
    from monai.transforms import (
        Compose,
        EnsureChannelFirst,
        LoadImage,
        Resize,
        SpatialPad,
    )
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False


# =============================================================================
# COMMON UTILITIES
# =============================================================================

SPLIT_DIRS = ['train', 'test', 'test_new', 'val', 'validation']


def get_patient_dirs(data_dir: Path) -> List[Path]:
    """Get list of patient directories (handles flat and split structures)."""
    patient_dirs = []

    for item in sorted(data_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            if item.name in SPLIT_DIRS:
                for patient in sorted(item.iterdir()):
                    if patient.is_dir() and not patient.name.startswith('.'):
                        patient_dirs.append(patient)
            else:
                if has_nifti_files(item):
                    patient_dirs.append(item)

    return patient_dirs


def has_nifti_files(directory: Path) -> bool:
    """Check if directory contains NIfTI files."""
    return any(directory.glob("*.nii.gz")) or any(directory.glob("*.nii"))


def get_nifti_files(patient_dir: Path) -> Dict[str, Path]:
    """Get all NIfTI files for a patient."""
    files = {}
    for nii_path in sorted(patient_dir.glob("*.nii.gz")):
        modality = nii_path.stem.replace('.nii', '')
        files[modality] = nii_path
    for nii_path in sorted(patient_dir.glob("*.nii")):
        modality = nii_path.stem
        if modality not in files:
            files[modality] = nii_path
    return files


def get_slice_count(nifti_path: Path) -> int:
    """Get slice count from a NIfTI file."""
    img = nib.load(nifti_path)
    return img.shape[2]


def get_volume_info(nifti_path: Path) -> Tuple[Tuple, int]:
    """Get shape and slice count from NIfTI file."""
    img = nib.load(nifti_path)
    shape = img.shape
    n_slices = shape[2] if len(shape) >= 3 else shape[0]
    return shape, n_slices


def print_header(title: str, dry_run: bool = False) -> None:
    """Print section header."""
    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{'='*60}")
    print(f"{prefix}{title}")
    print(f"{'='*60}")


def print_summary(results: List[Dict], dry_run: bool = False) -> None:
    """Print processing summary."""
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if 'skipped' in r['status'])
    error_count = sum(1 for r in results if 'error' in r['status'])

    print_header("SUMMARY", dry_run)
    print(f"Patients processed: {len(results)}")
    print(f"  - Modified: {success_count}")
    print(f"  - Skipped: {skipped_count}")
    print(f"  - Errors: {error_count}")

    if error_count > 0:
        print("\nErrors:")
        for r in results:
            if 'error' in r['status']:
                print(f"  {r.get('patient', 'unknown')}: {r['status']}")

    if dry_run:
        print("\n[DRY RUN] No files were modified. Run without --dry_run to apply changes.")


# =============================================================================
# ALIGN COMMAND
# =============================================================================

def normalize_volume(nifti_path: Path, target_slices: int, dry_run: bool = True) -> Tuple[str, int, int]:
    """Normalize a single NIfTI volume to target slice count."""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    original_slices = data.shape[2]

    if original_slices == target_slices:
        return ('unchanged', original_slices, target_slices)

    if original_slices > target_slices:
        new_data = data[:, :, :target_slices]
        action = 'truncated'
    else:
        pad_amount = target_slices - original_slices
        new_data = np.pad(data, ((0, 0), (0, 0), (0, pad_amount)), mode='constant', constant_values=0)
        action = 'padded'

    if not dry_run:
        new_img = nib.Nifti1Image(new_data.astype(data.dtype), img.affine, img.header)
        nib.save(new_img, nifti_path)

    return (action, original_slices, target_slices)


def process_patient_normalize(patient_dir: Path, target_slices: int, dry_run: bool = True) -> Dict:
    """Normalize all modalities for a patient to same slice count."""
    result = {
        'patient': patient_dir.name,
        'status': 'success',
        'original_slices': {},
        'effective_target': 0,
        'changes': [],
    }

    nii_files = get_nifti_files(patient_dir)
    if not nii_files:
        result['status'] = 'error: no NIfTI files found'
        return result

    # First pass: get slice counts
    for modality, nii_path in nii_files.items():
        try:
            result['original_slices'][modality] = get_slice_count(nii_path)
        except Exception as e:
            result['status'] = f'error reading {modality}: {e}'
            return result

    # Find minimum and effective target
    min_slices = min(result['original_slices'].values())
    effective_target = min(min_slices, target_slices)
    result['effective_target'] = effective_target

    # Second pass: normalize all modalities
    for modality, nii_path in nii_files.items():
        try:
            action, orig, new = normalize_volume(nii_path, effective_target, dry_run)
            if action != 'unchanged':
                result['changes'].append(f"{modality}: {orig} -> {new} ({action})")
        except Exception as e:
            result['status'] = f'error processing {modality}: {e}'
            return result

    if not result['changes']:
        result['status'] = 'skipped: all modalities already match'

    return result


def cmd_align(args):
    """Align command handler."""
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    patient_dirs = get_patient_dirs(data_dir)
    if args.patient:
        patient_dirs = [p for p in patient_dirs if p.name == args.patient]
        if not patient_dirs:
            print(f"Error: Patient not found: {args.patient}")
            return

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Processing {len(patient_dirs)} patients...")
    print(f"Target slices: {args.target_slices}")
    print("-" * 60)

    results = []
    for patient_dir in tqdm(patient_dirs, desc="Processing"):
        result = process_patient_normalize(patient_dir, args.target_slices, args.dry_run)
        results.append(result)

    # Print changes
    print_header("CHANGES", args.dry_run)
    for r in results:
        if r['changes']:
            slice_info = ", ".join(f"{m}={s}" for m, s in r['original_slices'].items())
            print(f"\n{r['patient']} (original: {slice_info}, target: {r['effective_target']}):")
            for change in r['changes']:
                print(f"  {change}")

    print_summary(results, args.dry_run)


# =============================================================================
# PAD COMMAND
# =============================================================================

def pad_volume(nifti_path: Path, target_slices: int, pad_start: bool = False, dry_run: bool = True) -> Tuple[int, int]:
    """Pad a NIfTI volume to target number of slices."""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    original_slices = data.shape[-1]

    if original_slices >= target_slices:
        return original_slices, original_slices

    slices_to_add = target_slices - original_slices
    pad_shape = list(data.shape)
    pad_shape[-1] = slices_to_add
    padding = np.zeros(pad_shape, dtype=data.dtype)

    if pad_start:
        padded_data = np.concatenate([padding, data], axis=-1)
    else:
        padded_data = np.concatenate([data, padding], axis=-1)

    if not dry_run:
        affine = img.affine.copy()
        if pad_start:
            voxel_size = np.abs(affine[2, 2])
            affine[2, 3] -= slices_to_add * voxel_size
        new_img = nib.Nifti1Image(padded_data.astype(data.dtype), affine, img.header)
        nib.save(new_img, nifti_path)

    return original_slices, padded_data.shape[-1]


def process_patient_pad(patient_dir: Path, target_slices: int, pad_start: bool = False, dry_run: bool = True) -> Dict:
    """Pad all modalities for a patient."""
    result = {
        'patient': patient_dir.name,
        'status': 'success',
        'original_slices': 0,
        'new_slices': 0,
        'slices_added': 0,
        'modalities_processed': []
    }

    nii_files = get_nifti_files(patient_dir)
    if not nii_files:
        result['status'] = 'error: no NIfTI files found'
        return result

    # Use first file as reference
    first_file = list(nii_files.values())[0]
    original_slices = get_slice_count(first_file)
    result['original_slices'] = original_slices

    if original_slices >= target_slices:
        result['status'] = f'skipped: already has {original_slices} slices (>= {target_slices})'
        result['new_slices'] = original_slices
        return result

    result['slices_added'] = target_slices - original_slices
    result['new_slices'] = target_slices

    for mod_name, mod_path in nii_files.items():
        try:
            pad_volume(mod_path, target_slices, pad_start, dry_run)
            result['modalities_processed'].append(mod_name)
        except Exception as e:
            result['status'] = f'error processing {mod_name}: {e}'
            return result

    return result


def cmd_pad(args):
    """Pad command handler."""
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    patient_dirs = get_patient_dirs(data_dir)
    if args.patient:
        patient_dirs = [p for p in patient_dirs if p.name == args.patient]

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(patient_dirs)} patients...")
    print(f"Target slices: {args.target_slices}")
    print(f"Pad position: {'start' if args.pad_start else 'end'}")

    results = []
    for patient_dir in tqdm(patient_dirs, desc="Processing"):
        result = process_patient_pad(patient_dir, args.target_slices, args.pad_start, args.dry_run)
        results.append(result)

    # Print modified patients
    print_header("MODIFIED", args.dry_run)
    for r in results:
        if r['status'] == 'success':
            print(f"  {r['patient']}: {r['original_slices']} -> {r['new_slices']} slices (added {r['slices_added']})")

    print_summary(results, args.dry_run)


# =============================================================================
# TRIM-AUTO COMMAND
# =============================================================================

def find_first_data_slice(volume: np.ndarray, threshold: float = 0.01) -> int:
    """Find index of first slice with data."""
    for i in range(volume.shape[-1]):
        slice_data = volume[..., i]
        if np.count_nonzero(slice_data) / slice_data.size > threshold:
            return i
    return 0


def find_last_data_slice(volume: np.ndarray, threshold: float = 0.01) -> int:
    """Find index of last slice with data."""
    for i in range(volume.shape[-1] - 1, -1, -1):
        slice_data = volume[..., i]
        if np.count_nonzero(slice_data) / slice_data.size > threshold:
            return i
    return volume.shape[-1] - 1


def trim_volume(nifti_path: Path, start_slice: int, end_slice: Optional[int] = None, dry_run: bool = True) -> Tuple[int, int]:
    """Trim slices from a NIfTI volume."""
    img = nib.load(nifti_path)
    data = img.get_fdata()
    original_slices = data.shape[-1]

    if end_slice is None:
        end_slice = original_slices - 1

    trimmed_data = data[..., start_slice:end_slice + 1]

    if not dry_run and (start_slice > 0 or end_slice < original_slices - 1):
        affine = img.affine.copy()
        voxel_size = np.abs(affine[2, 2])
        affine[2, 3] += start_slice * voxel_size
        new_img = nib.Nifti1Image(trimmed_data.astype(data.dtype), affine, img.header)
        nib.save(new_img, nifti_path)

    return original_slices, trimmed_data.shape[-1]


def process_patient_trim(patient_dir: Path, threshold: float = 0.01, trim_trailing: bool = False, dry_run: bool = True) -> Dict:
    """Trim empty slices from all modalities for a patient."""
    result = {
        'patient': patient_dir.name,
        'status': 'success',
        'slices_removed_start': 0,
        'slices_removed_end': 0,
        'original_slices': 0,
        'new_slices': 0,
        'modalities_processed': []
    }

    nii_files = get_nifti_files(patient_dir)
    if not nii_files:
        result['status'] = 'error: no NIfTI files found'
        return result

    # Use first image modality as reference (prefer bravo)
    ref_path = nii_files.get('bravo') or list(nii_files.values())[0]
    img = nib.load(ref_path)
    data = img.get_fdata()
    result['original_slices'] = data.shape[-1]

    start_slice = find_first_data_slice(data, threshold)
    result['slices_removed_start'] = start_slice

    if trim_trailing:
        end_slice = find_last_data_slice(data, threshold)
        result['slices_removed_end'] = result['original_slices'] - 1 - end_slice
    else:
        end_slice = None

    if end_slice is not None:
        result['new_slices'] = end_slice - start_slice + 1
    else:
        result['new_slices'] = result['original_slices'] - start_slice

    if start_slice == 0 and (end_slice is None or end_slice == result['original_slices'] - 1):
        result['status'] = 'skipped: no empty slices'
        return result

    for mod_name, mod_path in nii_files.items():
        try:
            trim_volume(mod_path, start_slice, end_slice, dry_run)
            result['modalities_processed'].append(mod_name)
        except Exception as e:
            result['status'] = f'error processing {mod_name}: {e}'
            return result

    return result


def cmd_trim_auto(args):
    """Trim-auto command handler."""
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    patient_dirs = get_patient_dirs(data_dir)
    if args.patient:
        patient_dirs = [p for p in patient_dirs if p.name == args.patient]

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(patient_dirs)} patients...")
    print(f"Threshold: {args.threshold}")
    print(f"Trim trailing: {args.trim_trailing}")

    results = []
    for patient_dir in tqdm(patient_dirs, desc="Processing"):
        result = process_patient_trim(patient_dir, args.threshold, args.trim_trailing, args.dry_run)
        results.append(result)

    # Print modified patients
    print_header("MODIFIED", args.dry_run)
    for r in results:
        if r['status'] == 'success':
            print(f"  {r['patient']}: {r['original_slices']} -> {r['new_slices']} slices "
                  f"(removed {r['slices_removed_start']} start, {r['slices_removed_end']} end)")

    print_summary(results, args.dry_run)


# =============================================================================
# TRIM-MANUAL COMMAND
# =============================================================================

def trim_volume_simple(nifti_path: Path, remove_start: int, remove_end: int = 0) -> None:
    """Trim slices from start/end of NIfTI volume."""
    img = nib.load(nifti_path)
    data = img.get_fdata()

    if remove_end > 0:
        data = data[:, :, remove_start:-remove_end]
    else:
        data = data[:, :, remove_start:]

    new_img = nib.Nifti1Image(data.astype(np.float32), img.affine)
    nib.save(new_img, nifti_path)


def cmd_trim_manual(args):
    """Trim-manual command handler."""
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    patients = sorted([d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

    print_header("INTERACTIVE SLICE TRIMMING", args.dry_run)
    print(f"Directory: {data_dir}")
    print(f"Patients:  {len(patients)}")
    print("\nFor each patient, enter slices to remove from START.")
    print("Press Enter for 0 (no change), or 'q' to quit.\n")

    changes = {}

    for patient_dir in patients:
        nii_files = sorted(patient_dir.glob("*.nii.gz"))
        if not nii_files:
            print(f"[SKIP] {patient_dir.name}: No NIfTI files found")
            continue

        shape, n_slices = get_volume_info(nii_files[0])
        print(f"{patient_dir.name} ({n_slices} slices, shape {shape})")

        try:
            user_input = input("  Remove from start [0]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nAborted by user.")
            return

        if user_input.lower() == 'q':
            print("\nQuitting...")
            break

        remove_start = int(user_input) if user_input else 0

        if remove_start > 0:
            changes[patient_dir] = remove_start
            if not args.dry_run:
                for nii_file in nii_files:
                    old_shape, _ = get_volume_info(nii_file)
                    trim_volume_simple(nii_file, remove_start)
                    new_shape, _ = get_volume_info(nii_file)
                    modality = nii_file.stem.replace('.nii', '')
                    print(f"    {modality}: {old_shape} -> {new_shape}")
                print(f"  [TRIMMED] Removed first {remove_start} slices")
            else:
                print(f"  [DRY RUN] Would remove first {remove_start} slices")
        else:
            print("  [SKIP] No changes")

    print_header("SUMMARY", args.dry_run)
    print(f"Patients modified: {len(changes)}")
    for patient_dir, remove_start in changes.items():
        print(f"  {patient_dir.name}: removed {remove_start} slices from start")


# =============================================================================
# SPLIT COMMAND
# =============================================================================

def cmd_split(args):
    """Split command handler."""
    data_dir = Path(args.data_dir)
    test_dir = data_dir / 'test'
    val_dir = data_dir / 'val'
    test_new_dir = data_dir / 'test_new'

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return
    if val_dir.exists():
        print("Error: val/ directory already exists. Remove it first.")
        return
    if test_new_dir.exists():
        print("Error: test_new/ directory already exists. Remove it first.")
        return

    # Get and split patients
    patients = sorted([d for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

    random.seed(args.seed)
    patients_shuffled = patients.copy()
    random.shuffle(patients_shuffled)

    n_val = int(len(patients_shuffled) * args.val_ratio)
    val_patients = sorted(patients_shuffled[:n_val])
    test_patients = sorted(patients_shuffled[n_val:])

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Splitting test dataset")
    print(f"Validation ratio: {args.val_ratio}, Seed: {args.seed}")
    print(f"\nOriginal test patients: {len(patients)}")
    print(f"  -> val/: {len(val_patients)} patients")
    print(f"  -> test_new/: {len(test_patients)} patients\n")

    print("Validation patients:")
    for p in val_patients:
        print(f"  {p.name}")
    print("\nTest patients (test_new):")
    for p in test_patients:
        print(f"  {p.name}")

    if args.dry_run:
        print("\n[DRY RUN] No files were copied.")
        return

    # Create directories and copy
    val_dir.mkdir(parents=True, exist_ok=True)
    test_new_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying {len(val_patients)} patients to val/...")
    for patient in val_patients:
        shutil.copytree(patient, val_dir / patient.name)
        print(f"  Copied {patient.name}")

    print(f"Copying {len(test_patients)} patients to test_new/...")
    for patient in test_patients:
        shutil.copytree(patient, test_new_dir / patient.name)
        print(f"  Copied {patient.name}")

    print_header("DONE")
    print(f"Created: {val_dir} ({len(val_patients)} patients)")
    print(f"Created: {test_new_dir} ({len(test_patients)} patients)")


# =============================================================================
# RESIZE COMMAND
# =============================================================================

def cmd_resize(args):
    """Resize command handler (pad + resize to 256x256)."""
    if not HAS_MONAI:
        print("Error: MONAI and torch are required for the resize command.")
        print("Install with: pip install monai torch")
        return

    input_base = Path(args.input)
    output_base = Path(args.output)
    intermediate_size = (240, 240)
    target_size = (256, 256)

    if not input_base.exists():
        print(f"Error: Input directory does not exist: {input_base}")
        return

    print_header("IMAGE PREPROCESSING")
    print(f"Input:  {input_base}")
    print(f"Output: {output_base}")
    print("Pipeline: Pad to 240x240 (centered) -> Resize to 256x256")

    # Count patients
    for split in ['test', 'train']:
        split_dir = input_base / split
        if split_dir.exists():
            patients = [d for d in split_dir.iterdir() if d.is_dir()]
            print(f"{split.capitalize()}: {len(patients)} patients")

    if args.dry_run:
        print("\n[DRY RUN] No files will be created.")
        return

    # Confirmation
    if output_base.exists():
        response = input("\nOutput directory exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    else:
        response = input(f"\nCreate output at {output_base}? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Create transforms
    def create_transform(is_seg: bool) -> Compose:
        mode = 'nearest' if is_seg else 'bilinear'
        return Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            SpatialPad(spatial_size=(intermediate_size[0], intermediate_size[1], -1), mode='constant', method='symmetric'),
            Resize(spatial_size=(target_size[0], target_size[1], -1), mode=mode)
        ])

    # Process each split
    for split in ['test', 'train']:
        split_dir = input_base / split
        if not split_dir.exists():
            continue

        print(f"\nProcessing {split} split...")
        patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        for patient_dir in tqdm(patient_dirs, desc=f"Processing {split}"):
            output_patient_dir = output_base / split / patient_dir.name
            output_patient_dir.mkdir(parents=True, exist_ok=True)

            for input_file in sorted(patient_dir.glob("*.nii.gz")):
                modality = input_file.stem.replace('.nii', '')
                is_seg = 'seg' in modality.lower()

                transform = create_transform(is_seg)
                processed = transform(str(input_file))

                if isinstance(processed, torch.Tensor):
                    processed = processed.cpu().numpy()
                if processed.shape[0] == 1:
                    processed = processed[0]

                processed = processed.astype(np.float32)
                if is_seg:
                    processed = np.where(processed > 0.5, 1.0, 0.0).astype(np.float32)

                output_nii = nib.Nifti1Image(processed, affine=np.eye(4))
                nib.save(output_nii, output_patient_dir / input_file.name)

    print_header("COMPLETE")
    print(f"Output: {output_base}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified preprocessing tool for NIfTI medical images",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Align command (was normalize)
    p_align = subparsers.add_parser('align', help='Align all modalities to same slice count')
    p_align.add_argument('--data_dir', type=str, required=True, help='Data directory')
    p_align.add_argument('--target_slices', '-t', type=int, default=150, help='Target slice count (default: 150)')
    p_align.add_argument('--patient', type=str, help='Process only this patient')
    p_align.add_argument('--dry_run', '-n', action='store_true', help='Show what would be done')

    # Pad command
    p_pad = subparsers.add_parser('pad', help='Pad volumes to target slice count')
    p_pad.add_argument('--data_dir', type=str, required=True, help='Data directory')
    p_pad.add_argument('--target_slices', '-t', type=int, required=True, help='Target slice count')
    p_pad.add_argument('--pad_start', action='store_true', help='Pad at start instead of end')
    p_pad.add_argument('--patient', type=str, help='Process only this patient')
    p_pad.add_argument('--dry_run', '-n', action='store_true', help='Show what would be done')

    # Trim-auto command (was trim)
    p_trim = subparsers.add_parser('trim-auto', help='Auto-detect and trim empty slices')
    p_trim.add_argument('--data_dir', type=str, required=True, help='Data directory')
    p_trim.add_argument('--threshold', type=float, default=0.01, help='Empty slice threshold (default: 0.01)')
    p_trim.add_argument('--trim_trailing', action='store_true', help='Also trim trailing empty slices')
    p_trim.add_argument('--patient', type=str, help='Process only this patient')
    p_trim.add_argument('--dry_run', '-n', action='store_true', help='Show what would be done')

    # Trim-manual command (was trim-interactive)
    p_trimi = subparsers.add_parser('trim-manual', help='Interactive per-patient slice trimming')
    p_trimi.add_argument('--data_dir', '-d', type=str, required=True, help='Data directory')
    p_trimi.add_argument('--dry_run', '-n', action='store_true', help='Show what would be done')

    # Split command
    p_split = subparsers.add_parser('split', help='Split test into val and test_new')
    p_split.add_argument('--data_dir', type=str, required=True, help='Data directory')
    p_split.add_argument('--val_ratio', type=float, default=0.5, help='Validation ratio (default: 0.5)')
    p_split.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    p_split.add_argument('--dry_run', '-n', action='store_true', help='Show what would be done')

    # Resize command (was process)
    p_resize = subparsers.add_parser('resize', help='Resize images to 256x256 (pad + resize)')
    p_resize.add_argument('--input', '-i', type=str, required=True, help='Input directory')
    p_resize.add_argument('--output', '-o', type=str, required=True, help='Output directory')
    p_resize.add_argument('--dry_run', '-n', action='store_true', help='Show what would be done')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        'align': cmd_align,
        'pad': cmd_pad,
        'trim-auto': cmd_trim_auto,
        'trim-manual': cmd_trim_manual,
        'split': cmd_split,
        'resize': cmd_resize,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
