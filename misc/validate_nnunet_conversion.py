"""Validate nnU-Net raw dataset after conversion.

Checks everything nnU-Net reads before preprocessing:
- Affine consistency (spacing must match across all files)
- Label values (must be exactly {0, 1})
- Image intensity ranges
- File completeness (every image has a label)
- Symlink validity (synthetic images)
- Tumor presence in labels

Run AFTER convert_dataset.py, BEFORE nnUNetv2_plan_and_preprocess.

Usage:
    python misc/validate_nnunet_conversion.py \
        --dataset-dir /path/to/nnUNet_raw/Dataset501_BrainMet
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np


def _check_affines(dataset_dir: str) -> tuple[int, int]:
    """Check that all NIfTI files have consistent voxel spacing."""
    errors = 0
    checked = 0
    spacings = {}

    for subdir in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        path = os.path.join(dataset_dir, subdir)
        if not os.path.isdir(path):
            continue
        for fname in sorted(os.listdir(path)):
            if not fname.endswith('.nii.gz'):
                continue
            fpath = os.path.join(path, fname)
            try:
                nii = nib.load(fpath)
                spacing = tuple(np.round(np.abs(np.diag(nii.affine)[:3]), 4))
                spacings.setdefault(spacing, []).append(f'{subdir}/{fname}')
                checked += 1
            except Exception as e:
                print(f'  ERROR loading {subdir}/{fname}: {e}')
                errors += 1

    if len(spacings) == 1:
        sp = list(spacings.keys())[0]
        print(f'  OK: All {checked} files have consistent spacing {sp}')
    else:
        print(f'  FAIL: Found {len(spacings)} different spacings!')
        for sp, files in spacings.items():
            print(f'    {sp}: {len(files)} files (e.g. {files[0]})')
        errors += 1

    return errors, checked


def _check_labels(dataset_dir: str, subset: str) -> tuple[int, int, int]:
    """Check label files have correct values and contain tumor."""
    labels_dir = os.path.join(dataset_dir, f'labels{subset}')
    if not os.path.isdir(labels_dir):
        return 0, 0, 0

    errors = 0
    empty = 0
    checked = 0

    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith('.nii.gz'):
            continue
        fpath = os.path.join(labels_dir, fname)
        nii = nib.load(fpath)
        data = nii.get_fdata()
        raw = np.asanyarray(nii.dataobj)
        checked += 1

        unique = np.unique(raw)
        if not np.array_equal(unique, [0]) and not np.array_equal(unique, [0, 1]) and not np.array_equal(unique, [1]):
            print(f'  FAIL: labels{subset}/{fname}: unexpected values {unique} (expected {{0, 1}})')
            errors += 1

        if raw.dtype != np.uint8:
            print(f'  WARN: labels{subset}/{fname}: dtype={raw.dtype} (expected uint8)')

        tumor = (data > 0).sum()
        if tumor == 0:
            empty += 1

    return errors, checked, empty


def _check_images(dataset_dir: str, subset: str) -> tuple[int, int]:
    """Check image files for valid ranges and symlinks."""
    images_dir = os.path.join(dataset_dir, f'images{subset}')
    if not os.path.isdir(images_dir):
        return 0, 0

    errors = 0
    checked = 0

    for fname in sorted(os.listdir(images_dir)):
        if not fname.endswith('.nii.gz'):
            continue
        fpath = os.path.join(images_dir, fname)
        checked += 1

        # Check symlinks
        if os.path.islink(fpath):
            target = os.readlink(fpath)
            if not os.path.exists(fpath):
                print(f'  FAIL: images{subset}/{fname}: broken symlink -> {target}')
                errors += 1
                continue

        try:
            nii = nib.load(fpath)
            data = nii.get_fdata()

            if data.min() < -0.01:
                print(f'  WARN: images{subset}/{fname}: negative values (min={data.min():.4f})')

            if data.max() > 1.01:
                print(f'  WARN: images{subset}/{fname}: values > 1 (max={data.max():.4f}) — raw data not normalized?')

        except Exception as e:
            print(f'  FAIL: images{subset}/{fname}: {e}')
            errors += 1

    return errors, checked


def _check_pairing(dataset_dir: str, subset: str) -> int:
    """Check every image has a corresponding label and vice versa."""
    images_dir = os.path.join(dataset_dir, f'images{subset}')
    labels_dir = os.path.join(dataset_dir, f'labels{subset}')
    errors = 0

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        return 0

    # Extract case IDs from images (remove _XXXX channel suffix)
    image_cases = set()
    for f in os.listdir(images_dir):
        if f.endswith('.nii.gz'):
            # BrainMet_005_0000.nii.gz -> BrainMet_005
            case_id = f.rsplit('_', 1)[0]
            image_cases.add(case_id)

    label_cases = set()
    for f in os.listdir(labels_dir):
        if f.endswith('.nii.gz'):
            # BrainMet_005.nii.gz -> BrainMet_005
            case_id = f.replace('.nii.gz', '')
            label_cases.add(case_id)

    missing_labels = image_cases - label_cases
    missing_images = label_cases - image_cases

    if missing_labels:
        print(f'  FAIL: {len(missing_labels)} images without labels in {subset}: {sorted(missing_labels)[:5]}...')
        errors += 1
    if missing_images:
        print(f'  FAIL: {len(missing_images)} labels without images in {subset}: {sorted(missing_images)[:5]}...')
        errors += 1
    if not missing_labels and not missing_images:
        print(f'  OK: {len(image_cases)} cases in {subset}, all paired')

    return errors


def _check_dataset_json(dataset_dir: str) -> int:
    """Check dataset.json is valid."""
    path = os.path.join(dataset_dir, 'dataset.json')
    if not os.path.exists(path):
        print('  FAIL: dataset.json not found')
        return 1

    with open(path) as f:
        dj = json.load(f)

    required = ['channel_names', 'labels', 'numTraining', 'file_ending']
    missing = [k for k in required if k not in dj]
    if missing:
        print(f'  FAIL: dataset.json missing keys: {missing}')
        return 1

    if dj['labels'] != {'background': 0, 'tumor': 1}:
        print(f'  WARN: unexpected labels: {dj["labels"]}')

    print(f'  OK: dataset.json — {dj["numTraining"]} training cases, '
          f'channels={dj["channel_names"]}, ending={dj["file_ending"]}')
    return 0


def _spot_check_details(dataset_dir: str, n: int = 3) -> None:
    """Print detailed info for a few files (real + synthetic)."""
    labels_dir = os.path.join(dataset_dir, 'labelsTr')
    if not os.path.isdir(labels_dir):
        return

    files = sorted(os.listdir(labels_dir))
    real = [f for f in files if not f.startswith('BrainMetSyn')]
    syn = [f for f in files if f.startswith('BrainMetSyn')]
    check = real[:n] + syn[:n]

    for fname in check:
        fpath = os.path.join(labels_dir, fname)
        nii = nib.load(fpath)
        data = nii.get_fdata()
        spacing = np.round(np.abs(np.diag(nii.affine)[:3]), 4)
        tumor = (data > 0).sum()
        print(f'  {fname}: shape={data.shape}, spacing={spacing}, '
              f'dtype={nii.header.get_data_dtype()}, tumor={tumor} ({100*tumor/data.size:.4f}%)')


def main():
    parser = argparse.ArgumentParser(description='Validate nnU-Net raw dataset')
    parser.add_argument('--dataset-dir', required=True,
                        help='Path to Dataset501_BrainMet directory')
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f'ERROR: {args.dataset_dir} not found')
        sys.exit(1)

    total_errors = 0

    print('=' * 60)
    print('nnU-Net Raw Dataset Validation')
    print(f'Dataset: {args.dataset_dir}')
    print('=' * 60)

    # 1. dataset.json
    print('\n1. DATASET JSON')
    total_errors += _check_dataset_json(args.dataset_dir)

    # 2. Affine consistency
    print('\n2. AFFINE / SPACING CONSISTENCY')
    errs, n = _check_affines(args.dataset_dir)
    total_errors += errs

    # 3. Labels
    for subset, name in [('Tr', 'Training'), ('Ts', 'Test')]:
        print(f'\n3. {name.upper()} LABELS')
        errs, checked, empty = _check_labels(args.dataset_dir, subset)
        total_errors += errs
        if checked:
            print(f'  Checked {checked} labels: {empty} empty (no tumor), {errs} errors')

    # 4. Images
    for subset, name in [('Tr', 'Training'), ('Ts', 'Test')]:
        print(f'\n4. {name.upper()} IMAGES')
        errs, checked = _check_images(args.dataset_dir, subset)
        total_errors += errs
        if checked:
            print(f'  Checked {checked} images: {errs} errors')

    # 5. Pairing
    print('\n5. IMAGE-LABEL PAIRING')
    for subset in ('Tr', 'Ts'):
        total_errors += _check_pairing(args.dataset_dir, subset)

    # 6. Spot check
    print('\n6. SPOT CHECK (first 3 real + 3 synthetic)')
    _spot_check_details(args.dataset_dir)

    # Summary
    print('\n' + '=' * 60)
    if total_errors == 0:
        print('PASSED: All checks OK. Safe to run nnUNetv2_plan_and_preprocess.')
    else:
        print(f'FAILED: {total_errors} error(s) found. Fix before preprocessing.')
    print('=' * 60)

    sys.exit(1 if total_errors else 0)


if __name__ == '__main__':
    main()
