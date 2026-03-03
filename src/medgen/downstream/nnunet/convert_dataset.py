"""Convert MedGen data format to nnU-Net v2 raw dataset format.

Creates ONE dataset with all data, preprocessed once:
    imagesTr: 105 real train + 525 synthetic (630 total)
    imagesTs: 51 test (25 from val/ + 26 from test_new/)

Experiment-specific splits_final.json controls what's used per experiment
(see splits.py). This avoids duplicating preprocessed data.

Our format:
    Real: brainmetshare-3/{train,val,test_new}/Mets_XXX/{bravo,seg}.nii.gz
    Synthetic: generated/exp_name/XXXXX/{bravo,seg}.nii.gz

nnU-Net format:
    nnUNet_raw/Dataset501_BrainMet/
        imagesTr/CaseName_0000.nii.gz   (channel 0 = bravo)
        labelsTr/CaseName.nii.gz        (uint8, values 0/1)
        imagesTs/CaseName_0000.nii.gz   (test images)
        labelsTs/CaseName.nii.gz        (test labels)
        dataset.json

Key decisions:
    - Symlink images (save disk), copy+convert labels (float32 -> uint8)
    - ALL 105 real train + ALL 525 synthetic go into imagesTr
    - Test set (val/ + test_new/ = 51) goes into imagesTs/labelsTs
    - Synthetic cases named BrainMetSyn_XXXXX to distinguish from real
"""
import argparse
import json
import logging
import os
import shutil

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

DATASET_ID = 501
DATASET_NAME = 'BrainMet'

# Modality presets: name -> list of NIfTI filenames (channel order matters)
MODALITY_PRESETS = {
    'bravo': ['bravo'],
    'dual': ['t1_pre', 't1_gd'],
}

def _resolve_modalities(modality: str | list[str]) -> list[str]:
    """Resolve modality argument to list of NIfTI file stems.

    Args:
        modality: Preset name ('bravo', 'dual') or explicit list (['t1_pre', 't1_gd']).

    Returns:
        List of modality file stems, e.g., ['bravo'] or ['t1_pre', 't1_gd'].
    """
    if isinstance(modality, list):
        return modality
    if modality in MODALITY_PRESETS:
        return MODALITY_PRESETS[modality]
    # Treat as single modality name
    return [modality]


def _find_patients(data_dir: str, split: str) -> list[str]:
    """Find patient directories in a data split.

    Returns:
        Sorted list of patient directory names (e.g., ['Mets_005', ...]).
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    return sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )


def _find_synthetic_samples(synthetic_dir: str) -> list[str]:
    """Find synthetic sample subdirectories (3D format: XXXXX/{bravo,seg}.nii.gz).

    Returns:
        Sorted list of sample subdirectory names (e.g., ['00000', '00001', ...]).
    """
    if not os.path.isdir(synthetic_dir):
        raise FileNotFoundError(f"Synthetic directory not found: {synthetic_dir}")

    return sorted(
        d for d in os.listdir(synthetic_dir)
        if os.path.isdir(os.path.join(synthetic_dir, d))
        and os.path.exists(os.path.join(synthetic_dir, d, 'seg.nii.gz'))
    )


def _convert_label(src_path: str, dst_path: str) -> None:
    """Convert a segmentation NIfTI from float32 to uint8 binary mask."""
    nii = nib.load(src_path)
    data = nii.get_fdata()
    binary = (data > 0.5).astype(np.uint8)
    out = nib.Nifti1Image(binary, nii.affine, nii.header)
    out.header.set_data_dtype(np.uint8)
    if os.path.exists(dst_path):
        os.remove(dst_path)
    nib.save(out, dst_path)


def _symlink_or_copy(src: str, dst: str) -> None:
    """Create symlink, falling back to copy if symlink fails."""
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    src_abs = os.path.abspath(src)
    try:
        os.symlink(src_abs, dst)
    except OSError:
        shutil.copy2(src_abs, dst)


def _add_real_patients(
    real_dir: str,
    split: str,
    output_dir: str,
    subset: str,
    modalities: list[str] | None = None,
) -> list[str]:
    """Add real patients from a split to the nnU-Net dataset.

    Args:
        real_dir: Root of brainmetshare-3.
        split: 'train', 'val', or 'test_new'.
        output_dir: nnU-Net dataset root.
        subset: 'Tr' or 'Ts' (nnU-Net convention).
        modalities: List of modality file stems (e.g., ['bravo'] or ['t1_pre', 't1_gd']).
            Each becomes a channel: _0000.nii.gz, _0001.nii.gz, etc.

    Returns:
        List of case identifiers added (e.g., ['BrainMet_005', ...]).
    """
    if modalities is None:
        modalities = ['bravo']

    patients = _find_patients(real_dir, split)
    case_ids = []

    images_dir = os.path.join(output_dir, f'images{subset}')
    labels_dir = os.path.join(output_dir, f'labels{subset}')

    for patient_name in patients:
        patient_dir = os.path.join(real_dir, split, patient_name)
        case_id = patient_name.replace('Mets_', 'BrainMet_')

        seg_src = os.path.join(patient_dir, 'seg.nii.gz')
        if not os.path.exists(seg_src):
            logger.warning(f"Missing seg.nii.gz for {patient_name}, skipping")
            continue

        # Check all modality files exist
        img_srcs = []
        missing = False
        for mod in modalities:
            src = os.path.join(patient_dir, f'{mod}.nii.gz')
            if not os.path.exists(src):
                logger.warning(f"Missing {mod}.nii.gz for {patient_name}, skipping")
                missing = True
                break
            img_srcs.append(src)
        if missing:
            continue

        # Symlink each modality as a separate channel
        for ch_idx, src in enumerate(img_srcs):
            _symlink_or_copy(src, os.path.join(images_dir, f'{case_id}_{ch_idx:04d}.nii.gz'))

        _convert_label(seg_src, os.path.join(labels_dir, f'{case_id}.nii.gz'))
        case_ids.append(case_id)

    return case_ids


def _add_synthetic_samples(
    synthetic_dir: str,
    output_dir: str,
    modalities: list[str] | None = None,
) -> list[str]:
    """Add ALL synthetic samples to imagesTr/labelsTr.

    Args:
        synthetic_dir: Directory with generated 3D samples.
        output_dir: nnU-Net dataset root.
        modalities: List of modality file stems. Each becomes a channel.

    Returns:
        List of case identifiers added (e.g., ['BrainMetSyn_00000', ...]).
    """
    if modalities is None:
        modalities = ['bravo']

    all_samples = _find_synthetic_samples(synthetic_dir)

    images_dir = os.path.join(output_dir, 'imagesTr')
    labels_dir = os.path.join(output_dir, 'labelsTr')
    case_ids = []

    for sample_name in all_samples:
        sample_dir = os.path.join(synthetic_dir, sample_name)
        case_id = f'BrainMetSyn_{sample_name}'

        seg_src = os.path.join(sample_dir, 'seg.nii.gz')

        # Check all modality files exist
        img_srcs = []
        missing = False
        for mod in modalities:
            src = os.path.join(sample_dir, f'{mod}.nii.gz')
            if not os.path.exists(src):
                logger.warning(f"Missing {mod}.nii.gz for synthetic {sample_name}, skipping")
                missing = True
                break
            img_srcs.append(src)
        if missing:
            continue

        for ch_idx, src in enumerate(img_srcs):
            _symlink_or_copy(src, os.path.join(images_dir, f'{case_id}_{ch_idx:04d}.nii.gz'))

        _convert_label(seg_src, os.path.join(labels_dir, f'{case_id}.nii.gz'))
        case_ids.append(case_id)

    return case_ids


def _write_dataset_json(
    output_dir: str,
    num_training: int,
    modalities: list[str],
) -> None:
    """Write dataset.json required by nnU-Net."""
    channel_names = {str(i): mod.upper() for i, mod in enumerate(modalities)}
    dataset_json = {
        'channel_names': channel_names,
        'labels': {'background': 0, 'tumor': 1},
        'numTraining': num_training,
        'file_ending': '.nii.gz',
    }
    path = os.path.join(output_dir, 'dataset.json')
    with open(path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    logger.info(f"Wrote {path} (channels: {channel_names})")


def create_dataset(
    real_dir: str,
    synthetic_dir: str | None,
    nnunet_raw: str,
    dataset_id: int = DATASET_ID,
    modality: str | list[str] = 'bravo',
) -> dict:
    """Create ONE nnU-Net raw dataset with all real + synthetic data.

    imagesTr: 105 real train patients + synthetic samples (if provided)
    imagesTs: 51 test patients (val/ + test_new/)

    Experiment-specific training subsets are controlled via splits_final.json
    (see splits.py), not by separate datasets.

    Args:
        real_dir: Root of brainmetshare-3 dataset.
        synthetic_dir: Path to generated 3D samples, or None for real-only.
        nnunet_raw: nnU-Net raw data root (nnUNet_raw/).
        dataset_id: Dataset ID (default: 501).
        modality: Modality preset ('bravo', 'dual') or explicit list
            (['t1_pre', 't1_gd']). Each modality becomes an input channel.

    Returns:
        Dict with 'real_train_cases', 'synthetic_cases', 'test_cases', 'output_dir'.
    """
    modalities = _resolve_modalities(modality)
    output_dir = os.path.join(nnunet_raw, f'Dataset{dataset_id}_{DATASET_NAME}')

    for subdir in ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    logger.info(f"Creating dataset in {output_dir}")
    logger.info(f"Modalities: {modalities} ({len(modalities)} channels)")

    # imagesTr: real train + synthetic (if provided)
    real_train_cases = _add_real_patients(
        real_dir, 'train', output_dir, 'Tr', modalities,
    )
    logger.info(f"Added {len(real_train_cases)} real train patients to imagesTr")

    synthetic_cases = []
    if synthetic_dir:
        synthetic_cases = _add_synthetic_samples(synthetic_dir, output_dir, modalities)
        logger.info(f"Added {len(synthetic_cases)} synthetic samples to imagesTr")

    # imagesTs: val (25) + test_new (26) = 51 test patients
    test_cases = []
    for split in ('val', 'test_new'):
        cases = _add_real_patients(real_dir, split, output_dir, 'Ts', modalities)
        test_cases.extend(cases)
        logger.info(f"Added {len(cases)} patients from {split}/ to imagesTs")

    # dataset.json
    num_training = len(real_train_cases) + len(synthetic_cases)
    _write_dataset_json(output_dir, num_training, modalities)

    # Save case lists for splits.py
    case_info = {
        'real_train_cases': real_train_cases,
        'synthetic_cases': synthetic_cases,
        'test_cases': test_cases,
        'modalities': modalities,
    }
    info_path = os.path.join(output_dir, 'case_info.json')
    with open(info_path, 'w') as f:
        json.dump(case_info, f, indent=2)
    logger.info(f"Wrote case info to {info_path}")

    logger.info(
        f"Dataset: {len(real_train_cases)} real train + "
        f"{len(synthetic_cases)} synthetic in imagesTr, "
        f"{len(test_cases)} test in imagesTs"
    )

    return {
        'real_train_cases': real_train_cases,
        'synthetic_cases': synthetic_cases,
        'test_cases': test_cases,
        'output_dir': output_dir,
    }


def main() -> None:
    """CLI entry point for dataset conversion."""
    parser = argparse.ArgumentParser(
        description='Convert MedGen data to nnU-Net format (single dataset)',
    )
    parser.add_argument('--real-dir', required=True,
                        help='Root of brainmetshare-3 dataset')
    parser.add_argument('--synthetic-dir', default=None,
                        help='Path to generated 3D samples (omit for real-only)')
    parser.add_argument('--nnunet-raw', required=True,
                        help='nnU-Net raw data directory')
    parser.add_argument('--dataset-id', type=int, default=DATASET_ID,
                        help=f'Dataset ID (default: {DATASET_ID})')
    parser.add_argument('--modality', default='bravo',
                        help='Modality preset (bravo, dual) or comma-separated '
                             'list (t1_pre,t1_gd). Default: bravo')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Parse modality: "dual" -> preset, "t1_pre,t1_gd" -> explicit list
    modality = args.modality
    if ',' in modality:
        modality = [m.strip() for m in modality.split(',')]

    result = create_dataset(
        real_dir=args.real_dir,
        synthetic_dir=args.synthetic_dir,
        nnunet_raw=args.nnunet_raw,
        dataset_id=args.dataset_id,
        modality=modality,
    )

    print(f"\nDataset created: {result['output_dir']}")
    print(f"  Real train:  {len(result['real_train_cases'])} cases")
    print(f"  Synthetic:   {len(result['synthetic_cases'])} cases")
    print(f"  Test:        {len(result['test_cases'])} cases")
    print(f"\nNext: run nnUNetv2_plan_and_preprocess -d {args.dataset_id} -c 3d_fullres")


if __name__ == '__main__':
    main()
