"""Build an nnU-Net dataset from a RANDOM 105/51 split of the 156 Stanford patients.

Pools all patients from train/, val/, and test_new/ (156 total) and randomly
splits them into 105 train + 51 test using a fixed seed. Produces a standard
nnU-Net dataset (imagesTr, labelsTr, imagesTs, labelsTs, dataset.json,
case_info.json) identical in layout to what `convert_dataset.py` creates for
the official Grøvik 2020 split — only the split membership differs.

Motivation
----------
Ottesen 2025 (JMRI) reports volumetric Dice = 0.66 ± 0.01 on Stanford after
*"the annotated Stanford data were split randomly into a training and test
dataset containing 105 and 51 cases, respectively"*. This is NOT the official
Grøvik 2020 hold-out split that we use for exp3 (0.32 Dice). This script
replicates Ottesen 2025's random-split protocol so we can verify whether the
~2× Dice gap is attributable to test-cohort difficulty.

Usage
-----

    python -m medgen.scripts.convert_random_split \\
        --real-dir /path/to/brainmetshare-3 \\
        --nnunet-raw /path/to/nnUNet_raw \\
        --dataset-id 640 \\
        --seed 42 \\
        --modality bravo

Output
------
- Dataset{ID}_BrainMet/imagesTr (105 cases × N modalities)
- Dataset{ID}_BrainMet/labelsTr (105 labels)
- Dataset{ID}_BrainMet/imagesTs (51 cases × N modalities)
- Dataset{ID}_BrainMet/labelsTs (51 labels)
- Dataset{ID}_BrainMet/dataset.json
- Dataset{ID}_BrainMet/case_info.json (includes the chosen split for reproducibility)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random

from medgen.downstream.nnunet import convert_dataset as cd

logger = logging.getLogger(__name__)


def _process_patient(real_dir: str, split_origin: str, patient_name: str,
                     output_dir: str, subset: str,
                     modalities: list[str]) -> str | None:
    """Process one patient into the nnU-Net dataset layout. Returns case_id or None.

    Mirrors the body of `convert_dataset._add_real_patients`, but operates on a
    SINGLE patient at a known split origin rather than scanning a whole split.
    """
    patient_dir = os.path.join(real_dir, split_origin, patient_name)
    case_id = patient_name.replace('Mets_', 'BrainMet_')

    seg_src = os.path.join(patient_dir, 'seg.nii.gz')
    if not os.path.exists(seg_src):
        logger.warning(f"Missing seg.nii.gz for {patient_name}, skipping")
        return None

    img_srcs = []
    for mod in modalities:
        src = os.path.join(patient_dir, f'{mod}.nii.gz')
        if not os.path.exists(src):
            logger.warning(f"Missing {mod}.nii.gz for {patient_name}, skipping")
            return None
        img_srcs.append(src)

    images_dir = os.path.join(output_dir, f'images{subset}')
    labels_dir = os.path.join(output_dir, f'labels{subset}')
    for ch_idx, (mod, src) in enumerate(zip(modalities, img_srcs, strict=True)):
        dst = os.path.join(images_dir, f'{case_id}_{ch_idx:04d}.nii.gz')
        cd._normalize_to_unit(src, dst, clip_max=cd.CLIP_MAX[mod])
    cd._convert_label(seg_src, os.path.join(labels_dir, f'{case_id}.nii.gz'))
    return case_id


def create_random_split_dataset(
    real_dir: str,
    nnunet_raw: str,
    dataset_id: int,
    seed: int,
    modality: str | list[str] = 'bravo',
    n_train: int = 105,
    n_test: int = 51,
) -> dict:
    """Build the nnU-Net dataset from a random 105/51 split of all 156 patients."""
    modalities = cd._resolve_modalities(modality)
    output_dir = os.path.join(nnunet_raw, f'Dataset{dataset_id}_{cd.DATASET_NAME}')

    for sub in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # Pool all 156 patients across the three official BrainMetShare-3 splits.
    pooled: list[tuple[str, str]] = []
    for split_origin in ('train', 'val', 'test_new'):
        for p in cd._find_patients(real_dir, split_origin):
            pooled.append((split_origin, p))
    logger.info(
        f"Pooled {len(pooled)} patients from train/val/test_new "
        f"(expected 156 for BrainMetShare-3)"
    )
    if len(pooled) != n_train + n_test:
        logger.warning(
            f"Pool size {len(pooled)} != n_train+n_test={n_train + n_test}. "
            f"Will proceed but truncate."
        )

    rng = random.Random(seed)
    rng.shuffle(pooled)

    train_pool = pooled[:n_train]
    test_pool = pooled[n_train:n_train + n_test]
    logger.info(
        f"Random split (seed={seed}): {len(train_pool)} train + "
        f"{len(test_pool)} test"
    )

    train_cases: list[str] = []
    for origin, p in train_pool:
        cid = _process_patient(real_dir, origin, p, output_dir, 'Tr', modalities)
        if cid:
            train_cases.append(cid)
    logger.info(f"Added {len(train_cases)} train cases to imagesTr")

    test_cases: list[str] = []
    for origin, p in test_pool:
        cid = _process_patient(real_dir, origin, p, output_dir, 'Ts', modalities)
        if cid:
            test_cases.append(cid)
    logger.info(f"Added {len(test_cases)} test cases to imagesTs")

    cd._write_dataset_json(output_dir, len(train_cases), modalities)

    case_info = {
        'random_split_seed': seed,
        'n_train': len(train_cases),
        'n_test': len(test_cases),
        'real_train_cases': train_cases,
        'synthetic_cases': [],
        'test_cases': test_cases,
        'train_pool_origins': [
            {'origin': o, 'patient': p, 'case_id': p.replace('Mets_', 'BrainMet_')}
            for o, p in train_pool
        ],
        'test_pool_origins': [
            {'origin': o, 'patient': p, 'case_id': p.replace('Mets_', 'BrainMet_')}
            for o, p in test_pool
        ],
        'modalities': modalities,
    }
    info_path = os.path.join(output_dir, 'case_info.json')
    with open(info_path, 'w') as f:
        json.dump(case_info, f, indent=2)
    logger.info(f"Wrote case info (with split provenance) to {info_path}")

    return {
        'train_cases': train_cases,
        'test_cases': test_cases,
        'output_dir': output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-dir', required=True,
                        help='Root of brainmetshare-3 dataset')
    parser.add_argument('--nnunet-raw', required=True,
                        help='nnU-Net raw data directory')
    parser.add_argument('--dataset-id', type=int, required=True,
                        help='Dataset ID (e.g., 640)')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for the 105/51 split (reproducibility)')
    parser.add_argument('--modality', default='bravo',
                        help='Modality preset (bravo, dual, triple, quad) or '
                             'comma-separated list. Default: bravo')
    parser.add_argument('--n-train', type=int, default=105)
    parser.add_argument('--n-test', type=int, default=51)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    modality = args.modality
    if ',' in modality:
        modality = [m.strip() for m in modality.split(',')]

    result = create_random_split_dataset(
        real_dir=args.real_dir,
        nnunet_raw=args.nnunet_raw,
        dataset_id=args.dataset_id,
        seed=args.seed,
        modality=modality,
        n_train=args.n_train,
        n_test=args.n_test,
    )
    logger.info(f"Random-split dataset built at {result['output_dir']}")


if __name__ == '__main__':
    main()
