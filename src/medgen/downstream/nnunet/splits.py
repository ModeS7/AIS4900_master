"""Generate experiment-specific splits_final.json for nnU-Net 5-fold CV.

Single dataset (Dataset501) contains all 105 real + 525 synthetic in imagesTr.
This module generates different splits_final.json files per experiment, where:

    - The 105 real patients are randomly partitioned into 5 folds (seed=42)
    - Each fold's val set is ALWAYS the same ~21 real patients
    - Only the train set changes between experiments:
        - baseline:  ~84 real
        - synthetic:  525 synthetic (val still real)
        - mixed_N:   ~84 real + N synthetic

The base 5-fold partition is generated ONCE from a fixed seed, so fold 0
always validates on the same 21 patients regardless of experiment.

Must be run AFTER nnUNetv2_plan_and_preprocess (which creates the preprocessed dir).

Usage:
    # Generate splits for baseline experiment
    python -m medgen.downstream.nnunet.splits \
        --nnunet-raw /path/to/nnUNet_raw \
        --nnunet-preprocessed /path/to/nnUNet_preprocessed \
        --experiment baseline

    # Generate splits for mixed with 210 synthetic
    python -m medgen.downstream.nnunet.splits \
        --nnunet-raw /path/to/nnUNet_raw \
        --nnunet-preprocessed /path/to/nnUNet_preprocessed \
        --experiment mixed --n-synthetic 210
"""
import argparse
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

DATASET_ID = 501
N_FOLDS = 5
FOLD_SEED = 42

# Experiment name -> description (for logging)
EXPERIMENTS = {
    'baseline': 'Real-only (105 train, BRAVO)',
    'baseline_dual': 'Real-only (105 train, dual: t1_pre + t1_gd)',
    'synthetic': 'Synthetic-only (525 train, real val)',
    'mixed': 'Mixed real + synthetic',
}


def _find_preprocessed_dir(nnunet_preprocessed: str, dataset_id: int) -> str:
    """Find the preprocessed dataset directory."""
    for entry in os.listdir(nnunet_preprocessed):
        if entry.startswith(f'Dataset{dataset_id}_'):
            return os.path.join(nnunet_preprocessed, entry)
    raise FileNotFoundError(
        f"No preprocessed directory for Dataset{dataset_id} in "
        f"{nnunet_preprocessed}. Run nnUNetv2_plan_and_preprocess first."
    )


def _load_case_info(nnunet_raw: str, dataset_id: int) -> dict:
    """Load case_info.json written by convert_dataset.py."""
    for entry in os.listdir(nnunet_raw):
        if entry.startswith(f'Dataset{dataset_id}_'):
            info_path = os.path.join(nnunet_raw, entry, 'case_info.json')
            if os.path.exists(info_path):
                with open(info_path) as f:
                    return json.load(f)
            raise FileNotFoundError(f"case_info.json not found in {entry}")
    raise FileNotFoundError(f"No dataset for ID {dataset_id} in {nnunet_raw}")


def generate_base_folds(
    real_train_cases: list[str],
    n_folds: int = N_FOLDS,
    seed: int = FOLD_SEED,
) -> list[tuple[list[str], list[str]]]:
    """Generate the base 5-fold partition of real patients.

    Randomly shuffles the 105 real patients with a fixed seed, then splits
    into 5 roughly equal folds. Returns (train_real, val) for each fold.

    This is the ONLY place fold assignment happens. All experiments reuse
    the same base folds to ensure identical validation sets.

    Args:
        real_train_cases: List of 105 real patient case IDs.
        n_folds: Number of folds (default: 5).
        seed: Random seed for reproducible shuffling.

    Returns:
        List of (train_cases, val_cases) tuples, one per fold.
    """
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(real_train_cases))
    fold_arrays = np.array_split(shuffled, n_folds)

    folds = []
    for i in range(n_folds):
        val_cases = sorted(fold_arrays[i].tolist())
        train_cases = sorted(
            c for j, arr in enumerate(fold_arrays) if j != i
            for c in arr.tolist()
        )
        folds.append((train_cases, val_cases))

    return folds


def generate_experiment_splits(
    experiment: str,
    real_train_cases: list[str],
    synthetic_cases: list[str],
    n_synthetic: int | None = None,
    seed: int = FOLD_SEED,
    synthetic_seed: int = 42,
) -> list[dict[str, list[str]]]:
    """Generate splits_final.json content for a specific experiment.

    Args:
        experiment: 'baseline', 'synthetic', or 'mixed'.
        real_train_cases: All 105 real train case IDs.
        synthetic_cases: All 525 synthetic case IDs.
        n_synthetic: Number of synthetic samples for mixed (None = all 525).
        seed: Seed for fold generation (must match across experiments).
        synthetic_seed: Seed for synthetic subset selection.

    Returns:
        List of 5 dicts, each with 'train' and 'val' keys.
    """
    base_folds = generate_base_folds(real_train_cases, seed=seed)

    # Select synthetic subset if needed
    if n_synthetic is not None and n_synthetic < len(synthetic_cases):
        rng = np.random.default_rng(synthetic_seed)
        syn_subset = sorted(rng.choice(synthetic_cases, size=n_synthetic, replace=False))
    else:
        syn_subset = sorted(synthetic_cases)

    splits = []
    for train_real, val_real in base_folds:
        if experiment in ('baseline', 'baseline_dual'):
            train = sorted(train_real)
        elif experiment == 'synthetic':
            train = sorted(syn_subset)
        elif experiment == 'mixed':
            train = sorted(train_real + syn_subset)
        else:
            raise ValueError(f"Unknown experiment: {experiment}")

        splits.append({'train': train, 'val': sorted(val_real)})

    return splits


def install_splits(
    splits: list[dict],
    nnunet_preprocessed: str,
    dataset_id: int = DATASET_ID,
) -> str:
    """Write splits_final.json to the preprocessed dataset directory.

    Args:
        splits: List of fold dicts (from generate_experiment_splits).
        nnunet_preprocessed: nnU-Net preprocessed data root.
        dataset_id: Dataset ID.

    Returns:
        Path to the written splits_final.json.
    """
    preprocessed_dir = _find_preprocessed_dir(nnunet_preprocessed, dataset_id)
    splits_path = os.path.join(preprocessed_dir, 'splits_final.json')

    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)

    # Log summary
    for i, fold in enumerate(splits):
        logger.info(
            f"  Fold {i}: {len(fold['train'])} train, {len(fold['val'])} val"
        )
    logger.info(f"Wrote {splits_path}")

    return splits_path


def main() -> None:
    """CLI entry point for generating experiment splits."""
    parser = argparse.ArgumentParser(
        description='Generate nnU-Net splits_final.json for an experiment',
    )
    parser.add_argument('--nnunet-raw', required=True,
                        help='nnU-Net raw data directory')
    parser.add_argument('--nnunet-preprocessed', required=True,
                        help='nnU-Net preprocessed data directory')
    parser.add_argument('--dataset-id', type=int, default=DATASET_ID)
    parser.add_argument('--experiment', required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiment type')
    parser.add_argument('--n-synthetic', type=int, default=None,
                        help='Number of synthetic samples for mixed (default: all)')
    parser.add_argument('--seed', type=int, default=FOLD_SEED,
                        help=f'Fold generation seed (default: {FOLD_SEED})')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    case_info = _load_case_info(args.nnunet_raw, args.dataset_id)

    desc = EXPERIMENTS[args.experiment]
    if args.experiment == 'mixed' and args.n_synthetic:
        desc += f' (n_synthetic={args.n_synthetic})'
    logger.info(f"Generating splits for: {desc}")

    splits = generate_experiment_splits(
        experiment=args.experiment,
        real_train_cases=case_info['real_train_cases'],
        synthetic_cases=case_info['synthetic_cases'],
        n_synthetic=args.n_synthetic,
        seed=args.seed,
    )

    path = install_splits(splits, args.nnunet_preprocessed, args.dataset_id)
    print(f"Installed: {path}")


if __name__ == '__main__':
    main()
