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
import contextlib
import json
import logging
import os
import re
import tempfile

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


def _load_synthetic_manifest(
    path: str,
    available_cases: list[str],
) -> list[str]:
    """Load one complete, ordered synthetic-case manifest.

    Rows may be nnU-Net case IDs (``BrainMetSyn_00000``) or raw five-digit
    candidate IDs (``00000``). The manifest must be an exact permutation of
    ``available_cases`` so prefix selection can never silently omit or add a
    synthetic case.
    """
    with open(path) as f:
        rows = [line.strip() for line in f if line.strip()]

    if not rows:
        raise ValueError(f"Synthetic manifest is empty: {path}")

    available = set(available_cases)
    normalized: list[str] = []
    for row in rows:
        if row in available:
            case_id = row
        elif re.fullmatch(r"[0-9]{5}", row):
            case_id = f"BrainMetSyn_{row}"
        else:
            raise ValueError(f"Malformed synthetic manifest row: {row!r}")
        if case_id not in available:
            raise ValueError(
                f"Synthetic manifest case is absent from case_info.json: {case_id}"
            )
        normalized.append(case_id)

    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Synthetic manifest contains duplicate cases: {path}")
    if set(normalized) != available:
        missing = sorted(available - set(normalized))
        raise ValueError(
            "Synthetic manifest is not a complete permutation of case_info.json: "
            f"manifest={len(normalized)}, available={len(available)}, "
            f"missing={missing[:5]}"
        )
    return normalized


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
    synthetic_order: list[str] | None = None,
) -> list[dict[str, list[str]]]:
    """Generate splits_final.json content for a specific experiment.

    Args:
        experiment: 'baseline', 'synthetic', or 'mixed'.
        real_train_cases: All 105 real train case IDs.
        synthetic_cases: All 525 synthetic case IDs.
        n_synthetic: Number of synthetic samples for mixed (None = all 525).
        seed: Seed for fold generation (must match across experiments).
        synthetic_seed: Seed for synthetic subset selection.
        synthetic_order: Optional complete case order. When supplied, the
            first ``n_synthetic`` cases are selected instead of using the
            historical seeded random subset.

    Returns:
        List of 5 dicts, each with 'train' and 'val' keys.
    """
    base_folds = generate_base_folds(real_train_cases, seed=seed)

    # Explicit order is opt-in. Historical callers retain the original seeded
    # random subset behavior byte-for-byte when no order is supplied.
    if synthetic_order is not None:
        if not synthetic_order:
            raise ValueError("synthetic_order must not be empty")
        if len(synthetic_order) != len(set(synthetic_order)):
            raise ValueError("synthetic_order contains duplicate cases")
        available = set(synthetic_cases)
        ordered = set(synthetic_order)
        if ordered != available:
            raise ValueError(
                "synthetic_order must be a complete permutation of "
                "synthetic_cases"
            )
        requested = len(synthetic_order) if n_synthetic is None else n_synthetic
        if requested < 0 or requested > len(synthetic_order):
            raise ValueError(
                f"n_synthetic={requested} is outside 0--{len(synthetic_order)}"
            )
        syn_subset = list(synthetic_order[:requested])
    elif n_synthetic is not None and n_synthetic < len(synthetic_cases):
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

    .. deprecated::
        Use :func:`create_isolated_preprocessed_dir` instead to avoid
        race conditions when running multiple experiments concurrently.

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


def create_isolated_preprocessed_dir(
    experiment_name: str,
    splits: list[dict],
    nnunet_preprocessed: str,
    dataset_id: int = DATASET_ID,
) -> str:
    """Create a per-experiment isolated preprocessed directory.

    Creates a shadow directory that symlinks all shared data (preprocessed
    .npz files, gt_segmentations, plan JSONs) from the original preprocessed
    dir but contains its own ``splits_final.json``. This prevents race
    conditions when multiple experiments run concurrently on the same dataset.

    Directory layout::

        {nnunet_preprocessed}_{experiment_name}/
        └── Dataset501_BrainMet/
            ├── splits_final.json          ← real file (experiment-specific)
            ├── nnUNetPlans_3d_fullres/    ← symlink to original
            ├── gt_segmentations/          ← symlink to original
            └── *.json                     ← symlinks to original

    Args:
        experiment_name: Unique experiment identifier (e.g. 'exp3_baseline').
        splits: List of fold dicts (from generate_experiment_splits).
        nnunet_preprocessed: Original nnU-Net preprocessed root
            (e.g. ``/cluster/.../nnUNet_preprocessed``).
        dataset_id: Dataset ID.

    Returns:
        Path to the new isolated preprocessed root (to use as
        ``nnUNet_preprocessed`` env var).
    """
    # Find original dataset dir
    original_dataset_dir = _find_preprocessed_dir(nnunet_preprocessed, dataset_id)
    dataset_dirname = os.path.basename(original_dataset_dir)

    # Create isolated parent: nnUNet_preprocessed_{experiment_name}/
    isolated_root = f"{nnunet_preprocessed}_{experiment_name}"
    isolated_dataset_dir = os.path.join(isolated_root, dataset_dirname)
    os.makedirs(isolated_dataset_dir, exist_ok=True)

    # Symlink all contents from original EXCEPT splits_final.json
    for entry in os.listdir(original_dataset_dir):
        src = os.path.join(original_dataset_dir, entry)
        dst = os.path.join(isolated_dataset_dir, entry)

        if entry == 'splits_final.json':
            continue  # We'll write our own

        # Race-safe symlink creation: another fold of the same experiment
        # (different SLURM job) may create this symlink between our
        # lexists check below and this call. Suppressing
        # FileExistsError is safe — every fold of one experiment maps the
        # same src→dst (the target is determined by the entry name, not by
        # the calling job). Always validate the resulting target so a stale
        # isolated directory cannot silently reuse another dataset cache.
        if not os.path.lexists(dst):
            with contextlib.suppress(FileExistsError):
                os.symlink(src, dst)
        if not os.path.islink(dst):
            raise RuntimeError(
                f"Isolated preprocessing entry is not a symlink: {dst}"
            )
        if os.path.realpath(dst) != os.path.realpath(src):
            raise RuntimeError(
                "Isolated preprocessing entry points to stale data: "
                f"{dst} -> {os.path.realpath(dst)}; expected {src}"
            )

    # Multiple folds of one experiment can start at once. They all write the
    # same split content, but a direct write can still expose a truncated JSON
    # file to another process. Write privately and publish atomically.
    splits_path = os.path.join(isolated_dataset_dir, 'splits_final.json')
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=isolated_dataset_dir,
            prefix='.splits_final.',
            suffix='.tmp',
            delete=False,
        ) as f:
            temporary_path = f.name
            json.dump(splits, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, splits_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_path)

    # Verify the complete content, not only case counts.
    with open(splits_path) as f:
        readback = json.load(f)
    assert readback == splits, (
        f"Split verification failed for {splits_path}. Another process may be "
        "using the same experiment name with different splits."
    )

    # Log summary
    for i, fold in enumerate(splits):
        logger.info(
            f"  Fold {i}: {len(fold['train'])} train, {len(fold['val'])} val"
        )
    logger.info(f"Isolated preprocessed dir: {isolated_dataset_dir}")

    return isolated_root


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
    parser.add_argument('--synthetic-manifest', default=None,
                        help='Ordered complete synthetic-case manifest; selects prefixes')
    parser.add_argument('--seed', type=int, default=FOLD_SEED,
                        help=f'Fold generation seed (default: {FOLD_SEED})')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    case_info = _load_case_info(args.nnunet_raw, args.dataset_id)
    synthetic_order = None
    if args.synthetic_manifest is not None:
        synthetic_order = _load_synthetic_manifest(
            args.synthetic_manifest,
            case_info['synthetic_cases'],
        )

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
        synthetic_order=synthetic_order,
    )

    # Use the per-experiment isolated preprocessed dir to avoid the splits_final.json
    # race condition that bit us before (see docs/common-pitfalls.md).
    path = create_isolated_preprocessed_dir(
        experiment_name=args.experiment,
        splits=splits,
        nnunet_preprocessed=args.nnunet_preprocessed,
        dataset_id=args.dataset_id,
    )
    logger.info(f"Installed isolated preprocessed dir: {path}")


if __name__ == '__main__':
    main()
