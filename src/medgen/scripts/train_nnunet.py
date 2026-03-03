"""Training entry point for nnU-Net v2 with TensorBoard logging.

Uses argparse (NOT Hydra) — nnU-Net has its own config system.

Workflow per experiment:
    1. Set nnU-Net env vars (shared raw + preprocessed, per-experiment results)
    2. Install experiment-specific splits_final.json
    3. Register custom TensorBoard trainer
    4. Run nnU-Net training for specified fold(s)

Usage:
    # Train baseline (real-only), all 5 folds
    python -m medgen.scripts.train_nnunet \
        --experiment baseline \
        --nnunet-base /cluster/work/modestas/nnunet

    # Train mixed with 210 synthetic, fold 0 only
    python -m medgen.scripts.train_nnunet \
        --experiment mixed --n-synthetic 210 \
        --fold 0 \
        --nnunet-base /cluster/work/modestas/nnunet

    # Continue interrupted training
    python -m medgen.scripts.train_nnunet \
        --experiment baseline --fold 0 \
        --nnunet-base /cluster/work/modestas/nnunet \
        --continue-training
"""
import argparse
import os
import shutil


def _setup_env(nnunet_base: str, nnunet_results: str, experiment_name: str) -> None:
    """Set nnU-Net environment variables.

    Raw and preprocessed are under nnunet_base (dataset storage).
    Results are under nnunet_results (per-experiment).
    """
    os.environ['nnUNet_raw'] = os.path.join(nnunet_base, 'nnUNet_raw')  # noqa: SIM112
    os.environ['nnUNet_preprocessed'] = os.path.join(nnunet_base, 'nnUNet_preprocessed')  # noqa: SIM112
    os.environ['nnUNet_results'] = os.path.join(  # noqa: SIM112
        nnunet_results, experiment_name,
    )

    for key in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        os.makedirs(os.environ[key], exist_ok=True)
        print(f"  {key}={os.environ[key]}")


def _register_trainer() -> None:
    """Register custom TensorBoard trainer into nnunetv2 package.

    Symlinks our trainer.py into nnunetv2/training/nnUNetTrainer/ so that
    nnU-Net's class discovery finds it. Falls back to copy if symlink fails.
    """
    import nnunetv2.training.nnUNetTrainer as trainer_pkg

    trainer_dir = os.path.dirname(trainer_pkg.__file__)
    our_trainer = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'downstream', 'nnunet', 'trainer.py',
    )
    target = os.path.join(trainer_dir, 'nnUNetTrainerTensorBoard.py')

    if os.path.exists(target):
        if os.path.islink(target) and os.readlink(target) == our_trainer:
            print(f"  Trainer already registered: {target}")
            return
        os.remove(target)

    try:
        os.symlink(our_trainer, target)
        print(f"  Symlinked trainer: {target} -> {our_trainer}")
    except OSError:
        shutil.copy2(our_trainer, target)
        print(f"  Copied trainer to: {target}")


def _get_experiment_name(experiment: str, n_synthetic: int | None) -> str:
    """Build a unique experiment name for results directory."""
    if experiment == 'baseline':
        return 'exp3_baseline'
    elif experiment == 'baseline_dual':
        return 'exp4_baseline_dual'
    elif experiment == 'synthetic':
        if n_synthetic is not None:
            return f'exp6_synthetic_{n_synthetic}'
        return 'exp6_synthetic'
    elif experiment == 'mixed':
        n = n_synthetic if n_synthetic is not None else 525
        return f'exp7_mixed_{n}syn'
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train nnU-Net with TensorBoard logging',
    )
    parser.add_argument('--experiment', required=True,
                        choices=['baseline', 'baseline_dual', 'synthetic', 'mixed'],
                        help='Experiment type')
    parser.add_argument('--n-synthetic', type=int, default=None,
                        help='Number of synthetic samples for mixed (default: all 525)')
    parser.add_argument('--dataset-id', type=int, default=501,
                        help='Dataset ID (default: 501)')
    parser.add_argument('--configuration', default='3d_fullres',
                        choices=['2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'])
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold number 0-4 (default: all 5 folds)')
    parser.add_argument('--nnunet-base', required=True,
                        help='Base directory for nnU-Net data (raw + preprocessed)')
    parser.add_argument('--nnunet-results', required=True,
                        help='Base directory for nnU-Net results/checkpoints')
    parser.add_argument('--trainer', default='nnUNetTrainerTensorBoard',
                        help='Trainer class name')
    parser.add_argument('--plans', default='nnUNetPlans',
                        help='Plans identifier')
    parser.add_argument('--continue-training', action='store_true',
                        help='Continue from last checkpoint')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu', 'mps'])

    args = parser.parse_args()

    experiment_name = _get_experiment_name(args.experiment, args.n_synthetic)
    folds = [args.fold] if args.fold is not None else list(range(5))

    print("=== nnU-Net Training Setup ===")
    print(f"  Experiment: {experiment_name}")
    print(f"  Folds: {folds}")

    # 1. Set environment variables (per-experiment results dir)
    print("\nEnvironment:")
    _setup_env(args.nnunet_base, args.nnunet_results, experiment_name)

    # 2. Install experiment-specific splits
    print("\nInstalling splits:")
    from medgen.downstream.nnunet.splits import (
        _load_case_info,
        generate_experiment_splits,
        install_splits,
    )

    case_info = _load_case_info(os.environ['nnUNet_raw'], args.dataset_id)  # noqa: SIM112
    splits = generate_experiment_splits(
        experiment=args.experiment,
        real_train_cases=case_info['real_train_cases'],
        synthetic_cases=case_info['synthetic_cases'],
        n_synthetic=args.n_synthetic,
    )
    install_splits(splits, os.environ['nnUNet_preprocessed'], args.dataset_id)  # noqa: SIM112

    # 3. Register custom trainer
    print("\nRegistering trainer:")
    _register_trainer()

    # 4. Run training for each fold
    from nnunetv2.run.run_training import run_training

    for fold in folds:
        print(f"\n{'='*50}")
        print(f"=== Training fold {fold} ===")
        print(f"  Train: {len(splits[fold]['train'])} cases")
        print(f"  Val:   {len(splits[fold]['val'])} cases")
        print(f"{'='*50}\n")

        run_training(
            dataset_name_or_id=args.dataset_id,
            configuration=args.configuration,
            fold=fold,
            trainer_name=args.trainer,
            plans_identifier=args.plans,
            num_gpus=1,
            pretrained_weights=None,
            use_compressed_data=False,
            export_validation_probabilities=False,
            continue_training=args.continue_training,
            only_run_validation=False,
            disable_checkpointing=False,
            device=args.device,
        )

    print(f"\n=== Training complete: {experiment_name} ===")


if __name__ == '__main__':
    main()
