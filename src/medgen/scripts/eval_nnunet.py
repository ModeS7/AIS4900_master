"""Evaluation entry point for nnU-Net predictions.

Runs nnU-Net 5-fold ensemble inference on the 51-patient test set,
then evaluates using MedGen metrics.

Usage:
    # Run inference + evaluation for baseline experiment
    python -m medgen.scripts.eval_nnunet \
        --experiment baseline \
        --nnunet-base /cluster/work/modestas/nnunet

    # Evaluate existing predictions (skip inference)
    python -m medgen.scripts.eval_nnunet \
        --pred-dir /path/to/predictions \
        --gt-dir /path/to/labelsTs \
        --output results.json
"""
import argparse
import os

import torch


def _setup_env(nnunet_base: str, nnunet_results: str, experiment_name: str) -> None:
    """Set nnU-Net environment variables.

    Points nnUNet_preprocessed to the per-experiment isolated dir if it exists
    (created by train_nnunet.py), otherwise falls back to the shared dir.
    """
    os.environ['nnUNet_raw'] = os.path.join(nnunet_base, 'nnUNet_raw')  # noqa: SIM112

    # Use isolated preprocessed dir if it exists (created during training)
    isolated = os.path.join(nnunet_base, f'nnUNet_preprocessed_{experiment_name}')
    if os.path.isdir(isolated):
        os.environ['nnUNet_preprocessed'] = isolated  # noqa: SIM112
    else:
        os.environ['nnUNet_preprocessed'] = os.path.join(nnunet_base, 'nnUNet_preprocessed')  # noqa: SIM112

    os.environ['nnUNet_results'] = os.path.join(  # noqa: SIM112
        nnunet_results, experiment_name,
    )


def _find_dataset_dir(nnunet_raw: str, dataset_id: int) -> str:
    """Find the dataset directory by ID."""
    for entry in os.listdir(nnunet_raw):
        if entry.startswith(f'Dataset{dataset_id}_'):
            return os.path.join(nnunet_raw, entry)
    raise FileNotFoundError(f"No dataset found for ID {dataset_id} in {nnunet_raw}")


def _get_experiment_name(experiment: str, n_synthetic: int | None) -> str:
    """Build experiment name matching train_nnunet.py convention."""
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


def _find_model_dir(
    dataset_id: int,
    configuration: str,
    trainer: str,
    plans: str,
) -> str:
    """Find the nnU-Net model directory for a dataset."""
    results_dir = os.environ['nnUNet_results']  # noqa: SIM112
    for entry in os.listdir(results_dir):
        if entry.startswith(f'Dataset{dataset_id}_'):
            model_dir = os.path.join(
                results_dir, entry,
                f'{trainer}__{plans}__{configuration}',
            )
            if os.path.isdir(model_dir):
                return model_dir
    raise FileNotFoundError(
        f"No trained model found for Dataset{dataset_id} with "
        f"{trainer}__{plans}__{configuration} in {results_dir}"
    )


def _run_inference(
    dataset_id: int,
    configuration: str,
    trainer: str,
    plans: str,
    output_dir: str,
    input_dir: str,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> str:
    """Run nnU-Net 5-fold ensemble inference on test images.

    Returns:
        Path to predictions directory.
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    model_dir = _find_model_dir(dataset_id, configuration, trainer, plans)

    # Auto-detect available folds (only use folds that have checkpoint_best.pth)
    available_folds = tuple(
        f for f in folds
        if os.path.isfile(os.path.join(model_dir, f'fold_{f}', 'checkpoint_best.pth'))
    )
    if not available_folds:
        raise FileNotFoundError(
            f"No trained folds found with checkpoint_best.pth in {model_dir}"
        )
    if available_folds != folds:
        print(f"Note: requested folds {folds}, but only {available_folds} have checkpoints")
    folds = available_folds

    print(f"Model directory: {model_dir}")
    print(f"Folds: {folds}")
    print(f"Input: {input_dir}")
    print(f"Output: {pred_dir}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose=False,
        verbose_preprocessing=False,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_dir,
        use_folds=folds,
        checkpoint_name='checkpoint_best.pth',
    )

    # Suppress nnU-Net's hard-coded print() statements during inference
    import io
    import sys
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        predictor.predict_from_files(
            list_of_lists_or_source_folder=input_dir,
            output_folder_or_list_of_truncated_output_files=pred_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=4,
            num_processes_segmentation_export=4,
        )
    finally:
        sys.stdout = _stdout

    print(f"Inference complete: {len(os.listdir(pred_dir))} predictions")
    return pred_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate nnU-Net predictions with MedGen metrics',
    )

    # Mode 1: Run inference + evaluate
    parser.add_argument('--experiment',
                        choices=['baseline', 'baseline_dual', 'synthetic', 'mixed'],
                        help='Experiment type (runs inference + eval)')
    parser.add_argument('--experiment-name', default=None,
                        help='Override auto-generated experiment name (results subdir)')
    parser.add_argument('--n-synthetic', type=int, default=None,
                        help='Number of synthetic samples (for mixed)')
    parser.add_argument('--dataset-id', type=int, default=501)
    parser.add_argument('--configuration', default='3d_fullres')
    parser.add_argument('--nnunet-base',
                        help='Base directory for nnU-Net data (raw + preprocessed)')
    parser.add_argument('--nnunet-results',
                        help='Base directory for nnU-Net results/checkpoints')
    parser.add_argument('--trainer', default='nnUNetTrainerTensorBoard')
    parser.add_argument('--plans', default='nnUNetPlans')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help='Folds to ensemble (default: all 5)')

    # Mode 2: Evaluate existing predictions
    parser.add_argument('--pred-dir',
                        help='Directory with existing predictions (skip inference)')
    parser.add_argument('--gt-dir',
                        help='Directory with ground truth labels')

    # Common
    parser.add_argument('--output', default=None,
                        help='Output JSON path for results')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--fov-mm', type=float, default=240.0)

    args = parser.parse_args()

    if args.pred_dir and args.gt_dir:
        # Mode 2: evaluate existing predictions
        pred_dir = args.pred_dir
        gt_dir = args.gt_dir
    elif args.experiment and args.nnunet_base and args.nnunet_results:
        # Mode 1: run inference then evaluate
        experiment_name = args.experiment_name or _get_experiment_name(args.experiment, args.n_synthetic)
        _setup_env(args.nnunet_base, args.nnunet_results, experiment_name)

        nnunet_raw = os.environ['nnUNet_raw']  # noqa: SIM112
        dataset_dir = _find_dataset_dir(nnunet_raw, args.dataset_id)
        gt_dir = os.path.join(dataset_dir, 'labelsTs')
        input_dir = os.path.join(dataset_dir, 'imagesTs')

        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"No test images found: {input_dir}")

        print(f"=== nnU-Net Inference: {experiment_name} ===")
        pred_dir = _run_inference(
            dataset_id=args.dataset_id,
            configuration=args.configuration,
            trainer=args.trainer,
            plans=args.plans,
            output_dir=os.path.join(
                os.environ['nnUNet_results'], f'eval_{experiment_name}',  # noqa: SIM112
            ),
            input_dir=input_dir,
            folds=tuple(args.folds),
        )
    else:
        parser.error(
            "Provide either (--experiment + --nnunet-base + --nnunet-results) or "
            "(--pred-dir + --gt-dir)"
        )

    # Evaluate
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("\n=== Running MedGen Evaluation ===")
    from medgen.downstream.nnunet.evaluate import evaluate_predictions

    output_path = args.output
    if output_path is None and args.experiment:
        experiment_name = args.experiment_name or _get_experiment_name(args.experiment, args.n_synthetic)
        output_path = os.path.join(
            os.environ.get('nnUNet_results', '.'),  # noqa: SIM112
            f'eval_{experiment_name}.json',
        )

    # Find TensorBoard dir to log test metrics alongside training curves
    tensorboard_dir = None
    if args.experiment and args.nnunet_base:
        try:
            model_dir = _find_model_dir(
                args.dataset_id, args.configuration,
                args.trainer, args.plans,
            )
            folds = args.folds
            if len(folds) == 1:
                # Single fold: log to that fold's TB dir
                tensorboard_dir = os.path.join(
                    model_dir, f'fold_{folds[0]}', 'tensorboard',
                )
            else:
                # Multi-fold ensemble: log to eval output dir
                experiment_name = args.experiment_name or _get_experiment_name(
                    args.experiment, args.n_synthetic,
                )
                tensorboard_dir = os.path.join(
                    os.environ['nnUNet_results'],  # noqa: SIM112
                    f'eval_{experiment_name}', 'tensorboard',
                )
        except FileNotFoundError:
            pass  # No model dir found, skip TB logging

    evaluate_predictions(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        output_path=output_path,
        tensorboard_dir=tensorboard_dir,
        image_size=args.image_size,
        fov_mm=args.fov_mm,
        spatial_dims=3,
    )


if __name__ == '__main__':
    main()
