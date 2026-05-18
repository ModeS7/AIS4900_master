"""Standalone per-fold nnU-Net inference (no medgen import).

Runs the 51 test cases through each of the 5 trained fold models individually
(no ensembling). Writes per-fold softmax .npz files for downstream threshold
analysis. Bypasses `medgen.scripts.eval_nnunet` because the medgen package
__init__ eagerly imports the diffusion pipeline (ema_pytorch, MONAI plugins,
etc.) which we don't need and which isn't installed in the nnunet venv.

Usage:  .venv_nnunet/bin/python misc/run_per_fold_local.py
"""
from __future__ import annotations

import io
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NNUNET_BASE = os.path.join(REPO, 'data', 'nnunet_local')
NNUNET_RAW = os.path.join(NNUNET_BASE, 'nnUNet_raw')
NNUNET_RESULTS = os.path.join(REPO, 'runs')
EXPERIMENT_NAME = 'exp3_baseline_v2_d600'
DATASET_NAME = 'Dataset600_BrainMet'
CONFIG = 'nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres'

MODEL_DIR = os.path.join(NNUNET_RESULTS, EXPERIMENT_NAME, DATASET_NAME, CONFIG)
INPUT_DIR = os.path.join(NNUNET_RAW, DATASET_NAME, 'imagesTs')
PER_FOLD_DIR = os.path.join(NNUNET_RESULTS, EXPERIMENT_NAME,
                            f'eval_{EXPERIMENT_NAME}', 'per_fold_test')

# nnU-Net env vars are required even though we only do inference. nnU-Net
# also writes some intermediate state to nnUNet_preprocessed at predict time
# in some configurations; setting it to a writable temp path is safest.
os.environ.setdefault('nnUNet_raw', NNUNET_RAW)
os.environ.setdefault('nnUNet_preprocessed',
                      os.path.join(NNUNET_BASE, 'nnUNet_preprocessed'))
os.environ.setdefault('nnUNet_results',
                      os.path.join(NNUNET_RESULTS, EXPERIMENT_NAME))
os.makedirs(os.environ['nnUNet_preprocessed'], exist_ok=True)

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402


def run_fold(fold: int) -> None:
    out_dir = os.path.join(PER_FOLD_DIR, f'fold_{fold}', 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    n_existing = sum(1 for f in os.listdir(out_dir) if f.endswith('.npz'))
    if n_existing == 51:
        print(f'=== Fold {fold}: already complete (51 .npz), skipping ===')
        return

    print(f'=== Fold {fold} → {out_dir} ===')
    t0 = time.time()

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose=False,
        verbose_preprocessing=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=MODEL_DIR,
        use_folds=(fold,),
        checkpoint_name='checkpoint_best.pth',
    )

    # Suppress nnU-Net's per-case print spam to keep the log readable.
    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        predictor.predict_from_files(
            list_of_lists_or_source_folder=INPUT_DIR,
            output_folder_or_list_of_truncated_output_files=out_dir,
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=4,
            num_processes_segmentation_export=4,
        )
    finally:
        sys.stdout = _stdout

    elapsed = time.time() - t0
    n_npz = sum(1 for f in os.listdir(out_dir) if f.endswith('.npz'))
    n_nii = sum(1 for f in os.listdir(out_dir) if f.endswith('.nii.gz'))
    print(f'    {n_npz} .npz + {n_nii} .nii.gz produced in {elapsed:.1f}s')


def main() -> None:
    print('=== Per-fold local inference: '
          + EXPERIMENT_NAME + ' ===')
    print(f'  Model: {MODEL_DIR}')
    print(f'  Input: {INPUT_DIR}')
    print(f'  Output base: {PER_FOLD_DIR}')
    print(f'  Device: {"cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}')
    print('')
    for fold in range(5):
        run_fold(fold)
        print('')

    print('=== All folds complete ===')
    for fold in range(5):
        out = os.path.join(PER_FOLD_DIR, f'fold_{fold}', 'predictions')
        n = sum(1 for f in os.listdir(out) if f.endswith('.npz')) if os.path.isdir(out) else 0
        print(f'  fold_{fold}: {n} .npz')


if __name__ == '__main__':
    main()
