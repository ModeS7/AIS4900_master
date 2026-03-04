"""Diagnose nnU-Net training issues.

Run on cluster to check labels, preprocessing, plans, and training logs.

Usage:
    python misc/diagnose_nnunet.py \
        --nnunet-base /cluster/work/modestas/MedicalDataSets/nnunet \
        --nnunet-results /cluster/work/modestas/AIS4900_master/runs/downstream/nnunet
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np


def check_raw_labels(nnunet_raw: str, dataset_id: int = 501) -> None:
    """Check label values in the raw nnU-Net dataset."""
    print("\n=== 1. RAW LABELS ===")
    dataset_dir = None
    for entry in os.listdir(nnunet_raw):
        if entry.startswith(f'Dataset{dataset_id}_'):
            dataset_dir = os.path.join(nnunet_raw, entry)
            break
    if not dataset_dir:
        print(f"ERROR: No dataset for ID {dataset_id}")
        return

    labels_dir = os.path.join(dataset_dir, 'labelsTr')
    if not os.path.isdir(labels_dir):
        print(f"ERROR: No labelsTr directory")
        return

    label_files = sorted(f for f in os.listdir(labels_dir) if f.endswith('.nii.gz'))
    print(f"Found {len(label_files)} label files")

    # Separate real vs synthetic
    real_labels = [f for f in label_files if not f.startswith('BrainMetSyn')]
    syn_labels = [f for f in label_files if f.startswith('BrainMetSyn')]
    print(f"  Real: {len(real_labels)}, Synthetic: {len(syn_labels)}")

    # Check first 3 real + first 2 synthetic
    check_labels = real_labels[:3] + syn_labels[:2]
    for fname in check_labels:
        path = os.path.join(labels_dir, fname)
        nii = nib.load(path)

        # Check header
        hdr = nii.header
        slope, inter = hdr.get_slope_inter()
        dtype = hdr.get_data_dtype()

        # Load data
        data = nii.get_fdata()
        raw_data = np.asanyarray(nii.dataobj)

        print(f"\n  {fname}:")
        print(f"    Header dtype: {dtype}, slope={slope}, inter={inter}")
        print(f"    Raw data: dtype={raw_data.dtype}, unique={np.unique(raw_data)}")
        print(f"    get_fdata(): dtype={data.dtype}, unique={np.unique(data)}")
        print(f"    Shape: {data.shape}")
        print(f"    Tumor voxels (>0.5): {(data > 0.5).sum()} / {data.size} "
              f"({100*(data > 0.5).sum()/data.size:.4f}%)")

    # Check images too
    images_dir = os.path.join(dataset_dir, 'imagesTr')
    if os.path.isdir(images_dir):
        img_files = sorted(f for f in os.listdir(images_dir) if f.endswith('.nii.gz'))
        print(f"\n  Images: {len(img_files)} files")
        # Check first image
        if img_files:
            path = os.path.join(images_dir, img_files[0])
            if os.path.islink(path):
                print(f"    {img_files[0]}: symlink -> {os.readlink(path)}")
            nii = nib.load(path)
            data = nii.get_fdata()
            print(f"    Shape: {data.shape}, dtype: {data.dtype}")
            print(f"    Range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"    Mean: {data.mean():.4f}, Std: {data.std():.4f}")


def check_preprocessed(nnunet_preprocessed: str, dataset_id: int = 501) -> None:
    """Check preprocessed data."""
    print("\n=== 2. PREPROCESSED DATA ===")
    pp_dir = None
    for entry in os.listdir(nnunet_preprocessed):
        if entry.startswith(f'Dataset{dataset_id}_'):
            pp_dir = os.path.join(nnunet_preprocessed, entry)
            break
    if not pp_dir:
        print(f"ERROR: No preprocessed dir for Dataset{dataset_id}")
        return

    print(f"Preprocessed dir: {pp_dir}")
    print(f"Contents: {sorted(os.listdir(pp_dir))}")

    # Check splits
    splits_path = os.path.join(pp_dir, 'splits_final.json')
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
        print(f"\nSplits: {len(splits)} folds")
        for i, fold in enumerate(splits):
            train_cases = fold['train']
            val_cases = fold['val']
            n_real = sum(1 for c in train_cases if not c.startswith('BrainMetSyn'))
            n_syn = sum(1 for c in train_cases if c.startswith('BrainMetSyn'))
            print(f"  Fold {i}: {len(train_cases)} train ({n_real} real + {n_syn} syn), "
                  f"{len(val_cases)} val")
    else:
        print("WARNING: No splits_final.json found!")

    # Check gt_segmentations (nnU-Net stores labels separately)
    gt_dir = os.path.join(pp_dir, 'gt_segmentations')
    if os.path.isdir(gt_dir):
        gt_files = sorted(f for f in os.listdir(gt_dir) if f.endswith('.nii.gz'))
        print(f"\n  gt_segmentations: {len(gt_files)} files")
        n_empty = 0
        for fname in gt_files[:5]:
            path = os.path.join(gt_dir, fname)
            nii = nib.load(path)
            data = np.asanyarray(nii.dataobj)
            tumor_count = (data > 0).sum()
            if tumor_count == 0:
                n_empty += 1
            print(f"    {fname}: shape={data.shape}, dtype={data.dtype}, "
                  f"unique={np.unique(data)}, tumor_voxels={tumor_count}")
        # Count total empty
        for fname in gt_files[5:]:
            data = np.asanyarray(nib.load(os.path.join(gt_dir, fname)).dataobj)
            if (data > 0).sum() == 0:
                n_empty += 1
        print(f"  Empty labels (no tumor): {n_empty}/{len(gt_files)}")
    else:
        print("\n  WARNING: No gt_segmentations directory!")

    # Check dataset_fingerprint.json
    fp_path = os.path.join(pp_dir, 'dataset_fingerprint.json')
    if os.path.exists(fp_path):
        with open(fp_path) as f:
            fp = json.load(f)
        print(f"\n  Dataset fingerprint:")
        for k, v in fp.items():
            print(f"    {k}: {json.dumps(v)[:120]}")

    # Check preprocessed samples (npz/npy files)
    for config_dir_name in os.listdir(pp_dir):
        config_path = os.path.join(pp_dir, config_dir_name)
        if not os.path.isdir(config_path) or config_dir_name.startswith('.'):
            continue
        npz_files = [f for f in os.listdir(config_path)
                     if f.endswith('.npz') or f.endswith('.npy')]
        if not npz_files:
            continue

        print(f"\n  Config: {config_dir_name} ({len(npz_files)} samples)")

        # Check first 3 preprocessed samples
        for fname in sorted(npz_files)[:3]:
            path = os.path.join(config_path, fname)
            try:
                d = np.load(path)
                data = d['data']  # [C, D, H, W] or [C, H, W]
                seg = d['seg'] if 'seg' in d else None
                print(f"\n    {fname}:")
                print(f"      data: shape={data.shape}, dtype={data.dtype}, "
                      f"range=[{data.min():.4f}, {data.max():.4f}]")
                if seg is not None:
                    print(f"      seg:  shape={seg.shape}, dtype={seg.dtype}, "
                          f"unique={np.unique(seg)}")
                    tumor_pct = 100 * (seg > 0).sum() / seg.size
                    print(f"      Tumor voxels: {(seg > 0).sum()} ({tumor_pct:.4f}%)")
                else:
                    print("      seg: NOT FOUND IN NPZ!")
            except Exception as e:
                print(f"    {fname}: ERROR loading: {e}")


def check_plans(nnunet_preprocessed: str, dataset_id: int = 501) -> None:
    """Check nnU-Net plans."""
    print("\n=== 3. NNUNET PLANS ===")
    pp_dir = None
    for entry in os.listdir(nnunet_preprocessed):
        if entry.startswith(f'Dataset{dataset_id}_'):
            pp_dir = os.path.join(nnunet_preprocessed, entry)
            break
    if not pp_dir:
        return

    plans_path = os.path.join(pp_dir, 'nnUNetPlans.json')
    if not os.path.exists(plans_path):
        print("ERROR: No nnUNetPlans.json found!")
        return

    with open(plans_path) as f:
        plans = json.load(f)

    # Show key configuration
    for config_name, config in plans.get('configurations', {}).items():
        print(f"\n  Configuration: {config_name}")
        print(f"    Patch size:    {config.get('patch_size')}")
        print(f"    Batch size:    {config.get('batch_size')}")
        print(f"    Spacing:       {config.get('spacing')}")
        print(f"    Median shape:  {config.get('median_image_size_in_voxels')}")
        print(f"    Architecture:  {config.get('UNet_class_name')}")

        arch = config.get('architecture', {})
        if arch:
            print(f"    Network arch:")
            print(f"      n_stages:    {arch.get('n_stages')}")
            print(f"      features:    {arch.get('features_per_stage')}")
            print(f"      kernel_sizes: {arch.get('kernel_sizes')}")

    # Dataset fingerprint
    fp = plans.get('dataset_fingerprint', {})
    if fp:
        print(f"\n  Dataset fingerprint:")
        for k, v in fp.items():
            if isinstance(v, (list, dict)):
                print(f"    {k}: {json.dumps(v)[:100]}")
            else:
                print(f"    {k}: {v}")


def check_training_logs(nnunet_results: str, dataset_id: int = 501) -> None:
    """Check training logs for exp3_baseline."""
    print("\n=== 4. TRAINING LOGS ===")
    exp_dir = os.path.join(nnunet_results, 'exp3_baseline')
    if not os.path.isdir(exp_dir):
        print(f"ERROR: No exp3_baseline dir at {exp_dir}")
        return

    # Find dataset dir
    for entry in os.listdir(exp_dir):
        if entry.startswith(f'Dataset{dataset_id}_'):
            model_base = os.path.join(exp_dir, entry)
            break
    else:
        print(f"ERROR: No Dataset{dataset_id} dir in {exp_dir}")
        return

    # Check each trainer/plans/config combo
    for trainer_dir_name in os.listdir(model_base):
        trainer_path = os.path.join(model_base, trainer_dir_name)
        if not os.path.isdir(trainer_path):
            continue

        print(f"\n  Trainer: {trainer_dir_name}")

        for fold_name in sorted(os.listdir(trainer_path)):
            fold_path = os.path.join(trainer_path, fold_name)
            if not os.path.isdir(fold_path) or not fold_name.startswith('fold_'):
                continue

            print(f"\n    {fold_name}:")

            # Check training log
            log_files = [f for f in os.listdir(fold_path) if f.startswith('training_log')]
            for log_file in sorted(log_files):
                log_path = os.path.join(fold_path, log_file)
                with open(log_path) as f:
                    lines = f.readlines()

                print(f"      Log: {log_file} ({len(lines)} lines)")

                # Show last 20 lines (most recent epochs)
                print("      Last 20 lines:")
                for line in lines[-20:]:
                    print(f"        {line.rstrip()}")

            # Check checkpoints
            ckpts = [f for f in os.listdir(fold_path)
                     if f.endswith('.pth') or f.endswith('.pt')]
            print(f"      Checkpoints: {sorted(ckpts)}")


def check_registered_trainer(nnunet_results: str) -> None:
    """Check what trainer code is actually being used."""
    print("\n=== 5. REGISTERED TRAINER ===")
    try:
        import nnunetv2.training.nnUNetTrainer as trainer_pkg
        trainer_dir = os.path.dirname(trainer_pkg.__file__)
        target = os.path.join(trainer_dir, 'nnUNetTrainerTensorBoard.py')

        if os.path.exists(target):
            if os.path.islink(target):
                print(f"  Symlink: {target}")
                print(f"  Points to: {os.readlink(target)}")
            else:
                print(f"  File (NOT symlink): {target}")
                print(f"  Size: {os.path.getsize(target)} bytes")

            # Check if it has bf16
            with open(target) as f:
                content = f.read()
            if 'bfloat16' in content:
                print("  WARNING: Trainer still contains bf16 code!")
            elif 'train_step' in content and 'def train_step' in content:
                print("  WARNING: Trainer still overrides train_step!")
            else:
                print("  OK: No train_step/validation_step override, no bf16")
        else:
            print(f"  NOT FOUND: {target}")
    except ImportError:
        print("  Cannot import nnunetv2 (not installed?)")


def main():
    parser = argparse.ArgumentParser(description='Diagnose nnU-Net training issues')
    parser.add_argument('--nnunet-base', required=True,
                        help='Base dir (contains nnUNet_raw, nnUNet_preprocessed)')
    parser.add_argument('--nnunet-results', required=True,
                        help='Results dir')
    parser.add_argument('--dataset-id', type=int, default=501)
    args = parser.parse_args()

    nnunet_raw = os.path.join(args.nnunet_base, 'nnUNet_raw')
    nnunet_preprocessed = os.path.join(args.nnunet_base, 'nnUNet_preprocessed')

    print("=" * 60)
    print("nnU-Net Diagnostic Report")
    print("=" * 60)

    check_raw_labels(nnunet_raw, args.dataset_id)
    check_preprocessed(nnunet_preprocessed, args.dataset_id)
    check_plans(nnunet_preprocessed, args.dataset_id)
    check_training_logs(args.nnunet_results, args.dataset_id)
    check_registered_trainer(args.nnunet_results)

    print("\n" + "=" * 60)
    print("Done. Look for:")
    print("  - Label values != {0, 1} (data corruption)")
    print("  - seg NOT FOUND in preprocessed npz (missing labels)")
    print("  - Tumor voxels = 0% (empty labels)")
    print("  - Dice stuck near 0 in training log (model not learning)")
    print("  - LR not decreasing (scheduler issue)")
    print("  - bf16 or train_step override in trainer (stale code)")
    print("=" * 60)


if __name__ == '__main__':
    main()
