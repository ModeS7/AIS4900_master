#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nnunet_prep"
#SBATCH --output=IDUN/output/train/downstream/nnunet/prep_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/prep_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# One-time: convert dataset + preprocess for nnU-Net
# Creates ONE dataset (501) with all 105 real + 525 synthetic.
# Experiment-specific splits are installed before each training run.

set -e  # Exit on first error

cd ${SLURM_SUBMIT_DIR}

echo "=== nnU-Net Dataset Conversion & Preprocessing ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

# === Paths ===
NNUNET_BASE="/cluster/work/modestas/MedicalDataSets/nnunet"
NNUNET_RESULTS="/cluster/work/modestas/AIS4900_master/runs/downstream/nnunet"
REAL_DIR="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
SYNTHETIC_DIR="/cluster/work/modestas/MedicalDataSets/generated/exp1_1_3d_256_525_baseline"

export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_RESULTS}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

# === Step 1: Convert to nnU-Net format ===
echo ""
echo "=== Step 1: Converting dataset ==="
echo "imagesTr: 105 real train + all synthetic"
echo "imagesTs: 25 val + 26 test_new = 51 test"

# Clean up previous data
rm -rf "${nnUNet_raw}/Dataset501_BrainMet"
rm -rf "${nnUNet_raw}/Dataset502_BrainMet"

# Dataset 501: BRAVO (1 channel)
echo "--- Dataset 501: BRAVO ---"
python -m medgen.downstream.nnunet.convert_dataset \
    --real-dir "${REAL_DIR}" \
    --synthetic-dir "${SYNTHETIC_DIR}" \
    --nnunet-raw "${nnUNet_raw}" \
    --dataset-id 501 \
    --modality bravo

# Dataset 502: Dual (2 channels: t1_pre + t1_gd), real-only
echo "--- Dataset 502: Dual ---"
python -m medgen.downstream.nnunet.convert_dataset \
    --real-dir "${REAL_DIR}" \
    --nnunet-raw "${nnUNet_raw}" \
    --dataset-id 502 \
    --modality dual

# === Step 2: Validate conversion ===
echo ""
echo "=== Step 2: Validating converted datasets ==="
python misc/validate_nnunet_conversion.py \
    --dataset-dir "${nnUNet_raw}/Dataset501_BrainMet"

echo ""
python misc/validate_nnunet_conversion.py \
    --dataset-dir "${nnUNet_raw}/Dataset502_BrainMet"

# set -e will abort here if validation fails

# === Step 3: Plan and preprocess ===
echo ""
echo "=== Step 3: nnU-Net plan and preprocess ==="
rm -rf "${nnUNet_preprocessed}/Dataset501_BrainMet"
rm -rf "${nnUNet_preprocessed}/Dataset502_BrainMet"
# Use -np 4 to reduce blosc2 concurrency issues on cluster filesystem
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -c 3d_fullres -np 4
nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity -c 3d_fullres -np 4

# === Step 4: Validate preprocessed data ===
echo ""
echo "=== Step 4: Checking preprocessed data ==="
python -c "
import numpy as np, os, json

for did in [501, 502]:
    pp_base = '${nnUNet_preprocessed}'
    pp_dir = None
    for entry in os.listdir(pp_base):
        if entry.startswith(f'Dataset{did}_'):
            pp_dir = os.path.join(pp_base, entry)
            break
    if not pp_dir:
        print(f'ERROR: No preprocessed dir for Dataset{did}')
        continue

    print(f'=== Dataset {did} preprocessed ===')

    # Check plans
    plans_path = os.path.join(pp_dir, 'nnUNetPlans.json')
    if os.path.exists(plans_path):
        with open(plans_path) as f:
            plans = json.load(f)
        for cfg_name, cfg in plans.get('configurations', {}).items():
            print(f'  {cfg_name}: patch={cfg.get(\"patch_size\")}, '
                  f'batch={cfg.get(\"batch_size\")}, spacing={cfg.get(\"spacing\")}')

    # Check preprocessed npz files
    for config_dir in os.listdir(pp_dir):
        config_path = os.path.join(pp_dir, config_dir)
        if not os.path.isdir(config_path):
            continue
        npz_files = sorted(f for f in os.listdir(config_path) if f.endswith('.npz'))
        if not npz_files:
            continue

        print(f'  Config {config_dir}: {len(npz_files)} samples')
        empty_seg = 0
        for fname in npz_files:
            d = np.load(os.path.join(config_path, fname))
            seg = d.get('seg')
            if seg is None:
                print(f'    FAIL: {fname} has no seg key!')
            elif (seg > 0).sum() == 0:
                empty_seg += 1

        # Spot check first 3
        for fname in npz_files[:3]:
            d = np.load(os.path.join(config_path, fname))
            data, seg = d['data'], d['seg']
            print(f'    {fname}: data={data.shape} [{data.min():.2f},{data.max():.2f}], '
                  f'seg={seg.shape} unique={np.unique(seg)} tumor={(seg>0).sum()}')

        if empty_seg:
            print(f'    WARNING: {empty_seg}/{len(npz_files)} samples have EMPTY seg!')
        else:
            print(f'    OK: all {len(npz_files)} samples have tumor voxels')
    print()
"

echo ""
echo "=== Done! ==="
echo "Ready for: python -m medgen.scripts.train_nnunet --experiment baseline --nnunet-base ${NNUNET_BASE}"
