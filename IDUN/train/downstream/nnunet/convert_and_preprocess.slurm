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

# Convert real + two synthetic datasets to nnU-Net format and preprocess.
#
# Dataset 501: BRAVO real + ImageNet-optimized synthetic (32 steps, shift 1.0)
# Dataset 502: BRAVO real + RadImageNet-optimized synthetic (79 steps, shift 1.0)
#
# Both share the same real data (105 train, 51 test).
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
SYNTHETIC_IMAGENET="/cluster/work/modestas/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"
SYNTHETIC_RADIMAGENET="/cluster/work/modestas/MedicalDataSets/generated/exp1_1_bravo_radimagenet_525"

export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_RESULTS}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

# === Verify synthetic data exists ===
for DIR in "${SYNTHETIC_IMAGENET}" "${SYNTHETIC_RADIMAGENET}"; do
    N=$(find "${DIR}" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | wc -l)
    echo "Synthetic: ${DIR} — ${N} volumes"
    if [ "$N" -lt 525 ]; then
        echo "WARNING: Expected 525 volumes, found ${N}. Generation may still be running."
    fi
done

# === Clean up ALL previous datasets ===
echo ""
echo "=== Cleaning previous nnU-Net data ==="
rm -rf "${nnUNet_raw}/Dataset501_BrainMet"
rm -rf "${nnUNet_raw}/Dataset502_BrainMet"
rm -rf "${nnUNet_preprocessed}/Dataset501_BrainMet"
rm -rf "${nnUNet_preprocessed}/Dataset502_BrainMet"
echo "Deleted Dataset501 and Dataset502 (raw + preprocessed)"

# === Step 1: Convert to nnU-Net format ===
echo ""
echo "=========================================="
echo "  Step 1: Converting datasets"
echo "=========================================="

# Dataset 501: BRAVO + ImageNet-optimized synthetic
echo ""
echo "--- Dataset 501: BRAVO + ImageNet synthetic ---"
python -m medgen.downstream.nnunet.convert_dataset \
    --real-dir "${REAL_DIR}" \
    --synthetic-dir "${SYNTHETIC_IMAGENET}" \
    --nnunet-raw "${nnUNet_raw}" \
    --dataset-id 501 \
    --modality bravo

# Dataset 502: BRAVO + RadImageNet-optimized synthetic
echo ""
echo "--- Dataset 502: BRAVO + RadImageNet synthetic ---"
python -m medgen.downstream.nnunet.convert_dataset \
    --real-dir "${REAL_DIR}" \
    --synthetic-dir "${SYNTHETIC_RADIMAGENET}" \
    --nnunet-raw "${nnUNet_raw}" \
    --dataset-id 502 \
    --modality bravo

# === Step 2: Plan and preprocess ===
echo ""
echo "=========================================="
echo "  Step 2: nnU-Net plan and preprocess"
echo "=========================================="

# ResEnc L planner (residual encoder, better for small lesions)
echo "--- Dataset 501 (ImageNet) ---"
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -c 3d_fullres -np 4
nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncL -c 3d_fullres -np 4

echo "--- Dataset 502 (RadImageNet) ---"
nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity -c 3d_fullres -np 4
nnUNetv2_plan_and_preprocess -d 502 -pl nnUNetPlannerResEncL -c 3d_fullres -np 4

# === Step 3: Validate ===
echo ""
echo "=========================================="
echo "  Step 3: Validation"
echo "=========================================="

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

    label = {501: 'ImageNet', 502: 'RadImageNet'}[did]
    print(f'=== Dataset {did} ({label}) ===')

    for plans_name in ['nnUNetPlans.json', 'nnUNetResEncUNetLPlans.json']:
        plans_path = os.path.join(pp_dir, plans_name)
        if not os.path.exists(plans_path):
            continue
        with open(plans_path) as f:
            plans = json.load(f)
        print(f'  Plans: {plans_name}')
        for cfg_name, cfg in plans.get('configurations', {}).items():
            print(f'    {cfg_name}: patch={cfg.get(\"patch_size\")}, '
                  f'batch={cfg.get(\"batch_size\")}, spacing={cfg.get(\"spacing\")}')

    for config_dir in os.listdir(pp_dir):
        config_path = os.path.join(pp_dir, config_dir)
        if not os.path.isdir(config_path):
            continue
        npz_files = sorted(f for f in os.listdir(config_path) if f.endswith('.npz'))
        if not npz_files:
            continue
        empty_seg = sum(1 for f in npz_files
                        if (np.load(os.path.join(config_path, f)).get('seg', np.array([1])) > 0).sum() == 0)
        print(f'  {config_dir}: {len(npz_files)} samples, {empty_seg} empty segs')
    print()
"

echo ""
echo "=== Done! Ready for training ==="
echo "Dataset 501: real + ImageNet synthetic"
echo "Dataset 502: real + RadImageNet synthetic"
