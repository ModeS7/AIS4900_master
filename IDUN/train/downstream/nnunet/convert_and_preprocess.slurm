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

# === Step 2: Plan and preprocess ===
echo ""
echo "=== Step 2: nnU-Net plan and preprocess ==="
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -c 3d_fullres
nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity -c 3d_fullres

echo ""
echo "=== Done! ==="
echo "Preprocessed once. Splits are installed per-experiment by train_nnunet.py."
echo "Ready for: python -m medgen.scripts.train_nnunet --experiment baseline --nnunet-base ${NNUNET_BASE}"
