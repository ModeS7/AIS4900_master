#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_d640_rs"
#SBATCH --output=IDUN/output/train/downstream/nnunet/convert_d640_random_split_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/convert_d640_random_split_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Build Dataset 640 = BRAVO real (105 train + 51 test) under a RANDOM split.
# Replicates Ottesen 2025's "split randomly into 105 train + 51 test" protocol
# (PMC12063759 §2.1) instead of the official Grøvik 2020 BrainMetShare-3
# hold-out split that exp3 (Dataset 600) uses.
#
# Single fixed seed (42) for reproducibility — the picked patient lists are
# saved into case_info.json so the split can be re-applied or audited.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

set -e
cd ${SLURM_SUBMIT_DIR}

echo "=== Dataset 640 build: random 105/51 split (seed=42) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"

export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}"

echo ""
echo "=== Cleaning previous Dataset 640 ==="
rm -rf "${nnUNet_raw}/Dataset640_BrainMet"
rm -rf "${nnUNet_preprocessed}/Dataset640_BrainMet"

echo ""
echo "=== Convert with random 105/51 split (seed=42) ==="
python -m medgen.scripts.convert_random_split \
    --real-dir "${REAL_DIR}" \
    --nnunet-raw "${nnUNet_raw}" \
    --dataset-id 640 \
    --seed 42 \
    --modality bravo

echo ""
echo "=== Plan + preprocess Dataset 640 ==="
nnUNetv2_plan_and_preprocess -d 640 --verify_dataset_integrity -c 3d_fullres -np 4
nnUNetv2_plan_and_preprocess -d 640 -pl nnUNetPlannerResEncL -c 3d_fullres -np 4

echo ""
echo "=== Done. Submit exp3_random_split_d640.slurm next. ==="
