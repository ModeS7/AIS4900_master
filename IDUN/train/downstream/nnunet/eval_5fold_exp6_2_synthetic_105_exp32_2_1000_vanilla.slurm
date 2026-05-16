#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_e5_32_2_v"
#SBATCH --output=IDUN/output/train/downstream/nnunet_vanilla/eval_5fold/exp32_2_1000_vanilla_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet_vanilla/eval_5fold/exp32_2_1000_vanilla_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# 5-fold ensemble evaluation of exp6_2 synthetic-105 candidate exp32_2_1000.
# Requires all 5 fold trainings on Dataset 612 to have completed first.
# Outputs ensemble Dice/Recall/Precision/HD95 to runs/downstream/nnunet_vanilla/.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== 5-fold ensemble eval: exp6_2 synthetic-105 exp32_2_1000 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet_vanilla"

time python -m medgen.scripts.eval_nnunet \
    --experiment synthetic --n-synthetic 105 \
    --dataset-id 612 \
    --experiment-name exp6_2_synthetic_105_exp32_2_1000_vanilla \
    --folds 0 1 2 3 4 \
    --nnunet-base "${NNUNET_BASE}" \
    --nnunet-results "${NNUNET_RESULTS}" \
    --trainer nnUNetTrainerVanilla \
    --plans nnUNetPlans
