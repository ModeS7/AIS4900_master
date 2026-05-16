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
#SBATCH --job-name="nn_e5_d602_v"
#SBATCH --output=IDUN/output/train/downstream/nnunet_vanilla/eval_5fold/exp5_baseline_triple_d602_vanilla_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet_vanilla/eval_5fold/exp5_baseline_triple_d602_vanilla_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# 5-fold ensemble evaluation of exp5 Group-A triple-modality real baseline (Dataset 602).
# train_nnunet has no built-in 'baseline_triple' experiment, so we mirror training: pass
# --experiment baseline with --experiment-name override and --dataset-id 602.
# Requires all 5 fold trainings on Dataset 602 to have completed first.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== 5-fold ensemble eval: exp5_baseline_triple_d602_vanilla ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet_vanilla"

time python -m medgen.scripts.eval_nnunet \
    --experiment baseline \
    --dataset-id 602 \
    --experiment-name exp5_baseline_triple_d602_vanilla \
    --folds 0 1 2 3 4 \
    --nnunet-base "${NNUNET_BASE}" \
    --nnunet-results "${NNUNET_RESULTS}" \
    --trainer nnUNetTrainerVanilla \
    --plans nnUNetPlans
