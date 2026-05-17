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
#SBATCH --job-name="nn_e5_mix525"
#SBATCH --output=IDUN/output/train/downstream/nnunet/eval_5fold/exp7_7_mixed_525syn_d614_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/eval_5fold/exp7_7_mixed_525syn_d614_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# 5-fold ensemble evaluation of exp7_7: Mixed 105 real + 525 synthetic (Dataset 614).
# Requires all 5 fold trainings (exp7_7_mixed_525syn_d614) to have completed first.
# Outputs ensemble Dice/Recall/Precision/HD95 to runs/downstream/nnunet/.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== 5-fold ensemble eval: exp7_7 mixed 525syn (d614) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"

time python -m medgen.scripts.eval_nnunet \
    --experiment mixed --n-synthetic 525 \
    --dataset-id 614 \
    --experiment-name exp7_7_mixed_525syn_d614 \
    --folds 0 1 2 3 4 \
    --nnunet-base "${NNUNET_BASE}" \
    --nnunet-results "${NNUNET_RESULTS}" \
    --trainer nnUNetTrainerBrainMets \
    --plans nnUNetResEncUNetLPlans
