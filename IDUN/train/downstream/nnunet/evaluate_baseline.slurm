#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --job-name="nn_eval_base"
#SBATCH --output=IDUN/output/train/downstream/nnunet/eval/baseline_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/eval/baseline_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Evaluate exp3 baseline (real-only BRAVO) on 51-volume test set
# 5-fold ensemble inference + MedGen metrics

cd ${SLURM_SUBMIT_DIR}

echo "=== nnU-Net: Evaluate exp3 baseline ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="/cluster/work/modestas/MedicalDataSets/nnunet"
NNUNET_RESULTS="/cluster/work/modestas/AIS4900_master/runs/downstream/nnunet"

python -m medgen.scripts.eval_nnunet \
    --experiment baseline \
    --dataset-id 501 \
    --folds 0 1 2 3 4 \
    --nnunet-base "${NNUNET_BASE}" \
    --nnunet-results "${NNUNET_RESULTS}" \
    --trainer nnUNetTrainerBrainMets \
    --plans nnUNetResEncUNetLPlans

conda deactivate
