#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_mix50"
#SBATCH --output=IDUN/output/train/downstream/nnunet/exp7_5/%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/exp7_5/%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

# nnU-Net exp7_5: Mixed 105 real + 50 synthetic, 5-fold CV
# FOLD=0 sbatch exp7_5_mixed_50syn.slurm

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
FOLD=${FOLD:-0}

handle_timeout() {
    echo "=== Wall time approaching. Resubmitting with --continue-training ==="
    kill -TERM $PID 2>/dev/null
    wait $PID 2>/dev/null

    NEXT_COUNT=$((CHAIN_COUNT + 1))
    if [ "$NEXT_COUNT" -ge "$CHAIN_MAX" ]; then
        echo "Chain limit reached (${CHAIN_MAX}). Stopping."
        exit 1
    fi

    sbatch --export=ALL,FOLD=${FOLD},CHAIN_COUNT=${NEXT_COUNT} "$0"
    echo "Resubmitted chain segment ${NEXT_COUNT}/${CHAIN_MAX}"
    exit 0
}

trap handle_timeout SIGUSR1

cd ${SLURM_SUBMIT_DIR}

echo "=== nnU-Net: Mixed 105 real + 50 syn (fold ${FOLD}) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="/cluster/work/modestas/MedicalDataSets/nnunet"
NNUNET_RESULTS="/cluster/work/modestas/AIS4900_master/runs/downstream/nnunet"
CONTINUE_FLAG=""
if [ "$CHAIN_COUNT" -gt 0 ]; then
    CONTINUE_FLAG="--continue-training"
fi

time python -m medgen.scripts.train_nnunet \
    --experiment mixed \
    --n-synthetic 50 \
    --fold ${FOLD} \
    --nnunet-base "${NNUNET_BASE}" \
    --nnunet-results "${NNUNET_RESULTS}" \
    --trainer nnUNetTrainerBrainMets \
    --plans nnUNetResEncUNetLPlans \
    ${CONTINUE_FLAG} &

PID=$!
wait $PID
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "=== Evaluating mixed 50syn (fold ${FOLD}) ==="
    python -m medgen.scripts.eval_nnunet \
        --experiment mixed \
        --n-synthetic 50 \
        --folds ${FOLD} \
        --nnunet-base "${NNUNET_BASE}" \
        --nnunet-results "${NNUNET_RESULTS}" \
        --trainer nnUNetTrainerBrainMets \
        --plans nnUNetResEncUNetLPlans \
        || echo "WARNING: Evaluation failed"
fi
