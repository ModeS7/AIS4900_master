#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_s525_48ch200"
#SBATCH --output=IDUN/output/train/downstream/nnunet/exp6_2_exp48c_handoff_exp32_525_200ep/%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/exp6_2_exp48c_handoff_exp32_525_200ep/%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

# nnU-Net exp6_2: synth-only with FULL 525 vols of exp48c_handoff_exp32 synth,
# trained for 200 epochs (50k iterations) instead of the standard 1000 (250k).
# Roughly 1/5 the compute of the existing 1000-epoch synth-105 run.
#
# Dataset 614 (real + 525 brain-masked synth from exp48c_handoff_exp32).
# Trainer: nnUNetTrainerBrainMets_200epochs (BrainMets settings + num_epochs=200).
# Same Group A architecture as the existing exp6_2_synthetic_105_* runs.
#
# FOLD=0 sbatch exp6_2_synthetic_525_exp48c_handoff_exp32_200ep.slurm

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

CHAIN_MAX=10
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

echo "=== nnU-Net: exp6_2 synthetic-525 exp48c_handoff_exp32 200ep | fold ${FOLD} ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
CONTINUE_FLAG=""
if [ "$CHAIN_COUNT" -gt 0 ]; then
    CONTINUE_FLAG="--continue-training"
fi

# --experiment synthetic with NO --n-synthetic flag uses the full 525 synth pool.
time python -m medgen.scripts.train_nnunet \
    --experiment synthetic \
    --dataset-id 614 \
    --experiment-name exp6_2_synthetic_525_exp48c_handoff_exp32_200ep \
    --fold ${FOLD} \
    --nnunet-base "${NNUNET_BASE}" \
    --nnunet-results "${NNUNET_RESULTS}" \
    --trainer nnUNetTrainerBrainMets_200epochs \
    --plans nnUNetResEncUNetLPlans \
    ${CONTINUE_FLAG} &

PID=$!
wait $PID
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "=== Evaluating exp6_2_synthetic_525_exp48c_handoff_exp32_200ep (fold ${FOLD}) ==="
    python -m medgen.scripts.eval_nnunet \
        --experiment synthetic \
        --dataset-id 614 \
        --experiment-name exp6_2_synthetic_525_exp48c_handoff_exp32_200ep \
        --folds ${FOLD} \
        --nnunet-base "${NNUNET_BASE}" \
        --nnunet-results "${NNUNET_RESULTS}" \
        --trainer nnUNetTrainerBrainMets_200epochs \
        --plans nnUNetResEncUNetLPlans \
        || echo "WARNING: Evaluation failed"
fi
