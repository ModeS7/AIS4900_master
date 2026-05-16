#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_s525_48ch_v"
#SBATCH --output=IDUN/output/train/downstream/nnunet_vanilla/exp6_2_exp48c_handoff_exp32_525_vanilla/%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet_vanilla/exp6_2_exp48c_handoff_exp32_525_vanilla/%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

# ────────────────────────────────────────────────────────────────────────────
# DO NOT SUBMIT YET. The original-trainer 525-vol handoff job is still in
# flight under nnUNetTrainerBrainMets + nnUNetResEncUNetLPlans. Submitting
# this vanilla retrain now would compete for the same GPU pool with no
# benefit. Let the in-flight job finish first, then submit this alongside
# the rest of the vanilla marathon.
# ────────────────────────────────────────────────────────────────────────────

# nnU-Net exp6_2: synth-only with FULL 525 vols of exp48c_handoff_exp32 synth,
# trained for nnU-Net default 1000 epochs (250k iterations).
# Same compute as 105×1000ep but with 5× more cases sampled per training step.
#
# Dataset 614 (real + 525 brain-masked synth from exp48c_handoff_exp32).
# Trainer: nnUNetTrainerVanilla (upstream nnU-Net defaults, default 1000 epochs).
# Same Group A architecture as the existing exp6_2_synthetic_105_* runs.
# Per-case sampling density: ~1429 patches/case across full training.
#
# FOLD=0 sbatch exp6_2_synthetic_525_exp48c_handoff_exp32.slurm

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

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

echo "=== nnU-Net: exp6_2 synthetic-525 exp48c_handoff_exp32 1000ep | fold ${FOLD} ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet_vanilla"
CONTINUE_FLAG=""
if [ "$CHAIN_COUNT" -gt 0 ]; then
    CONTINUE_FLAG="--continue-training"
fi

# --experiment synthetic with NO --n-synthetic flag uses the full 525 synth pool.
# Default trainer = nnUNetTrainerVanilla (1000 epochs).
time python -m medgen.scripts.train_nnunet \
    --experiment synthetic \
    --dataset-id 614 \
    --experiment-name exp6_2_synthetic_525_exp48c_handoff_exp32_vanilla \
    --fold ${FOLD} \
    --nnunet-base "${NNUNET_BASE}" \
    --nnunet-results "${NNUNET_RESULTS}" \
    --trainer nnUNetTrainerVanilla \
    --plans nnUNetPlans \
    ${CONTINUE_FLAG} &

PID=$!
wait $PID
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "=== Evaluating exp6_2_synthetic_525_exp48c_handoff_exp32 (fold ${FOLD}) ==="
    python -m medgen.scripts.eval_nnunet \
        --experiment synthetic \
        --dataset-id 614 \
        --experiment-name exp6_2_synthetic_525_exp48c_handoff_exp32_vanilla \
        --folds ${FOLD} \
        --nnunet-base "${NNUNET_BASE}" \
        --nnunet-results "${NNUNET_RESULTS}" \
        --trainer nnUNetTrainerVanilla \
        --plans nnUNetPlans \
        || echo "WARNING: Evaluation failed"
fi
