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
#SBATCH --job-name="nn_b345"
#SBATCH --output=IDUN/output/train/downstream/nnunet/exp345_baseline_quad_d603/%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/exp345_baseline_quad_d603/%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

# Real-only quad-modal (t1_pre + t1_gd + flair + bravo) baseline under
# Group A plans (Dataset 603). This is the 4-channel setup that mirrors
# Grøvik 2020 / Ottesen 2022 — the apples-to-apples comparison for the
# published per-volume Dice numbers (Grøvik 0.79 ± 0.12; Ottesen 0.85
# slice-wise sagittal). Test set is the same 51 BrainMetShare-3 patients
# used by exp3/4/5.
#
# Run dataset prep first:   sbatch convert_d603_quad.slurm
# Then submit each fold:    FOLD=0 sbatch exp345_baseline_quad_d603.slurm
#                           FOLD=1 sbatch exp345_baseline_quad_d603.slurm
#                           ...

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

echo "=== nnU-Net: exp345_baseline_quad d603 (real quad-modal) | fold ${FOLD} ==="
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

# train_nnunet has no built-in 'baseline_quad' experiment, so we mirror the
# exp4/exp5 pattern: pass --experiment baseline with --experiment-name
# override and --dataset-id 603 (the quad-modality real dataset).
time python -m medgen.scripts.train_nnunet \
    --experiment baseline \
    --dataset-id 603 \
    --experiment-name exp345_baseline_quad_d603 \
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
    echo "=== Evaluating exp345_baseline_quad_d603 (fold ${FOLD}) ==="
    python -m medgen.scripts.eval_nnunet \
        --experiment baseline \
        --dataset-id 603 \
        --experiment-name exp345_baseline_quad_d603 \
        --folds ${FOLD} \
        --nnunet-base "${NNUNET_BASE}" \
        --nnunet-results "${NNUNET_RESULTS}" \
        --trainer nnUNetTrainerBrainMets \
        --plans nnUNetResEncUNetLPlans \
        || echo "WARNING: Evaluation failed"
fi
