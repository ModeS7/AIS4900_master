#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp19_1_wdm"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp19_1/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp19_1/%j.err
#SBATCH --signal=B:SIGUSR1@120

# WDM 256x256x160: Per-subband N(0,1) normalization
# RFlow velocity prediction, 74M params, batch=1, LR=1e-4
# Wavelet → [B,8,80,128,128] — grad checkpointing to fit 80GB

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp19_1"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/${EXP_NAME}_${TIMESTAMP}"
fi

CHAIN_ARGS="hydra.run.dir=${CHAIN_RUN_DIR}"
CKPT="${CHAIN_RUN_DIR}/checkpoint_latest.pt"
if [ -f "${CKPT}" ] && [ -s "${CKPT}" ]; then
    CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CKPT}"
    echo "Resuming from: ${CKPT}"
fi

handle_timeout() {
    echo "=== Wall time approaching (SIGUSR1). Saving and resubmitting... ==="
    kill -USR1 $PID 2>/dev/null
    wait $PID 2>/dev/null

    if [ -f "${CHAIN_RUN_DIR}/.training_complete" ]; then
        echo "Training complete. Not resubmitting."
        exit 0
    fi

    NEXT_COUNT=$((CHAIN_COUNT + 1))
    if [ "$NEXT_COUNT" -ge "$CHAIN_MAX" ]; then
        echo "Chain limit reached (${CHAIN_MAX}). Stopping."
        exit 1
    fi

    sbatch --export=ALL,CHAIN_RUN_DIR="${CHAIN_RUN_DIR}",CHAIN_COUNT=${NEXT_COUNT} "$0"
    echo "Resubmitted chain segment ${NEXT_COUNT}/${CHAIN_MAX}"
    exit 0
}

trap handle_timeout SIGUSR1

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp19_1 WDM Wavelet RFlow velocity: Bravo 256x256x160 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=wdm_3d \
    wavelet.enabled=true \
    wavelet.normalize=true \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.batch_size=1 \
    training.gradient_checkpointing=true \
    training.learning_rate=1e-4 \
    training.gradient_clip_norm=0.5 \
    training.warmup_epochs=10 \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
