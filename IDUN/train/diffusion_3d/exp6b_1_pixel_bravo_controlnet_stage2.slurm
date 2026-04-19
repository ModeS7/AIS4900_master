#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp6b_1_cn_s2"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp6b_1/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp6b_1/%j.err
#SBATCH --signal=B:SIGUSR1@120

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# Exp6b_1: 3D Pixel-Space Diffusion — Bravo ControlNet Stage 2 (256x256x160)
# Architecture:
#   - UNet: 270M (FROZEN from exp6a_1 Stage 1)
#   - ControlNet: Mirrors UNet encoder (~135M trainable params)
# Mode: bravo — UNet in=1 (unconditional), ControlNet injects seg via residuals
# Strategy: RFlow
# Purpose: Train ControlNet for seg-conditioned bravo generation
#
# Prerequisites: exp6a_1 Stage 1 checkpoint (unconditional UNet, 256x256x160)

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp6b_1_pixel_bravo_controlnet_stage2_"

STAGE1_CHECKPOINT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_best.pt"

if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    STAGE1_CHECKPOINT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_latest.pt"
fi

if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Stage 1 checkpoint not found for exp6a_1"
    exit 1
fi

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/${EXP_NAME}${TIMESTAMP}"
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

echo "=== Exp6b_1 3D Pixel Diffusion: Bravo ControlNet Stage 2 256x256x160 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
echo "Run dir: ${CHAIN_RUN_DIR}"
echo "Stage 1 checkpoint: ${STAGE1_CHECKPOINT}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=default_3d \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.warmup_epochs=10 \
    training.gradient_clip_norm=0.5 \
    controlnet.enabled=true \
    controlnet.freeze_unet=true \
    controlnet.gradient_checkpointing=true \
    pretrained_checkpoint="${STAGE1_CHECKPOINT}" \
    training.name=${EXP_NAME} \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
