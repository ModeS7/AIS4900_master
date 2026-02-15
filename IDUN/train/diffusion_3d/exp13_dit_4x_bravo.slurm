#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp13_dit_4x"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp13_dit_4x_bravo/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp13_dit_4x_bravo/%j.err
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --exclude=idun-06-02

# === Auto-chain configuration ===
CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp13_dit_4x_bravo"

# First run: create unique run directory. Subsequent: reuse from env var.
if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/${EXP_NAME}_${TIMESTAMP}"
fi

# Build resume arguments
CHAIN_ARGS="hydra.run.dir=${CHAIN_RUN_DIR}"
CKPT="${CHAIN_RUN_DIR}/checkpoint_latest.pt"
if [ -f "${CKPT}" ] && [ -s "${CKPT}" ]; then
    CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CKPT}"
    echo "Resuming from: ${CKPT}"
fi

# === Signal handler for auto-chaining ===
handle_timeout() {
    echo "=== Wall time approaching (SIGUSR1). Saving and resubmitting... ==="
    kill -USR1 $PID 2>/dev/null
    wait $PID 2>/dev/null

    # Don't resubmit if training+eval completed
    if [ -f "${CHAIN_RUN_DIR}/.training_complete" ]; then
        echo "Training and evaluation complete. Not resubmitting."
        exit 0
    fi

    # Check chain limit
    NEXT_COUNT=$((CHAIN_COUNT + 1))
    if [ "$NEXT_COUNT" -ge "$CHAIN_MAX" ]; then
        echo "Chain limit reached (${CHAIN_MAX}). Stopping."
        exit 1
    fi

    # Resubmit
    sbatch --export=ALL,CHAIN_RUN_DIR="${CHAIN_RUN_DIR}",CHAIN_COUNT=${NEXT_COUNT} \
           "$0"
    echo "Resubmitted chain segment ${NEXT_COUNT}/${CHAIN_MAX}"
    exit 0
}

trap handle_timeout SIGUSR1

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Exp13: DiT-S LDM @ VQ-VAE 4x, patch=1 (163,840 tokens) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# VQ-VAE 4x compression checkpoint (exp8_1)
COMPRESSION_CHECKPOINT=/cluster/work/modestas/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt

# ============================================================================
# DiT-S LDM: VQ-VAE 4x bravo_seg_cond
# ============================================================================
# Compression: VQ-VAE 4x (exp8_1)
#   - Pixel: 256x256x160 -> Latent: 64x64x40
#
# Architecture: DiT-S (33M params)
#   - hidden_size: 384, depth: 12, heads: 6
#   - patch_size: 1 -> 163,840 tokens (no patchification)
#   - Profiled: ~45.7 GB on A100 80GB
#
# Mode: bravo_seg_cond
#   - in_channels:  8 = [bravo_latent(4), seg_latent(4)]
#   - out_channels: 4 = predicted velocity for bravo_latent
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    mode=bravo_seg_cond \
    model=dit_3d \
    model.variant=S \
    model.patch_size=1 \
    model.image_size=64 \
    model.depth_size=40 \
    latent.enabled=true \
    latent.compression_checkpoint=${COMPRESSION_CHECKPOINT} \
    latent.compression_type=vqvae \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
