#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp7_2_sit_xl_p8"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp7_2_sit_xl_256_patch8/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp7_2_sit_xl_256_patch8/%j.err
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --exclude=idun-06-02

# === Auto-chain configuration ===
CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp7_2_sit_xl_256_patch8"

# First run: create unique run directory. Subsequent: reuse from env var.
if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/${EXP_NAME}_${TIMESTAMP}"
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

echo "=== Exp7_2: 3D-DiT-XL @ 160x256x256, patch=8 (20,480 tokens) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# 3D-DiT-XL: Bravo mode @ 160x256x256, patch_size=8
# ============================================================================
# Architecture: DiT-XL (675M params)
#   - hidden_size: 1152, depth: 28, heads: 16
#   - patch_size: 8 -> 20,480 tokens
#   - Profiled: ~12.3 GB on A100 80GB (67.7 GB headroom)
#
# Mode: bravo
#   - in_channels: 2 = [noisy_bravo, seg_mask]
#   - out_channels: 1 = predicted velocity for bravo
#
# Resolution: 256x256x160 (pixel space, full resolution)
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=dit_3d \
    model.variant=XL \
    model.patch_size=8 \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.use_compile=true \
    training.gradient_checkpointing=false \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
