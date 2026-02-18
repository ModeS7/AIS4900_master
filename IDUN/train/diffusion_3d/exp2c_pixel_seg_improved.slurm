#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp2c_seg"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp2c/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp2c/%j.err
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --exclude=idun-06-02

# Seg conditioned 3D pixel diffusion with improved size bin conditioning (128x128x160)

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp2c_pixel_seg_improved"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/${EXP_NAME}_${TIMESTAMP}"
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

echo "=== Exp2c 3D Pixel Diffusion: Seg (Improved Size Bin Conditioning) 128x128x160 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# ============================================================================
# 3D Pixel-Space Diffusion: Seg Mask with Improved Size Bin Conditioning
# ============================================================================
# Changes from exp2_1:
#   A. Deeper projection MLP: embedding_dim=64, hidden=512, 4 hidden layers
#      7×Embedding(11,64) → [B,448] → 512→512→512→512→256(zero-init) → FiLM
#      Adds ~741K params (~0.3% of 270M UNet)
#   B. Regional weighting: loss weighted by tumor region size
#   C. CFG dropout: 15% → 25% for stronger conditioned/unconditioned contrast
#
# Resolution: 128x128x160 (same as exp2)
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    mode=seg_conditioned_3d \
    volume.height=128 \
    volume.width=128 \
    training.learning_rate=5e-5 \
    training.gradient_clip_norm=0.5 \
    training.warmup_epochs=10 \
    mode.size_bins.embedding_dim=64 \
    mode.size_bins.projection_hidden_dim=512 \
    mode.size_bins.projection_num_layers=4 \
    mode.cfg_dropout_prob=0.25 \
    training.regional_weighting.enabled=true \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
