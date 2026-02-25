#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp2e_seg"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp2e/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp2e/%j.err
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --exclude=idun-06-02

# Seg conditioned 3D pixel diffusion with multi-level auxiliary bin prediction (128x128x160)

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp2e_pixel_seg_multilevel_aux"

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

echo "=== Exp2e 3D Pixel Diffusion: Seg (Multi-Level Auxiliary Bin Prediction) 128x128x160 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# ============================================================================
# 3D Pixel-Space Diffusion: Seg Mask with Multi-Level Auxiliary Bin Prediction
# ============================================================================
# Base: same as exp2d (aux bin prediction on bottleneck)
# Change: aux_loss_multilevel=true — prediction heads at 3 decoder levels:
#   mid (middle_block, 512ch), dec0 (up_blocks[0], 512ch), dec1 (up_blocks[1], 256ch)
#   Forces bin info through decoder path where skip connections carry spatial info
#   Per-level TensorBoard curves: AuxBin/mid, AuxBin/dec0, AuxBin/dec1
#
# Resolution: 128x128x160 (same as exp2c/exp2d)
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    mode=seg_conditioned_3d \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.learning_rate=5e-5 \
    training.gradient_clip_norm=0.5 \
    training.warmup_epochs=10 \
    mode.size_bins.embedding_dim=64 \
    mode.size_bins.projection_hidden_dim=512 \
    mode.size_bins.projection_num_layers=4 \
    mode.size_bins.aux_loss_weight=0.1 \
    mode.size_bins.aux_loss_multilevel=true \
    mode.cfg_dropout_prob=0.25 \
    training.regional_weighting.enabled=true \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
