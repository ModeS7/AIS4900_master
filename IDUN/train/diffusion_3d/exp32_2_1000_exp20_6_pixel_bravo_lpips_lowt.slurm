#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="ex32k_17m"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp32_2_1000_exp20_6/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp32_2_1000_exp20_6/%j.err
#SBATCH --signal=B:SIGUSR1@120

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# exp32_2_1000-style fine-tune applied to exp20_6 (tiny 17M UNet) base.
# NOTE: the original exp20_6 SLURM and run-dir name use "_tiny_20m" but the actual
# parameter count is ~17M (see docs/experiment_results_3d.md:1003). The "20m" in
# the filename is a misnomer kept for path compatibility with the existing
# checkpoint directory.
#
# Mirrors exp32_2_1000_pixel_bravo_lpips_lowt.slurm:
#   - Loads exp20_6 (17M) model+optimizer, resets scheduler for fresh cosine cycle
#   - 1000-epoch run with LPIPS at low timesteps (perceptual_max_timestep=250)
#   - LPIPS weight 0.1, applied for t<250 of 1000 (linearly scaled)
#
# Architecture override copied verbatim from exp20_6_pixel_bravo_tiny_20m.slurm so
# the LPIPS run stays compatible with the 17M-tiny checkpoint:
#   channels=[8,16,32,64,128,128], attention_levels=[F,F,F,F,T,T],
#   num_res_blocks=[1,1,1,2,2,2], num_head_channels=8, norm_num_groups=8.
# use_compile is ON — torch.compile is compatible with LPIPS perceptual hooks
# (verified working in earlier exp32_2 runs). With 17M params + LPIPS,
# ~22h for 500 epochs (per user), so 1000 epochs needs ~3-4 chain segments at 12h each.

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp32_2_1000_exp20_6_pixel_bravo_lpips_lowt"
PRETRAINED="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp20_6_pixel_bravo_tiny_20m_20260314-072201/checkpoint_latest.pt"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/${EXP_NAME}_${TIMESTAMP}"
fi

CHAIN_ARGS="hydra.run.dir=${CHAIN_RUN_DIR}"
CKPT="${CHAIN_RUN_DIR}/checkpoint_latest.pt"
if [ -f "${CKPT}" ] && [ -s "${CKPT}" ]; then
    CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CKPT}"
    echo "Resuming from own checkpoint: ${CKPT}"
else
    CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${PRETRAINED} +training.reset_scheduler=true"
    echo "Continuing from: ${PRETRAINED} (reset scheduler for new cosine cycle)"
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

echo "=== Exp32_2_1000-on-exp20_6: Continue LPIPS low-t fine-tune for 1000 epochs (20M UNet) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
echo "Pretrained: ${PRETRAINED}"
echo "Run dir: ${CHAIN_RUN_DIR}"
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
    'model.channels=[8, 16, 32, 64, 128, 128]' \
    'model.attention_levels=[false, false, false, false, true, true]' \
    'model.num_res_blocks=[1, 1, 1, 2, 2, 2]' \
    model.num_head_channels=8 \
    model.norm_num_groups=8 \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=1000 \
    training.learning_rate=1e-5 \
    training.warmup_epochs=10 \
    training.gradient_clip_norm=0.5 \
    training.perceptual_weight=0.1 \
    training.perceptual_max_timestep=250 \
    training.gradient_checkpointing=false \
    training.use_compile=true \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
