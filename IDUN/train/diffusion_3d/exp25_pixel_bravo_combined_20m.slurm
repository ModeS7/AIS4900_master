#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp25_combined_20m"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp25/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp25/%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# Exp25: Combined best techniques — ~20M UNet (exp20_6) @ 256x256x160, 2000 epochs
# Same stack as exp24 but on tiny model with 4x more training:
#   - ScoreAug (exp23): rotation, flip, translation, cutout, omega conditioning
#   - Adjusted offset noise (exp1l): strength=0.1, adjusted=true
#   - PosthocEMA (exp1o): sigma_rels=[0.05,0.28], checkpoint every 5000 steps
#   - Uniform timestep (exp1p): sample_method=uniform
#   - Attention dropout (exp1r): dropout_cattn=0.1
#   - Weight decay (exp1s): 0.05
# Model: [8,16,32,64,128,128] ~17M params, GC ON, torch.compile OFF (ScoreAug dynamic shapes break compile)

CHAIN_MAX=40
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp25_pixel_bravo_combined_20m"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/${EXP_NAME}_${TIMESTAMP}"
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

echo "=== Exp25 Combined 20M: ScoreAug + AdjOffset + PosthocEMA + Uniform T + Dropout + WD ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
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
    training.epochs=2000 \
    training.learning_rate=1e-4 \
    training.augment=false \
    training.use_min_snr=false \
    training.gradient_checkpointing=true \
    training.use_compile=false \
    training.warmup_epochs=10 \
    training.gradient_clip_norm=0.5 \
    training.score_aug.enabled=true \
    training.score_aug.rotation=true \
    training.score_aug.flip=true \
    training.score_aug.translation=true \
    training.score_aug.cutout=true \
    training.score_aug.brightness=false \
    training.score_aug.use_omega_conditioning=true \
    training.offset_noise.enabled=true \
    training.offset_noise.strength=0.1 \
    training.offset_noise.adjusted=true \
    training.use_ema=true \
    training.ema.mode=post_hoc \
    'training.ema.sigma_rels=[0.05,0.28]' \
    training.ema.checkpoint_every_num_steps=5000 \
    training.ema.update_every=10 \
    strategy.sample_method=uniform \
    model.dropout_cattn=0.1 \
    training.optimizer.weight_decay=0.05 \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
