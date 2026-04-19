#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp26_wdm_sa"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp26/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp26/%j.err
#SBATCH --signal=B:SIGUSR1@120

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# Exp26: WDM 270M + ScoreAug, 1000 epochs
# Based on exp19_2 (best WDM: FID 67.32)
# Stack:
#   - Base: WDM DDPM x0-prediction, [32,64,256,512,512], wavelet N(0,1) norm
#   - ScoreAug: rotation, flip, translation, cutout, omega conditioning

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp26_wdm_scoreaug"

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

echo "=== Exp26 WDM 270M + ScoreAug 1000ep: Bravo 256x256x160 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
echo "Run dir: ${CHAIN_RUN_DIR}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=ddpm \
    strategy.prediction_type=sample \
    strategy.schedule=linear_beta \
    mode=bravo \
    model=wdm_3d \
    'model.channels=[32,64,256,512,512]' \
    'model.attention_levels=[false,false,false,true,true]' \
    'model.num_res_blocks=[1,1,1,2,2]' \
    model.num_head_channels=16 \
    model.norm_num_groups=16 \
    wavelet.enabled=true \
    wavelet.normalize=true \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=1000 \
    training.batch_size=1 \
    training.gradient_checkpointing=true \
    training.learning_rate=1e-4 \
    training.gradient_clip_norm=0.5 \
    training.warmup_epochs=10 \
    training.score_aug.enabled=true \
    training.score_aug.rotation=true \
    training.score_aug.flip=true \
    training.score_aug.translation=true \
    training.score_aug.cutout=true \
    training.score_aug.brightness=false \
    training.score_aug.use_omega_conditioning=true \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
