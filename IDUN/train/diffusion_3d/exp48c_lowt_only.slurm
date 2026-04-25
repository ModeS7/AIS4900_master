#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp48c"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp48c/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp48c/%j.err
#SBATCH --signal=B:SIGUSR1@120

# exp48c: low-t-only training, pure lpips_huber (Karras Huber+LPIPS) across
# trained window. Curriculum restricts t∈[0, 0.25].

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp48c_lowt_only_lpips_huber"
PRETRAINED="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt"

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
    if [ ! -f "${PRETRAINED}" ]; then
        echo "ERROR: pretrained checkpoint not found at ${PRETRAINED}"
        exit 1
    fi
    CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${PRETRAINED} +training.reset_scheduler=true"
    echo "Continuing from exp32_2_1000: ${PRETRAINED} (reset scheduler)"
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

echo "=== Exp48c: low-t-only training, pure lpips_huber ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
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
    training.learning_rate=1e-5 \
    training.warmup_epochs=10 \
    training.gradient_clip_norm=0.5 \
    strategy.loss_type=lpips_huber \
    training.curriculum.enabled=true \
    training.curriculum.warmup_epochs=1 \
    training.curriculum.min_t_start=0.0 \
    training.curriculum.min_t_end=0.0 \
    training.curriculum.max_t_start=0.25 \
    training.curriculum.max_t_end=0.25 \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
