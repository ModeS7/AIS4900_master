#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp37_3"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp37_3/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp37_3/%j.err
#SBATCH --signal=B:SIGUSR1@120

# exp37_3: Short high-weight aux fine-tune of exp1_1_1000.
# Addresses observed regression in exp37_1/exp37_2 (peak at ~epoch 120, drift back
# toward MSE-optimum by epoch 260+). Hypothesis: shorter schedule with steeper
# LR decay freezes weights near the peak before MSE gradient dominates.
#   - 150 epochs (vs 500 in exp37_1/2)
#   - LR 2e-5 -> 1e-7 cosine (aggressive tail decay)
#   - LPIPS weight 0.5 (vs 0.4), FFL weight 0.7 (vs 0.5) — heavier push
#   - Same t-schedules as exp37_2: LPIPS [0.05,0.20,0.70], FFL [0.10,0.30,0.80]

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp37_3_pixel_bravo_lpips_ffl_short"
PRETRAINED="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/${EXP_NAME}_${TIMESTAMP}"
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

echo "=== Exp37_3: short heavy-aux fine-tune (150 epochs) ==="
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
    training.epochs=150 \
    training.learning_rate=2e-5 \
    training.warmup_epochs=10 \
    training.eta_min=1e-7 \
    training.gradient_clip_norm=0.5 \
    training.perceptual_weight=0.5 \
    'training.perceptual_t_schedule=[0.05,0.20,0.70]' \
    training.focal_frequency_weight=0.7 \
    'training.focal_frequency_t_schedule=[0.10,0.30,0.80]' \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
