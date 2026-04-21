#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp38"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp38/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp38/%j.err
#SBATCH --signal=B:SIGUSR1@120

# exp38: Continue fine-tune from exp37_3 with MSE gated to zero at low-t.
# Prior finding: exp37_3 and all MSE-trained variants ceiling at ~60% real
# at very_high band because MSE's posterior-mean gradient fights LPIPS/FFL
# at low t (the HF-deficit phase). Remove MSE in that phase so auxiliary
# losses own the detail gradient.
#
#   mse_t_schedule=[0.05, 0.15, 1.0]:
#     - weight = 0 for t < 0.05  (zero MSE in vessel-emergence phase)
#     - ramp 0 → 1 for t ∈ [0.05, 0.15]
#     - weight = 1 for t ≥ 0.15 (full MSE in structure/mid phase)
#     - no ramp-down at high t (MSE fine there — structure needs it)
#
# LPIPS and FFL t-schedules unchanged from exp37_3 to isolate the MSE-gating
# effect. 60 epochs, LR 1e-5 (gentler than exp37_3's 2e-5 — avoids destabilizing
# the already-refined weights).

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp38_zero_mse_lowt"
PRETRAINED="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042/checkpoint_latest.pt"

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

echo "=== Exp38: zero-MSE at low-t (continue from exp37_3) ==="
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
    training.epochs=60 \
    training.learning_rate=1e-5 \
    training.warmup_epochs=3 \
    training.eta_min=1e-7 \
    training.gradient_clip_norm=0.5 \
    training.perceptual_weight=0.5 \
    'training.perceptual_t_schedule=[0.05,0.20,0.70]' \
    training.focal_frequency_weight=0.7 \
    'training.focal_frequency_t_schedule=[0.10,0.30,0.80]' \
    'training.mse_t_schedule=[0.05,0.15,1.0]' \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
