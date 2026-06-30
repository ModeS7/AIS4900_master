#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp49g"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp49g/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp49g/%j.err
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --exclude=idun-06-02

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# exp49g: exp1 baseline + detail-band augmentation. Cutoff sweep arm g of h.
#
# Identical to exp1_pixel_bravo (from SCRATCH, 128x128x160, 270M default_3d UNet,
# RFlow, lr 1e-4, use_compile) EXCEPT augmentation is enabled at low noise levels:
#   training.augmentation_t_schedule=[0.05, 0.10, 0.90]
# Augmentation suite = medium_no_brightness (flip + rotate90 + affine + contrast;
# NO brightness scaling, to avoid leaking an intensity shift into generated volumes).
# Applied IN-TRAINER to the clean image+seg BEFORE noise (training.augment=false so
# the dataloader does not also augment), gated to the detail band only:
#   t < 0.05            -> aug OFF (cleanest detail untouched)
#   0.05 <= t < 0.10    -> ramp 0 -> 1
#   0.10 <= t < 0.90 -> aug fully ON
#   t >= 0.90 (cutoff) -> aug OFF (structure-formation noise levels untouched)
# MONAI t convention: t=0 clean, t=1 noise. Only the cutoff t_off=0.90 varies
# across exp49a..exp49h (0.30, 0.40, ..., 1.00); everything else is held == exp1.

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp49g_aug_cutoff_t090"

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

echo "=== Exp49g: exp1 baseline + detail-band aug, cutoff t_off=0.90 (128x128x160) ==="
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
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.use_compile=true \
    training.gradient_checkpointing=false \
    training.augment=false \
    training.augmentation_level=medium_no_brightness \
    'training.augmentation_t_schedule=[0.05,0.10,0.90]' \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
