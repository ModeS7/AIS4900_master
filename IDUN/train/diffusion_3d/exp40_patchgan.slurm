#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp40"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp40/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp40/%j.err
#SBATCH --signal=B:SIGUSR1@120

# exp40: Adversarial fine-tune from exp37_3 with a 2.5D PatchGAN discriminator
# on the center-slice of predicted_clean. Adds pressure that MSE/LPIPS/FFL
# only approximate — discriminator directly penalizes blurry outputs.
#
# Safety rails (baked into config defaults):
#   - adv_weight=0.02 (low — GAN gradient dwarfs MSE)
#   - adv_weight_ramp_steps=2000 — ramps from 0 so G isn't shocked
#   - warmup_steps=500 — D trains alone before coupling to G
#   - step_frequency=2 — G:D = 2:1, lets G keep up with a strengthening D
#   - adv_t_schedule=[0.0, 0.02, 0.15] — gate adversarial signal to low-t only
#     (the HF-deficit phase; high-t predicted_clean is too uncertain for GAN)
#   - Self-conditioning consistency disabled — can interact badly with GAN
#   - LR 5e-6 (lower than exp38) — adversarial gradient amplifies drift
#
# 100 epochs, continues from exp37_3 weights.

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp40_patchgan"
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

echo "=== Exp40: PatchGAN 2.5D adversarial fine-tune (from exp37_3) ==="
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
    training.epochs=100 \
    training.learning_rate=5e-6 \
    training.warmup_epochs=5 \
    training.eta_min=1e-8 \
    training.gradient_clip_norm=0.5 \
    training.perceptual_weight=0.5 \
    'training.perceptual_t_schedule=[0.05,0.20,0.70]' \
    training.focal_frequency_weight=0.7 \
    'training.focal_frequency_t_schedule=[0.10,0.30,0.80]' \
    training.self_cond.enabled=false \
    training.discriminator.enabled=true \
    training.discriminator.spatial_dims=2 \
    training.discriminator.channels=1 \
    training.discriminator.num_layers=3 \
    training.discriminator.num_channels=32 \
    training.discriminator.learning_rate=1.0e-4 \
    training.discriminator.warmup_steps=500 \
    training.discriminator.step_frequency=2 \
    training.discriminator.adv_weight=0.02 \
    training.discriminator.adv_weight_ramp_steps=2000 \
    'training.discriminator.adv_t_schedule=[0.0,0.02,0.15]' \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
