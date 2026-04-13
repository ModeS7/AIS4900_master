#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp1o_1_pixel_bravo"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp1o_1/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp1o_1/%j.err
#SBATCH --signal=B:SIGUSR1@120

# Exp1o_1: 3D Pixel-Space Diffusion — Bravo (256x256x160) + Post-hoc EMA
# Architecture: 270M UNet (default_3d)
#   - channels: [16, 32, 64, 256, 512, 512]
#   - attention_levels: [false, false, false, false, true, true]
#   - num_res_blocks: [1, 1, 1, 2, 2, 2]
# Mode: bravo (in=2, out=1), raw [0,1] normalization
# Strategy: RFlow
# EMA: Post-hoc (Karras EDM2) with sigma_rels=[0.05, 0.10] (matching paper)
#   - Paper: γ₁=16.97 (σ_rel=0.05), γ₂=6.94 (σ_rel=0.10) — close gammas for stable interpolation
#   - Checkpoints every 5000 steps to phema_checkpoints/
#   - After training: run synthesize_phema.py to find optimal sigma_rel

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp1o_1_pixel_bravo_"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/${EXP_NAME}${TIMESTAMP}"
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

echo "=== Exp1o_1 3D Pixel Diffusion: Bravo 256x256x160 + Post-hoc EMA ==="
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
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.warmup_epochs=10 \
    training.gradient_clip_norm=0.5 \
    training.use_ema=true \
    training.ema.mode=post_hoc \
    'training.ema.sigma_rels=[0.05,0.10]' \
    training.ema.checkpoint_every_num_steps=5000 \
    training.ema.update_every=10 \
    training.name=${EXP_NAME} \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
