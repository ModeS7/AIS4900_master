#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp39"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp39/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp39/%j.err
#SBATCH --signal=B:SIGUSR1@120

# exp39: IR-SDE restoration network trained from scratch with on-the-fly
# synthetic degradation (Gaussian blur + noise). Targets the MSE-posterior-
# mean-blur phenotype that exp37_3 produces. Runs at inference as a second
# stage: exp37_3 generates the blurry brain, IR-SDE restores HF detail.
#
# No warm-start: the 2-channel input semantics (x_t + degraded) differ from
# BRAVO (x_t + seg), so exp37_3 weights would be misleading. Train fresh.
#
# Uses standard BRAVO training data (paths.data_dir) directly — no
# precomputed degradation pairs needed. 80 epochs, LR 3e-5 cosine.

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp39_irsde_restore"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/${EXP_NAME}_${TIMESTAMP}"
fi

CHAIN_ARGS="hydra.run.dir=${CHAIN_RUN_DIR}"
CKPT="${CHAIN_RUN_DIR}/checkpoint_latest.pt"
if [ -f "${CKPT}" ] && [ -s "${CKPT}" ]; then
    CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CKPT}"
    echo "Resuming from own checkpoint: ${CKPT}"
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

echo "=== Exp39: IR-SDE restoration (synthetic degradation, from scratch) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Synthetic-degradation restoration: uses mode=restoration + CLI overrides to
# enable synthetic degradation (rather than a separate mode config file, which
# Hydra would place at cfg.mode.* not cfg.restoration.* where the loader reads).
# The loader auto-falls-back to paths.data_dir when synthetic mode is detected.
time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=irsde \
    mode=restoration \
    model=default_3d \
    model.spatial_dims=3 \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=80 \
    training.learning_rate=3e-5 \
    training.warmup_epochs=5 \
    training.eta_min=1e-7 \
    training.gradient_clip_norm=0.5 \
    '+restoration.degradation.type=synthetic' \
    '+restoration.degradation.blur_sigma=[0.8,1.6]' \
    '+restoration.degradation.noise_sigma=[0.005,0.02]' \
    +restoration.degradation.downsample_upsample_prob=0.3 \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
