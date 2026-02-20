#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp9_dcae_f128_1_5"
#SBATCH --output=IDUN/output/train/compression/exp9_3_dcae_f128_1_5/%j.out
#SBATCH --error=IDUN/output/train/compression/exp9_3_dcae_f128_1_5/%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@300

# === Auto-chain configuration ===
CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp9_dcae_f128_1_5"

# First run: create unique run directory. Subsequent: reuse from env var.
if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/compression_2d/multi_modality/${EXP_NAME}_${TIMESTAMP}"
fi

# Build resume arguments
CHAIN_ARGS="hydra.run.dir=${CHAIN_RUN_DIR}"
CKPT="${CHAIN_RUN_DIR}/checkpoint_latest.pt"
if [ -f "${CKPT}" ] && [ -s "${CKPT}" ]; then
    CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CKPT}"
    echo "Resuming from: ${CKPT}"
fi

# === Signal handler for auto-chaining ===
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

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Exp9.3 DC-AE 1.5 f128c512 (128x compression) Multi-Modality ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Exp9.3 DC-AE 1.5: f128c512 with Structured Latent Space
# Input: 256x256x1 -> Latent: 2x2x512
# Structured latent: channel_steps=[16, 20, 24, ..., 512] (125 options)
# Per paper: DC-AE 1.5 recommended for large channel counts
time python -m medgen.scripts.train_compression --config-name=dcae \
    paths=cluster \
    mode=multi_modality \
    dcae=f128 \
    dcae.structured_latent.enabled=true \
    training.epochs=125 \
    training.batch_size=16 \
    training.learning_rate=1e-4 \
    training.use_compile=false \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
