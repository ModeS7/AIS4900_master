#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp1_chained"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

# === Auto-chain configuration ===
CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
LOG_DIR="IDUN/output/train/diffusion_3d/exp1_chained"
BASE_DIR="/cluster/work/modestas/AIS4900_master"

# First run: create unique run directory. Subsequent: reuse from env var.
if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="${BASE_DIR}/runs/diffusion_3d/bravo/exp1_chained_${TIMESTAMP}"
fi

# Build resume arguments
CHAIN_ARGS="hydra.run.dir=${CHAIN_RUN_DIR}"
CKPT="${CHAIN_RUN_DIR}/checkpoint_latest.pt"
if [ -f "${CKPT}" ]; then
    if python -c "import zipfile; zipfile.ZipFile('${CKPT}').close()" 2>/dev/null; then
        CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CKPT}"
        echo "Resuming from: ${CKPT}"
    else
        echo "WARNING: checkpoint invalid, starting fresh"
    fi
fi

# === Signal handler for auto-chaining ===
handle_timeout() {
    echo "=== Wall time approaching (SIGUSR1). Saving and resubmitting... ==="
    kill -USR1 $PID 2>/dev/null
    wait $PID 2>/dev/null

    # Don't resubmit if training+eval completed
    if [ -f "${CHAIN_RUN_DIR}/.training_complete" ]; then
        echo "Training and evaluation complete. Not resubmitting."
        exit 0
    fi

    # Check chain limit
    NEXT_COUNT=$((CHAIN_COUNT + 1))
    if [ "$NEXT_COUNT" -ge "$CHAIN_MAX" ]; then
        echo "Chain limit reached (${CHAIN_MAX}). Stopping."
        exit 1
    fi

    # Resubmit
    mkdir -p "${LOG_DIR}"
    sbatch --export=ALL,CHAIN_RUN_DIR="${CHAIN_RUN_DIR}",CHAIN_COUNT=${NEXT_COUNT} \
           --output="${LOG_DIR}/%j.out" \
           --error="${LOG_DIR}/%j.err" \
           "$0"
    echo "Resubmitted chain segment ${NEXT_COUNT}/${CHAIN_MAX}"
    exit 0
}

trap handle_timeout SIGUSR1

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}
mkdir -p "${LOG_DIR}"

echo "=== Exp1 3D Pixel Diffusion: Bravo ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# === Training (backgrounded for signal handling) ===
time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=default_3d \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.epochs=100 \
    training.learning_rate=1e-4 \
    training.name=exp1_chained_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID

conda deactivate
