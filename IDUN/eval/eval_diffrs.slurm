#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="eval_diffrs"
#SBATCH --output=IDUN/output/eval/eval_diffrs_%j.out
#SBATCH --error=IDUN/output/eval/eval_diffrs_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02
#SBATCH --signal=B:SIGUSR1@300

# === Auto-chain configuration ===
CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}

# === Signal handler for auto-chaining ===
handle_timeout() {
    echo "=== Wall time approaching (SIGUSR1). Killing generation and resubmitting... ==="
    kill -TERM $PID 2>/dev/null
    wait $PID 2>/dev/null

    NEXT_COUNT=$((CHAIN_COUNT + 1))
    if [ "$NEXT_COUNT" -ge "$CHAIN_MAX" ]; then
        echo "Chain limit reached (${CHAIN_MAX}). Stopping."
        exit 1
    fi

    sbatch --export=ALL,CHAIN_COUNT=${NEXT_COUNT} "$0"
    echo "Resubmitted chain segment ${NEXT_COUNT}/${CHAIN_MAX}"
    exit 0
}

trap handle_timeout SIGUSR1

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== DiffRS Evaluation ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# === PATHS ===
BRAVO_CHECKPOINT="runs/diffusion_3d/bravo/exp1_1_pixel_bravo_rflow_256x160_20260119-212710/checkpoint_latest.pt"
DIFFRS_HEAD="runs/diffrs/diffrs_head.pt"
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUTPUT_DIR="/cluster/work/modestas/MedicalDataSets/diffrs"

# Check if DiffRS head exists
DIFFRS_FLAG=""
if [ -f "${DIFFRS_HEAD}" ]; then
    echo "DiffRS head found: ${DIFFRS_HEAD}"
    DIFFRS_FLAG="--diffrs-head ${DIFFRS_HEAD}"
else
    echo "WARNING: No DiffRS head found at ${DIFFRS_HEAD}"
    echo "Running baseline evaluation only."
fi

# Run evaluation (with --resume for auto-chain support)
time python -m medgen.scripts.eval_diffrs \
    --bravo-model "${BRAVO_CHECKPOINT}" \
    ${DIFFRS_FLAG} \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-volumes 25 \
    --seed 42 \
    --resume &

PID=$!
wait $PID

echo "=== Evaluation complete ==="
echo "Results:"
cat "${OUTPUT_DIR}/results.csv" 2>/dev/null || echo "No results CSV found"

conda deactivate
