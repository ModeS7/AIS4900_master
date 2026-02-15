#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name="tensorboard"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/tensorboard/%j.out
#SBATCH --error=IDUN/output/tensorboard/%j.err
#SBATCH --signal=B:SIGUSR1@60

# === Auto-chain configuration ===
CHAIN_MAX=50
CHAIN_COUNT=${CHAIN_COUNT:-0}

# === Signal handler for auto-chaining ===
handle_timeout() {
    echo "=== Wall time approaching (SIGUSR1). Resubmitting TensorBoard... ==="
    kill $PID 2>/dev/null
    wait $PID 2>/dev/null

    NEXT_COUNT=$((CHAIN_COUNT + 1))
    if [ "$NEXT_COUNT" -ge "$CHAIN_MAX" ]; then
        echo "Chain limit reached (${CHAIN_MAX}). Stopping."
        exit 0
    fi

    sbatch --export=ALL,CHAIN_COUNT=${NEXT_COUNT} "$0"
    echo "Resubmitted chain segment ${NEXT_COUNT}/${CHAIN_MAX}"
    exit 0
}

trap handle_timeout SIGUSR1

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

LOGDIR=/cluster/work/modestas/AIS4900_master/runs
PORT=6006

echo "============================================"
echo "TensorBoard"
echo "============================================"
echo "Job:   ${SLURM_JOB_ID}"
echo "Node:  ${SLURM_JOB_NODELIST}"
echo "Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
echo "Port:  ${PORT}"
echo "Log:   ${LOGDIR}"
echo ""
echo "Connect via SSH tunnel:"
echo "  ssh -L ${PORT}:${SLURM_JOB_NODELIST}:${PORT} modestas@idun-login1.hpc.ntnu.no"
echo "  Then open: http://localhost:${PORT}"
echo "============================================"

tensorboard --logdir=${LOGDIR} --port=${PORT} --bind_all &

PID=$!
wait $PID
