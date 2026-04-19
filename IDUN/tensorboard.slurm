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

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# === Auto-chain configuration ===
CHAIN_MAX=50
CHAIN_COUNT=${CHAIN_COUNT:-0}

# === Signal handler for auto-chaining ===
# Try to land on the same node so the URL stays the same
handle_timeout() {
    echo "=== Wall time approaching (SIGUSR1). Resubmitting TensorBoard... ==="
    kill $PID 2>/dev/null
    wait $PID 2>/dev/null

    NEXT_COUNT=$((CHAIN_COUNT + 1))
    if [ "$NEXT_COUNT" -ge "$CHAIN_MAX" ]; then
        echo "Chain limit reached (${CHAIN_MAX}). Stopping."
        exit 0
    fi

    PREV_NODE="${SLURM_JOB_NODELIST}"

    # Try to get the same node for 60s (keeps the URL stable)
    NEW_JOB=$(sbatch --export=ALL,CHAIN_COUNT=${NEXT_COUNT},PREV_NODE=${PREV_NODE} \
        --nodelist="${PREV_NODE}" --immediate=60 "$0" 2>&1)

    if echo "$NEW_JOB" | grep -q "Submitted"; then
        echo "Resubmitted on SAME node ${PREV_NODE}: ${NEW_JOB}"
    else
        # Fallback: any available node
        NEW_JOB=$(sbatch --export=ALL,CHAIN_COUNT=${NEXT_COUNT},PREV_NODE=${PREV_NODE} "$0" 2>&1)
        echo "Same node unavailable, resubmitted on ANY node: ${NEW_JOB}"
    fi

    exit 0
}

trap handle_timeout SIGUSR1

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

LOGDIR=${CLUSTER_BASE}/AIS4900_master/runs
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
echo "Open in browser:"
echo "  https://apps.hpc.ntnu.no/rnode/${SLURM_JOB_NODELIST}/${PORT}/"
if [ -n "${PREV_NODE:-}" ] && [ "${PREV_NODE}" != "${SLURM_JOB_NODELIST}" ]; then
    echo ""
    echo "WARNING: Node changed from ${PREV_NODE} -> ${SLURM_JOB_NODELIST}"
    echo "         Update your browser URL!"
fi
echo "============================================"

tensorboard --logdir=${LOGDIR} --port=${PORT} --bind_all &

PID=$!
wait $PID
