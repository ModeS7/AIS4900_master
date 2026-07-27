#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name="rebuild_tb"
#SBATCH --output=IDUN/output/train/downstream/nnunet/rebuild_tb_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/rebuild_tb_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# Rebuild clean per-fold TensorBoard curves for exp17 from the logger history
# embedded in each checkpoint_final.pth. Reads checkpoints only; writes one
# fresh event file per fold's tensorboard/ dir; moves any pre-existing event
# files to a LOCAL tensorboard/superseded/ (no external quarantine).
#
# Preview first:  set DRY_RUN=1 (below) or pass --dry-run.

set -Eeuo pipefail

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
RESULTS_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
GLOB="${GLOB:-exp17_*_d663}"
DRY_RUN_FLAG=""
[ "${DRY_RUN:-0}" = "1" ] && DRY_RUN_FLAG="--dry-run"

# Safety: don't rebuild while any exp17 training job is still writing.
active="$(squeue -h -u "$USER" -o '%j' | grep -E '^exp17_' || true)"
[ -z "$active" ] || { echo "FATAL: exp17 jobs still active:"; echo "$active"; exit 1; }
[ -d "$RESULTS_ROOT" ] || { echo "FATAL: results root missing: $RESULTS_ROOT"; exit 1; }

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=""

python -m medgen.scripts.rebuild_tensorboard_from_checkpoints \
    --results-root "$RESULTS_ROOT" \
    --glob "$GLOB" \
    ${DRY_RUN_FLAG}
