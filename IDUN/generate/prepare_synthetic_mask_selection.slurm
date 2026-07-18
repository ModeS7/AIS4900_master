#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name="prep_synthmask_selection"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/prep_synthmask_selection_%j.out
#SBATCH --error=IDUN/output/generate/prep_synthmask_selection_%j.err

# Run once after the 525-mask pool exists.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
readonly POOL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/source_selection_train105_seed42_euler100/common_seg_masks_525"
readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly FINAL_DIR="${EVAL_ROOT}/ordered_common_seg_masks_seed42"

[[ -d "$POOL_ROOT" ]] || fatal "525-mask pool not found: $POOL_ROOT"
[[ ! -e "$FINAL_DIR" ]] || fatal "prepared mask view already exists: $FINAL_DIR"

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m medgen.scripts.prepare_synthetic_mask_selection \
    --pool-root "$POOL_ROOT" \
    --output-root "$FINAL_DIR"

[[ -s "$FINAL_DIR/selection.json" ]] || fatal "selection.json was not created"
[[ "$(find "$FINAL_DIR/candidates525" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 525 ]] || fatal "candidate view does not contain 525 masks"

echo "Prepared fixed seed-42 mask selection: $FINAL_DIR"
