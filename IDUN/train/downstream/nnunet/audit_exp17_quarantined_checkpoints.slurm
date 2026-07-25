#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="audit17_ckpt"
#SBATCH --output=IDUN/output/train/downstream/nnunet/audit_exp17_checkpoints_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/audit_exp17_checkpoints_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Read-only integrity and lineage audit of the exp17 checkpoints quarantined
# after overlapping resume allocations. This job never changes a checkpoint.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly QUARANTINE="${EXP17_QUARANTINE:-/cluster/work/${USER}/quarantine/exp17_hybrid_duplicate_20260725-185311}"
readonly RESULTS_ROOT="${QUARANTINE}/results"
readonly SHADOWS_ROOT="${QUARANTINE}/preprocessed_shadows"
readonly OUTPUT="${QUARANTINE}/checkpoint_audit_${SLURM_JOB_ID}.json"

[[ -d "$RESULTS_ROOT" ]] || fatal "quarantined results are missing: $RESULTS_ROOT"
[[ -d "$SHADOWS_ROOT" ]] || fatal "quarantined shadows are missing: $SHADOWS_ROOT"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m medgen.scripts.audit_nnunet_checkpoints \
    --results-root "$RESULTS_ROOT" \
    --shadows-root "$SHADOWS_ROOT" \
    --output "$OUTPUT" \
    --arm exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663=25 \
    --arm exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663=50 \
    --arm exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663=105 \
    --arm exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663=210

echo "Completed read-only exp17 checkpoint audit: $OUTPUT"
