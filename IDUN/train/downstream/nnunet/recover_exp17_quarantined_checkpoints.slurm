#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="recover17_ckpt"
#SBATCH --output=IDUN/output/train/downstream/nnunet/recover_exp17_checkpoints_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/recover_exp17_checkpoints_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Restore only the checkpoints selected by the hardened audit. The default is
# a dry run. Set APPLY=1 only after reviewing the dry-run output.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly QUARANTINE="${EXP17_QUARANTINE:-${CLUSTER_BASE}/quarantine/exp17_hybrid_duplicate_20260725-185311}"
readonly AUDIT_JSON="${EXP17_AUDIT_JSON:-${QUARANTINE}/checkpoint_audit_24940260.json}"
readonly ACTIVE_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly ACTIVE_LOGS="${REPO_ROOT}/IDUN/output/train/downstream/nnunet"
readonly APPLY="${APPLY:-0}"

[[ "$APPLY" == 0 || "$APPLY" == 1 ]] || fatal "APPLY must be 0 or 1"
[[ -s "$AUDIT_JSON" ]] || fatal "audit report is missing: $AUDIT_JSON"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=(
    --audit "$AUDIT_JSON"
    --active-results "$ACTIVE_RESULTS"
    --active-logs-root "$ACTIVE_LOGS"
)
if [[ "$APPLY" == 1 ]]; then
    args+=(--apply)
fi

python -m medgen.scripts.recover_exp17_checkpoints "${args[@]}"
