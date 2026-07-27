#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="rebuild17_tb"
#SBATCH --output=IDUN/output/train/downstream/nnunet/rebuild_exp17_tensorboard_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/rebuild_exp17_tensorboard_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# Reconstruct one clean 1000-epoch TensorBoard history for every completed
# exp17 fold from the logger history embedded in checkpoint_final.pth.  This
# job never modifies active training results or the quarantine.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly RESULTS_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly OUTPUT_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/eval/exp17_hybrid_tensorboard_reconstructed"
readonly RECOVERY_MARKER="${RESULTS_ROOT}/.exp17_recovery_manifests/recovery_checkpoint_audit_24940260.json"

active="$(
    squeue -h -u "$USER" -o '%j|%T' \
        | grep -E '^(exp17_1_h25|exp17_2_h50|exp17_3_h105|exp17_4_h210)\|' \
        || true
)"
[[ -z "$active" ]] || fatal "exp17 training jobs are still active: $active"
[[ -d "$RESULTS_ROOT" ]] || fatal "results root is missing: $RESULTS_ROOT"
[[ -s "$RECOVERY_MARKER" ]] || fatal "exp17 recovery marker is missing: $RECOVERY_MARKER"
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || \
    fatal "refusing to overwrite existing output: $OUTPUT_ROOT"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=""

python -m medgen.scripts.rebuild_exp17_tensorboard \
    --results-root "$RESULTS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --recovery-manifest "$RECOVERY_MARKER"

echo "TensorBoard log directory: $OUTPUT_ROOT"
echo "Manifest: $OUTPUT_ROOT/reconstruction_manifest.json"
