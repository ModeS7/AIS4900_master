#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name="finish_common_masks"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/finish_common_masks_%j.out
#SBATCH --error=IDUN/output/generate/finish_common_masks_%j.err

# Finish job 24795335 after its completed 525-mask generation was stopped by
# the now-removed Git-commit check in the post-generation audit.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly PANEL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/source_selection_train105_seed42_euler100"
readonly STAGING_DIR="${PANEL_ROOT}/.staging/common_seg_masks_525.job-24795335"
readonly FINAL_DIR="${PANEL_ROOT}/common_seg_masks_525"
readonly ATLAS_PATH="${REPO_ROOT}/data/brain_atlas_256x256x160.nii.gz"
readonly SEG_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"

[[ -d "$STAGING_DIR" ]] || fatal "completed staging directory not found: $STAGING_DIR"
[[ ! -e "$FINAL_DIR" ]] || fatal "final directory already exists: $FINAL_DIR"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m medgen.scripts.audit_seg_mask_pool \
    --dataset-root "$STAGING_DIR" \
    --published-root "$FINAL_DIR" \
    --atlas "$ATLAS_PATH" \
    --checkpoint "$SEG_CKPT" \
    --expected-count 525 \
    --height 256 \
    --width 256 \
    --output-depth 150 \
    --generation-depth 160 \
    --seed 42 \
    --steps 100 \
    --max-attempts 50 \
    --fov-mm 240.0 \
    --output "$STAGING_DIR/mask_pool_audit.json"

mv -T -- "$STAGING_DIR" "$FINAL_DIR"
echo "Completed mask pool: $FINAL_DIR"
