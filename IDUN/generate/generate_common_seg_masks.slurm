#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_common_masks"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/generate_common_seg_masks_%j.out
#SBATCH --error=IDUN/output/generate/generate_common_seg_masks_%j.err

# One frozen synthetic mask bank for all downstream image-generator runs.
# This is a new controlled protocol, not a reproduction of the old 40-step
# production pools: exp14_1 latest, seed 42, 100 unshifted Euler steps.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly NUM_MASKS=525
readonly SEED=42
readonly STEPS=100
readonly MAX_ATTEMPTS=50
readonly COMPONENT_CONNECTIVITY=26
readonly EVAL_ID="source_selection_train105_seed42_euler100"
readonly POOL_NAME="common_seg_masks_525"

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
readonly REPO_ROOT
[[ -f "$REPO_ROOT/pyproject.toml" ]] || fatal "repository root not found: $REPO_ROOT"
cd "$REPO_ROOT"

readonly SEG_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
readonly ATLAS_PATH="${REPO_ROOT}/data/brain_atlas_256x256x160.nii.gz"
readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly FINAL_DIR="${EVAL_ROOT}/${POOL_NAME}"
readonly JOB_TOKEN="${SLURM_JOB_ID:-preflight}"

[[ -f "$SEG_CKPT" ]] || fatal "pinned segmentation checkpoint not found: $SEG_CKPT"
[[ "$(basename "$SEG_CKPT")" == "checkpoint_latest.pt" ]] || fatal \
    "segmentation checkpoint is not checkpoint_latest.pt"
[[ -f "$ATLAS_PATH" ]] || fatal "tracked train-only atlas not found: $ATLAS_PATH"
[[ ! -e "$FINAL_DIR" ]] || fatal "final mask pool already exists: $FINAL_DIR"

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${JOB_TOKEN}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Frozen common segmentation-mask pool ==="
echo "checkpoint:       $SEG_CKPT"
echo "atlas:            $ATLAS_PATH"
echo "protocol:         525 masks, seed 42, Euler 100, shift 1.0"
echo "artifact filters: non-empty, strict train-derived brain-support containment"
echo "lesion components: ${COMPONENT_CONNECTIVITY}-connectivity"
echo "disabled:         per-slice burden limit, segmentation PCA, brain PCA, image-derived cleanup"
echo "output:           $FINAL_DIR"

time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=seg_conditioned \
    image_size=256 \
    depth=160 \
    trim_slices=10 \
    fov_mm=240.0 \
    seg_model="$SEG_CKPT" \
    image_model=null \
    seed="$SEED" \
    batch_size=1 \
    num_images="$NUM_MASKS" \
    current_image=0 \
    num_steps="$STEPS" \
    num_steps_seg="$STEPS" \
    ode_solver=euler \
    shift_ratio=1.0 \
    shift_ratio_seg=1.0 \
    cfg_scale_seg=1.0 \
    cfg_scale_seg_end=null \
    cfg_mode=standard \
    validate_size_bins=false \
    seg_component_connectivity="$COMPONENT_CONNECTIVITY" \
    max_retries="$MAX_ATTEMPTS" \
    max_white_percentage=null \
    brain_atlas_path="$ATLAS_PATH" \
    brain_tolerance=0.0 \
    brain_dilate_pixels=0 \
    brain_pca_path=null \
    seg_pca_path=null \
    validate_brain_mask=false \
    mask_outside_brain=false \
    diffrs_checkpoint=null \
    paths.generated_dir="$EVAL_ROOT" \
    output_subdir="$POOL_NAME" \
    verbose=true

[[ -d "$FINAL_DIR" ]] || fatal "generator did not create output: $FINAL_DIR"
MASK_COUNT="$(find "$FINAL_DIR" -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz | wc -l)"
[[ "$MASK_COUNT" -eq "$NUM_MASKS" ]] || fatal \
    "generated pool contains $MASK_COUNT masks, expected $NUM_MASKS"
[[ -s "$FINAL_DIR/bins.csv" ]] || fatal "generated pool is missing bins.csv"
BIN_COUNT="$(tail -n +2 "$FINAL_DIR/bins.csv" | wc -l)"
[[ "$BIN_COUNT" -eq "$NUM_MASKS" ]] || fatal \
    "bins.csv contains $BIN_COUNT rows, expected $NUM_MASKS"

echo "Completed frozen mask pool: $FINAL_DIR"
