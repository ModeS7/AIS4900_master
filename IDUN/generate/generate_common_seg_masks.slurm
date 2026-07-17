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
readonly MAX_WHITE_PER_SLICE=0.04
readonly COMPONENT_CONNECTIVITY=26
readonly PANEL_ID="source_selection_train105_seed42_euler100"
readonly POOL_NAME="common_seg_masks_525"
readonly EXPECTED_ATLAS_SHA256="c0c00cfc37177cd555d956a02554895cc9041280f7f32df20cb9ae21388c75ba"

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

: "${MASK_SOURCE_COMMIT:?Set MASK_SOURCE_COMMIT to the Git commit at submission time}"
SOURCE_COMMIT="$MASK_SOURCE_COMMIT"
readonly SOURCE_COMMIT
readonly -a SOURCE_PATHS=(
    pyproject.toml
    configs
    src/medgen
    IDUN/generate
    data/brain_atlas_256x256x160.nii.gz
)

verify_source_tree() {
    local current_commit
    current_commit="$(git rev-parse --verify HEAD)" || fatal "could not resolve repository HEAD"
    [[ "$current_commit" == "$SOURCE_COMMIT" ]] || fatal \
        "repository commit changed: expected $SOURCE_COMMIT, found $current_commit"
    if [[ -n "$(git status --porcelain --untracked-files=all -- "${SOURCE_PATHS[@]}")" ]]; then
        fatal "mask-pool source paths contain tracked or untracked changes"
    fi
}

verify_source_tree

readonly SEG_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
readonly ATLAS_PATH="${REPO_ROOT}/data/brain_atlas_256x256x160.nii.gz"
readonly PANEL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${PANEL_ID}"
readonly FINAL_DIR="${PANEL_ROOT}/${POOL_NAME}"
readonly STAGING_ROOT="${PANEL_ROOT}/.staging"
readonly JOB_TOKEN="${SLURM_JOB_ID:-preflight}"
readonly STAGING_NAME="${POOL_NAME}.job-${JOB_TOKEN}"
readonly STAGING_DIR="${STAGING_ROOT}/${STAGING_NAME}"

[[ -f "$SEG_CKPT" ]] || fatal "pinned segmentation checkpoint not found: $SEG_CKPT"
[[ "$(basename "$SEG_CKPT")" == "checkpoint_latest.pt" ]] || fatal \
    "segmentation checkpoint is not checkpoint_latest.pt"
[[ -f "$ATLAS_PATH" ]] || fatal "tracked train-only atlas not found: $ATLAS_PATH"
ATLAS_SHA256="$(sha256sum "$ATLAS_PATH" | awk '{print $1}')"
[[ "$ATLAS_SHA256" == "$EXPECTED_ATLAS_SHA256" ]] || fatal \
    "atlas hash changed: expected $EXPECTED_ATLAS_SHA256, found $ATLAS_SHA256"

: "${MASK_SEG_CKPT_SHA256:?Set MASK_SEG_CKPT_SHA256 to the checkpoint hash at submission time}"
SEG_CKPT_SHA256="$MASK_SEG_CKPT_SHA256"
readonly SEG_CKPT_SHA256
[[ "$SEG_CKPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || fatal "invalid checkpoint SHA-256"
[[ "$(sha256sum "$SEG_CKPT" | awk '{print $1}')" == "$SEG_CKPT_SHA256" ]] || fatal \
    "checkpoint hash differs from the submitted/pinned value"
[[ ! -e "$FINAL_DIR" ]] || fatal "final mask pool already exists: $FINAL_DIR"
[[ ! -e "$STAGING_DIR" ]] || fatal "job staging directory already exists: $STAGING_DIR"

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${JOB_TOKEN}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

python - "$SEG_CKPT" <<'PY'
import json
import sys

from medgen.scripts.generate import (
    _checkpoint_generation_signature,
    _validate_checkpoint_signature,
)

checkpoint = sys.argv[1]
signature = _checkpoint_generation_signature(checkpoint)
_validate_checkpoint_signature(
    signature,
    checkpoint_path=checkpoint,
    in_channels=1,
    out_channels=1,
    spatial_dims=3,
    strategy='rflow',
)
if signature.get('wrapper_type') != 'raw':
    raise RuntimeError(f"Expected unconditional raw mask checkpoint, found {signature}")
if signature.get('mode') != 'seg':
    raise RuntimeError(f"Expected mode='seg' mask checkpoint, found {signature}")
print(json.dumps(signature, sort_keys=True))
PY

python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Frozen common segmentation-mask pool ==="
echo "checkpoint:       $SEG_CKPT"
echo "checkpoint sha:   $SEG_CKPT_SHA256"
echo "atlas:            $ATLAS_PATH"
echo "atlas sha:        $ATLAS_SHA256"
echo "protocol:         525 masks, seed 42, Euler 100, shift 1.0"
echo "artifact filters: non-empty, <4% foreground per slice, strict train-atlas containment"
echo "lesion components: ${COMPONENT_CONNECTIVITY}-connectivity"
echo "disabled:         size-bin validation, brain PCA, segmentation PCA, image-derived cleanup"
echo "staging output:   $STAGING_DIR"
echo "final output:     $FINAL_DIR"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "Preflight passed; generation skipped because PREFLIGHT_ONLY=1"
    exit 0
fi

mkdir -p "$STAGING_ROOT"

time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    expected_strategy=rflow \
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
    max_white_percentage="$MAX_WHITE_PER_SLICE" \
    brain_atlas_path="$ATLAS_PATH" \
    brain_tolerance=0.0 \
    brain_dilate_pixels=0 \
    brain_pca_path=null \
    seg_pca_path=null \
    validate_brain_mask=false \
    mask_outside_brain=false \
    diffrs_checkpoint=null \
    provenance_hash_checkpoints=true \
    paths.generated_dir="$STAGING_ROOT" \
    output_subdir="$STAGING_NAME" \
    verbose=true

[[ -d "$STAGING_DIR" ]] || fatal "generator did not create staging output: $STAGING_DIR"

python -m medgen.scripts.audit_seg_mask_pool \
    --dataset-root "$STAGING_DIR" \
    --published-root "$FINAL_DIR" \
    --atlas "$ATLAS_PATH" \
    --checkpoint "$SEG_CKPT" \
    --expected-count "$NUM_MASKS" \
    --height 256 \
    --width 256 \
    --output-depth 150 \
    --generation-depth 160 \
    --seed "$SEED" \
    --steps "$STEPS" \
    --max-white-per-slice "$MAX_WHITE_PER_SLICE" \
    --max-attempts "$MAX_ATTEMPTS" \
    --fov-mm 240.0 \
    --expected-git-commit "$SOURCE_COMMIT" \
    --output "$STAGING_DIR/mask_pool_audit.json"

verify_source_tree
[[ "$(sha256sum "$SEG_CKPT" | awk '{print $1}')" == "$SEG_CKPT_SHA256" ]] || fatal \
    "checkpoint changed while generation was running"
[[ "$(sha256sum "$ATLAS_PATH" | awk '{print $1}')" == "$EXPECTED_ATLAS_SHA256" ]] || fatal \
    "atlas changed while generation was running"
[[ ! -e "$FINAL_DIR" ]] || fatal "final output appeared during generation: $FINAL_DIR"
mv -T -- "$STAGING_DIR" "$FINAL_DIR"

echo "Published frozen mask pool atomically: $FINAL_DIR"
echo "Audit report: $FINAL_DIR/mask_pool_audit.json"
