#!/usr/bin/env bash
# Shared, fail-closed runner for the 14 fixed source-selection generators.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

if [[ $# -ne 2 && $# -ne 4 ]]; then
    fatal "usage: $0 LABEL LOW_RUN [HIGH_RUN HANDOFF_T]"
fi

readonly LABEL="$1"
readonly LOW_RUN="$2"
readonly HIGH_RUN="${3:-}"
readonly HANDOFF_T="${4:-}"
[[ "$LABEL" =~ ^[a-zA-Z0-9_.-]+$ ]] || fatal "unsafe label: $LABEL"
[[ "$LOW_RUN" =~ ^[a-zA-Z0-9_.-]+$ ]] || fatal "unsafe low-t run name: $LOW_RUN"
if [[ -n "$HIGH_RUN" ]]; then
    [[ "$HIGH_RUN" =~ ^[a-zA-Z0-9_.-]+$ ]] || fatal "unsafe high-t run name: $HIGH_RUN"
    [[ "$HANDOFF_T" == "0.25" ]] || fatal "panel handoff must be 0.25"
else
    [[ -z "$HANDOFF_T" ]] || fatal "handoff_t requires a high-t checkpoint"
fi

readonly NUM_IMAGES=105
readonly SEED=42
readonly STEPS=100
readonly PANEL_ID="${PANEL_ID:-source_selection_train105_seed42_euler100}"
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

SOURCE_COMMIT="${PANEL_SOURCE_COMMIT:-$(git rev-parse --verify HEAD)}" || \
    fatal "could not resolve the repository commit"
readonly SOURCE_COMMIT
readonly -a PANEL_SOURCE_PATHS=(pyproject.toml configs src/medgen IDUN/generate)

verify_source_tree() {
    local current_commit
    current_commit="$(git rev-parse --verify HEAD)" || fatal "could not resolve the repository commit"
    [[ "$current_commit" == "$SOURCE_COMMIT" ]] || fatal \
        "repository commit changed after job submission: expected $SOURCE_COMMIT, found $current_commit"
    if [[ -n "$(git status --porcelain --untracked-files=all -- "${PANEL_SOURCE_PATHS[@]}")" ]]; then
        fatal "panel source files changed; expected committed content under: ${PANEL_SOURCE_PATHS[*]}"
    fi
}

verify_source_tree

readonly BRAVO_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"
readonly TRAIN_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/train"
readonly PANEL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${PANEL_ID}"
readonly FINAL_DIR="${PANEL_ROOT}/${LABEL}"
readonly LOW_CKPT="${BRAVO_ROOT}/${LOW_RUN}/checkpoint_latest.pt"
readonly MANIFEST_TOOL="${REPO_ROOT}/IDUN/generate/panel_manifest.py"
HIGH_CKPT=""
if [[ -n "$HIGH_RUN" ]]; then
    HIGH_CKPT="${BRAVO_ROOT}/${HIGH_RUN}/checkpoint_latest.pt"
fi
readonly HIGH_CKPT

[[ -d "$TRAIN_DIR" ]] || fatal "training reference not found: $TRAIN_DIR"
[[ -f "$LOW_CKPT" ]] || fatal "pinned low-t checkpoint not found: $LOW_CKPT"
[[ "$(basename "$LOW_CKPT")" == "checkpoint_latest.pt" ]] || fatal "low-t checkpoint is not latest"
if [[ -n "$HIGH_CKPT" ]]; then
    [[ -f "$HIGH_CKPT" ]] || fatal "pinned high-t checkpoint not found: $HIGH_CKPT"
    [[ "$(basename "$HIGH_CKPT")" == "checkpoint_latest.pt" ]] || fatal "high-t checkpoint is not latest"
fi
[[ -f "$MANIFEST_TOOL" ]] || fatal "manifest helper not found: $MANIFEST_TOOL"
[[ ! -e "$FINAL_DIR" ]] || fatal "final output already exists; refusing to resume or overwrite: $FINAL_DIR"

mapfile -t SEG_IDS < <(find "$TRAIN_DIR" -mindepth 2 -maxdepth 2 -type f -name 'seg.nii.gz' -printf '%h\n' | sed 's|.*/||' | LC_ALL=C sort)
mapfile -t BRAVO_IDS < <(find "$TRAIN_DIR" -mindepth 2 -maxdepth 2 -type f -name 'bravo.nii.gz' -printf '%h\n' | sed 's|.*/||' | LC_ALL=C sort)
[[ ${#SEG_IDS[@]} -eq $NUM_IMAGES ]] || fatal "expected ${NUM_IMAGES} training masks, found ${#SEG_IDS[@]}"
[[ ${#BRAVO_IDS[@]} -eq $NUM_IMAGES ]] || fatal "expected ${NUM_IMAGES} training BRAVO volumes, found ${#BRAVO_IDS[@]}"
for ((index = 0; index < NUM_IMAGES; index++)); do
    [[ "${SEG_IDS[index]}" == "${BRAVO_IDS[index]}" ]] || fatal "training mask/BRAVO ID mismatch at index ${index}"
done

echo "Preflight OK [${LABEL}]: 105 paired training cases and pinned checkpoint_latest.pt"
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    exit 0
fi

readonly JOB_TOKEN="${SLURM_JOB_ID:?SLURM_JOB_ID must be set for generation}"
readonly STAGING_ROOT="${PANEL_ROOT}/.staging"
readonly STAGING_NAME="${LABEL}.job-${JOB_TOKEN}"
readonly STAGING_DIR="${STAGING_ROOT}/${STAGING_NAME}"
[[ ! -e "$STAGING_DIR" ]] || fatal "job staging output already exists: $STAGING_DIR"
mkdir -p "$STAGING_ROOT"

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

GEN_ARGS=(
    paths=cluster
    spatial_dims=3
    expected_strategy=rflow
    gen_mode=bravo
    image_size=256
    depth=160
    trim_slices=10
    fov_mm=240.0
    image_model="$LOW_CKPT"
    image_model_high_t=null
    real_seg_dir="$TRAIN_DIR"
    expected_real_cases=105
    expected_real_depth=150
    seed=42
    num_images=105
    current_image=0
    num_steps_bravo=100
    ode_solver=euler
    shift_ratio_bravo=1.0
    cfg_scale_bravo=1.0
    validate_size_bins=false
    validate_brain_mask=false
    brain_atlas_path=null
    brain_pca_path=null
    diffrs_checkpoint=null
    mask_outside_brain=false
    mask_outside_brain_dilate_pixels=0
    require_real_bravo_pairs=true
    validate_real_seg_masks=true
    provenance_hash_checkpoints=true
    paths.generated_dir="$STAGING_ROOT"
    output_subdir="$STAGING_NAME"
    verbose=true
)
if [[ -n "$HIGH_CKPT" ]]; then
    GEN_ARGS+=(image_model_high_t="$HIGH_CKPT" handoff_t=0.25)
fi

echo "=== Generating ${LABEL}: 105 BRAVO volumes, seed 42, Euler 100, train masks ==="
echo "low-t checkpoint:  $LOW_CKPT"
if [[ -n "$HIGH_CKPT" ]]; then
    echo "high-t checkpoint: $HIGH_CKPT (t > 0.25; 75/100 Euler evaluations)"
fi
echo "staging output:    $STAGING_DIR"
time python -m medgen.scripts.generate "${GEN_ARGS[@]}"

[[ -d "$STAGING_DIR" ]] || fatal "generator did not create staging output: $STAGING_DIR"
mapfile -t OUTPUT_IDS < <(find "$STAGING_DIR" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort)
[[ ${#OUTPUT_IDS[@]} -eq $NUM_IMAGES ]] || fatal "expected ${NUM_IMAGES} output directories, found ${#OUTPUT_IDS[@]}"
for ((index = 0; index < NUM_IMAGES; index++)); do
    printf -v expected_id '%05d' "$index"
    [[ "${OUTPUT_IDS[index]}" == "$expected_id" ]] || fatal "unexpected output ID ${OUTPUT_IDS[index]} at index ${index}"
    [[ -s "$STAGING_DIR/$expected_id/seg.nii.gz" ]] || fatal "missing output mask for $expected_id"
    [[ -s "$STAGING_DIR/$expected_id/bravo.nii.gz" ]] || fatal "missing output BRAVO for $expected_id"
done
[[ -s "$STAGING_DIR/generation_manifest.json" ]] || fatal "missing generation_manifest.json"

# The jobs use the live checkout. Recheck after the long generation run so an
# edit made while jobs were queued or running cannot be published silently.
verify_source_tree

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    GIT_DIRTY=true
else
    GIT_DIRTY=false
fi

MANIFEST_ARGS=(
    write
    --output "$STAGING_DIR/panel_job_manifest.json"
    --label "$LABEL"
    --input-root "$TRAIN_DIR"
    --dataset-root "$STAGING_DIR"
    --final-dataset-root "$FINAL_DIR"
    --low-checkpoint "$LOW_CKPT"
    --git-commit "$SOURCE_COMMIT"
    --git-dirty "$GIT_DIRTY"
)
if [[ -n "$HIGH_CKPT" ]]; then
    MANIFEST_ARGS+=(--high-checkpoint "$HIGH_CKPT" --handoff-t 0.25)
fi
python "$MANIFEST_TOOL" "${MANIFEST_ARGS[@]}"

[[ ! -e "$FINAL_DIR" ]] || fatal "final output appeared during generation: $FINAL_DIR"
mv -- "$STAGING_DIR" "$FINAL_DIR"
echo "Published complete dataset atomically: $FINAL_DIR"
