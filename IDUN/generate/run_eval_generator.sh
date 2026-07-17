#!/usr/bin/env bash
# Shared runner for the 14 fixed source-selection generators.

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
    [[ "$HANDOFF_T" == "0.25" ]] || fatal "handoff must be 0.25"
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

readonly BRAVO_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"
readonly TRAIN_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/train"
readonly PANEL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${PANEL_ID}"
readonly FINAL_DIR="${PANEL_ROOT}/${LABEL}"
readonly LOW_CKPT="${BRAVO_ROOT}/${LOW_RUN}/checkpoint_latest.pt"
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
[[ ! -e "$FINAL_DIR" ]] || fatal "final output already exists; refusing to resume or overwrite: $FINAL_DIR"

mapfile -t SEG_IDS < <(find "$TRAIN_DIR" -mindepth 2 -maxdepth 2 -type f -name 'seg.nii.gz' -printf '%h\n' | sed 's|.*/||' | LC_ALL=C sort)
mapfile -t BRAVO_IDS < <(find "$TRAIN_DIR" -mindepth 2 -maxdepth 2 -type f -name 'bravo.nii.gz' -printf '%h\n' | sed 's|.*/||' | LC_ALL=C sort)
[[ ${#SEG_IDS[@]} -eq $NUM_IMAGES ]] || fatal "expected ${NUM_IMAGES} training masks, found ${#SEG_IDS[@]}"
[[ ${#BRAVO_IDS[@]} -eq $NUM_IMAGES ]] || fatal "expected ${NUM_IMAGES} training BRAVO volumes, found ${#BRAVO_IDS[@]}"
for ((index = 0; index < NUM_IMAGES; index++)); do
    [[ "${SEG_IDS[index]}" == "${BRAVO_IDS[index]}" ]] || fatal "training mask/BRAVO ID mismatch at index ${index}"
done

echo "Preflight OK [${LABEL}]: 105 paired training cases and pinned checkpoint_latest.pt"
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
    paths.generated_dir="$PANEL_ROOT"
    output_subdir="$LABEL"
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
echo "output:            $FINAL_DIR"
time python -m medgen.scripts.generate "${GEN_ARGS[@]}"

[[ -d "$FINAL_DIR" ]] || fatal "generator did not create output: $FINAL_DIR"
SEG_COUNT="$(find "$FINAL_DIR" -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz | wc -l)"
BRAVO_COUNT="$(find "$FINAL_DIR" -mindepth 2 -maxdepth 2 -type f -name bravo.nii.gz | wc -l)"
[[ "$SEG_COUNT" -eq "$NUM_IMAGES" ]] || fatal \
    "generated dataset contains $SEG_COUNT masks, expected $NUM_IMAGES"
[[ "$BRAVO_COUNT" -eq "$NUM_IMAGES" ]] || fatal \
    "generated dataset contains $BRAVO_COUNT BRAVO volumes, expected $NUM_IMAGES"
echo "Completed dataset: $FINAL_DIR"
