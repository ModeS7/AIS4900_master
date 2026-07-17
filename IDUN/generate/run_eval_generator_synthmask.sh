#!/usr/bin/env bash
# Generate one 105-volume BRAVO dataset from the shared synthetic masks.

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
readonly MAX_IMAGE_ATTEMPTS=50
readonly BRAIN_MARGIN_MM=3
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask105_seed42_euler100}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly BRAVO_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"
readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly SCREENING_MASKS="${EVAL_ROOT}/ordered_common_seg_masks_seed42/screening105"
readonly SELECTION_FILE="${EVAL_ROOT}/ordered_common_seg_masks_seed42/selection.json"
readonly LOW_CKPT="${BRAVO_ROOT}/${LOW_RUN}/checkpoint_latest.pt"
readonly FINAL_DIR="${EVAL_ROOT}/${LABEL}"
HIGH_CKPT=""
if [[ -n "$HIGH_RUN" ]]; then
    HIGH_CKPT="${BRAVO_ROOT}/${HIGH_RUN}/checkpoint_latest.pt"
fi
readonly HIGH_CKPT

[[ -f "$REPO_ROOT/pyproject.toml" ]] || fatal "repository root not found: $REPO_ROOT"
[[ -d "$SCREENING_MASKS" ]] || fatal "prepared 105-mask view not found: $SCREENING_MASKS"
[[ -s "$SELECTION_FILE" ]] || fatal "mask selection file not found: $SELECTION_FILE"
[[ -f "$LOW_CKPT" ]] || fatal "low-t checkpoint not found: $LOW_CKPT"
if [[ -n "$HIGH_CKPT" ]]; then
    [[ -f "$HIGH_CKPT" ]] || fatal "high-t checkpoint not found: $HIGH_CKPT"
fi
[[ ! -e "$FINAL_DIR" ]] || fatal "final output already exists: $FINAL_DIR"

mapfile -t MASK_IDS < <(find "$SCREENING_MASKS" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort)
[[ ${#MASK_IDS[@]} -eq $NUM_IMAGES ]] || fatal "expected 105 prepared masks, found ${#MASK_IDS[@]}"
for ((index = 0; index < NUM_IMAGES; index++)); do
    printf -v expected_id '%05d' "$index"
    [[ "${MASK_IDS[index]}" == "$expected_id" ]] || fatal "unexpected mask ID ${MASK_IDS[index]}"
    [[ -s "$SCREENING_MASKS/$expected_id/seg.nii.gz" ]] || fatal "missing mask $expected_id"
done

echo "Preflight OK [${LABEL}]: 105 shared synthetic masks and checkpoint_latest.pt"
cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"

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
    real_seg_dir="$SCREENING_MASKS"
    expected_real_cases=105
    expected_real_depth=150
    seed="$SEED"
    num_images="$NUM_IMAGES"
    current_image=0
    num_steps_bravo="$STEPS"
    ode_solver=euler
    shift_ratio_bravo=1.0
    cfg_scale_bravo=1.0
    validate_size_bins=false
    validate_real_seg_masks=true
    require_real_bravo_pairs=false
    validate_brain_mask=true
    conditioning_brain_qc_mode=reject
    brain_threshold=0.05
    brain_containment_margin_mm="$BRAIN_MARGIN_MM"
    max_image_attempts_per_mask="$MAX_IMAGE_ATTEMPTS"
    brain_atlas_path=null
    brain_pca_path=null
    seg_pca_path=null
    diffrs_checkpoint=null
    mask_outside_brain=false
    paths.generated_dir="$EVAL_ROOT"
    output_subdir="$LABEL"
    verbose=true
)
if [[ -n "$HIGH_CKPT" ]]; then
    GEN_ARGS+=(image_model_high_t="$HIGH_CKPT" handoff_t="$HANDOFF_T")
fi

echo "=== Generating ${LABEL}: 105 BRAVO volumes, seed 42, Euler 100 ==="
echo "masks:            $SCREENING_MASKS"
echo "low-t checkpoint: $LOW_CKPT"
if [[ -n "$HIGH_CKPT" ]]; then
    echo "high-t checkpoint: $HIGH_CKPT (t > $HANDOFF_T)"
fi
time python -m medgen.scripts.generate "${GEN_ARGS[@]}"

[[ -d "$FINAL_DIR" ]] || fatal "generator did not create output: $FINAL_DIR"
SEG_COUNT="$(find "$FINAL_DIR" -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz | wc -l)"
BRAVO_COUNT="$(find "$FINAL_DIR" -mindepth 2 -maxdepth 2 -type f -name bravo.nii.gz | wc -l)"
[[ "$SEG_COUNT" -eq "$NUM_IMAGES" ]] || fatal \
    "generated dataset contains $SEG_COUNT masks, expected $NUM_IMAGES"
[[ "$BRAVO_COUNT" -eq "$NUM_IMAGES" ]] || fatal \
    "generated dataset contains $BRAVO_COUNT BRAVO volumes, expected $NUM_IMAGES"
echo "Completed dataset: $FINAL_DIR"
