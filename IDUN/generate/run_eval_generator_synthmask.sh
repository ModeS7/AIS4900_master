#!/usr/bin/env bash
# Evaluate one BRAVO generator on 525 shared synthetic-mask candidates.

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

readonly NUM_CANDIDATES=525
readonly SEED=42
readonly STEPS=100
readonly MAX_IMAGE_ATTEMPTS=3
readonly START_CANDIDATE_INPUT="${SYNTHMASK_START_CANDIDATE:-0}"
[[ "$START_CANDIDATE_INPUT" =~ ^(0|[1-9][0-9]*)$ ]] || fatal \
    "SYNTHMASK_START_CANDIDATE must be a non-negative integer: $START_CANDIDATE_INPUT"
readonly START_CANDIDATE_VALUE=$((10#$START_CANDIDATE_INPUT))
(( START_CANDIDATE_VALUE < NUM_CANDIDATES )) || fatal \
    "SYNTHMASK_START_CANDIDATE must be below $NUM_CANDIDATES: $START_CANDIDATE_VALUE"
readonly NUM_CANDIDATES_TO_PROCESS=$((NUM_CANDIDATES - START_CANDIDATE_VALUE))
readonly OUTPUT_LABEL="${SYNTHMASK_OUTPUT_LABEL:-$LABEL}"
[[ "$OUTPUT_LABEL" =~ ^[a-zA-Z0-9_.-]+$ ]] || fatal "unsafe output label: $OUTPUT_LABEL"
if (( START_CANDIDATE_VALUE > 0 )) && [[ "$OUTPUT_LABEL" == "$LABEL" ]]; then
    fatal "a partial candidate range must use a separate SYNTHMASK_OUTPUT_LABEL"
fi
# Maximum observed training-pair distance was 2.322613 mm. Round upward to 0.001 mm.
readonly BRAIN_MARGIN_MM=2.323
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly BRAVO_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"
readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly CANDIDATE_MASKS="${EVAL_ROOT}/ordered_common_seg_masks_seed42/candidates525"
readonly SELECTION_FILE="${EVAL_ROOT}/ordered_common_seg_masks_seed42/selection.json"
readonly LOW_CKPT="${BRAVO_ROOT}/${LOW_RUN}/checkpoint_latest.pt"
readonly BRAIN_SUPPORT_PCA="${REPO_ROOT}/data/brain_support_pca_train105_var95_256x256x160.npz"
readonly FINAL_DIR="${EVAL_ROOT}/${OUTPUT_LABEL}"
HIGH_CKPT=""
if [[ -n "$HIGH_RUN" ]]; then
    HIGH_CKPT="${BRAVO_ROOT}/${HIGH_RUN}/checkpoint_latest.pt"
fi
readonly HIGH_CKPT

[[ -f "$REPO_ROOT/pyproject.toml" ]] || fatal "repository root not found: $REPO_ROOT"
[[ -d "$CANDIDATE_MASKS" ]] || fatal "prepared $NUM_CANDIDATES-mask view not found: $CANDIDATE_MASKS"
[[ -s "$SELECTION_FILE" ]] || fatal "mask selection file not found: $SELECTION_FILE"
[[ -f "$LOW_CKPT" ]] || fatal "low-t checkpoint not found: $LOW_CKPT"
[[ -s "$BRAIN_SUPPORT_PCA" ]] || fatal "brain-support PCA not found: $BRAIN_SUPPORT_PCA"
if [[ -n "$HIGH_CKPT" ]]; then
    [[ -f "$HIGH_CKPT" ]] || fatal "high-t checkpoint not found: $HIGH_CKPT"
fi
[[ ! -e "$FINAL_DIR" ]] || fatal "final output already exists: $FINAL_DIR"

mapfile -t MASK_IDS < <(find "$CANDIDATE_MASKS" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort)
[[ ${#MASK_IDS[@]} -eq $NUM_CANDIDATES ]] || fatal \
    "expected $NUM_CANDIDATES prepared masks, found ${#MASK_IDS[@]}"
for ((index = 0; index < NUM_CANDIDATES; index++)); do
    printf -v expected_id '%05d' "$index"
    [[ "${MASK_IDS[index]}" == "$expected_id" ]] || fatal "unexpected mask ID ${MASK_IDS[index]}"
    [[ -s "$CANDIDATE_MASKS/$expected_id/seg.nii.gz" ]] || fatal "missing mask $expected_id"
done

echo "Preflight OK [${LABEL}]: $NUM_CANDIDATES shared synthetic-mask candidates and checkpoint_latest.pt"
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
    real_seg_dir="$CANDIDATE_MASKS"
    expected_real_cases="$NUM_CANDIDATES"
    expected_real_depth=150
    seed="$SEED"
    num_images="$NUM_CANDIDATES"
    current_image="$START_CANDIDATE_VALUE"
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
    skip_failed_fixed_masks=true
    brain_atlas_path=null
    brain_support_pca_path="$BRAIN_SUPPORT_PCA"
    brain_pca_path=null
    seg_pca_path=null
    diffrs_checkpoint=null
    mask_outside_brain=false
    paths.generated_dir="$EVAL_ROOT"
    output_subdir="$OUTPUT_LABEL"
    verbose=true
)
if [[ -n "$HIGH_CKPT" ]]; then
    GEN_ARGS+=(image_model_high_t="$HIGH_CKPT" handoff_t="$HANDOFF_T")
fi

echo "=== Evaluating ${LABEL}: candidates ${START_CANDIDATE_VALUE}..$((NUM_CANDIDATES - 1)) "\
"(${NUM_CANDIDATES_TO_PROCESS} total), up to ${MAX_IMAGE_ATTEMPTS} image attempts each ==="
if [[ "$OUTPUT_LABEL" != "$LABEL" ]]; then
    echo "output shard:      $OUTPUT_LABEL"
fi
echo "masks:            $CANDIDATE_MASKS"
echo "brain support:    $BRAIN_SUPPORT_PCA (train105 PCA, 0.5 threshold, ${BRAIN_MARGIN_MM} mm margin)"
echo "low-t checkpoint: $LOW_CKPT"
if [[ -n "$HIGH_CKPT" ]]; then
    echo "high-t checkpoint: $HIGH_CKPT (t > $HANDOFF_T)"
fi
time python -m medgen.scripts.generate "${GEN_ARGS[@]}"

[[ -d "$FINAL_DIR" ]] || fatal "generator did not create output: $FINAL_DIR"
[[ -s "$FINAL_DIR/bins.csv" ]] || fatal "generator did not create bins.csv"
readonly EXPECTED_BINS_HEADER='id,bin_0,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,total_tumors'
[[ "$(head -n 1 "$FINAL_DIR/bins.csv")" == "$EXPECTED_BINS_HEADER" ]] || fatal \
    "unexpected bins.csv header in $FINAL_DIR"
SEG_COUNT="$(find "$FINAL_DIR" -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz | wc -l)"
BRAVO_COUNT="$(find "$FINAL_DIR" -mindepth 2 -maxdepth 2 -type f -name bravo.nii.gz | wc -l)"
BINS_COUNT="$(( $(wc -l < "$FINAL_DIR/bins.csv") - 1 ))"
[[ "$SEG_COUNT" -eq "$BRAVO_COUNT" ]] || fatal \
    "generated dataset contains $SEG_COUNT masks but $BRAVO_COUNT BRAVO volumes"
[[ "$BINS_COUNT" -eq "$BRAVO_COUNT" ]] || fatal \
    "bins.csv contains $BINS_COUNT samples but the dataset contains $BRAVO_COUNT BRAVO volumes"
[[ "$BRAVO_COUNT" -le "$NUM_CANDIDATES_TO_PROCESS" ]] || fatal \
    "generated dataset contains $BRAVO_COUNT BRAVO volumes for only "\
    "$NUM_CANDIDATES_TO_PROCESS requested candidates"

mapfile -t GENERATED_IDS < <(
    find "$FINAL_DIR" -mindepth 1 -maxdepth 1 -type d \
        -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort
)
mapfile -t BINS_IDS < <(awk -F, 'NR > 1 { print $1 }' "$FINAL_DIR/bins.csv")
[[ ${#GENERATED_IDS[@]} -eq "$BRAVO_COUNT" ]] || fatal \
    "found ${#GENERATED_IDS[@]} numeric sample directories but $BRAVO_COUNT BRAVO volumes"
[[ ${#BINS_IDS[@]} -eq ${#GENERATED_IDS[@]} ]] || fatal \
    "bins.csv ID count does not match generated sample directory count"
for ((index = 0; index < ${#GENERATED_IDS[@]}; index++)); do
    candidate_id="${GENERATED_IDS[index]}"
    candidate_number=$((10#$candidate_id))
    (( candidate_number >= START_CANDIDATE_VALUE && candidate_number < NUM_CANDIDATES )) || fatal \
        "generated candidate $candidate_id is outside requested range "\
        "${START_CANDIDATE_VALUE}..$((NUM_CANDIDATES - 1))"
    [[ "${BINS_IDS[index]}" == "$candidate_id" ]] || fatal \
        "bins.csv ID ${BINS_IDS[index]} does not match generated candidate $candidate_id"
done

readonly COMPLETION_MARKER="$FINAL_DIR/.candidate_range_complete"
{
    echo "label=$LABEL"
    echo "output_label=$OUTPUT_LABEL"
    echo "start_candidate=$START_CANDIDATE_VALUE"
    echo "stop_candidate=$NUM_CANDIDATES"
    echo "processed_candidates=$NUM_CANDIDATES_TO_PROCESS"
    echo "accepted_candidates=$BRAVO_COUNT"
} > "${COMPLETION_MARKER}.tmp"
mv -- "${COMPLETION_MARKER}.tmp" "$COMPLETION_MARKER"

echo "Completed candidate range: accepted $BRAVO_COUNT/$NUM_CANDIDATES_TO_PROCESS, "\
"skipped $((NUM_CANDIDATES_TO_PROCESS - BRAVO_COUNT))"
echo "Output: $FINAL_DIR"
