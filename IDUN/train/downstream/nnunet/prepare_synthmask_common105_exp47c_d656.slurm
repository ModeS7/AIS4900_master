#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="prep_c105_e47c"
#SBATCH --output=IDUN/output/train/downstream/nnunet/exp16_2_common105_exp47c_d656/prepare_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/exp16_2_common105_exp47c_d656/prepare_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Prepare exp47c for the controlled exp16_2 comparison without generating any
# new image. This creates the frozen 105-case symlink view, converts Dataset656,
# and applies the exact Dataset600 plan used by exp3 and exp6_2.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

require_marker_value() {
    local marker="$1"
    local key="$2"
    local expected="$3"
    local -a values=()

    mapfile -t values < <(
        awk -F= -v key="$key" \
            '$1 == key { print substr($0, index($0, "=") + 1) }' "$marker"
    )
    [[ ${#values[@]} -eq 1 ]] || fatal \
        "$marker must contain exactly one ${key}=... entry"
    [[ "${values[0]}" == "$expected" ]] || fatal \
        "$marker has ${key}=${values[0]}; expected ${key}=${expected}"
}

readonly LABEL="exp47c"
readonly DATASET_ID=656
readonly EXPECTED_CASES=105
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly PANEL_MASK_MARKER="${EVAL_ROOT}/.brain_mask_stage3_all14_complete"
readonly CANONICAL_DIR="${EVAL_ROOT}/${LABEL}"
readonly DATASET_MASK_MARKER="${CANONICAL_DIR}/.brain_mask_stage3_applied"
readonly CANDIDATE_MASKS="${EVAL_ROOT}/ordered_common_seg_masks_seed42/candidates525"
readonly MATCHED_ROOT="${EVAL_ROOT}/matched_common105"
readonly COMMON_IDS_FILE="${MATCHED_ROOT}/common_candidate_ids.txt"
readonly MATCHED_VIEW="${MATCHED_ROOT}/${LABEL}"
readonly VIEW_LOCK="${MATCHED_ROOT}/.${LABEL}_view.lock"
readonly STAGED_VIEW="${MATCHED_ROOT}/.${LABEL}.prepare_${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
readonly RAW_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet"
readonly PREPROCESSED_DIR="${NNUNET_BASE}/nnUNet_preprocessed/Dataset${DATASET_ID}_BrainMet"
readonly PREPROCESS_MARKER="${PREPROCESSED_DIR}/.exp16_2_d600_preprocess_complete"
readonly CONVERT_SCRIPT="${REPO_ROOT}/IDUN/train/downstream/nnunet/convert_synthmask_common105_panel.slurm"
readonly PREPROCESS_SCRIPT="${REPO_ROOT}/IDUN/train/downstream/nnunet/preprocess_synthmask_common105_panel.slurm"

[[ -d "$MATCHED_ROOT" && ! -L "$MATCHED_ROOT" ]] || fatal \
    "matched common-105 root is missing or is a symlink: $MATCHED_ROOT"
[[ -s "$COMMON_IDS_FILE" ]] || fatal \
    "frozen common-105 ID list is missing: $COMMON_IDS_FILE"
[[ -d "$CANONICAL_DIR" && ! -L "$CANONICAL_DIR" ]] || fatal \
    "canonical exp47c dataset is missing: $CANONICAL_DIR"
[[ -s "$PANEL_MASK_MARKER" ]] || fatal \
    "all-14 brain masking is incomplete: $PANEL_MASK_MARKER"
[[ -s "$DATASET_MASK_MARKER" ]] || fatal \
    "exp47c brain masking is incomplete: $DATASET_MASK_MARKER"
require_marker_value "$PANEL_MASK_MARKER" mode in_place_atomic
require_marker_value "$PANEL_MASK_MARKER" threshold 0.05
require_marker_value "$PANEL_MASK_MARKER" fill_holes true
require_marker_value "$PANEL_MASK_MARKER" dilate_pixels 2
require_marker_value "$DATASET_MASK_MARKER" threshold 0.05
require_marker_value "$DATASET_MASK_MARKER" fill_holes true
require_marker_value "$DATASET_MASK_MARKER" dilate_pixels 2

mapfile -t COMMON_IDS < "$COMMON_IDS_FILE"
[[ ${#COMMON_IDS[@]} -eq $EXPECTED_CASES ]] || fatal \
    "frozen common-105 list has ${#COMMON_IDS[@]} rows; expected $EXPECTED_CASES"
[[ "$(printf '%s\n' "${COMMON_IDS[@]}" | LC_ALL=C sort -u | wc -l)" \
    -eq $EXPECTED_CASES ]] || fatal "frozen common-105 IDs are not unique"

validate_view() {
    local view="$1"
    local candidate_id case_count

    [[ -d "$view" && ! -L "$view" ]] || fatal \
        "matched exp47c view is missing or is a symlink: $view"
    case_count="$(
        find -P "$view" -mindepth 1 -maxdepth 1 -type d \
            -name '[0-9][0-9][0-9][0-9][0-9]' | wc -l
    )"
    [[ "$case_count" -eq $EXPECTED_CASES ]] || fatal \
        "matched exp47c view has $case_count cases; expected $EXPECTED_CASES"

    while IFS= read -r candidate_id; do
        [[ "$candidate_id" =~ ^[0-9]{5}$ ]] || fatal \
            "invalid frozen candidate ID: $candidate_id"
        [[ -s "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" ]] || fatal \
            "shared candidate mask is missing: $candidate_id"
        [[ -s "$CANONICAL_DIR/$candidate_id/bravo.nii.gz" ]] || fatal \
            "exp47c is missing frozen BRAVO candidate $candidate_id"
        [[ "$view/$candidate_id/seg.nii.gz" \
            -ef "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" ]] || fatal \
            "matched segmentation link is wrong: $candidate_id"
        [[ "$view/$candidate_id/bravo.nii.gz" \
            -ef "$CANONICAL_DIR/$candidate_id/bravo.nii.gz" ]] || fatal \
            "matched BRAVO link is wrong: $candidate_id"
        [[ "$(find -P "$view/$candidate_id" -mindepth 1 -maxdepth 1 | wc -l)" \
            -eq 2 ]] || fatal \
            "matched case contains files other than BRAVO and segmentation: $candidate_id"
    done < "$COMMON_IDS_FILE"
}

exec 9>"$VIEW_LOCK"
flock -n 9 || fatal "another job is modifying the exp47c common-105 view"

if [[ -e "$MATCHED_VIEW" || -L "$MATCHED_VIEW" ]]; then
    validate_view "$MATCHED_VIEW"
    echo "Matched exp47c common-105 view already exists and is valid."
else
    [[ ! -e "$STAGED_VIEW" && ! -L "$STAGED_VIEW" ]] || fatal \
        "staging path already exists: $STAGED_VIEW"
    cleanup_staged_view() {
        rm -rf -- "$STAGED_VIEW"
    }
    trap cleanup_staged_view EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    mkdir "$STAGED_VIEW"
    for candidate_id in "${COMMON_IDS[@]}"; do
        [[ "$candidate_id" =~ ^[0-9]{5}$ ]] || fatal \
            "invalid frozen candidate ID: $candidate_id"
        [[ -s "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" ]] || fatal \
            "shared candidate mask is missing: $candidate_id"
        [[ -s "$CANONICAL_DIR/$candidate_id/bravo.nii.gz" ]] || fatal \
            "exp47c is missing frozen BRAVO candidate $candidate_id"
        mkdir "$STAGED_VIEW/$candidate_id"
        ln -s -- \
            "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" \
            "$STAGED_VIEW/$candidate_id/seg.nii.gz"
        ln -s -- \
            "$CANONICAL_DIR/$candidate_id/bravo.nii.gz" \
            "$STAGED_VIEW/$candidate_id/bravo.nii.gz"
    done
    validate_view "$STAGED_VIEW"
    mv -- "$STAGED_VIEW" "$MATCHED_VIEW"
    trap - EXIT HUP INT TERM
    echo "Created matched exp47c common-105 view from existing images."
fi

export EXP16_LABEL_OVERRIDE="$LABEL"
export EXP16_DATASET_ID_OVERRIDE="$DATASET_ID"

if [[ -e "$RAW_DIR" || -L "$RAW_DIR" ]]; then
    [[ -d "$RAW_DIR" && -s "$RAW_DIR/source_selection.txt" ]] || fatal \
        "Dataset${DATASET_ID} raw directory is incomplete: $RAW_DIR"
    grep -Fxq "generator=${LABEL}" "$RAW_DIR/source_selection.txt" || fatal \
        "existing Dataset${DATASET_ID} raw data is not exp47c"
    grep -Fxq "dataset_id=${DATASET_ID}" "$RAW_DIR/source_selection.txt" || fatal \
        "existing Dataset${DATASET_ID} raw provenance has the wrong ID"
    echo "Dataset${DATASET_ID} raw data already exists; preprocessing will validate it."
else
    [[ ! -e "$PREPROCESSED_DIR" && ! -L "$PREPROCESSED_DIR" ]] || fatal \
        "preprocessed Dataset${DATASET_ID} exists without its raw dataset"
    bash "$CONVERT_SCRIPT"
fi

if [[ -s "$PREPROCESS_MARKER" ]]; then
    require_marker_value "$PREPROCESS_MARKER" dataset_id "$DATASET_ID"
    require_marker_value "$PREPROCESS_MARKER" generator "$LABEL"
    require_marker_value "$PREPROCESS_MARKER" target_plans nnUNetResEncUNetLPlansD600
    echo "Dataset${DATASET_ID} controlled preprocessing is already complete."
else
    bash "$PREPROCESS_SCRIPT"
fi

echo "exp47c Dataset${DATASET_ID} is ready for the five-fold exp16_2 training job."
