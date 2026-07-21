#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="prep_c105_e47a"
#SBATCH --output=IDUN/output/train/downstream/nnunet/exp16_2_common105_exp47a_d654/prepare_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/exp16_2_common105_exp47a_d654/prepare_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Prepare recovered exp47a for the controlled exp16_2 comparison without
# generating any image. This validates the recovery-created frozen 105-case
# view, converts Dataset654, and applies the exact Dataset600 plan used by exp3
# and exp6_2.

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

readonly LABEL="exp47a"
readonly DATASET_ID=654
readonly EXPECTED_CASES=105
readonly RECOVERED_CANDIDATE="00400"
readonly EXPECTED_CANONICAL_CASES=155
readonly EXPECTED_PANEL_VOLUMES=2125
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
readonly CASE_RECOVERY_MARKER="${CANONICAL_DIR}/${RECOVERED_CANDIDATE}/.recovery_provenance"
readonly VIEW_RECOVERY_MARKER="${MATCHED_VIEW}/.recovery_provenance"
readonly RECOVERY_LOCK="${EVAL_ROOT}/.brain_mask_stage3_all14.lock"
readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
readonly RAW_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet"
readonly PREPROCESSED_DIR="${NNUNET_BASE}/nnUNet_preprocessed/Dataset${DATASET_ID}_BrainMet"
readonly PREPROCESS_MARKER="${PREPROCESSED_DIR}/.exp16_2_d600_preprocess_complete"
readonly CONVERT_SCRIPT="${REPO_ROOT}/IDUN/train/downstream/nnunet/convert_synthmask_common105_panel.slurm"
readonly PREPROCESS_SCRIPT="${REPO_ROOT}/IDUN/train/downstream/nnunet/preprocess_synthmask_common105_panel.slurm"

exec 9>"$RECOVERY_LOCK"
flock -n 9 || fatal "exp47a recovery or brain masking is still modifying the panel"

[[ -d "$MATCHED_ROOT" && ! -L "$MATCHED_ROOT" ]] || fatal \
    "matched common-105 root is missing or is a symlink: $MATCHED_ROOT"
[[ -s "$COMMON_IDS_FILE" ]] || fatal \
    "frozen common-105 ID list is missing: $COMMON_IDS_FILE"
[[ -d "$CANONICAL_DIR" && ! -L "$CANONICAL_DIR" ]] || fatal \
    "canonical exp47a dataset is missing: $CANONICAL_DIR"
[[ -s "$PANEL_MASK_MARKER" ]] || fatal \
    "all-14 brain masking is incomplete: $PANEL_MASK_MARKER"
[[ -s "$DATASET_MASK_MARKER" ]] || fatal \
    "exp47a brain masking is incomplete: $DATASET_MASK_MARKER"
require_marker_value "$PANEL_MASK_MARKER" mode in_place_atomic
require_marker_value "$PANEL_MASK_MARKER" threshold 0.05
require_marker_value "$PANEL_MASK_MARKER" fill_holes true
require_marker_value "$PANEL_MASK_MARKER" dilate_pixels 2
require_marker_value "$DATASET_MASK_MARKER" threshold 0.05
require_marker_value "$DATASET_MASK_MARKER" fill_holes true
require_marker_value "$DATASET_MASK_MARKER" dilate_pixels 2
require_marker_value "$DATASET_MASK_MARKER" accepted_cases "$EXPECTED_CANONICAL_CASES"
require_marker_value "$DATASET_MASK_MARKER" extension_candidate "$RECOVERED_CANDIDATE"
require_marker_value "$PANEL_MASK_MARKER" accepted_volumes "$EXPECTED_PANEL_VOLUMES"
require_marker_value "$PANEL_MASK_MARKER" extension_label "$LABEL"
require_marker_value "$PANEL_MASK_MARKER" extension_candidate "$RECOVERED_CANDIDATE"

mapfile -t COMMON_IDS < "$COMMON_IDS_FILE"
[[ ${#COMMON_IDS[@]} -eq $EXPECTED_CASES ]] || fatal \
    "frozen common-105 list has ${#COMMON_IDS[@]} rows; expected $EXPECTED_CASES"
[[ "$(printf '%s\n' "${COMMON_IDS[@]}" | LC_ALL=C sort -u | wc -l)" \
    -eq $EXPECTED_CASES ]] || fatal "frozen common-105 IDs are not unique"

validate_view() {
    local view="$1"
    local candidate_id case_count

    [[ -d "$view" && ! -L "$view" ]] || fatal \
        "matched exp47a view is missing or is a symlink: $view"
    case_count="$(
        find -P "$view" -mindepth 1 -maxdepth 1 -type d \
            -name '[0-9][0-9][0-9][0-9][0-9]' | wc -l
    )"
    [[ "$case_count" -eq $EXPECTED_CASES ]] || fatal \
        "matched exp47a view has $case_count cases; expected $EXPECTED_CASES"

    while IFS= read -r candidate_id; do
        [[ "$candidate_id" =~ ^[0-9]{5}$ ]] || fatal \
            "invalid frozen candidate ID: $candidate_id"
        [[ -s "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" ]] || fatal \
            "shared candidate mask is missing: $candidate_id"
        [[ -s "$CANONICAL_DIR/$candidate_id/bravo.nii.gz" ]] || fatal \
            "exp47a is missing frozen BRAVO candidate $candidate_id"
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

[[ -s "$CASE_RECOVERY_MARKER" ]] || fatal \
    "canonical exp47a/00400 recovery provenance is missing: $CASE_RECOVERY_MARKER"
[[ -s "$VIEW_RECOVERY_MARKER" ]] || fatal \
    "exp47a common-105 recovery provenance is missing: $VIEW_RECOVERY_MARKER"
cmp -s "$CASE_RECOVERY_MARKER" "$VIEW_RECOVERY_MARKER" || fatal \
    "canonical and common-105 recovery provenance differ"
[[ "$CANONICAL_DIR/$RECOVERED_CANDIDATE/seg.nii.gz" \
    -ef "$CANDIDATE_MASKS/$RECOVERED_CANDIDATE/seg.nii.gz" ]] || fatal \
    "canonical recovered segmentation is not frozen candidate $RECOVERED_CANDIDATE"

recovery_job="$(
    awk -F= '$1 == "job_id" { print substr($0, index($0, "=") + 1) }' \
        "$CASE_RECOVERY_MARKER"
)"
[[ "$recovery_job" =~ ^[0-9]+$ ]] || fatal \
    "canonical recovery provenance has an invalid job_id: $recovery_job"

require_marker_value "$CASE_RECOVERY_MARKER" candidate_id "$RECOVERED_CANDIDATE"
require_marker_value "$CASE_RECOVERY_MARKER" job_id "$recovery_job"
require_marker_value "$CASE_RECOVERY_MARKER" selection_rule first_passing_draw_in_4_to_50
require_marker_value "$CASE_RECOVERY_MARKER" brain_threshold 0.05
require_marker_value "$CASE_RECOVERY_MARKER" brain_fill_holes true
require_marker_value "$CASE_RECOVERY_MARKER" brain_dilate_pixels 2
require_marker_value "$VIEW_RECOVERY_MARKER" candidate_id "$RECOVERED_CANDIDATE"
require_marker_value "$VIEW_RECOVERY_MARKER" job_id "$recovery_job"
require_marker_value "$DATASET_MASK_MARKER" extension_job_id "$recovery_job"
require_marker_value "$PANEL_MASK_MARKER" extension_job_id "$recovery_job"

mask_sha256="$(sha256sum "$CANONICAL_DIR/$RECOVERED_CANDIDATE/seg.nii.gz" | awk '{print $1}')"
bravo_sha256="$(sha256sum "$CANONICAL_DIR/$RECOVERED_CANDIDATE/bravo.nii.gz" | awk '{print $1}')"
require_marker_value "$CASE_RECOVERY_MARKER" mask_sha256 "$mask_sha256"
require_marker_value "$CASE_RECOVERY_MARKER" masked_bravo_sha256 "$bravo_sha256"

validate_view "$MATCHED_VIEW"
echo "Validated the recovery-created exp47a common-105 view from job $recovery_job."
flock -u 9
exec 9>&-

export EXP16_LABEL_OVERRIDE="$LABEL"
export EXP16_DATASET_ID_OVERRIDE="$DATASET_ID"

if [[ -e "$RAW_DIR" || -L "$RAW_DIR" ]]; then
    [[ -d "$RAW_DIR" && -s "$RAW_DIR/source_selection.txt" ]] || fatal \
        "Dataset${DATASET_ID} raw directory is incomplete: $RAW_DIR"
    grep -Fxq "generator=${LABEL}" "$RAW_DIR/source_selection.txt" || fatal \
        "existing Dataset${DATASET_ID} raw data is not exp47a"
    grep -Fxq "dataset_id=${DATASET_ID}" "$RAW_DIR/source_selection.txt" || fatal \
        "existing Dataset${DATASET_ID} raw provenance has the wrong ID"
    [[ "$RAW_DIR/source_selection.txt" -nt "$VIEW_RECOVERY_MARKER" ]] || fatal \
        "existing Dataset${DATASET_ID} raw data predates the exp47a recovery"
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
    [[ "$PREPROCESS_MARKER" -nt "$RAW_DIR/source_selection.txt" ]] || fatal \
        "Dataset${DATASET_ID} preprocessing predates its converted raw data"
    [[ "$PREPROCESS_MARKER" -nt "$VIEW_RECOVERY_MARKER" ]] || fatal \
        "Dataset${DATASET_ID} preprocessing predates the exp47a recovery"
    echo "Dataset${DATASET_ID} controlled preprocessing is already complete."
else
    bash "$PREPROCESS_SCRIPT"
fi

echo "exp47a Dataset${DATASET_ID} is ready for the five-fold exp16_2 training job."
