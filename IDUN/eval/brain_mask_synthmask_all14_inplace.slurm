#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="bm_sm_all14"
#SBATCH --output=IDUN/output/eval/brain_mask_sm_all14_%j.out
#SBATCH --error=IDUN/output/eval/brain_mask_sm_all14_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Apply the historical posthoc Stage-3-equivalent brain-mask zeroing directly
# to every accepted BRAVO in the canonical 14-generator panel. No derivative
# dataset is created. Segmentation masks and bins.csv are never modified.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly NUM_CANDIDATES=525
readonly BINS_HEADER='id,bin_0,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,total_tumors'
readonly MASK_MODE='in_place_atomic'
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly COMMON96_ROOT="${EVAL_ROOT}/matched_common96_all14"
readonly COMMON105_ROOT="${EVAL_ROOT}/matched_common105"
readonly COMPLETE_MARKER="${EVAL_ROOT}/.brain_mask_stage3_all14_complete"
readonly LOCK_FILE="${EVAL_ROOT}/.brain_mask_stage3_all14.lock"

readonly LABELS=(
    exp1_1_1000
    exp1_1_1000plus
    exp32_2_1000
    exp32_3_1000
    exp47a
    exp47b
    exp47c
    exp47d
    exp47e
    exp1_to_exp48a_t025
    exp1_to_exp48b_t025
    exp1_to_exp48c_t025
    exp1_to_exp48d_t025
    exp1_to_exp48e_t025
)
readonly EXPECTED_COUNTS=(151 134 171 140 154 138 163 158 159 153 149 151 150 153)
readonly COMMON105_LABELS=(
    exp1_1_1000
    exp1_1_1000plus
    exp32_2_1000
    exp1_to_exp48c_t025
    exp1_to_exp48d_t025
)

[[ -d "$EVAL_ROOT" ]] || fatal "evaluation root not found: $EVAL_ROOT"
[[ -d "$COMMON96_ROOT" ]] || fatal "matched common-96 view not found: $COMMON96_ROOT"
[[ -d "$COMMON105_ROOT" ]] || fatal "matched common-105 view not found: $COMMON105_ROOT"

# Prevent two in-place jobs from touching the panel concurrently.
exec 9>"$LOCK_FILE"
flock -n 9 || fatal "another all-14 brain-masking job holds $LOCK_FILE"

validate_dataset() {
    local label="$1"
    local expected_count="$2"
    local dataset="${EVAL_ROOT}/${label}"
    local candidate_id candidate_number index
    local -a case_ids bin_ids

    [[ -d "$dataset" && ! -L "$dataset" ]] || \
        fatal "canonical dataset is missing or is a symlink: $dataset"
    [[ -s "$dataset/bins.csv" ]] || fatal "bins.csv is missing: $label"
    [[ "$(head -n 1 "$dataset/bins.csv")" == "$BINS_HEADER" ]] || \
        fatal "unexpected bins.csv header: $label"

    mapfile -t case_ids < <(
        find -P "$dataset" -mindepth 1 -maxdepth 1 -type d \
            -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort
    )
    mapfile -t bin_ids < <(tail -n +2 "$dataset/bins.csv" | cut -d, -f1)
    [[ ${#case_ids[@]} -eq $expected_count ]] || \
        fatal "$label has ${#case_ids[@]} accepted cases; expected $expected_count"
    [[ ${#bin_ids[@]} -eq $expected_count ]] || \
        fatal "$label bins.csv has ${#bin_ids[@]} rows; expected $expected_count"

    for ((index = 0; index < expected_count; index++)); do
        candidate_id="${case_ids[index]}"
        [[ "${bin_ids[index]}" == "$candidate_id" ]] || \
            fatal "$label bins.csv IDs do not match its case directories"
        candidate_number=$((10#$candidate_id))
        (( candidate_number >= 0 && candidate_number < NUM_CANDIDATES )) || \
            fatal "$label has an out-of-range candidate ID: $candidate_id"
        [[ -s "$dataset/$candidate_id/bravo.nii.gz" && \
           ! -L "$dataset/$candidate_id/bravo.nii.gz" ]] || \
            fatal "$label canonical BRAVO is missing or is a symlink: $candidate_id"
        [[ -s "$dataset/$candidate_id/seg.nii.gz" ]] || \
            fatal "$label segmentation is missing: $candidate_id"
    done
}

validate_common_view() {
    local root="$1"
    local expected_cases="$2"
    shift 2
    local label candidate_id linked canonical
    local -a ids=( )

    mapfile -t ids < "$root/common_candidate_ids.txt"
    [[ ${#ids[@]} -eq $expected_cases ]] || \
        fatal "$root has ${#ids[@]} common IDs; expected $expected_cases"

    for label in "$@"; do
        [[ -d "$root/$label" ]] || fatal "matched view is missing label: $root/$label"
        for candidate_id in "${ids[@]}"; do
            linked="$root/$label/$candidate_id/bravo.nii.gz"
            canonical="$EVAL_ROOT/$label/$candidate_id/bravo.nii.gz"
            [[ -L "$linked" && -s "$linked" && "$linked" -ef "$canonical" ]] || \
                fatal "matched BRAVO does not target its canonical file: $label/$candidate_id"
        done
    done
}

for index in "${!LABELS[@]}"; do
    validate_dataset "${LABELS[index]}" "${EXPECTED_COUNTS[index]}"
done
validate_common_view "$COMMON96_ROOT" 96 "${LABELS[@]}"
validate_common_view "$COMMON105_ROOT" 105 "${COMMON105_LABELS[@]}"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== In-place brain masking for all 14 canonical generator datasets ==="
echo "Root:      $EVAL_ROOT"
echo "Threshold: 0.05"
echo "Dilation:  2 pixels"

for index in "${!LABELS[@]}"; do
    label="${LABELS[index]}"
    dataset="${EVAL_ROOT}/${label}"
    echo ""
    echo "--- $label (${EXPECTED_COUNTS[index]} BRAVOs) ---"
    python -m medgen.scripts.brain_mask_existing \
        --src "$dataset" \
        --in-place \
        --threshold 0.05 \
        --dilate-pixels 2 \
        --expected-shape 256 256 150

    marker="${dataset}/.brain_mask_stage3_applied"
    marker_tmp="${marker}.tmp_${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
    {
        echo "method=medgen.scripts.brain_mask_existing"
        echo "mode=${MASK_MODE}"
        echo "threshold=0.05"
        echo "fill_holes=true"
        echo "dilate_pixels=2"
        echo "accepted_cases=${EXPECTED_COUNTS[index]}"
        echo "job_id=${SLURM_JOB_ID}"
    } > "$marker_tmp"
    mv -f -- "$marker_tmp" "$marker"
done

# Atomic replacement leaves these symlink views attached to the canonical paths.
validate_common_view "$COMMON96_ROOT" 96 "${LABELS[@]}"
validate_common_view "$COMMON105_ROOT" 105 "${COMMON105_LABELS[@]}"

marker_tmp="${COMPLETE_MARKER}.tmp_${SLURM_JOB_ID}"
{
    echo "eval_id=${EVAL_ID}"
    echo "mode=${MASK_MODE}"
    echo "threshold=0.05"
    echo "fill_holes=true"
    echo "dilate_pixels=2"
    echo "generators=${#LABELS[@]}"
    echo "accepted_volumes=2124"
    echo "labels=${LABELS[*]}"
    echo "job_id=${SLURM_JOB_ID}"
} > "$marker_tmp"
mv -f -- "$marker_tmp" "$COMPLETE_MARKER"

echo ""
echo "Completed in-place masking of all 2,124 BRAVOs."
echo "No derivative dataset was created; segmentation masks were not changed."
echo "Common-96 and common-105 views still resolve to the canonical datasets."
