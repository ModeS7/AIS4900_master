#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="bm_rs_all14"
#SBATCH --output=IDUN/output/eval/brain_mask_realseg_all14_%j.out
#SBATCH --error=IDUN/output/eval/brain_mask_realseg_all14_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Apply the historical Stage-3 brain-mask zeroing to all 14 datasets generated
# from the same 105 real training segmentation masks. BRAVOs are replaced
# atomically in place; segmentation masks and bins.csv are never modified.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly NUM_CASES=105
readonly TOTAL_BRAVOS=1470
readonly BINS_HEADER='id,bin_0,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,total_tumors'
readonly MASK_MODE='in_place_atomic'
readonly EVAL_ID="${EVAL_ID:-source_selection_train105_seed42_euler100}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly TRAIN_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/train"
readonly ORIGINAL_REPORT="${CLUSTER_BASE}/AIS4900_master/runs/eval/${EVAL_ID}/all_metrics_train105.json"
readonly COMPLETE_MARKER="${EVAL_ROOT}/.brain_mask_stage3_all14_complete"
readonly CONDITIONING_MAP="${EVAL_ROOT}/.real_seg_conditioning_case_map.tsv"
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

(( ${#LABELS[@]} * NUM_CASES == TOTAL_BRAVOS )) || fatal \
    "internal panel size does not equal $TOTAL_BRAVOS"

[[ -d "$EVAL_ROOT" ]] || fatal "real-seg evaluation panel not found: $EVAL_ROOT"
[[ -d "$TRAIN_DIR" ]] || fatal "real training data not found: $TRAIN_DIR"
[[ -s "$ORIGINAL_REPORT" ]] || fatal \
    "preserved unmasked metric report not found: $ORIGINAL_REPORT"
[[ ! -e "$COMPLETE_MARKER" ]] || fatal \
    "real-seg panel is already marked as brain-masked: $COMPLETE_MARKER"

# Prevent concurrent in-place jobs from touching the same panel.
exec 9>"$LOCK_FILE"
flock -n 9 || fatal "another real-seg brain-masking job holds $LOCK_FILE"

seg_tree_sha256() {
    local dataset="$1"
    (
        cd "$dataset"
        find . -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz -print0 |
            LC_ALL=C sort -z | xargs -0 sha256sum
    ) | sha256sum | awk '{print $1}'
}

validate_dataset() {
    local label="$1"
    local dataset="${EVAL_ROOT}/${label}"
    local expected_id index
    local -a case_ids bin_ids bin_rows all_dirs

    [[ -d "$dataset" && ! -L "$dataset" ]] || \
        fatal "canonical dataset is missing or is a symlink: $dataset"
    [[ -s "$dataset/bins.csv" && ! -L "$dataset/bins.csv" ]] || \
        fatal "regular bins.csv is missing: $label"
    [[ "$(head -n 1 "$dataset/bins.csv")" == "$BINS_HEADER" ]] || \
        fatal "unexpected bins.csv header: $label"

    mapfile -t all_dirs < <(
        find -P "$dataset" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
            LC_ALL=C sort
    )
    mapfile -t case_ids < <(
        find -P "$dataset" -mindepth 1 -maxdepth 1 -type d \
            -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort
    )
    mapfile -t bin_ids < <(tail -n +2 "$dataset/bins.csv" | cut -d, -f1)
    mapfile -t bin_rows < <(tail -n +2 "$dataset/bins.csv")

    [[ ${#all_dirs[@]} -eq $NUM_CASES ]] || \
        fatal "$label has ${#all_dirs[@]} case directories; expected $NUM_CASES"
    [[ ${#case_ids[@]} -eq $NUM_CASES ]] || \
        fatal "$label has ${#case_ids[@]} numeric case directories; expected $NUM_CASES"
    [[ ${#bin_ids[@]} -eq $NUM_CASES ]] || \
        fatal "$label bins.csv has ${#bin_ids[@]} rows; expected $NUM_CASES"

    for ((index = 0; index < NUM_CASES; index++)); do
        printf -v expected_id '%05d' "$index"
        [[ "${case_ids[index]}" == "$expected_id" ]] || \
            fatal "$label case IDs are not exactly 00000..00104"
        [[ "${bin_ids[index]}" == "$expected_id" ]] || \
            fatal "$label bins.csv IDs do not match 00000..00104"
        [[ "${bin_rows[index]}" == "${expected_id},0,0,0,0,0,0,0,0" ]] || \
            fatal "$label bins.csv is not the expected all-zero real-mask placeholder"
        [[ -s "$dataset/$expected_id/bravo.nii.gz" && \
           ! -L "$dataset/$expected_id/bravo.nii.gz" ]] || \
            fatal "$label canonical BRAVO is missing or is a symlink: $expected_id"
        [[ -s "$dataset/$expected_id/seg.nii.gz" && \
           ! -L "$dataset/$expected_id/seg.nii.gz" ]] || \
            fatal "$label segmentation is missing or is a symlink: $expected_id"
    done
}

declare -A SEG_SHA256_BEFORE BINS_SHA256_BEFORE
for label in "${LABELS[@]}"; do
    validate_dataset "$label"
    SEG_SHA256_BEFORE["$label"]="$(seg_tree_sha256 "$EVAL_ROOT/$label")"
    BINS_SHA256_BEFORE["$label"]="$(sha256sum "$EVAL_ROOT/$label/bins.csv" | awk '{print $1}')"
done

mapfile -t SOURCE_CASE_IDS < <(
    find "$TRAIN_DIR" -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz \
        -printf '%h\n' | sed 's|.*/||' | LC_ALL=C sort
)
[[ ${#SOURCE_CASE_IDS[@]} -eq $NUM_CASES ]] || fatal \
    "real training data contains ${#SOURCE_CASE_IDS[@]} segmentations; expected $NUM_CASES"
conditioning_map_tmp="${CONDITIONING_MAP}.tmp_${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
{
    echo -e "generated_id\treal_case_id"
    for ((index = 0; index < NUM_CASES; index++)); do
        printf '%05d\t%s\n' "$index" "${SOURCE_CASE_IDS[index]}"
    done
} > "$conditioning_map_tmp"
mv -f -- "$conditioning_map_tmp" "$CONDITIONING_MAP"
readonly CONDITIONING_MAP_SHA256="$(sha256sum "$CONDITIONING_MAP" | awk '{print $1}')"
readonly ORIGINAL_REPORT_SHA256="$(sha256sum "$ORIGINAL_REPORT" | awk '{print $1}')"
readonly REPOSITORY_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONSAFEPATH=1

echo "=== In-place brain masking for the real-seg all-14 panel ==="
echo "Root:      $EVAL_ROOT"
echo "Cases:     ${#LABELS[@]} x ${NUM_CASES} = ${TOTAL_BRAVOS}"
echo "Threshold: 0.05"
echo "Dilation:  2 pixels"

for label in "${LABELS[@]}"; do
    dataset="${EVAL_ROOT}/${label}"
    echo ""
    echo "--- $label (${NUM_CASES} BRAVOs) ---"
    python -m medgen.scripts.brain_mask_existing \
        --src "$dataset" \
        --in-place \
        --threshold 0.05 \
        --dilate-pixels 2 \
        --expected-shape 256 256 150

    validate_dataset "$label"
    [[ "$(seg_tree_sha256 "$dataset")" == "${SEG_SHA256_BEFORE[$label]}" ]] || \
        fatal "$label segmentation files changed during BRAVO masking"
    [[ "$(sha256sum "$dataset/bins.csv" | awk '{print $1}')" == \
       "${BINS_SHA256_BEFORE[$label]}" ]] || \
        fatal "$label bins.csv changed during BRAVO masking"
    marker="${dataset}/.brain_mask_stage3_applied"
    marker_tmp="${marker}.tmp_${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
    {
        echo "method=medgen.scripts.brain_mask_existing"
        echo "mode=${MASK_MODE}"
        echo "threshold=0.05"
        echo "fill_holes=true"
        echo "dilate_pixels=2"
        echo "conditioning=real_train_segmentation"
        echo "accepted_cases=${NUM_CASES}"
        echo "seg_tree_sha256=${SEG_SHA256_BEFORE[$label]}"
        echo "bins_sha256=${BINS_SHA256_BEFORE[$label]}"
        echo "job_id=${SLURM_JOB_ID}"
    } > "$marker_tmp"
    mv -f -- "$marker_tmp" "$marker"
done

marker_tmp="${COMPLETE_MARKER}.tmp_${SLURM_JOB_ID}"
{
    echo "eval_id=${EVAL_ID}"
    echo "mode=${MASK_MODE}"
    echo "threshold=0.05"
    echo "fill_holes=true"
    echo "dilate_pixels=2"
    echo "conditioning=real_train_segmentation"
    echo "generators=${#LABELS[@]}"
    echo "generated_cases_per_dataset=${NUM_CASES}"
    echo "generated_volumes=${TOTAL_BRAVOS}"
    echo "conditioning_map=${CONDITIONING_MAP}"
    echo "conditioning_map_sha256=${CONDITIONING_MAP_SHA256}"
    echo "unmasked_report=${ORIGINAL_REPORT}"
    echo "unmasked_report_sha256=${ORIGINAL_REPORT_SHA256}"
    echo "repository_commit=${REPOSITORY_COMMIT}"
    echo "labels=${LABELS[*]}"
    echo "job_id=${SLURM_JOB_ID}"
} > "$marker_tmp"
mv -f -- "$marker_tmp" "$COMPLETE_MARKER"

echo ""
echo "Completed in-place masking of all ${TOTAL_BRAVOS} real-mask-conditioned BRAVOs."
echo "Segmentation masks and bins.csv were not modified."
