#!/usr/bin/env bash
# Merge one stopped exp48 head (0..299) with its completed H100 tail (300..524).

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

if [[ $# -ne 2 ]]; then
    fatal "usage: $0 LABEL HEAD_STDOUT_LOG"
fi

readonly LABEL="$1"
readonly HEAD_LOG="$2"
[[ "$LABEL" =~ ^exp1_to_exp48[a-e]_t025$ ]] || fatal "unsupported exp48 label: $LABEL"

readonly NUM_CANDIDATES=525
readonly START_CANDIDATE=300
readonly RANGE_SUFFIX=00300
readonly TAIL_LABEL="${LABEL}__tail_from_${RANGE_SUFFIX}"
readonly EXPECTED_BINS_HEADER='id,bin_0,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,total_tumors'
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly HEAD_DIR="${EVAL_ROOT}/${LABEL}"
readonly TAIL_DIR="${EVAL_ROOT}/${TAIL_LABEL}"
readonly TAIL_MARKER="${TAIL_DIR}/.candidate_range_complete"
readonly MERGE_TAG="${MERGE_TAG:-${SLURM_JOB_ID:-manual}}"
[[ "$MERGE_TAG" =~ ^[a-zA-Z0-9_.-]+$ ]] || fatal "unsafe merge tag: $MERGE_TAG"
readonly STAGING_DIR="${EVAL_ROOT}/${LABEL}.__merge_staging_${RANGE_SUFFIX}_${MERGE_TAG}"
readonly BACKUP_DIR="${EVAL_ROOT}/${LABEL}.__head_before_tail_${RANGE_SUFFIX}_${MERGE_TAG}"

[[ "${SYNTHMASK_HEAD_STOP_CONFIRMED:-}" == true ]] || fatal \
    "head-stop confirmation is missing; submit the merge through its Slurm wrapper"
[[ -d "$HEAD_DIR" ]] || fatal "head dataset not found: $HEAD_DIR"
[[ -d "$TAIL_DIR" ]] || fatal "tail dataset not found: $TAIL_DIR"
[[ -s "$HEAD_LOG" ]] || fatal "head stdout log not found: $HEAD_LOG"
[[ -s "$TAIL_DIR/bins.csv" ]] || fatal "tail bins.csv not found: $TAIL_DIR/bins.csv"
[[ -s "$TAIL_MARKER" ]] || fatal "tail completion marker not found: $TAIL_MARKER"
[[ ! -e "$STAGING_DIR" ]] || fatal "merge staging path already exists: $STAGING_DIR"
[[ ! -e "$BACKUP_DIR" ]] || fatal "head backup path already exists: $BACKUP_DIR"

HEAD_PROGRESS="$(
    sed -n 's/.*Progress: \([0-9][0-9]*\)\/525.*/\1/p' "$HEAD_LOG" | tail -n 1
)"
[[ "$HEAD_PROGRESS" =~ ^[0-9]+$ ]] || fatal "no candidate progress found in $HEAD_LOG"
(( HEAD_PROGRESS >= START_CANDIDATE )) || fatal \
    "head log reached only $HEAD_PROGRESS/$NUM_CANDIDATES; need at least "\
    "$START_CANDIDATE/$NUM_CANDIDATES"

grep -Fxq "label=$LABEL" "$TAIL_MARKER" || fatal "tail marker label mismatch"
grep -Fxq "output_label=$TAIL_LABEL" "$TAIL_MARKER" || fatal "tail marker output label mismatch"
grep -Fxq "start_candidate=$START_CANDIDATE" "$TAIL_MARKER" || fatal \
    "tail marker start candidate mismatch"
grep -Fxq "stop_candidate=$NUM_CANDIDATES" "$TAIL_MARKER" || fatal \
    "tail marker stop candidate mismatch"
grep -Fxq "processed_candidates=$((NUM_CANDIDATES - START_CANDIDATE))" \
    "$TAIL_MARKER" || fatal "tail marker processed-candidate count mismatch"

[[ "$(head -n 1 "$TAIL_DIR/bins.csv")" == "$EXPECTED_BINS_HEADER" ]] || fatal \
    "unexpected tail bins.csv header"

mapfile -t TAIL_IDS < <(
    find "$TAIL_DIR" -mindepth 1 -maxdepth 1 -type d \
        -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort
)
grep -Fxq "accepted_candidates=${#TAIL_IDS[@]}" "$TAIL_MARKER" || fatal \
    "tail marker accepted-candidate count mismatch"
mapfile -t TAIL_BINS_ROWS < <(tail -n +2 "$TAIL_DIR/bins.csv")
[[ ${#TAIL_BINS_ROWS[@]} -eq ${#TAIL_IDS[@]} ]] || fatal \
    "tail bins.csv has ${#TAIL_BINS_ROWS[@]} rows but tail contains ${#TAIL_IDS[@]} samples"

for ((index = 0; index < ${#TAIL_IDS[@]}; index++)); do
    candidate_id="${TAIL_IDS[index]}"
    candidate_number=$((10#$candidate_id))
    (( candidate_number >= START_CANDIDATE && candidate_number < NUM_CANDIDATES )) || fatal \
        "tail candidate $candidate_id is outside ${START_CANDIDATE}..$((NUM_CANDIDATES - 1))"
    [[ -s "$TAIL_DIR/$candidate_id/seg.nii.gz" ]] || fatal \
        "tail candidate $candidate_id has no complete segmentation"
    [[ -s "$TAIL_DIR/$candidate_id/bravo.nii.gz" ]] || fatal \
        "tail candidate $candidate_id has no complete BRAVO volume"
    [[ "${TAIL_BINS_ROWS[index]}" == "$candidate_id,0,0,0,0,0,0,0,0" ]] || fatal \
        "tail bins.csv row does not match fixed-mask candidate $candidate_id"
done

mapfile -t HEAD_NUMERIC_IDS < <(
    find "$HEAD_DIR" -mindepth 1 -maxdepth 1 -type d \
        -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort
)
HEAD_IDS=()
for candidate_id in "${HEAD_NUMERIC_IDS[@]}"; do
    candidate_number=$((10#$candidate_id))
    (( candidate_number < NUM_CANDIDATES )) || fatal "invalid head candidate ID: $candidate_id"
    if (( candidate_number >= START_CANDIDATE )); then
        # The tail owns this entire range. Ignore any complete or partial overlap
        # left by the original job after it logged progress 300.
        continue
    fi
    [[ -s "$HEAD_DIR/$candidate_id/seg.nii.gz" ]] || fatal \
        "owned head candidate $candidate_id has no complete segmentation"
    [[ -s "$HEAD_DIR/$candidate_id/bravo.nii.gz" ]] || fatal \
        "owned head candidate $candidate_id has no complete BRAVO volume"
    HEAD_IDS+=("$candidate_id")
done

mkdir -p "$STAGING_DIR"
readonly STAGING_BINS_TMP="$STAGING_DIR/bins.csv.tmp"
printf '%s\n' "$EXPECTED_BINS_HEADER" > "$STAGING_BINS_TMP"

link_candidate() {
    local source_dir="$1"
    local candidate_id="$2"
    local destination_dir="$STAGING_DIR/$candidate_id"

    mkdir -p "$destination_dir"
    ln -- "$source_dir/$candidate_id/seg.nii.gz" "$destination_dir/seg.nii.gz"
    ln -- "$source_dir/$candidate_id/bravo.nii.gz" "$destination_dir/bravo.nii.gz"
    printf '%s,0,0,0,0,0,0,0,0\n' "$candidate_id" >> "$STAGING_BINS_TMP"
}

for candidate_id in "${HEAD_IDS[@]}"; do
    link_candidate "$HEAD_DIR" "$candidate_id"
done
for candidate_id in "${TAIL_IDS[@]}"; do
    link_candidate "$TAIL_DIR" "$candidate_id"
done

mv -- "$STAGING_BINS_TMP" "$STAGING_DIR/bins.csv"
{
    echo "label=$LABEL"
    echo "split_candidate=$START_CANDIDATE"
    echo "head_progress=$HEAD_PROGRESS"
    echo "head_log=$HEAD_LOG"
    echo "head_source=$BACKUP_DIR"
    echo "tail_source=$TAIL_DIR"
    echo "head_accepted=${#HEAD_IDS[@]}"
    echo "tail_accepted=${#TAIL_IDS[@]}"
    echo "merged_accepted=$((${#HEAD_IDS[@]} + ${#TAIL_IDS[@]}))"
} > "$STAGING_DIR/.tail_merge_manifest"

MERGED_SEG_COUNT="$(find "$STAGING_DIR" -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz | wc -l)"
MERGED_BRAVO_COUNT="$(find "$STAGING_DIR" -mindepth 2 -maxdepth 2 -type f -name bravo.nii.gz | wc -l)"
MERGED_BINS_COUNT="$(( $(wc -l < "$STAGING_DIR/bins.csv") - 1 ))"
readonly EXPECTED_MERGED_COUNT=$((${#HEAD_IDS[@]} + ${#TAIL_IDS[@]}))
[[ "$MERGED_SEG_COUNT" -eq "$EXPECTED_MERGED_COUNT" ]] || fatal \
    "staged segmentation count mismatch: $MERGED_SEG_COUNT != $EXPECTED_MERGED_COUNT"
[[ "$MERGED_BRAVO_COUNT" -eq "$EXPECTED_MERGED_COUNT" ]] || fatal \
    "staged BRAVO count mismatch: $MERGED_BRAVO_COUNT != $EXPECTED_MERGED_COUNT"
[[ "$MERGED_BINS_COUNT" -eq "$EXPECTED_MERGED_COUNT" ]] || fatal \
    "staged bins.csv count mismatch: $MERGED_BINS_COUNT != $EXPECTED_MERGED_COUNT"

restore_head_on_failure() {
    status=$?
    if (( status == 0 )); then
        status=1
    fi
    if [[ ! -e "$HEAD_DIR" && -d "$BACKUP_DIR" ]]; then
        mv -- "$BACKUP_DIR" "$HEAD_DIR" || true
    fi
    exit "$status"
}
trap restore_head_on_failure EXIT INT TERM

mv -- "$HEAD_DIR" "$BACKUP_DIR"
mv -- "$STAGING_DIR" "$HEAD_DIR"
trap - EXIT INT TERM

echo "Merged $LABEL at fixed split $START_CANDIDATE"
echo "Head accepted: ${#HEAD_IDS[@]} (source retained at $BACKUP_DIR)"
echo "Tail accepted: ${#TAIL_IDS[@]} (source retained at $TAIL_DIR)"
echo "Canonical output: $HEAD_DIR"
