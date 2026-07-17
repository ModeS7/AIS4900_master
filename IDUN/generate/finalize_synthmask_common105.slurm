#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name="finalize_sm_common105"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/finalize_sm_common105_%j.out
#SBATCH --error=IDUN/output/generate/finalize_sm_common105_%j.err

# Build 14 matched 105-case datasets from the candidates accepted by every generator.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly NUM_CANDIDATES=150
readonly NUM_SELECTED=105
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask150_seed42_euler100}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly CANDIDATE_MASKS="${EVAL_ROOT}/ordered_common_seg_masks_seed42/candidates150"
readonly FINAL_ROOT="${EVAL_ROOT}/matched_common105"
readonly COMMON_IDS_FILE="${FINAL_ROOT}/common_candidate_ids.txt"
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

[[ -d "$CANDIDATE_MASKS" ]] || fatal "candidate masks not found: $CANDIDATE_MASKS"
[[ ! -e "$FINAL_ROOT" ]] || fatal "matched dataset already exists: $FINAL_ROOT"

for label in "${LABELS[@]}"; do
    dataset_dir="${EVAL_ROOT}/${label}"
    [[ -d "$dataset_dir" ]] || fatal "generator dataset not found: $label"
    [[ -s "$dataset_dir/bins.csv" ]] || fatal "generator did not finish all 150 candidates: $label"
done

COMMON_IDS=()
for ((candidate = 0; candidate < NUM_CANDIDATES; candidate++)); do
    printf -v candidate_id '%05d' "$candidate"
    [[ -s "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" ]] || \
        fatal "shared candidate mask is missing: $candidate_id"

    accepted_by_all=true
    for label in "${LABELS[@]}"; do
        sample_dir="${EVAL_ROOT}/${label}/${candidate_id}"
        has_seg=false
        has_bravo=false
        [[ -s "$sample_dir/seg.nii.gz" ]] && has_seg=true
        [[ -s "$sample_dir/bravo.nii.gz" ]] && has_bravo=true
        if [[ "$has_seg" != "$has_bravo" ]]; then
            fatal "$label candidate $candidate_id is incomplete"
        fi
        if [[ "$has_bravo" == false ]]; then
            accepted_by_all=false
            break
        fi
    done
    if [[ "$accepted_by_all" == true ]]; then
        COMMON_IDS+=("$candidate_id")
    fi
done

[[ ${#COMMON_IDS[@]} -ge $NUM_SELECTED ]] || fatal \
    "only ${#COMMON_IDS[@]} candidates were accepted by all 14 generators; need $NUM_SELECTED"

SELECTED_IDS=()
for ((index = 0; index < NUM_SELECTED; index++)); do
    SELECTED_IDS+=("${COMMON_IDS[index]}")
done

mkdir -p "$FINAL_ROOT"
printf '%s\n' "${SELECTED_IDS[@]}" > "$COMMON_IDS_FILE"

for label in "${LABELS[@]}"; do
    final_dataset="${FINAL_ROOT}/${label}"
    mkdir -p "$final_dataset"
    for candidate_id in "${SELECTED_IDS[@]}"; do
        sample_dir="${final_dataset}/${candidate_id}"
        mkdir -p "$sample_dir"
        ln -s -- \
            "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" \
            "$sample_dir/seg.nii.gz"
        ln -s -- \
            "$EVAL_ROOT/$label/$candidate_id/bravo.nii.gz" \
            "$sample_dir/bravo.nii.gz"
        [[ -s "$sample_dir/seg.nii.gz" && -s "$sample_dir/bravo.nii.gz" ]] || \
            fatal "failed to create matched case $label/$candidate_id"
    done

    case_count="$(find "$final_dataset" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    [[ "$case_count" -eq $NUM_SELECTED ]] || fatal \
        "$label matched dataset contains $case_count cases, expected $NUM_SELECTED"
done

echo "Common candidates accepted by all 14 generators: ${#COMMON_IDS[@]}/$NUM_CANDIDATES"
echo "Selected the first $NUM_SELECTED candidates in the fixed seed-42 order"
echo "Matched datasets: $FINAL_ROOT"
echo "Selected candidate IDs: $COMMON_IDS_FILE"
