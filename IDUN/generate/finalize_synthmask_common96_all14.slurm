#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name="finalize_sm_c96_all14"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/finalize_sm_c96_all14_%j.out
#SBATCH --error=IDUN/output/generate/finalize_sm_c96_all14_%j.err

# Build the strict matched 96-case view accepted by all 14 generators.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly NUM_CANDIDATES=525
readonly EXPECTED_COMMON=96
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly CANDIDATE_MASKS="${EVAL_ROOT}/ordered_common_seg_masks_seed42/candidates525"
readonly FINAL_ROOT="${EVAL_ROOT}/matched_common96_all14"
readonly STAGING_ROOT="${FINAL_ROOT}.__staging_${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
readonly COMMON_IDS_FILE="${STAGING_ROOT}/common_candidate_ids.txt"
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
readonly NUM_GENERATORS=${#LABELS[@]}

[[ -d "$CANDIDATE_MASKS" ]] || fatal "candidate masks not found: $CANDIDATE_MASKS"
[[ ! -e "$FINAL_ROOT" ]] || fatal "matched dataset already exists: $FINAL_ROOT"
[[ ! -e "$STAGING_ROOT" ]] || fatal "matched staging path already exists: $STAGING_ROOT"

for label in "${LABELS[@]}"; do
    dataset_dir="${EVAL_ROOT}/${label}"
    [[ -d "$dataset_dir" ]] || fatal "generator dataset not found: $label"
    [[ -s "$dataset_dir/bins.csv" ]] || fatal \
        "generator did not finish all $NUM_CANDIDATES candidates: $label"
done

for label in exp1_to_exp48a_t025 exp1_to_exp48b_t025 exp1_to_exp48c_t025 \
    exp1_to_exp48d_t025 exp1_to_exp48e_t025; do
    [[ -s "$EVAL_ROOT/$label/.tail_merge_manifest" ]] || fatal \
        "$label has not been merged with its tail300 output"
done

COMMON_IDS=()
for ((candidate = 0; candidate < NUM_CANDIDATES; candidate++)); do
    printf -v candidate_id '%05d' "$candidate"
    [[ -s "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" ]] || fatal \
        "shared candidate mask is missing: $candidate_id"

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
        fi
    done

    if [[ "$accepted_by_all" == true ]]; then
        COMMON_IDS+=("$candidate_id")
    fi
done

[[ ${#COMMON_IDS[@]} -eq $EXPECTED_COMMON ]] || fatal \
    "expected exactly $EXPECTED_COMMON candidates accepted by all $NUM_GENERATORS "\
    "generators, found ${#COMMON_IDS[@]}"

mkdir "$STAGING_ROOT"
printf '%s\n' "${COMMON_IDS[@]}" > "$COMMON_IDS_FILE"

for label in "${LABELS[@]}"; do
    final_dataset="${STAGING_ROOT}/${label}"
    mkdir "$final_dataset"
    for candidate_id in "${COMMON_IDS[@]}"; do
        sample_dir="${final_dataset}/${candidate_id}"
        mkdir "$sample_dir"
        ln -s -- \
            "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" \
            "$sample_dir/seg.nii.gz"
        ln -s -- \
            "$EVAL_ROOT/$label/$candidate_id/bravo.nii.gz" \
            "$sample_dir/bravo.nii.gz"
        [[ "$sample_dir/seg.nii.gz" \
            -ef "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" ]] || fatal \
            "failed to link shared mask for $label/$candidate_id"
        [[ "$sample_dir/bravo.nii.gz" \
            -ef "$EVAL_ROOT/$label/$candidate_id/bravo.nii.gz" ]] || fatal \
            "failed to link original BRAVO for $label/$candidate_id"
    done

    case_count="$(find "$final_dataset" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    seg_count="$(
        find -L "$final_dataset" -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz | wc -l
    )"
    bravo_count="$(
        find -L "$final_dataset" -mindepth 2 -maxdepth 2 -type f -name bravo.nii.gz | wc -l
    )"
    [[ "$case_count" -eq $EXPECTED_COMMON ]] || fatal \
        "$label matched dataset contains $case_count cases, expected $EXPECTED_COMMON"
    [[ "$seg_count" -eq $EXPECTED_COMMON ]] || fatal \
        "$label matched dataset contains $seg_count masks, expected $EXPECTED_COMMON"
    [[ "$bravo_count" -eq $EXPECTED_COMMON ]] || fatal \
        "$label matched dataset contains $bravo_count BRAVO volumes, expected $EXPECTED_COMMON"
done

ids_sha256="$(sha256sum "$COMMON_IDS_FILE" | awk '{print $1}')"
{
    echo "eval_id=$EVAL_ID"
    echo "panel=all14_common96"
    echo "protocol=original_attempts_0_2"
    echo "candidate_pool=$NUM_CANDIDATES"
    echo "generators=$NUM_GENERATORS"
    echo "common_candidates=$EXPECTED_COMMON"
    echo "candidate_ids_sha256=$ids_sha256"
    echo "labels=${LABELS[*]}"
} > "$STAGING_ROOT/.selection_manifest"

mv -- "$STAGING_ROOT" "$FINAL_ROOT"

echo "Common candidates accepted by all $NUM_GENERATORS generators: "\
"${#COMMON_IDS[@]}/$NUM_CANDIDATES"
echo "Matched datasets: $FINAL_ROOT"
echo "Selected candidate IDs: $FINAL_ROOT/common_candidate_ids.txt"
