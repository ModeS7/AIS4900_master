#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="sm_all14_c96_metrics"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/sm_all14_c96_metrics_%j.out
#SBATCH --error=IDUN/output/generate/sm_all14_c96_metrics_%j.err

# Evaluate the strict 96 cases accepted by all 14 generators.
# Submit only after the separate common-96 finalizer has completed.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly NUM_CANDIDATES=525
readonly NUM_COMMON=96
readonly NUM_REFERENCE=105
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly TRAIN_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/train"
readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly CANDIDATE_MASKS="${EVAL_ROOT}/ordered_common_seg_masks_seed42/candidates525"
readonly MATCHED_ROOT="${EVAL_ROOT}/matched_common96_all14"
readonly COMMON_IDS_SOURCE="${MATCHED_ROOT}/common_candidate_ids.txt"
readonly RESULT_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/eval"
readonly FINAL_DIR="${RESULT_ROOT}/${EVAL_ID}_all14_common96"
readonly REPORT_NAME="all_metrics_train105_all14_common96.json"
readonly COMMON_IDS_RESULT_NAME="common_candidate_ids.txt"
readonly INPUT_MANIFEST_NAME="metric_inputs.txt"
readonly STAGING_DIR="${FINAL_DIR}.__staging_${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
readonly REPORT_STAGING="${STAGING_DIR}/${REPORT_NAME}"
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

[[ -f "$REPO_ROOT/pyproject.toml" ]] || fatal "repository root not found: $REPO_ROOT"
[[ -d "$TRAIN_DIR" ]] || fatal "training reference not found: $TRAIN_DIR"
[[ -d "$CANDIDATE_MASKS" ]] || fatal "candidate masks not found: $CANDIDATE_MASKS"
[[ -d "$MATCHED_ROOT" ]] || fatal \
    "matched all-14/common-96 dataset not found: $MATCHED_ROOT"
[[ -s "$COMMON_IDS_SOURCE" ]] || fatal \
    "common candidate ID list not found: $COMMON_IDS_SOURCE"
[[ ! -e "$FINAL_DIR" ]] || fatal "metric output already exists: $FINAL_DIR"
[[ ! -e "$STAGING_DIR" ]] || fatal "metric staging path already exists: $STAGING_DIR"

reference_count="$(
    find "$TRAIN_DIR" -mindepth 2 -maxdepth 2 -type f -name bravo.nii.gz | wc -l
)"
[[ "$reference_count" -eq $NUM_REFERENCE ]] || fatal \
    "training reference contains $reference_count BRAVO volumes, expected $NUM_REFERENCE"

mapfile -t COMMON_IDS < "$COMMON_IDS_SOURCE"
[[ ${#COMMON_IDS[@]} -eq $NUM_COMMON ]] || fatal \
    "common candidate ID list contains ${#COMMON_IDS[@]} entries, expected $NUM_COMMON"

previous_id=""
for candidate_id in "${COMMON_IDS[@]}"; do
    [[ "$candidate_id" =~ ^[0-9]{5}$ ]] || fatal "invalid candidate ID: $candidate_id"
    candidate_number=$((10#$candidate_id))
    (( candidate_number >= 0 && candidate_number < NUM_CANDIDATES )) || fatal \
        "candidate ID is outside 00000..00524: $candidate_id"
    if [[ -n "$previous_id" ]]; then
        [[ "$previous_id" < "$candidate_id" ]] || fatal \
            "common candidate IDs are duplicated or not in fixed numeric order"
    fi
    previous_id="$candidate_id"
    [[ -s "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" ]] || fatal \
        "shared candidate mask is missing: $candidate_id"
done

for label in exp1_to_exp48a_t025 exp1_to_exp48b_t025 exp1_to_exp48c_t025 \
    exp1_to_exp48d_t025 exp1_to_exp48e_t025; do
    [[ -s "$EVAL_ROOT/$label/.tail_merge_manifest" ]] || fatal \
        "$label has not been finalized with its tail300 output"
done

DATASET_ARGS=()
for label in "${LABELS[@]}"; do
    canonical_dir="${EVAL_ROOT}/${label}"
    dataset_dir="${MATCHED_ROOT}/${label}"
    [[ -d "$canonical_dir" ]] || fatal "canonical generator dataset not found: $label"
    [[ -s "$canonical_dir/bins.csv" ]] || fatal \
        "canonical generator dataset has no bins.csv: $label"
    [[ -d "$dataset_dir" ]] || fatal "matched dataset not found: $label"

    mapfile -t DATASET_IDS < <(
        find "$dataset_dir" -mindepth 1 -maxdepth 1 -type d \
            -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort
    )
    [[ ${#DATASET_IDS[@]} -eq $NUM_COMMON ]] || fatal \
        "$label contains ${#DATASET_IDS[@]} case directories, expected $NUM_COMMON"

    for ((index = 0; index < NUM_COMMON; index++)); do
        candidate_id="${COMMON_IDS[index]}"
        [[ "${DATASET_IDS[index]}" == "$candidate_id" ]] || fatal \
            "$label case IDs do not match the frozen common-96 list"
        [[ -s "$dataset_dir/$candidate_id/seg.nii.gz" ]] || fatal \
            "$label is missing segmentation candidate $candidate_id"
        [[ -s "$dataset_dir/$candidate_id/bravo.nii.gz" ]] || fatal \
            "$label is missing BRAVO candidate $candidate_id"
        [[ "$dataset_dir/$candidate_id/seg.nii.gz" \
            -ef "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" ]] || fatal \
            "$label segmentation $candidate_id does not reference the shared mask"
        [[ "$dataset_dir/$candidate_id/bravo.nii.gz" \
            -ef "$canonical_dir/$candidate_id/bravo.nii.gz" ]] || fatal \
            "$label BRAVO $candidate_id does not reference the original accepted output"
    done

    seg_count="$(
        find -L "$dataset_dir" -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz | wc -l
    )"
    bravo_count="$(
        find -L "$dataset_dir" -mindepth 2 -maxdepth 2 -type f -name bravo.nii.gz | wc -l
    )"
    [[ "$seg_count" -eq $NUM_COMMON ]] || fatal \
        "$label contains $seg_count segmentation volumes, expected $NUM_COMMON"
    [[ "$bravo_count" -eq $NUM_COMMON ]] || fatal \
        "$label contains $bravo_count BRAVO volumes, expected $NUM_COMMON"
    DATASET_ARGS+=(--dataset "${label}:${dataset_dir}")
done

mkdir -p "$RESULT_ROOT"
mkdir "$STAGING_DIR"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Computing train-reference metrics for all 14 datasets on the strict common 96 ==="
time python -m medgen.scripts.eval_all_metrics \
    --reference "train:${TRAIN_DIR}" \
    "${DATASET_ARGS[@]}" \
    --pattern bravo.nii.gz \
    --target-shape 150 256 256 \
    --pool-cap 0 \
    --ref-cap 0 \
    --diversity-cap 96 \
    --trim-slices 0 \
    --med3d-batch 2 \
    --pca-model none \
    --seed 42 \
    --device cuda \
    --output "$REPORT_STAGING"

[[ -s "$REPORT_STAGING" ]] || fatal "metric report was not created"
cp -- "$COMMON_IDS_SOURCE" "$STAGING_DIR/$COMMON_IDS_RESULT_NAME"
ids_sha256="$(sha256sum "$COMMON_IDS_SOURCE" | awk '{print $1}')"
{
    echo "eval_id=$EVAL_ID"
    echo "panel=all14_common96"
    echo "generated_cases=$NUM_COMMON"
    echo "reference=train105"
    echo "reference_cases=$NUM_REFERENCE"
    echo "candidate_ids_sha256=$ids_sha256"
    echo "labels=${LABELS[*]}"
} > "$STAGING_DIR/$INPUT_MANIFEST_NAME"

mv -- "$STAGING_DIR" "$FINAL_DIR"

echo "Completed metric report: $FINAL_DIR/$REPORT_NAME"
echo "Matched candidate IDs: $FINAL_DIR/$COMMON_IDS_RESULT_NAME"
