#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="synthmask_metrics"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/synthmask_metrics_%j.out
#SBATCH --error=IDUN/output/generate/synthmask_metrics_%j.err

# Submit after finalize_synthmask_common105.slurm has created the matched datasets.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly TRAIN_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/train"
readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly MATCHED_ROOT="${EVAL_ROOT}/matched_common105"
readonly COMMON_IDS_SOURCE="${MATCHED_ROOT}/common_candidate_ids.txt"
readonly RESULT_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/eval"
readonly FINAL_DIR="${RESULT_ROOT}/${EVAL_ID}"
readonly REPORT="${FINAL_DIR}/all_metrics_train105.json"
readonly COMMON_IDS_RESULT="${FINAL_DIR}/common_candidate_ids.txt"
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

[[ -d "$TRAIN_DIR" ]] || fatal "training reference not found: $TRAIN_DIR"
[[ -d "$MATCHED_ROOT" ]] || fatal "matched 105-case datasets not found: $MATCHED_ROOT"
[[ -s "$COMMON_IDS_SOURCE" ]] || fatal "common candidate ID list not found: $COMMON_IDS_SOURCE"
[[ ! -e "$FINAL_DIR" ]] || fatal "metric output already exists: $FINAL_DIR"

mapfile -t COMMON_IDS < "$COMMON_IDS_SOURCE"
[[ ${#COMMON_IDS[@]} -eq 105 ]] || fatal \
    "common candidate ID list contains ${#COMMON_IDS[@]} entries, expected 105"

DATASET_ARGS=()
for label in "${LABELS[@]}"; do
    dataset_dir="${MATCHED_ROOT}/${label}"
    [[ -d "$dataset_dir" ]] || fatal "matched dataset not found: $label"
    for candidate_id in "${COMMON_IDS[@]}"; do
        [[ "$candidate_id" =~ ^[0-9]{5}$ ]] || fatal "invalid candidate ID: $candidate_id"
        [[ -s "$dataset_dir/$candidate_id/seg.nii.gz" ]] || \
            fatal "$label is missing segmentation candidate $candidate_id"
        [[ -s "$dataset_dir/$candidate_id/bravo.nii.gz" ]] || \
            fatal "$label is missing BRAVO candidate $candidate_id"
    done
    count="$(find -L "$dataset_dir" -mindepth 2 -maxdepth 2 -type f -name bravo.nii.gz | wc -l)"
    [[ "$count" -eq 105 ]] || fatal "$label contains $count BRAVO volumes, expected 105"
    DATASET_ARGS+=(--dataset "${label}:${dataset_dir}")
done

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Computing train-reference metrics for all 14 datasets ==="
time python -m medgen.scripts.eval_all_metrics \
    --reference "train:${TRAIN_DIR}" \
    "${DATASET_ARGS[@]}" \
    --pattern bravo.nii.gz \
    --target-shape 150 256 256 \
    --pool-cap 0 \
    --ref-cap 0 \
    --diversity-cap 105 \
    --trim-slices 0 \
    --med3d-batch 2 \
    --pca-model none \
    --seed 42 \
    --device cuda \
    --output "$REPORT"

[[ -s "$REPORT" ]] || fatal "metric report was not created"
cp -- "$COMMON_IDS_SOURCE" "$COMMON_IDS_RESULT"
echo "Completed metric report: $REPORT"
echo "Matched candidate IDs: $COMMON_IDS_RESULT"
