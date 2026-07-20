#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="eval_gen_panel_metrics"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/eval_gen_panel_metrics_%j.out
#SBATCH --error=IDUN/output/generate/eval_gen_panel_metrics_%j.err

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

require_marker_value() {
    local marker="$1"
    local key="$2"
    local expected="$3"
    local -a values
    mapfile -t values < <(sed -n "s/^${key}=//p" "$marker")
    [[ ${#values[@]} -eq 1 ]] || fatal \
        "$marker must contain exactly one ${key}=... entry"
    [[ "${values[0]}" == "$expected" ]] || fatal \
        "$marker has ${key}=${values[0]}; expected ${key}=${expected}"
}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
readonly PANEL_ID="${PANEL_ID:-source_selection_train105_seed42_euler100}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly TRAIN_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/train"
readonly PANEL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${PANEL_ID}"
readonly BRAIN_MASK_COMPLETE_MARKER="${PANEL_ROOT}/.brain_mask_stage3_all14_complete"
readonly RESULT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/${PANEL_ID}"
readonly REPORT="${RESULT_DIR}/all_metrics_train105_brainmasked.json"
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
[[ -s "$BRAIN_MASK_COMPLETE_MARKER" ]] || fatal \
    "real-seg all-14 brain masking is not complete: $BRAIN_MASK_COMPLETE_MARKER"
require_marker_value "$BRAIN_MASK_COMPLETE_MARKER" eval_id "$PANEL_ID"
require_marker_value "$BRAIN_MASK_COMPLETE_MARKER" mode in_place_atomic
require_marker_value "$BRAIN_MASK_COMPLETE_MARKER" threshold 0.05
require_marker_value "$BRAIN_MASK_COMPLETE_MARKER" fill_holes true
require_marker_value "$BRAIN_MASK_COMPLETE_MARKER" dilate_pixels 2
require_marker_value "$BRAIN_MASK_COMPLETE_MARKER" conditioning real_train_segmentation
require_marker_value "$BRAIN_MASK_COMPLETE_MARKER" generators 14
require_marker_value "$BRAIN_MASK_COMPLETE_MARKER" generated_cases_per_dataset 105
require_marker_value "$BRAIN_MASK_COMPLETE_MARKER" generated_volumes 1470
[[ ! -e "$REPORT" ]] || fatal "metric report already exists; refusing to overwrite: $REPORT"
cd "$REPO_ROOT"

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATASET_ARGS=()
for label in "${LABELS[@]}"; do
    dataset_dir="${PANEL_ROOT}/${label}"
    count="$(find "$dataset_dir" -mindepth 2 -maxdepth 2 -type f -name bravo.nii.gz | wc -l)"
    [[ "$count" -eq 105 ]] || fatal "$label contains $count BRAVO volumes, expected 105"
    DATASET_ARGS+=(--dataset "${label}:${dataset_dir}")
done

echo "=== Train-reference metrics for all 14 in-place brain-masked real-seg datasets ==="
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
echo "Completed brain-masked metric report: $REPORT"
