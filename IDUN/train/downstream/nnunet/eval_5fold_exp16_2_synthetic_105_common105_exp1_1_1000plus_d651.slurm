#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|gpu80g"
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="eval16_c105_e1p"
#SBATCH --output=IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_synthetic_105_common105_exp1_1_1000plus_d651_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_synthetic_105_common105_exp1_1_1000plus_d651_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Evaluate the completed five-fold exp16_2 Dataset651 model on the official
# 51-case test set. This job never accepts a partial fold ensemble.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly DATASET_ID=651
readonly LABEL="exp1_1_1000plus"
readonly EXPERIMENT_NAME="exp16_2_synthetic_105_common105_${LABEL}_d${DATASET_ID}"
readonly TRAINER="nnUNetTrainerBrainMets"
readonly PLANS_NAME="nnUNetResEncUNetLPlansD600"
readonly CONFIGURATION="3d_fullres"
readonly EXPECTED_CASES=51
readonly FOLDS_CSV="0,1,2,3,4"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
readonly NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly RAW_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet"
readonly OFFICIAL_LABELS_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset600_BrainMet/labelsTs"
readonly EXPERIMENT_RESULTS="${NNUNET_RESULTS}/${EXPERIMENT_NAME}"
readonly MODEL_DIR="${EXPERIMENT_RESULTS}/Dataset${DATASET_ID}_BrainMet/${TRAINER}__${PLANS_NAME}__${CONFIGURATION}"
readonly EVAL_DIR="${EXPERIMENT_RESULTS}/eval_${EXPERIMENT_NAME}"
readonly PREDICTIONS_DIR="${EVAL_DIR}/predictions"
readonly EVAL_JSON="${EXPERIMENT_RESULTS}/eval_${EXPERIMENT_NAME}.json"
readonly COMPLETE_MARKER="${EVAL_DIR}/.exp16_2_5fold_eval_complete"
readonly STDOUT_PATH="${REPO_ROOT}/IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_synthetic_105_common105_exp1_1_1000plus_d651_${JOB_ID}.out"
readonly STDERR_PATH="${REPO_ROOT}/IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_synthetic_105_common105_exp1_1_1000plus_d651_${JOB_ID}.err"

[[ -d "$RAW_DIR/imagesTs" ]] || fatal "official test images are missing: $RAW_DIR/imagesTs"
[[ -d "$RAW_DIR/labelsTs" ]] || fatal "official test labels are missing: $RAW_DIR/labelsTs"
[[ -d "$OFFICIAL_LABELS_DIR" ]] || fatal \
    "Dataset600 official test labels are missing: $OFFICIAL_LABELS_DIR"
[[ -d "$MODEL_DIR" ]] || fatal "trained model directory is missing: $MODEL_DIR"
for fold in 0 1 2 3 4; do
    fold_dir="${MODEL_DIR}/fold_${fold}"
    [[ -s "${fold_dir}/checkpoint_best.pth" ]] || \
        fatal "fold ${fold} checkpoint_best.pth is missing"
    [[ -s "${fold_dir}/checkpoint_final.pth" ]] || \
        fatal "fold ${fold} checkpoint_final.pth is missing"
done

python - "$RAW_DIR/imagesTs" "$RAW_DIR/labelsTs" "$OFFICIAL_LABELS_DIR" \
    "$EXPECTED_CASES" <<'PYEOF'
import sys
from pathlib import Path

images_dir, labels_dir, official_labels_dir = map(Path, sys.argv[1:4])
expected_cases = int(sys.argv[4])
image_suffix = "_0000.nii.gz"
label_suffix = ".nii.gz"
image_ids = {
    path.name[:-len(image_suffix)]
    for path in images_dir.glob(f"*{image_suffix}")
}
label_ids = {
    path.name[:-len(label_suffix)]
    for path in labels_dir.glob(f"*{label_suffix}")
}
official_label_ids = {
    path.name[:-len(label_suffix)]
    for path in official_labels_dir.glob(f"*{label_suffix}")
}
if len(image_ids) != expected_cases or len(label_ids) != expected_cases:
    raise RuntimeError(
        f"official test set must contain {expected_cases} image/label IDs, "
        f"got images={len(image_ids)}, labels={len(label_ids)}"
    )
if image_ids != label_ids:
    raise RuntimeError(
        f"official test image/label IDs differ: "
        f"missing_labels={sorted(image_ids - label_ids)}, "
        f"missing_images={sorted(label_ids - image_ids)}"
    )
if len(official_label_ids) != expected_cases or label_ids != official_label_ids:
    raise RuntimeError(
        "dataset test IDs do not exactly match Dataset600 official labels: "
        f"missing={sorted(official_label_ids - label_ids)}, "
        f"extra={sorted(label_ids - official_label_ids)}"
    )
print(f"Official test-set preflight OK: {expected_cases} exact image/label IDs")
PYEOF

mkdir -p "$EVAL_DIR"
rm -f -- "$COMPLETE_MARKER"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== exp16_2 five-fold evaluation ==="
echo "Generator:  $LABEL"
echo "Dataset:    $DATASET_ID"
echo "Experiment: $EXPERIMENT_NAME"
echo "Folds:      $FOLDS_CSV"
echo "Output:     $EVAL_DIR"
nvidia-smi

time python -m medgen.scripts.eval_nnunet \
    --experiment synthetic \
    --n-synthetic 105 \
    --dataset-id "$DATASET_ID" \
    --experiment-name "$EXPERIMENT_NAME" \
    --folds 0 1 2 3 4 \
    --nnunet-base "$NNUNET_BASE" \
    --nnunet-results "$NNUNET_RESULTS" \
    --trainer "$TRAINER" \
    --plans "$PLANS_NAME"

[[ -s "$EVAL_JSON" ]] || fatal "evaluation JSON is missing: $EVAL_JSON"
[[ -d "$PREDICTIONS_DIR" ]] || fatal \
    "prediction directory is missing: $PREDICTIONS_DIR"
prediction_count="$(python - "$PREDICTIONS_DIR" "$OFFICIAL_LABELS_DIR" \
    "$EXPECTED_CASES" <<'PYEOF'
import sys
from pathlib import Path

predictions_dir, labels_dir = map(Path, sys.argv[1:3])
expected_cases = int(sys.argv[3])
suffix = ".nii.gz"
prediction_ids = {
    path.name[:-len(suffix)]
    for path in predictions_dir.glob(f"*{suffix}")
}
label_ids = {
    path.name[:-len(suffix)]
    for path in labels_dir.glob(f"*{suffix}")
}
if len(prediction_ids) != expected_cases:
    raise RuntimeError(
        f"expected exactly {expected_cases} prediction NIfTIs, "
        f"got {len(prediction_ids)}"
    )
if prediction_ids != label_ids:
    raise RuntimeError(
        f"prediction IDs do not exactly match official labels: "
        f"missing={sorted(label_ids - prediction_ids)}, "
        f"extra={sorted(prediction_ids - label_ids)}"
    )
print(len(prediction_ids))
PYEOF
)" || fatal "prediction verification failed"
[[ "$prediction_count" == "$EXPECTED_CASES" ]] || \
    fatal "prediction verification returned $prediction_count"

marker_tmp="${COMPLETE_MARKER}.tmp_${JOB_ID}"
trap 'rm -f -- "$marker_tmp"' EXIT
{
    echo "job_id=${JOB_ID}"
    echo "stdout=${STDOUT_PATH}"
    echo "stderr=${STDERR_PATH}"
    echo "experiment=${EXPERIMENT_NAME}"
    echo "dataset=Dataset${DATASET_ID}_BrainMet"
    echo "folds=${FOLDS_CSV}"
    echo "prediction_count=${prediction_count}"
} > "$marker_tmp"
mv -f -- "$marker_tmp" "$COMPLETE_MARKER"
trap - EXIT

echo "Completed strict five-fold evaluation: $COMPLETE_MARKER"
