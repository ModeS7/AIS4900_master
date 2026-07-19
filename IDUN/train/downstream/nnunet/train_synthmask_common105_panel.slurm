#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --exclude=idun-06-02
#SBATCH --array=0-69%14
#SBATCH --job-name="nn_syn_common105"
#SBATCH --output=IDUN/output/train/downstream/nnunet/train_common105_%A_%a.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/train_common105_%A_%a.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:SIGUSR1@120

# Train five pure-synthetic folds for each of Dataset650--Dataset663.
# Array mapping: five adjacent tasks per dataset, fold = task modulo five.
# Training uses all 105 synthetic cases; validation is the fixed real fold.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this file as its configured job array}"
(( TASK_ID >= 0 && TASK_ID < 70 )) || fatal "array index is outside 0--69: $TASK_ID"

readonly PANEL_INDEX="$((TASK_ID / 5))"
readonly FOLD="$((TASK_ID % 5))"
readonly DATASET_ID="$((650 + PANEL_INDEX))"
readonly CHAIN_MAX=20
readonly CHAIN_COUNT="${CHAIN_COUNT:-0}"
[[ "$CHAIN_COUNT" =~ ^[0-9]+$ ]] || fatal "invalid CHAIN_COUNT: $CHAIN_COUNT"

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
readonly NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly RAW_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet"
readonly PREPROCESSED_DIR="${NNUNET_BASE}/nnUNet_preprocessed/Dataset${DATASET_ID}_BrainMet"
readonly REFERENCE_PREPROCESSED_DIR="${NNUNET_BASE}/nnUNet_preprocessed/Dataset650_BrainMet"
readonly SOURCE_INFO="${RAW_DIR}/source_selection.txt"

[[ -d "$RAW_DIR" ]] || fatal "raw Dataset${DATASET_ID} not found: $RAW_DIR"
[[ -d "$PREPROCESSED_DIR" ]] || \
    fatal "preprocessed Dataset${DATASET_ID} not found: $PREPROCESSED_DIR"
[[ -d "$REFERENCE_PREPROCESSED_DIR" ]] || \
    fatal "reference preprocessed Dataset650 not found: $REFERENCE_PREPROCESSED_DIR"
[[ -s "$RAW_DIR/case_info.json" ]] || fatal "case_info.json not found: $RAW_DIR"
[[ -s "$SOURCE_INFO" ]] || fatal "source provenance not found: $SOURCE_INFO"
[[ -s "$PREPROCESSED_DIR/nnUNetResEncUNetLPlans.json" ]] || \
    fatal "ResEnc-L plans not found for Dataset${DATASET_ID}"

label_count="$(grep -c '^generator=' "$SOURCE_INFO" || true)"
[[ "$label_count" -eq 1 ]] || fatal "expected one generator entry in $SOURCE_INFO"
LABEL="$(sed -n 's/^generator=//p' "$SOURCE_INFO")"
[[ "$LABEL" =~ ^[a-zA-Z0-9_.-]+$ ]] || fatal "unsafe generator label: $LABEL"
readonly LABEL

readonly EXPERIMENT_NAME="exp6_2_synthetic_105_common105_${LABEL}_d${DATASET_ID}"
readonly MODEL_NAME="nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres"
readonly FOLD_RESULT_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/Dataset${DATASET_ID}_BrainMet/${MODEL_NAME}/fold_${FOLD}"

if (( CHAIN_COUNT == 0 )); then
    [[ ! -e "$FOLD_RESULT_DIR" ]] || fatal \
        "fold output already exists; refusing to overwrite: $FOLD_RESULT_DIR"
else
    [[ -s "$FOLD_RESULT_DIR/checkpoint_latest.pth" ]] || fatal \
        "continuation requested without checkpoint_latest.pth: $FOLD_RESULT_DIR"
fi

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python - "$RAW_DIR" "$PREPROCESSED_DIR" "$REFERENCE_PREPROCESSED_DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1])
preprocessed = Path(sys.argv[2])
reference_preprocessed = Path(sys.argv[3])
case_info = json.loads((raw / "case_info.json").read_text())
plans = json.loads((preprocessed / "nnUNetResEncUNetLPlans.json").read_text())
reference_plans = json.loads(
    (reference_preprocessed / "nnUNetResEncUNetLPlans.json").read_text()
)

assert len(case_info["real_train_cases"]) == 105
assert len(case_info["synthetic_cases"]) == 105
assert len(case_info["test_cases"]) == 51
assert case_info["modalities"] == ["bravo"]

configuration = plans["configurations"]["3d_fullres"]
reference_configuration = reference_plans["configurations"]["3d_fullres"]
for key in (
    "spacing",
    "patch_size",
    "batch_size",
    "architecture",
    "normalization_schemes",
    "use_mask_for_norm",
):
    assert configuration.get(key) == reference_configuration.get(key), (
        key,
        configuration.get(key),
        reference_configuration.get(key),
    )

cache = preprocessed / configuration["data_identifier"]
assert cache.is_dir(), cache
b2nd_count = len(list(cache.glob("*.b2nd")))
npz_count = len(list(cache.glob("*.npz")))
if b2nd_count:
    assert b2nd_count == 420, (cache, b2nd_count)
else:
    assert npz_count == 210, (cache, npz_count)

print(
    f"Validated {raw.name}: 105 synthetic training cases, "
    "five fixed real validation folds"
)
PYEOF

echo "=== Pure-synthetic nnU-Net training ==="
echo "Array task: ${TASK_ID}/69"
echo "Generator:  $LABEL"
echo "Dataset:    $DATASET_ID"
echo "Fold:       $FOLD"
echo "Experiment: $EXPERIMENT_NAME"
echo "Chain:      $CHAIN_COUNT/$CHAIN_MAX"
nvidia-smi

PID=""
handle_timeout() {
    trap - SIGUSR1
    echo "=== Wall time approaching; stopping this fold for continuation ==="

    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi

    if [[ -s "$FOLD_RESULT_DIR/checkpoint_final.pth" ]]; then
        echo "Training completed while handling the timeout signal; no resubmission needed."
        exit 0
    fi

    next_count="$((CHAIN_COUNT + 1))"
    (( next_count < CHAIN_MAX )) || fatal \
        "chain limit reached ($CHAIN_MAX) for Dataset${DATASET_ID} fold $FOLD"
    [[ -s "$FOLD_RESULT_DIR/checkpoint_latest.pth" ]] || fatal \
        "cannot continue because checkpoint_latest.pth was not written"

    sbatch \
        --array="$TASK_ID" \
        --export="ALL,CHAIN_COUNT=$next_count" \
        "$0"
    echo "Resubmitted only array task $TASK_ID as chain $next_count/$CHAIN_MAX"
    exit 0
}
trap handle_timeout SIGUSR1

TRAIN_ARGS=(
    --experiment synthetic
    --n-synthetic 105
    --dataset-id "$DATASET_ID"
    --experiment-name "$EXPERIMENT_NAME"
    --fold "$FOLD"
    --nnunet-base "$NNUNET_BASE"
    --nnunet-results "$NNUNET_RESULTS"
    --trainer nnUNetTrainerBrainMets
    --plans nnUNetResEncUNetLPlans
)
if (( CHAIN_COUNT > 0 )); then
    TRAIN_ARGS+=(--continue-training)
fi

python -m medgen.scripts.train_nnunet "${TRAIN_ARGS[@]}" &
PID=$!
set +e
wait "$PID"
train_exit=$?
set -e
PID=""

(( train_exit == 0 )) || fatal \
    "training exited with status $train_exit for Dataset${DATASET_ID} fold $FOLD"
[[ -s "$FOLD_RESULT_DIR/checkpoint_final.pth" ]] || fatal \
    "training exited successfully but checkpoint_final.pth is missing"
[[ -s "$FOLD_RESULT_DIR/checkpoint_best.pth" ]] || fatal \
    "training exited successfully but checkpoint_best.pth is missing"

echo "Completed Dataset${DATASET_ID} fold $FOLD: $FOLD_RESULT_DIR"
echo "Five-fold ensemble evaluation will be submitted separately after all folds finish."
