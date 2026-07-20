#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|gpu80g"
#SBATCH --exclude=idun-06-02
#SBATCH --array=0-4%5
#SBATCH --job-name="exp16_2_c105_e32"
#SBATCH --output=IDUN/output/train/downstream/nnunet/exp16_2_common105_exp32_2_1000_d652/%A_%a.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/exp16_2_common105_exp32_2_1000_d652/%A_%a.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:SIGUSR1@120

# One submission launches five folds for the frozen common-105 exp32 dataset.
# Each fold may use at most 20 chained 12-hour allocations.
# Uses Dataset600's transferred plan and the historical unseeded training path.
# Normal: sbatch IDUN/train/downstream/nnunet/exp16_2_synthetic_105_common105_exp32_2_1000_d652.slurm
# Resume fold 2 only:
# sbatch --array=2 --export=ALL,RESUME=1 IDUN/train/downstream/nnunet/exp16_2_synthetic_105_common105_exp32_2_1000_d652.slurm

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly DATASET_ID=652
readonly LABEL="exp32_2_1000"
readonly EXPERIMENT_NAME="exp16_2_synthetic_105_common105_${LABEL}_d${DATASET_ID}"
readonly FOLD="${SLURM_ARRAY_TASK_ID:?submit this file as its configured job array}"
readonly RESUME="${RESUME:-0}"
readonly CHAIN_MAX="${CHAIN_MAX:-20}"
readonly CHAIN_COUNT="${CHAIN_COUNT:-0}"
readonly PLANS_NAME="nnUNetResEncUNetLPlansD600"
readonly SCRIPT_RELATIVE="IDUN/train/downstream/nnunet/exp16_2_synthetic_105_common105_exp32_2_1000_d652.slurm"
(( FOLD >= 0 && FOLD < 5 )) || fatal "fold is outside 0--4: $FOLD"
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || fatal "RESUME must be 0 or 1"
[[ "$CHAIN_MAX" =~ ^[1-9][0-9]*$ ]] || fatal "CHAIN_MAX must be a positive integer"
[[ "$CHAIN_COUNT" =~ ^[0-9]+$ ]] || fatal "CHAIN_COUNT must be a non-negative integer"
(( CHAIN_COUNT < CHAIN_MAX )) || fatal "chain count reached limit: ${CHAIN_COUNT}/${CHAIN_MAX}"
(( CHAIN_COUNT == 0 || RESUME == 1 )) || fatal "a chained allocation must use RESUME=1"

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_RELATIVE}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
readonly NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly RAW_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet"
readonly PREPROCESSED_DIR="${NNUNET_BASE}/nnUNet_preprocessed/Dataset${DATASET_ID}_BrainMet"
readonly REAL_RAW_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset600_BrainMet"
readonly MODEL_NAME="nnUNetTrainerBrainMets__${PLANS_NAME}__3d_fullres"
readonly FOLD_RESULT_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/Dataset${DATASET_ID}_BrainMet/${MODEL_NAME}/fold_${FOLD}"
readonly PREPROCESS_MARKER="${PREPROCESSED_DIR}/.exp16_2_d600_preprocess_complete"

[[ -s "$RAW_DIR/source_selection.txt" ]] || fatal "missing source provenance: $RAW_DIR"
[[ "$(grep -c '^generator=' "$RAW_DIR/source_selection.txt" || true)" -eq 1 ]] || \
    fatal "expected one generator entry in source_selection.txt"
[[ "$(sed -n 's/^generator=//p' "$RAW_DIR/source_selection.txt")" == "$LABEL" ]] || \
    fatal "Dataset${DATASET_ID} does not contain $LABEL"
[[ -s "$PREPROCESSED_DIR/${PLANS_NAME}.json" ]] || \
    fatal "Dataset${DATASET_ID} has no Dataset600-controlled plan"
[[ -s "$PREPROCESS_MARKER" ]] || \
    fatal "Dataset${DATASET_ID} controlled preprocessing is incomplete"
[[ -s "$REAL_RAW_DIR/case_info.json" ]] || fatal "Dataset600 case_info.json is missing"

if [[ "$RESUME" == 0 ]]; then
    [[ ! -e "$FOLD_RESULT_DIR" ]] || \
        fatal "fold output exists; use RESUME=1 only for an interrupted fold: $FOLD_RESULT_DIR"
else
    [[ -s "$FOLD_RESULT_DIR/checkpoint_latest.pth" ]] || \
        fatal "cannot resume without checkpoint_latest.pth: $FOLD_RESULT_DIR"
    [[ ! -s "$FOLD_RESULT_DIR/checkpoint_final.pth" ]] || \
        fatal "fold is already complete: $FOLD_RESULT_DIR"
fi

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python - "$RAW_DIR" "$PREPROCESSED_DIR" "$REAL_RAW_DIR" "$LABEL" "$DATASET_ID" "$PLANS_NAME" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path

raw, preprocessed, real_raw = map(Path, sys.argv[1:4])
expected_label, expected_dataset_id = sys.argv[4], int(sys.argv[5])
plans_name = sys.argv[6]
case_info = json.loads((raw / "case_info.json").read_text())
real_info = json.loads((real_raw / "case_info.json").read_text())
source = dict(
    line.split("=", 1)
    for line in (raw / "source_selection.txt").read_text().splitlines()
    if "=" in line
)
common_ids = (raw / "common_candidate_ids.txt").read_text().splitlines()

assert source["generator"] == expected_label
assert int(source["dataset_id"]) == expected_dataset_id
assert len(common_ids) == len(set(common_ids)) == 105
assert case_info["synthetic_cases"] == [f"BrainMetSyn_{i}" for i in common_ids]
assert len(case_info["real_train_cases"]) == 105
assert len(case_info["test_cases"]) == 51
assert case_info["real_train_cases"] == real_info["real_train_cases"]
assert case_info["test_cases"] == real_info["test_cases"]
assert case_info["modalities"] == ["bravo"]

marker_path = preprocessed / ".exp16_2_d600_preprocess_complete"
marker = dict(
    line.split("=", 1)
    for line in marker_path.read_text().splitlines()
    if "=" in line
)
assert int(marker["dataset_id"]) == expected_dataset_id
assert marker["generator"] == expected_label
assert marker["source_dataset_id"] == "600"
assert marker["target_plans"] == plans_name

plan_path = preprocessed / f"{plans_name}.json"
plans = json.loads(plan_path.read_text())
actual_plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
if marker.get("target_plan_sha256") != actual_plan_sha256:
    raise RuntimeError(
        "controlled plan SHA-256 does not match the completed preprocessing marker: "
        f"expected {marker.get('target_plan_sha256')}, got {actual_plan_sha256}"
    )
configuration = plans["configurations"]["3d_fullres"]
required = {
    "patch_size": [160, 192, 160],
    "batch_size": 3,
    "spacing": [1.0, 0.9375, 0.9375],
    "normalization_schemes": ["ZScoreNormalization"],
    "use_mask_for_norm": [True],
    "preprocessor_name": "DefaultPreprocessor",
    "batch_dice": False,
}
if plans["plans_name"] != plans_name:
    raise RuntimeError(
        f"plans_name: expected {plans_name!r}, got {plans['plans_name']!r}"
    )
for key, expected in required.items():
    actual = configuration.get(key)
    if actual != expected:
        raise RuntimeError(f"{key}: expected {expected!r}, got {actual!r}")

architecture = configuration["architecture"]
expected_network = (
    "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"
)
if architecture["network_class_name"] != expected_network:
    raise RuntimeError(
        f"network_class_name: expected {expected_network!r}, "
        f"got {architecture['network_class_name']!r}"
    )
expected_features = [32, 64, 128, 256, 320, 320]
actual_features = architecture["arch_kwargs"]["features_per_stage"]
if actual_features != expected_features:
    raise RuntimeError(
        f"features_per_stage: expected {expected_features!r}, "
        f"got {actual_features!r}"
    )

cache = preprocessed / configuration["data_identifier"]
assert cache.is_dir(), cache
assert configuration["data_identifier"].startswith(plans_name), (
    f"cache identifier does not belong to {plans_name}: "
    f"{configuration['data_identifier']}"
)
assert marker_path.stat().st_mtime_ns >= plan_path.stat().st_mtime_ns, (
    "controlled preprocess marker predates its plan"
)
assert marker_path.stat().st_mtime_ns >= cache.stat().st_mtime_ns, (
    "controlled preprocess marker predates its cache"
)
b2nd_count = len(list(cache.glob("*.b2nd")))
npz_count = len(list(cache.glob("*.npz")))
assert b2nd_count == 420 or (b2nd_count == 0 and npz_count == 210), (
    cache, b2nd_count, npz_count,
)
print("Preflight OK: common 105, fixed real folds/test, exact Dataset600 plan/cache")
PYEOF

echo "=== Common-105 synthetic-only training ==="
echo "Generator:  $LABEL"
echo "Dataset:    $DATASET_ID"
echo "Fold:       $FOLD/4"
echo "Seed:       historical behavior (no explicit override)"
echo "Plans:      $PLANS_NAME"
echo "Experiment: $EXPERIMENT_NAME"
echo "Resume:     $RESUME"
echo "Chain:      $CHAIN_COUNT/$CHAIN_MAX"
nvidia-smi

TRAIN_ARGS=(
    --experiment synthetic
    --n-synthetic 105
    --dataset-id "$DATASET_ID"
    --experiment-name "$EXPERIMENT_NAME"
    --fold "$FOLD"
    --nnunet-base "$NNUNET_BASE"
    --nnunet-results "$NNUNET_RESULTS"
    --trainer nnUNetTrainerBrainMets
    --plans "$PLANS_NAME"
)
if [[ "$RESUME" == 1 ]]; then
    TRAIN_ARGS+=(--continue-training)
fi

TRAIN_PID=""
handle_timeout() {
    trap - SIGUSR1
    echo "=== Wall time approaching for fold $FOLD; preparing continuation ==="
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    if [[ -s "$FOLD_RESULT_DIR/checkpoint_final.pth" ]]; then
        echo "Fold $FOLD completed before timeout; no continuation needed."
        exit 0
    fi
    [[ -s "$FOLD_RESULT_DIR/checkpoint_latest.pth" ]] || \
        fatal "cannot chain without checkpoint_latest.pth: $FOLD_RESULT_DIR"
    local next_count=$((CHAIN_COUNT + 1))
    (( next_count < CHAIN_MAX )) || fatal "chain limit reached: ${CHAIN_MAX} allocations"
    local submission
    submission="$(sbatch --array="$FOLD" \
        --export="ALL,RESUME=1,CHAIN_COUNT=${next_count},CHAIN_MAX=${CHAIN_MAX}" \
        "$SCRIPT_PATH")" || fatal "failed to submit continuation for fold $FOLD"
    echo "$submission"
    echo "Submitted fold $FOLD continuation $next_count/$CHAIN_MAX"
    exit 0
}
trap handle_timeout SIGUSR1

python -m medgen.scripts.train_nnunet "${TRAIN_ARGS[@]}" &
TRAIN_PID=$!
if wait "$TRAIN_PID"; then
    train_exit=0
else
    train_exit=$?
fi
trap - SIGUSR1
(( train_exit == 0 )) || fatal "training exited with status $train_exit"

[[ -s "$FOLD_RESULT_DIR/checkpoint_final.pth" ]] || \
    fatal "checkpoint_final.pth is missing after training"
[[ -s "$FOLD_RESULT_DIR/checkpoint_best.pth" ]] || \
    fatal "checkpoint_best.pth is missing after training"

echo "Completed $LABEL fold $FOLD. Evaluation is intentionally separate."
