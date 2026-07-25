#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|gpu80g"
#SBATCH --exclude=idun-06-02
#SBATCH --array=0-4%5
#SBATCH --job-name="exp17_1_h25"
#SBATCH --output=IDUN/output/train/downstream/nnunet/exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663/%A_%a.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663/%A_%a.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:SIGUSR1@120

# One submission launches five folds for the exp17 hybrid arm with 25 synthetic cases.
# Every fold uses 84 real training cases, the same ordered synthetic prefix, and
# 21 real validation cases. Each fold may use 20 chained 12-hour allocations.
# Uses Dataset600's transferred plan and the historical unseeded training path.
# Normal: sbatch IDUN/train/downstream/nnunet/exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663.slurm
# Resume fold 2 only:
# sbatch --array=2 --export=ALL,RESUME=1 IDUN/train/downstream/nnunet/exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663.slurm

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly DATASET_ID=663
readonly N_SYNTHETIC=25
readonly LABEL="exp1_to_exp48c_t025"
readonly EXPERIMENT_NAME="exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663"
readonly FOLD="${SLURM_ARRAY_TASK_ID:?submit this file as its configured job array}"
readonly RESUME="${RESUME:-0}"
readonly CHAIN_MAX="${CHAIN_MAX:-20}"
readonly CHAIN_COUNT="${CHAIN_COUNT:-0}"
readonly PLANS_NAME="nnUNetResEncUNetLPlansD600"
readonly SCRIPT_RELATIVE="IDUN/train/downstream/nnunet/exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663.slurm"
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
readonly CONVERSION_MARKER="${RAW_DIR}/.exp17_d663_conversion_complete"
readonly SYNTHETIC_MANIFEST="${RAW_DIR}/synthetic_candidate_order.txt"
readonly PREPROCESS_MARKER="${PREPROCESSED_DIR}/.exp17_d600_preprocess_complete"
readonly HISTORICAL_SPLITS="${NNUNET_BASE}/nnUNet_preprocessed_exp3_baseline_v2_d600/Dataset600_BrainMet/splits_final.json"
readonly MODEL_RESULT_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/Dataset${DATASET_ID}_BrainMet/${MODEL_NAME}"
readonly FOLD_LOCK="${MODEL_RESULT_DIR}/.fold_${FOLD}.training.lock"

# Hold one advisory lock for the complete allocation. This prevents a manual
# resume and an automatic continuation from writing the same fold concurrently.
mkdir -p "$MODEL_RESULT_DIR"
exec 9>"$FOLD_LOCK"
flock -n 9 || fatal \
    "another allocation is already training ${EXPERIMENT_NAME} fold ${FOLD}"

[[ -s "$CONVERSION_MARKER" ]] || fatal "Dataset663 conversion is incomplete: $CONVERSION_MARKER"
[[ -s "$RAW_DIR/source_selection.txt" ]] || fatal "missing source provenance: $RAW_DIR"
[[ "$(grep -c '^experiment_family=' "$RAW_DIR/source_selection.txt" || true)" -eq 1 ]] || \
    fatal "expected one experiment_family entry in source_selection.txt"
[[ "$(sed -n 's/^experiment_family=//p' "$RAW_DIR/source_selection.txt")" == "exp17" ]] || \
    fatal "Dataset663 does not belong to exp17"
[[ "$(grep -c '^generator=' "$RAW_DIR/source_selection.txt" || true)" -eq 1 ]] || \
    fatal "expected one generator entry in source_selection.txt"
[[ "$(sed -n 's/^generator=//p' "$RAW_DIR/source_selection.txt")" == "$LABEL" ]] || \
    fatal "Dataset663 does not contain $LABEL"
[[ -s "$SYNTHETIC_MANIFEST" ]] || fatal "missing synthetic order: $SYNTHETIC_MANIFEST"
[[ -s "$PREPROCESSED_DIR/$PLANS_NAME.json" ]] || \
    fatal "Dataset663 has no Dataset600-controlled plan"
[[ -s "$PREPROCESS_MARKER" ]] || \
    fatal "Dataset663 controlled preprocessing is incomplete"
[[ -s "$REAL_RAW_DIR/case_info.json" ]] || fatal "Dataset600 case_info.json is missing"
[[ -s "$HISTORICAL_SPLITS" ]] || \
    fatal "historical exp3 splits are missing: $HISTORICAL_SPLITS"

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

python - "$RAW_DIR" "$PREPROCESSED_DIR" "$REAL_RAW_DIR" "$SYNTHETIC_MANIFEST" \
    "$LABEL" "$DATASET_ID" "$PLANS_NAME" "$N_SYNTHETIC" \
    "$HISTORICAL_SPLITS" <<'PYEOF'
import hashlib
import json
import re
import sys
from pathlib import Path

from medgen.downstream.nnunet.splits import generate_experiment_splits

raw, preprocessed, real_raw, manifest_path = map(Path, sys.argv[1:5])
expected_label = sys.argv[5]
expected_dataset_id = int(sys.argv[6])
plans_name = sys.argv[7]
n_synthetic = int(sys.argv[8])
historical_splits_path = Path(sys.argv[9])

case_info = json.loads((raw / "case_info.json").read_text())
dataset_info = json.loads((raw / "dataset.json").read_text())
real_info = json.loads((real_raw / "case_info.json").read_text())
source = dict(
    line.split("=", 1)
    for line in (raw / "source_selection.txt").read_text().splitlines()
    if "=" in line
)
candidate_ids = [
    line.strip()
    for line in manifest_path.read_text().splitlines()
    if line.strip()
]
historical_splits = json.loads(historical_splits_path.read_text())

assert source["experiment_family"] == "exp17"
assert source["generator"] == expected_label
assert source["generator_description"] == "weighted_huber_handoff"
if "dataset_id" in source:
    assert int(source["dataset_id"]) == expected_dataset_id
assert len(candidate_ids) == len(set(candidate_ids)) == 210
assert all(re.fullmatch(r"[0-9]{5}", value) for value in candidate_ids)
synthetic_order = [f"BrainMetSyn_{value}" for value in candidate_ids]
assert len(case_info["synthetic_cases"]) == 210
assert set(case_info["synthetic_cases"]) == set(synthetic_order)
assert len(case_info["real_train_cases"]) == 105
assert len(case_info["test_cases"]) == 51
assert case_info["real_train_cases"] == real_info["real_train_cases"]
assert case_info["test_cases"] == real_info["test_cases"]
assert case_info["modalities"] == ["bravo"]
assert dataset_info["numTraining"] == 315
assert dataset_info["channel_names"] == {"0": "BRAVO"}

marker_path = preprocessed / ".exp17_d600_preprocess_complete"
marker = dict(
    line.split("=", 1)
    for line in marker_path.read_text().splitlines()
    if "=" in line
)
assert int(marker["dataset_id"]) == expected_dataset_id
assert marker["generator"] == expected_label
assert marker["source_dataset_id"] == "600"
assert marker["target_plans"] == plans_name

actual_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if marker.get("synthetic_candidate_order") != str(manifest_path):
    raise RuntimeError("preprocess marker points to a different synthetic manifest")
if marker.get("synthetic_candidate_order_sha256") != actual_manifest_sha256:
    raise RuntimeError(
        "synthetic manifest SHA-256 does not match the completed preprocessing marker"
    )
conversion_marker_path = raw / ".exp17_d663_conversion_complete"
if marker.get("conversion_marker") != str(conversion_marker_path):
    raise RuntimeError("preprocess marker points to a different conversion marker")
actual_conversion_sha256 = hashlib.sha256(
    conversion_marker_path.read_bytes()
).hexdigest()
if marker.get("conversion_marker_sha256") != actual_conversion_sha256:
    raise RuntimeError(
        "conversion marker SHA-256 does not match the completed preprocessing marker"
    )
conversion = dict(
    line.split("=", 1)
    for line in conversion_marker_path.read_text().splitlines()
    if "=" in line
)
source_path = raw / "source_selection.txt"
actual_source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
if conversion.get("source_selection_sha256") != actual_source_sha256:
    raise RuntimeError(
        "source-selection SHA-256 does not match the completed conversion marker"
    )

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
assert b2nd_count == 630 or (b2nd_count == 0 and npz_count == 315), (
    cache, b2nd_count, npz_count,
)

splits = generate_experiment_splits(
    experiment="mixed",
    real_train_cases=case_info["real_train_cases"],
    synthetic_cases=case_info["synthetic_cases"],
    n_synthetic=n_synthetic,
    synthetic_order=synthetic_order,
)
assert len(splits) == 5
assert len(historical_splits) == 5
selected = set(synthetic_order[:n_synthetic])
all_synthetic = set(synthetic_order)
validation_union = set()
for fold, split in enumerate(splits):
    train = set(split["train"])
    val = set(split["val"])
    train_real = train - all_synthetic
    train_synthetic = train & all_synthetic
    assert len(train) == 84 + n_synthetic, (fold, len(train))
    assert len(val) == 21, (fold, len(val))
    assert train_synthetic == selected, fold
    assert not (val & all_synthetic), fold
    assert not (train_real & val), fold
    assert train_real | val == set(case_info["real_train_cases"]), fold
    assert split["val"] == historical_splits[fold]["val"], fold
    assert sorted(train_real) == historical_splits[fold]["train"], fold
    validation_union |= val
assert validation_union == set(case_info["real_train_cases"])
print(
    f"Preflight OK: 84 real + {n_synthetic} manifest-prefix synthetic train "
    "and 21 real validation per fold; exact Dataset600 plan/cache"
)
PYEOF

echo "=== exp17 hybrid training ==="
echo "Generator:           $LABEL"
echo "Dataset:             $DATASET_ID"
echo "Synthetic cases:     $N_SYNTHETIC (ordered manifest prefix)"
echo "Per-fold training:   $((84 + N_SYNTHETIC)) cases"
echo "Per-fold validation: 21 real cases"
echo "Fold:                $FOLD/4"
echo "Seed:                historical behavior (no explicit override)"
echo "Plans:               $PLANS_NAME"
echo "Experiment:          $EXPERIMENT_NAME"
echo "Resume:              $RESUME"
echo "Chain:               $CHAIN_COUNT/$CHAIN_MAX"
nvidia-smi

TRAIN_ARGS=(
    --experiment mixed
    --n-synthetic "$N_SYNTHETIC"
    --synthetic-manifest "$SYNTHETIC_MANIFEST"
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
        --dependency="afterany:${SLURM_JOB_ID}" \
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

echo "Completed $EXPERIMENT_NAME fold $FOLD. Evaluation is intentionally separate."
