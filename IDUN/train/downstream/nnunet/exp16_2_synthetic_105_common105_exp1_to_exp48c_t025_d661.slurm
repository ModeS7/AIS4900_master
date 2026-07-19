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
#SBATCH --job-name="exp16_2_c105_48c"
#SBATCH --output=IDUN/output/train/downstream/nnunet/exp16_2_common105_exp48c_%A_%a.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/exp16_2_common105_exp48c_%A_%a.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=FAIL

# Five independent runs for the frozen common-105 exp48c handoff dataset.
# Normal: sbatch IDUN/train/downstream/nnunet/exp16_2_synthetic_105_common105_exp1_to_exp48c_t025_d661.slurm
# Resume fold 2 only:
# sbatch --array=2 --export=ALL,RESUME=1 IDUN/train/downstream/nnunet/exp16_2_synthetic_105_common105_exp1_to_exp48c_t025_d661.slurm

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly DATASET_ID=661
readonly LABEL="exp1_to_exp48c_t025"
readonly EXPERIMENT_NAME="exp16_2_synthetic_105_common105_${LABEL}_d${DATASET_ID}"
readonly FOLD="${SLURM_ARRAY_TASK_ID:?submit this file as its configured job array}"
readonly SEED="$((4200 + FOLD))"
readonly RESUME="${RESUME:-0}"
(( FOLD >= 0 && FOLD < 5 )) || fatal "fold is outside 0--4: $FOLD"
[[ "$RESUME" == 0 || "$RESUME" == 1 ]] || fatal "RESUME must be 0 or 1"

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
readonly NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly RAW_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet"
readonly PREPROCESSED_DIR="${NNUNET_BASE}/nnUNet_preprocessed/Dataset${DATASET_ID}_BrainMet"
readonly REAL_RAW_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset600_BrainMet"
readonly REAL_PREPROCESSED_DIR="${NNUNET_BASE}/nnUNet_preprocessed/Dataset600_BrainMet"
readonly MODEL_NAME="nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres"
readonly FOLD_RESULT_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/Dataset${DATASET_ID}_BrainMet/${MODEL_NAME}/fold_${FOLD}"

[[ -s "$RAW_DIR/source_selection.txt" ]] || fatal "missing source provenance: $RAW_DIR"
[[ "$(grep -c '^generator=' "$RAW_DIR/source_selection.txt" || true)" -eq 1 ]] || \
    fatal "expected one generator entry in source_selection.txt"
[[ "$(sed -n 's/^generator=//p' "$RAW_DIR/source_selection.txt")" == "$LABEL" ]] || \
    fatal "Dataset${DATASET_ID} does not contain $LABEL"
[[ -s "$PREPROCESSED_DIR/nnUNetResEncUNetLPlans.json" ]] || \
    fatal "Dataset${DATASET_ID} has not been preprocessed with ResEnc-L"
[[ -s "$REAL_RAW_DIR/case_info.json" ]] || fatal "Dataset600 case_info.json is missing"
[[ -s "$REAL_PREPROCESSED_DIR/nnUNetResEncUNetLPlans.json" ]] || \
    fatal "Dataset600 ResEnc-L plans are missing"

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

python - "$RAW_DIR" "$PREPROCESSED_DIR" "$REAL_RAW_DIR" "$REAL_PREPROCESSED_DIR" "$LABEL" "$DATASET_ID" <<'PYEOF'
import json
import sys
from pathlib import Path

raw, preprocessed, real_raw, real_preprocessed = map(Path, sys.argv[1:5])
expected_label, expected_dataset_id = sys.argv[5], int(sys.argv[6])
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

plans = json.loads((preprocessed / "nnUNetResEncUNetLPlans.json").read_text())
reference = json.loads((real_preprocessed / "nnUNetResEncUNetLPlans.json").read_text())
configuration = plans["configurations"]["3d_fullres"]
reference_configuration = reference["configurations"]["3d_fullres"]
for key in (
    "spacing", "patch_size", "batch_size", "architecture",
    "normalization_schemes", "use_mask_for_norm",
):
    assert configuration.get(key) == reference_configuration.get(key), key

cache = preprocessed / configuration["data_identifier"]
assert cache.is_dir(), cache
b2nd_count = len(list(cache.glob("*.b2nd")))
npz_count = len(list(cache.glob("*.npz")))
assert b2nd_count == 420 or (b2nd_count == 0 and npz_count == 210), (
    cache, b2nd_count, npz_count,
)
print("Preflight OK: common 105, fixed real folds/test, Dataset600-matched ResEnc-L plans")
PYEOF

echo "=== Common-105 synthetic-only training ==="
echo "Generator:  $LABEL"
echo "Dataset:    $DATASET_ID"
echo "Fold:       $FOLD/4"
echo "Seed:       $SEED"
echo "Experiment: $EXPERIMENT_NAME"
echo "Resume:     $RESUME"
nvidia-smi

TRAIN_ARGS=(
    --experiment synthetic
    --n-synthetic 105
    --dataset-id "$DATASET_ID"
    --experiment-name "$EXPERIMENT_NAME"
    --fold "$FOLD"
    --seed "$SEED"
    --nnunet-base "$NNUNET_BASE"
    --nnunet-results "$NNUNET_RESULTS"
    --trainer nnUNetTrainerBrainMets
    --plans nnUNetResEncUNetLPlans
)
if [[ "$RESUME" == 1 ]]; then
    TRAIN_ARGS+=(--continue-training)
fi

time python -m medgen.scripts.train_nnunet "${TRAIN_ARGS[@]}"

[[ -s "$FOLD_RESULT_DIR/checkpoint_final.pth" ]] || \
    fatal "checkpoint_final.pth is missing after training"
[[ -s "$FOLD_RESULT_DIR/checkpoint_best.pth" ]] || \
    fatal "checkpoint_best.pth is missing after training"

echo "Completed $LABEL fold $FOLD. Evaluation is intentionally separate."
