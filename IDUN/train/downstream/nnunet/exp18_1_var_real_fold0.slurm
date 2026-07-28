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
#SBATCH --array=1-4%2
#SBATCH --job-name="exp18_1_varreal"
#SBATCH --output=IDUN/output/train/downstream/nnunet/exp18_1_var_real_fold0/%A_%a.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/exp18_1_var_real_fold0/%A_%a.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:SIGUSR1@120

# =============================================================================
# Training-variance screen, arm 1 of 2: the real-data reference.
#
# WHY THIS EXISTS
#   Every downstream condition in the study was trained exactly once, so the
#   reported differences contain an unknown amount of run-to-run variation. A
#   decomposition of the existing validation grid bounds that variation at
#   SD <= 0.0149, which is the same order as the augmentation effects being
#   claimed (-0.0117 to -0.0035) and larger than the +0.0084 bound the paper
#   reports. That upper bound is not good enough to defend the bound with. This
#   measures the noise floor directly.
#
# THE DESIGN, AND WHY IT IS CHEAP
#   Fold 0 only, repeated 4 times. The array indexes REPLICATES, not folds.
#   Training data, validation patients and every hyperparameter are identical
#   across replicates. The only thing that differs is the run itself.
#
#   No seed plumbing is required, and none is added. nnU-Net here uses the
#   historical unseeded path: seed 42 fixes fold membership and nothing else,
#   so model initialisation, augmentation and training order already differ
#   between invocations. A plain rerun IS an independent draw. The study's own
#   limitation is what makes this measurement possible.
#
#   Cost is 4 trainings per arm instead of 20. A single-model spread over 21
#   validation patients OVERESTIMATES the spread of a 5-model ensemble over 51
#   test patients, so a small result here is conclusive and a large result
#   means escalating to full 5-fold replication.
#
# WHAT IS DELIBERATELY NOT DONE
#   The 51 official test cases are never touched. The readout is fold 0's 21
#   real validation patients, which is the clean comparison anyway: same data,
#   same patients, only the run differs. Reserving the test cohort is the
#   study's standing rule and a variance screen is no reason to spend it.
#
#   Nothing existing is modified. Each replicate writes to its own experiment
#   name, so exp3_baseline_v2_d600 is untouched.
#
# SUBMIT
#   sbatch IDUN/train/downstream/nnunet/exp18_1_var_real_fold0.slurm
# RESUME one interrupted replicate
#   sbatch --array=3 --export=ALL,RESUME=1 IDUN/train/downstream/nnunet/exp18_1_var_real_fold0.slurm
# =============================================================================

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly DATASET_ID=600
readonly FOLD=0
readonly REPLICATE="${SLURM_ARRAY_TASK_ID:?submit this file as its configured job array}"
readonly RESUME="${RESUME:-0}"
readonly CHAIN_MAX="${CHAIN_MAX:-20}"
readonly CHAIN_COUNT="${CHAIN_COUNT:-0}"
readonly PLANS_NAME="nnUNetResEncUNetLPlans"
readonly REFERENCE_EXPERIMENT="exp3_baseline_v2_d600"
readonly EXPERIMENT_NAME="exp18_1_var_real_fold0_rep${REPLICATE}"
readonly SCRIPT_RELATIVE="IDUN/train/downstream/nnunet/exp18_1_var_real_fold0.slurm"

(( REPLICATE >= 1 && REPLICATE <= 4 )) || fatal "replicate is outside 1--4: $REPLICATE"
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
readonly PREPROCESSED_DIR="${NNUNET_BASE}/nnUNet_preprocessed/Dataset${DATASET_ID}_BrainMet"
readonly RAW_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet"
readonly MODEL_NAME="nnUNetTrainerBrainMets__${PLANS_NAME}__3d_fullres"
readonly MODEL_RESULT_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/Dataset${DATASET_ID}_BrainMet/${MODEL_NAME}"
readonly FOLD_RESULT_DIR="${MODEL_RESULT_DIR}/fold_${FOLD}"
readonly REFERENCE_RESULT_DIR="${NNUNET_RESULTS}/${REFERENCE_EXPERIMENT}/Dataset${DATASET_ID}_BrainMet/${MODEL_NAME}/fold_${FOLD}"
readonly FOLD_LOCK="${MODEL_RESULT_DIR}/.rep_${REPLICATE}.training.lock"

mkdir -p "$MODEL_RESULT_DIR"
exec 9>"$FOLD_LOCK"
flock -n 9 || fatal "another allocation is already training replicate ${REPLICATE}"

# --- the replicate must be comparable to the original run, or it measures nothing ----
[[ -s "$RAW_DIR/case_info.json" ]] || fatal "Dataset600 case_info.json is missing"
[[ -s "$PREPROCESSED_DIR/$PLANS_NAME.json" ]] || fatal "missing plan: $PLANS_NAME"
[[ -s "$PREPROCESSED_DIR/splits_final.json" ]] || fatal "missing splits_final.json"
[[ -d "$REFERENCE_RESULT_DIR" ]] || \
    fatal "original reference fold 0 not found, nothing to replicate: $REFERENCE_RESULT_DIR"

# Never clobber the study's own results.
[[ "$EXPERIMENT_NAME" != "$REFERENCE_EXPERIMENT" ]] || fatal "refusing to overwrite the reference"

if [[ "$RESUME" == 0 ]]; then
    [[ ! -e "$FOLD_RESULT_DIR" ]] || \
        fatal "replicate output exists; use RESUME=1 only for an interrupted run: $FOLD_RESULT_DIR"
else
    [[ -s "$FOLD_RESULT_DIR/checkpoint_latest.pth" ]] || \
        fatal "cannot resume without checkpoint_latest.pth: $FOLD_RESULT_DIR"
    [[ ! -s "$FOLD_RESULT_DIR/checkpoint_final.pth" ]] || \
        fatal "replicate is already complete: $FOLD_RESULT_DIR"
fi

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Preflight: this replicate must train on exactly the cases the original fold 0 used.
python - "$PREPROCESSED_DIR/splits_final.json" "$FOLD" <<'PYEOF'
import json, sys
splits = json.loads(open(sys.argv[1]).read())
fold = int(sys.argv[2])
assert len(splits) == 5, len(splits)
s = splits[fold]
assert len(s["train"]) == 84, len(s["train"])
assert len(s["val"]) == 21, len(s["val"])
assert not (set(s["train"]) & set(s["val"]))
print(f"Preflight OK: fold {fold} = 84 real train / 21 real validation, "
      f"identical to the original reference run")
PYEOF

echo "=== exp18 training-variance screen, arm 1 (real-data reference) ==="
echo "Replicate:           ${REPLICATE}/4"
echo "Fold:                ${FOLD} (fixed; the array indexes replicates)"
echo "Dataset:             ${DATASET_ID}"
echo "Plans:               ${PLANS_NAME}"
echo "Experiment:          ${EXPERIMENT_NAME}"
echo "Seed:                none (unseeded path -> each rerun is an independent draw)"
echo "Readout:             fold ${FOLD} validation Dice on 21 real patients"
echo "Test cohort:         untouched, by design"
echo "Resume:              ${RESUME}   Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

TRAIN_ARGS=(
    --experiment baseline
    --dataset-id "$DATASET_ID"
    --experiment-name "$EXPERIMENT_NAME"
    --fold "$FOLD"
    --nnunet-base "$NNUNET_BASE"
    --nnunet-results "$NNUNET_RESULTS"
    --trainer nnUNetTrainerBrainMets
    --plans "$PLANS_NAME"
)
[[ "$RESUME" == 1 ]] && TRAIN_ARGS+=(--continue-training)

TRAIN_PID=""
handle_timeout() {
    trap - SIGUSR1
    echo "=== Wall time approaching for replicate ${REPLICATE}; preparing continuation ==="
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    if [[ -s "$FOLD_RESULT_DIR/checkpoint_final.pth" ]]; then
        echo "Replicate ${REPLICATE} completed before timeout."
        exit 0
    fi
    [[ -s "$FOLD_RESULT_DIR/checkpoint_latest.pth" ]] || \
        fatal "cannot chain without checkpoint_latest.pth: $FOLD_RESULT_DIR"
    local next_count=$((CHAIN_COUNT + 1))
    (( next_count < CHAIN_MAX )) || fatal "chain limit reached: ${CHAIN_MAX}"
    sbatch --array="$REPLICATE" \
        --dependency="afterany:${SLURM_JOB_ID}" \
        --export="ALL,RESUME=1,CHAIN_COUNT=${next_count},CHAIN_MAX=${CHAIN_MAX}" \
        "$SCRIPT_PATH" || fatal "failed to submit continuation for replicate ${REPLICATE}"
    echo "Submitted replicate ${REPLICATE} continuation ${next_count}/${CHAIN_MAX}"
    exit 0
}
trap handle_timeout SIGUSR1

python -m medgen.scripts.train_nnunet "${TRAIN_ARGS[@]}" &
TRAIN_PID=$!
if wait "$TRAIN_PID"; then train_exit=0; else train_exit=$?; fi
trap - SIGUSR1
(( train_exit == 0 )) || fatal "training exited with status $train_exit"

[[ -s "$FOLD_RESULT_DIR/checkpoint_final.pth" ]] || fatal "checkpoint_final.pth missing"
[[ -s "$FOLD_RESULT_DIR/checkpoint_best.pth" ]] || fatal "checkpoint_best.pth missing"

echo ""
echo "=== Validation inference for replicate ${REPLICATE} ==="
python -m medgen.scripts.eval_nnunet \
    --experiment baseline \
    --dataset-id "$DATASET_ID" \
    --experiment-name "$EXPERIMENT_NAME" \
    --folds "$FOLD" \
    --nnunet-base "$NNUNET_BASE" \
    --nnunet-results "$NNUNET_RESULTS" \
    --trainer nnUNetTrainerBrainMets \
    --plans "$PLANS_NAME" \
    || echo "WARNING: validation evaluation failed; rerun eval_nnunet for this replicate"

echo "Completed ${EXPERIMENT_NAME}. Collect all replicates with analyse_training_variance.py."
