#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|gpu80g"
#SBATCH --exclude=idun-06-02
#SBATCH --array=0-34%10
#SBATCH --job-name="val16_best"
#SBATCH --output=IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_validation_best_%A_%a.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_validation_best_%A_%a.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=FAIL

# Validate checkpoint_best.pth for seven sources and five folds. Each array
# task has its own permanent result tree, so nnU-Net cannot overwrite a prior
# validation directory or include stale predictions.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly CONDITION_LABELS=(
    original_mse
    extended_mse
    perceptual_continuation
    strong_perceptual_continuation
    weighted_huber_transition
    weighted_huber_handoff
    pseudo_huber_perceptual_handoff
)
readonly GENERATOR_LABELS=(
    exp1_1_1000
    exp1_1_1000plus
    exp32_2_1000
    exp47a
    exp47c
    exp1_to_exp48c_t025
    exp1_to_exp48d_t025
)
readonly DATASET_IDS=(650 651 652 654 656 661 662)

readonly TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this file as its configured job array}"
readonly ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID must be set}"
readonly JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
(( TASK_ID >= 0 && TASK_ID < 35 )) || fatal "array task is outside 0--34: $TASK_ID"
readonly SOURCE_INDEX=$((TASK_ID / 5))
readonly FOLD=$((TASK_ID % 5))
readonly CONDITION_LABEL="${CONDITION_LABELS[$SOURCE_INDEX]}"
readonly GENERATOR_LABEL="${GENERATOR_LABELS[$SOURCE_INDEX]}"
readonly DATASET_ID="${DATASET_IDS[$SOURCE_INDEX]}"
readonly EXPERIMENT_NAME="exp16_2_synthetic_105_common105_${GENERATOR_LABEL}_d${DATASET_ID}"

readonly TRAINER="nnUNetTrainerBrainMets"
readonly PLANS_NAME="nnUNetResEncUNetLPlansD600"
readonly CONFIGURATION="3d_fullres"
readonly MODEL_NAME="${TRAINER}__${PLANS_NAME}__${CONFIGURATION}"
readonly EXPECTED_CASES=21

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
readonly NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly RAW_ROOT="${NNUNET_BASE}/nnUNet_raw"
readonly PREPROCESSED_ROOT="${NNUNET_BASE}/nnUNet_preprocessed_${EXPERIMENT_NAME}"
readonly PREPROCESSED_DATASET="${PREPROCESSED_ROOT}/Dataset${DATASET_ID}_BrainMet"
readonly SPLITS_PATH="${PREPROCESSED_DATASET}/splits_final.json"
readonly CANONICAL_MODEL_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/Dataset${DATASET_ID}_BrainMet/${MODEL_NAME}"
readonly CANONICAL_FOLD_DIR="${CANONICAL_MODEL_DIR}/fold_${FOLD}"

readonly VALIDATION_ROOT="${NNUNET_RESULTS}/exp16_2_validation_best/job_${ARRAY_JOB_ID}"
readonly TASK_ROOT="${VALIDATION_ROOT}/${EXPERIMENT_NAME}/fold_${FOLD}"
readonly STAGED_RESULTS="${TASK_ROOT}/nnunet_results"
readonly STAGED_MODEL_DIR="${STAGED_RESULTS}/Dataset${DATASET_ID}_BrainMet/${MODEL_NAME}"
readonly STAGED_FOLD_DIR="${STAGED_MODEL_DIR}/fold_${FOLD}"
readonly PREDICTION_DIR="${STAGED_FOLD_DIR}/validation"
readonly COMPLETE_MARKER="${TASK_ROOT}/.best_validation_complete.json"
readonly STDOUT_PATH="${REPO_ROOT}/IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_validation_best_${ARRAY_JOB_ID}_${TASK_ID}.out"
readonly STDERR_PATH="${REPO_ROOT}/IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_validation_best_${ARRAY_JOB_ID}_${TASK_ID}.err"
readonly REGISTRATION_LOCK="${NNUNET_RESULTS}/.locks/custom_trainer_registration.lock"

[[ -d "$RAW_ROOT/Dataset${DATASET_ID}_BrainMet" ]] || \
    fatal "raw dataset is missing: Dataset${DATASET_ID}_BrainMet"
[[ -s "$PREPROCESSED_DATASET/${PLANS_NAME}.json" ]] || \
    fatal "isolated controlled plan is missing: $PREPROCESSED_DATASET"
[[ -s "$PREPROCESSED_DATASET/dataset.json" ]] || \
    fatal "isolated dataset JSON is missing: $PREPROCESSED_DATASET"
[[ -d "$PREPROCESSED_DATASET/gt_segmentations" ]] || \
    fatal "isolated validation labels are missing: $PREPROCESSED_DATASET"
[[ -s "$SPLITS_PATH" ]] || fatal "isolated fixed splits are missing: $SPLITS_PATH"
[[ -d "$CANONICAL_MODEL_DIR" ]] || fatal "trained model is missing: $CANONICAL_MODEL_DIR"
[[ -s "$CANONICAL_FOLD_DIR/checkpoint_best.pth" ]] || \
    fatal "checkpoint_best.pth is missing: $CANONICAL_FOLD_DIR"
[[ -s "$CANONICAL_FOLD_DIR/checkpoint_final.pth" ]] || \
    fatal "checkpoint_final.pth is missing: $CANONICAL_FOLD_DIR"
[[ ! -e "$TASK_ROOT" ]] || fatal "validation task output already exists: $TASK_ROOT"

mkdir -p "$STAGED_FOLD_DIR" "$(dirname "$REGISTRATION_LOCK")"
ln -s -- "$CANONICAL_FOLD_DIR/checkpoint_best.pth" \
    "$STAGED_FOLD_DIR/checkpoint_best.pth"
ln -s -- "$CANONICAL_FOLD_DIR/checkpoint_final.pth" \
    "$STAGED_FOLD_DIR/checkpoint_final.pth"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export nnUNet_raw="$RAW_ROOT"
export nnUNet_preprocessed="$PREPROCESSED_ROOT"
export nnUNet_results="$STAGED_RESULTS"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

flock -x "$REGISTRATION_LOCK" \
    python -c 'from medgen.scripts.train_nnunet import _register_trainer; _register_trainer()'

echo "=== exp16_2 best-checkpoint validation ==="
echo "Condition:  $CONDITION_LABEL"
echo "Generator:  $GENERATOR_LABEL"
echo "Dataset:    $DATASET_ID"
echo "Fold:       $FOLD/4"
echo "Checkpoint: $CANONICAL_FOLD_DIR/checkpoint_best.pth"
echo "Output:     $TASK_ROOT"
nvidia-smi

time nnUNetv2_train "$DATASET_ID" "$CONFIGURATION" "$FOLD" \
    -tr "$TRAINER" \
    -p "$PLANS_NAME" \
    --val \
    --val_best

python -m medgen.scripts.select_exp16_2_validation_source verify-fold \
    --prediction-dir "$PREDICTION_DIR" \
    --reference-dir "$PREPROCESSED_DATASET/gt_segmentations" \
    --splits "$SPLITS_PATH" \
    --fold "$FOLD" \
    --expected-folds 5 \
    --expected-cases "$EXPECTED_CASES" \
    --checkpoint-best "$CANONICAL_FOLD_DIR/checkpoint_best.pth" \
    --checkpoint-final "$CANONICAL_FOLD_DIR/checkpoint_final.pth" \
    --output-marker "$COMPLETE_MARKER" \
    --label "$CONDITION_LABEL" \
    --experiment "$EXPERIMENT_NAME" \
    --dataset-id "$DATASET_ID" \
    --trainer "$TRAINER" \
    --plans "$PLANS_NAME" \
    --configuration "$CONFIGURATION" \
    --array-job-id "$ARRAY_JOB_ID" \
    --array-task-id "$TASK_ID" \
    --job-id "$JOB_ID" \
    --stdout "$STDOUT_PATH" \
    --stderr "$STDERR_PATH"

echo "=== Verified best-checkpoint fold validation ==="
echo "Marker: $COMPLETE_MARKER"
