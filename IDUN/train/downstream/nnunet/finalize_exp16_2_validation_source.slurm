#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="lock16_source"
#SBATCH --output=IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_validation_finalizer_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_validation_finalizer_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Pool the 35 completed best-checkpoint validation folds and lock the source
# with the highest exact mean patient-level complete-volume Dice. This is a
# separate CPU job and never reads the official test set.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly VALIDATION_ARRAY_JOB_ID="${VALIDATION_ARRAY_JOB_ID:?export the validation array job ID}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
readonly NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly VALIDATION_ROOT="${NNUNET_RESULTS}/exp16_2_validation_best/job_${VALIDATION_ARRAY_JOB_ID}"
readonly OUTPUT_DIR="${NNUNET_RESULTS}/exp16_2_validation_source_selection/job_${JOB_ID}"

[[ "$VALIDATION_ARRAY_JOB_ID" =~ ^[0-9]+$ ]] || \
    fatal "VALIDATION_ARRAY_JOB_ID must be numeric: $VALIDATION_ARRAY_JOB_ID"
[[ -d "$VALIDATION_ROOT" ]] || fatal "validation array output is missing: $VALIDATION_ROOT"
[[ ! -e "$OUTPUT_DIR" ]] || fatal "finalizer output already exists: $OUTPUT_DIR"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "=== exp16_2 validation source finalizer ==="
echo "Validation array: $VALIDATION_ARRAY_JOB_ID"
echo "Cohort:           five disjoint folds, 105 real patients per source"
echo "Endpoint:         mean patient-level complete-volume Dice"
echo "Output:           $OUTPUT_DIR"

time python -m medgen.scripts.select_exp16_2_validation_source finalize \
    --condition "original_mse|exp16_2_synthetic_105_common105_exp1_1_1000_d650|650" \
    --condition "extended_mse|exp16_2_synthetic_105_common105_exp1_1_1000plus_d651|651" \
    --condition "perceptual_continuation|exp16_2_synthetic_105_common105_exp32_2_1000_d652|652" \
    --condition "strong_perceptual_continuation|exp16_2_synthetic_105_common105_exp47a_d654|654" \
    --condition "weighted_huber_transition|exp16_2_synthetic_105_common105_exp47c_d656|656" \
    --condition "weighted_huber_handoff|exp16_2_synthetic_105_common105_exp1_to_exp48c_t025_d661|661" \
    --condition "pseudo_huber_perceptual_handoff|exp16_2_synthetic_105_common105_exp1_to_exp48d_t025_d662|662" \
    --validation-root "$VALIDATION_ROOT" \
    --nnunet-raw "$NNUNET_BASE/nnUNet_raw" \
    --nnunet-preprocessed "$NNUNET_BASE/nnUNet_preprocessed" \
    --output-dir "$OUTPUT_DIR" \
    --canonical-dataset-id 600 \
    --expected-folds 5 \
    --expected-cases-per-fold 21

echo "=== Validation ranking ==="
cat "$OUTPUT_DIR/validation_summary.csv"
echo "=== Locked hybrid source ==="
cat "$OUTPUT_DIR/LOCKED_SOURCE.txt"
