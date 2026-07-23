#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="dice16_valtest"
#SBATCH --output=IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_validation_test_dice_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/eval_5fold/exp16_2_validation_test_dice_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Read the existing saved predictions and print exactly four metrics per source:
# validation Dice, validation tiny Dice, test Dice, and test tiny Dice.

set -Eeuo pipefail

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
: "${VALIDATION_SELECTION_JOB_ID:=24929442}"
: "${TEST_METRICS_JOB_ID:=24929214}"

readonly RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly SELECTION_DIR="${RESULTS}/exp16_2_validation_source_selection/job_${VALIDATION_SELECTION_JOB_ID}"
readonly TEST_SUMMARY="${RESULTS}/exp16_2_synthetic_only_all_metrics/job_${TEST_METRICS_JOB_ID}/summary.json"
readonly OUTPUT_DIR="${RESULTS}/exp16_2_validation_test_dice/job_${JOB_ID}"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

python -m medgen.scripts.compare_exp16_2_validation_test_dice \
    --selection-dir "$SELECTION_DIR" \
    --test-summary "$TEST_SUMMARY" \
    --output-dir "$OUTPUT_DIR"
