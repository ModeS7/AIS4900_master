#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="metrics16_syn"
#SBATCH --output=IDUN/output/train/downstream/nnunet/recompute_exp16_2_synthetic_only_all_metrics_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/recompute_exp16_2_synthetic_only_all_metrics_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Compute the complete prediction-only metric panel after all seven independent
# exp16_2 five-fold evaluations finish. No checkpoint is loaded by this job.
# It reads the saved 51-case ensembles and reports patient, slice, voxel,
# surface, lesion, size-stratified, and paired statistics at full precision.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

marker_value() {
    local marker="$1"
    local key="$2"
    local -a values=()
    mapfile -t values < <(
        awk -F= -v key="$key" '$1 == key { print substr($0, index($0, "=") + 1) }' "$marker"
    )
    [[ ${#values[@]} -eq 1 ]] || fatal \
        "$marker must contain exactly one ${key}=... entry"
    printf '%s\n' "${values[0]}"
}

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly NNUNET_RAW="${CLUSTER_BASE}/MedicalDataSets/nnunet/nnUNet_raw"
readonly OUTPUT_DIR="${NNUNET_RESULTS}/exp16_2_synthetic_only_all_metrics/job_${JOB_ID}"
readonly MARKER_NAME=".exp16_2_5fold_eval_complete"
readonly EXPECTED_CASES=51
readonly EXPECTED_FOLDS="0,1,2,3,4"
readonly BASELINE_EXPERIMENT="exp3_baseline_v2_d600"
readonly BASELINE_STDOUT="${REPO_ROOT}/IDUN/output/train/downstream/nnunet/eval_5fold/exp3_baseline_v2_d600_24512664.out"

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

[[ ! -e "$OUTPUT_DIR" ]] || fatal "output already exists: $OUTPUT_DIR"
[[ -s "$BASELINE_STDOUT" ]] || fatal "baseline provenance is missing: $BASELINE_STDOUT"
[[ -s "${BASELINE_STDOUT%.out}.err" ]] || fatal \
    "baseline stderr provenance is missing: ${BASELINE_STDOUT%.out}.err"

ARGS=(
    --condition "real_reference|${BASELINE_EXPERIMENT}|600|${BASELINE_STDOUT}"
)
COMPARE_ARGS=()

for index in "${!CONDITION_LABELS[@]}"; do
    condition_label="${CONDITION_LABELS[$index]}"
    generator_label="${GENERATOR_LABELS[$index]}"
    dataset_id="${DATASET_IDS[$index]}"
    experiment="exp16_2_synthetic_105_common105_${generator_label}_d${dataset_id}"
    eval_dir="${NNUNET_RESULTS}/${experiment}/eval_${experiment}"
    marker="${eval_dir}/${MARKER_NAME}"

    [[ -s "$marker" ]] || fatal \
        "five-fold evaluation is incomplete for ${condition_label}: $marker"
    [[ "$(marker_value "$marker" experiment)" == "$experiment" ]] || fatal \
        "$marker has the wrong experiment"
    [[ "$(marker_value "$marker" dataset)" == "Dataset${dataset_id}_BrainMet" ]] || fatal \
        "$marker has the wrong dataset"
    [[ "$(marker_value "$marker" folds)" == "$EXPECTED_FOLDS" ]] || fatal \
        "$marker is not a five-fold ensemble"
    [[ "$(marker_value "$marker" prediction_count)" == "$EXPECTED_CASES" ]] || fatal \
        "$marker does not document ${EXPECTED_CASES} predictions"

    provenance_stdout="$(marker_value "$marker" stdout)"
    provenance_stderr="$(marker_value "$marker" stderr)"
    [[ -s "$provenance_stdout" ]] || fatal \
        "missing stdout provenance for ${condition_label}: $provenance_stdout"
    [[ -s "$provenance_stderr" ]] || fatal \
        "missing stderr provenance for ${condition_label}: $provenance_stderr"
    [[ "$provenance_stderr" == "${provenance_stdout%.out}.err" ]] || fatal \
        "stdout/stderr provenance pair disagrees for ${condition_label}"

    ARGS+=(--condition "${condition_label}|${experiment}|${dataset_id}|${provenance_stdout}")
    COMPARE_ARGS+=(--compare "$condition_label")
done

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

echo "=== exp16_2 synthetic-only complete metric panel ==="
echo "Conditions: real reference + ${#CONDITION_LABELS[@]} synthetic sources"
echo "Cases:      ${EXPECTED_CASES} exact official-test cases per source"
echo "Output:     ${OUTPUT_DIR}"

time python -m medgen.scripts.recompute_nnunet_metrics \
    --nnunet-results "$NNUNET_RESULTS" \
    --nnunet-raw "$NNUNET_RAW" \
    --output-dir "$OUTPUT_DIR" \
    --baseline real_reference \
    --expected-cases "$EXPECTED_CASES" \
    --min-component-voxels 5 \
    --detection-threshold 0.1 \
    --bootstrap-draws 10000 \
    --seed 0 \
    "${ARGS[@]}" \
    "${COMPARE_ARGS[@]}"

echo "=== Completed exp16_2 synthetic-only metric panel ==="
echo "Full JSON:        ${OUTPUT_DIR}/summary.json"
echo "All metrics CSV:  ${OUTPUT_DIR}/summary.csv"
echo "Per patient CSV:  ${OUTPUT_DIR}/per_case.csv"
echo "Per lesion CSV:   ${OUTPUT_DIR}/per_lesion.csv"
echo "Paired tests CSV: ${OUTPUT_DIR}/paired_statistics.csv"
