#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="eval_gen_panel_metrics"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/eval_gen_panel_metrics_%j.out
#SBATCH --error=IDUN/output/generate/eval_gen_panel_metrics_%j.err

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
readonly PANEL_ID="${PANEL_ID:-source_selection_train105_seed42_euler100}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly TRAIN_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/train"
readonly PANEL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${PANEL_ID}"
readonly RESULT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/${PANEL_ID}"
readonly REPORT="${RESULT_DIR}/all_metrics_train105.json"
readonly TEMP_REPORT="${RESULT_DIR}/.all_metrics_train105.job-${SLURM_JOB_ID}.tmp.json"
readonly MANIFEST_TOOL="${REPO_ROOT}/IDUN/generate/panel_manifest.py"
readonly LABELS=(
    exp1_1_1000
    exp1_1_1000plus
    exp32_2_1000
    exp32_3_1000
    exp47a
    exp47b
    exp47c
    exp47d
    exp47e
    exp1_to_exp48a_t025
    exp1_to_exp48b_t025
    exp1_to_exp48c_t025
    exp1_to_exp48d_t025
    exp1_to_exp48e_t025
)

[[ -f "$REPO_ROOT/pyproject.toml" ]] || fatal "repository root not found: $REPO_ROOT"
[[ -d "$TRAIN_DIR" ]] || fatal "training reference not found: $TRAIN_DIR"
[[ -f "$MANIFEST_TOOL" ]] || fatal "manifest helper not found: $MANIFEST_TOOL"
[[ ! -e "$REPORT" ]] || fatal "metric report already exists; refusing to overwrite: $REPORT"
[[ ! -e "$TEMP_REPORT" ]] || fatal "metric staging report already exists: $TEMP_REPORT"
mkdir -p "$RESULT_DIR"
cd "$REPO_ROOT"

readonly SOURCE_COMMIT="${PANEL_SOURCE_COMMIT:?PANEL_SOURCE_COMMIT must be exported by the panel launcher}"

verify_source_tree() {
    local current_commit
    current_commit="$(git rev-parse --verify HEAD)" || fatal "could not resolve the repository commit"
    [[ "$current_commit" == "$SOURCE_COMMIT" ]] || fatal \
        "repository commit changed after panel submission: expected $SOURCE_COMMIT, found $current_commit"
    if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
        fatal "repository worktree is not clean; refusing to evaluate the paper panel"
    fi
}

verify_source_tree

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

MANIFEST_ARGS=()
DATASET_ARGS=()
for label in "${LABELS[@]}"; do
    dataset_dir="${PANEL_ROOT}/${label}"
    MANIFEST_ARGS+=(--manifest "${dataset_dir}/panel_job_manifest.json" --expected-label "$label")
    DATASET_ARGS+=(--dataset "${label}:${dataset_dir}")
done

echo "=== Validating all 14 completed generator jobs before metric extraction ==="
python "$MANIFEST_TOOL" validate --panel-root "$PANEL_ROOT" \
    --expected-git-commit "$SOURCE_COMMIT" \
    "${MANIFEST_ARGS[@]}"

echo "=== One combined train-reference metric pass for all 14 datasets ==="
time python -m medgen.scripts.eval_all_metrics \
    --reference "train:${TRAIN_DIR}" \
    "${DATASET_ARGS[@]}" \
    --pattern bravo.nii.gz \
    --target-shape 150 256 256 \
    --pool-cap 0 \
    --ref-cap 0 \
    --diversity-cap 105 \
    --trim-slices 0 \
    --med3d-batch 2 \
    --pca-model none \
    --seed 42 \
    --source-git-commit "$SOURCE_COMMIT" \
    --device cuda \
    --output "$TEMP_REPORT"

verify_source_tree

REPORT_VALIDATION_ARGS=()
for label in "${LABELS[@]}"; do
    REPORT_VALIDATION_ARGS+=(--expected-label "$label")
done
python "$MANIFEST_TOOL" validate-report --report "$TEMP_REPORT" \
    --expected-git-commit "$SOURCE_COMMIT" \
    "${REPORT_VALIDATION_ARGS[@]}"
[[ ! -e "$REPORT" ]] || fatal "metric report appeared during evaluation: $REPORT"
mv -- "$TEMP_REPORT" "$REPORT"
echo "Published combined metric report atomically: $REPORT"
