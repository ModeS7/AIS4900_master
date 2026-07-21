#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|gpu80g"
#SBATCH --job-name="recover_sm_47a_00400"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/recover_sm_47a_00400_%j.out
#SBATCH --error=IDUN/output/generate/recover_sm_47a_00400_%j.err

# Recover only exp47a candidate 00400. The original evaluation exhausted
# deterministic draws 1--3, so this job evaluates draws 4--50 and fails closed
# if none passes the unchanged PCA brain-support containment rule.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

require_marker_value() {
    local marker="$1"
    local key="$2"
    local expected="$3"
    local -a values=()

    mapfile -t values < <(
        awk -F= -v key="$key" \
            '$1 == key { print substr($0, index($0, "=") + 1) }' "$marker"
    )
    [[ ${#values[@]} -eq 1 ]] || fatal \
        "$marker must contain exactly one ${key} entry"
    [[ "${values[0]}" == "$expected" ]] || fatal \
        "$marker has ${key}=${values[0]}; expected ${expected}"
}

readonly LABEL="exp47a"
readonly LOW_RUN="exp47a_lpips_strong_20260425-055252"
readonly CANDIDATE_ID="00400"
readonly CANDIDATE_INDEX=400
readonly NUM_SOURCE_MASKS=525
readonly NUM_IMAGES=401
readonly ORIGINAL_BASE_SEED=42
readonly ATTEMPT_STRIDE=1000000
readonly FIRST_RETRY_DRAW=4
readonly LAST_RETRY_DRAW=50
readonly RETRY_DRAWS=$((LAST_RETRY_DRAW - FIRST_RETRY_DRAW + 1))
readonly RETRY_BASE_SEED=$((
    ORIGINAL_BASE_SEED + (FIRST_RETRY_DRAW - 1) * ATTEMPT_STRIDE
))
readonly FIRST_RETRY_SEED=$((RETRY_BASE_SEED + CANDIDATE_INDEX))
readonly LAST_RETRY_SEED=$((
    RETRY_BASE_SEED + CANDIDATE_INDEX + (RETRY_DRAWS - 1) * ATTEMPT_STRIDE
))
readonly STEPS=100
readonly BRAIN_MARGIN_MM=2.323
readonly EXPECTED_CANONICAL_BEFORE=154
readonly EXPECTED_CANONICAL_AFTER=155
readonly EXPECTED_PANEL_BEFORE=2124
readonly EXPECTED_PANEL_AFTER=2125
readonly BINS_HEADER='id,bin_0,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,total_tumors'
readonly BINS_ROW="${CANDIDATE_ID},0,0,0,0,0,0,0,0"
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly BRAVO_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"
readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly CANDIDATE_MASKS="${EVAL_ROOT}/ordered_common_seg_masks_seed42/candidates525"
readonly CANDIDATE_MASK="${CANDIDATE_MASKS}/${CANDIDATE_ID}/seg.nii.gz"
readonly MATCHED105_ROOT="${EVAL_ROOT}/matched_common105"
readonly COMMON105_IDS="${MATCHED105_ROOT}/common_candidate_ids.txt"
readonly MATCHED105_VIEW="${MATCHED105_ROOT}/${LABEL}"
readonly LOW_CKPT="${BRAVO_ROOT}/${LOW_RUN}/checkpoint_latest.pt"
readonly BRAIN_SUPPORT_PCA="${REPO_ROOT}/data/brain_support_pca_train105_var95_256x256x160.npz"
readonly CANONICAL_DIR="${EVAL_ROOT}/${LABEL}"
readonly CANONICAL_CASE="${CANONICAL_DIR}/${CANDIDATE_ID}"
readonly CANONICAL_BINS="${CANONICAL_DIR}/bins.csv"
readonly DATASET_MASK_MARKER="${CANONICAL_DIR}/.brain_mask_stage3_applied"
readonly PANEL_MASK_MARKER="${EVAL_ROOT}/.brain_mask_stage3_all14_complete"
readonly LOCK_FILE="${EVAL_ROOT}/.brain_mask_stage3_all14.lock"
readonly TRANSACTION_DIR="${EVAL_ROOT}/.recover_${LABEL}_${CANDIDATE_ID}.transaction"
readonly TRANSACTION_PREP="${TRANSACTION_DIR}.prepare_${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
readonly SHARD_LABEL="${LABEL}_recovery_${CANDIDATE_ID}_a4to50_${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
readonly SHARD_DIR="${EVAL_ROOT}/${SHARD_LABEL}"
readonly SHARD_CASE="${SHARD_DIR}/${CANDIDATE_ID}"
readonly SHARD_MARKER="${SHARD_DIR}/.candidate_retry_complete"
readonly CASE_RECOVERY_MARKER="${CANONICAL_CASE}/.recovery_provenance"
readonly VIEW_RECOVERY_MARKER="${MATCHED105_VIEW}/.recovery_provenance"

atomic_copy_replace() {
    local source="$1"
    local target="$2"
    local temporary="${target}.replace_${SLURM_JOB_ID}"

    cp -- "$source" "$temporary"
    mv -f -- "$temporary" "$target"
}

validate_case_bin_parity() {
    local expected_count="$1"
    local context="$2"
    local -a case_ids=()
    local -a bin_ids=()

    mapfile -t case_ids < <(
        find -P "$CANONICAL_DIR" -mindepth 1 -maxdepth 1 -type d \
            -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort
    )
    mapfile -t bin_ids < <(tail -n +2 "$CANONICAL_BINS" | cut -d, -f1)
    [[ ${#case_ids[@]} -eq $expected_count ]] || fatal \
        "$context has ${#case_ids[@]} case directories; expected $expected_count"
    [[ ${#bin_ids[@]} -eq $expected_count ]] || fatal \
        "$context bins.csv has ${#bin_ids[@]} rows; expected $expected_count"
    [[ "$(printf '%s\n' "${case_ids[@]}")" == "$(printf '%s\n' "${bin_ids[@]}")" ]] || \
        fatal "$context case IDs and bins.csv IDs differ"
    [[ "$(printf '%s\n' "${bin_ids[@]}" | LC_ALL=C sort -u | wc -l)" -eq $expected_count ]] || \
        fatal "$context bins.csv contains duplicate or unsorted IDs"
}

validate_completed_recovery() {
    local recovery_job mask_sha256 bravo_sha256

    validate_case_bin_parity "$EXPECTED_CANONICAL_AFTER" "completed canonical dataset"
    [[ -s "$CANONICAL_CASE/seg.nii.gz" ]] || fatal \
        "completed recovery is missing the segmentation"
    [[ -s "$CANONICAL_CASE/bravo.nii.gz" ]] || fatal \
        "completed recovery is missing the BRAVO"
    [[ -s "$CASE_RECOVERY_MARKER" ]] || fatal \
        "completed recovery is missing case provenance"
    [[ "$(grep -c "^${CANDIDATE_ID}," "$CANONICAL_BINS")" -eq 1 ]] || fatal \
        "completed recovery must contain one bins.csv row for $CANDIDATE_ID"
    [[ "$CANONICAL_CASE/seg.nii.gz" -ef "$CANDIDATE_MASK" ]] || fatal \
        "completed recovery segmentation is not the frozen candidate mask"
    require_marker_value "$CASE_RECOVERY_MARKER" candidate_id "$CANDIDATE_ID"
    recovery_job="$(
        awk -F= '$1 == "job_id" { print substr($0, index($0, "=") + 1) }' \
            "$CASE_RECOVERY_MARKER"
    )"
    [[ -n "$recovery_job" ]] || fatal "completed recovery has no provenance job ID"
    require_marker_value "$CASE_RECOVERY_MARKER" job_id "$recovery_job"
    require_marker_value "$CASE_RECOVERY_MARKER" brain_threshold 0.05
    require_marker_value "$CASE_RECOVERY_MARKER" brain_fill_holes true
    require_marker_value "$CASE_RECOVERY_MARKER" brain_dilate_pixels 2
    mask_sha256="$(sha256sum "$CANONICAL_CASE/seg.nii.gz" | awk '{print $1}')"
    bravo_sha256="$(sha256sum "$CANONICAL_CASE/bravo.nii.gz" | awk '{print $1}')"
    require_marker_value "$CASE_RECOVERY_MARKER" mask_sha256 "$mask_sha256"
    require_marker_value "$CASE_RECOVERY_MARKER" masked_bravo_sha256 "$bravo_sha256"
    require_marker_value "$DATASET_MASK_MARKER" accepted_cases "$EXPECTED_CANONICAL_AFTER"
    require_marker_value "$DATASET_MASK_MARKER" extension_candidate "$CANDIDATE_ID"
    require_marker_value "$DATASET_MASK_MARKER" extension_job_id "$recovery_job"
    require_marker_value "$PANEL_MASK_MARKER" accepted_volumes "$EXPECTED_PANEL_AFTER"
    require_marker_value "$PANEL_MASK_MARKER" extension_label "$LABEL"
    require_marker_value "$PANEL_MASK_MARKER" extension_candidate "$CANDIDATE_ID"
    require_marker_value "$PANEL_MASK_MARKER" extension_job_id "$recovery_job"

    [[ -d "$MATCHED105_VIEW" && ! -L "$MATCHED105_VIEW" ]] || fatal \
        "completed recovery is missing the exp47a common-105 view"
    [[ -s "$VIEW_RECOVERY_MARKER" ]] || fatal \
        "completed common-105 view is missing recovery provenance"
    require_marker_value "$VIEW_RECOVERY_MARKER" candidate_id "$CANDIDATE_ID"
    require_marker_value "$VIEW_RECOVERY_MARKER" job_id "$recovery_job"
    local candidate_id view_count
    view_count="$(
        find -P "$MATCHED105_VIEW" -mindepth 1 -maxdepth 1 -type d \
            -name '[0-9][0-9][0-9][0-9][0-9]' | wc -l
    )"
    [[ "$view_count" -eq 105 ]] || fatal \
        "completed exp47a common-105 view has $view_count cases"
    while IFS= read -r candidate_id; do
        [[ "$MATCHED105_VIEW/$candidate_id/seg.nii.gz" \
            -ef "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" ]] || fatal \
            "common-105 segmentation link is invalid: $candidate_id"
        [[ "$MATCHED105_VIEW/$candidate_id/bravo.nii.gz" \
            -ef "$CANONICAL_DIR/$candidate_id/bravo.nii.gz" ]] || fatal \
            "common-105 BRAVO link is invalid: $candidate_id"
    done < "$COMMON105_IDS"
}

rollback_incomplete_transaction() {
    [[ -d "$TRANSACTION_DIR" && ! -L "$TRANSACTION_DIR" ]] || fatal \
        "invalid recovery transaction path: $TRANSACTION_DIR"
    [[ ! -e "$TRANSACTION_DIR/COMMITTED" ]] || fatal \
        "refusing to roll back a committed recovery transaction"
    [[ -s "$TRANSACTION_DIR/bins.before" ]] || fatal \
        "incomplete transaction lacks bins.csv backup"
    [[ -s "$TRANSACTION_DIR/dataset_marker.before" ]] || fatal \
        "incomplete transaction lacks dataset-marker backup"
    [[ -s "$TRANSACTION_DIR/panel_marker.before" ]] || fatal \
        "incomplete transaction lacks panel-marker backup"
    [[ "$(head -n 1 "$TRANSACTION_DIR/bins.before")" == "$BINS_HEADER" ]] || fatal \
        "transaction bins.csv backup has an unexpected header"
    [[ "$(( $(wc -l < "$TRANSACTION_DIR/bins.before") - 1 ))" -eq $EXPECTED_CANONICAL_BEFORE ]] || fatal \
        "transaction bins.csv backup has an unexpected row count"
    ! grep -q "^${CANDIDATE_ID}," "$TRANSACTION_DIR/bins.before" || fatal \
        "transaction bins.csv backup already contains $CANDIDATE_ID"
    require_marker_value \
        "$TRANSACTION_DIR/dataset_marker.before" accepted_cases "$EXPECTED_CANONICAL_BEFORE"
    require_marker_value \
        "$TRANSACTION_DIR/panel_marker.before" accepted_volumes "$EXPECTED_PANEL_BEFORE"

    if [[ -e "$CANONICAL_CASE" || -L "$CANONICAL_CASE" ]]; then
        [[ -s "$CASE_RECOVERY_MARKER" ]] || fatal \
            "candidate $CANDIDATE_ID exists without recovery provenance; refusing removal"
        require_marker_value "$CASE_RECOVERY_MARKER" candidate_id "$CANDIDATE_ID"
        rm -rf -- "$CANONICAL_CASE"
    fi
    if [[ -e "$MATCHED105_VIEW" || -L "$MATCHED105_VIEW" ]]; then
        [[ -s "$VIEW_RECOVERY_MARKER" ]] || fatal \
            "exp47a common-105 view exists without recovery provenance"
        require_marker_value "$VIEW_RECOVERY_MARKER" candidate_id "$CANDIDATE_ID"
        rm -rf -- "$MATCHED105_VIEW"
    fi
    atomic_copy_replace "$TRANSACTION_DIR/bins.before" "$CANONICAL_BINS"
    atomic_copy_replace "$TRANSACTION_DIR/dataset_marker.before" "$DATASET_MASK_MARKER"
    atomic_copy_replace "$TRANSACTION_DIR/panel_marker.before" "$PANEL_MASK_MARKER"
    rm -rf -- "$TRANSACTION_DIR"
    echo "Rolled back an incomplete earlier recovery transaction."
}

[[ -f "${REPO_ROOT}/pyproject.toml" ]] || fatal "repository root not found: $REPO_ROOT"
[[ -d "$EVAL_ROOT" ]] || fatal "evaluation root not found: $EVAL_ROOT"

# Lock before inspecting mutable canonical state.
exec 9>"$LOCK_FILE"
flock -n 9 || fatal "another panel-modifying job holds $LOCK_FILE"

[[ -s "$CANDIDATE_MASK" ]] || fatal "conditioning mask not found: $CANDIDATE_MASK"
[[ -s "$COMMON105_IDS" ]] || fatal "frozen common-105 ID list not found: $COMMON105_IDS"
grep -Fxq "$CANDIDATE_ID" "$COMMON105_IDS" || fatal \
    "$CANDIDATE_ID is not part of the frozen common-105 cohort"
[[ "$(wc -l < "$COMMON105_IDS")" -eq 105 ]] || fatal \
    "frozen common-105 ID list must contain exactly 105 rows"
[[ "$(LC_ALL=C sort -u "$COMMON105_IDS" | wc -l)" -eq 105 ]] || fatal \
    "frozen common-105 ID list contains duplicate IDs"
[[ -d "$MATCHED105_ROOT" && ! -L "$MATCHED105_ROOT" ]] || fatal \
    "matched common-105 root is missing or is a symlink: $MATCHED105_ROOT"
[[ -d "$CANONICAL_DIR" && ! -L "$CANONICAL_DIR" ]] || fatal \
    "canonical dataset is missing or is a symlink: $CANONICAL_DIR"
[[ -s "$CANONICAL_BINS" ]] || fatal "canonical bins.csv not found: $CANONICAL_BINS"
[[ "$(head -n 1 "$CANONICAL_BINS")" == "$BINS_HEADER" ]] || fatal \
    "unexpected canonical bins.csv header"
[[ -s "$DATASET_MASK_MARKER" ]] || fatal \
    "canonical dataset brain-mask marker not found: $DATASET_MASK_MARKER"
[[ -s "$PANEL_MASK_MARKER" ]] || fatal \
    "all-14 brain-mask marker not found: $PANEL_MASK_MARKER"

# A previous allocation may have stopped between individual filesystem
# renames. Reconcile that durable transaction before assessing canonical state.
if [[ -e "$TRANSACTION_DIR" || -L "$TRANSACTION_DIR" ]]; then
    [[ -d "$TRANSACTION_DIR" && ! -L "$TRANSACTION_DIR" ]] || fatal \
        "invalid recovery transaction path: $TRANSACTION_DIR"
    if [[ -e "$TRANSACTION_DIR/COMMITTED" ]]; then
        validate_completed_recovery
        rm -rf -- "$TRANSACTION_DIR"
        echo "Recovery was already committed by an earlier allocation."
        exit 0
    fi
    rollback_incomplete_transaction
fi

case_present=false
bins_row_present=false
[[ -e "$CANONICAL_CASE" || -L "$CANONICAL_CASE" ]] && case_present=true
grep -q "^${CANDIDATE_ID}," "$CANONICAL_BINS" && bins_row_present=true
if [[ "$case_present" == true || "$bins_row_present" == true ]]; then
    [[ "$case_present" == true && "$bins_row_present" == true ]] || fatal \
        "untracked partial recovery found for candidate $CANDIDATE_ID"
    validate_completed_recovery
    echo "Recovery is already complete for ${LABEL}/${CANDIDATE_ID}."
    exit 0
fi

[[ ! -e "$CANONICAL_CASE" && ! -L "$CANONICAL_CASE" ]] || fatal \
    "canonical candidate already exists: $CANONICAL_CASE"
[[ ! -e "$MATCHED105_VIEW" && ! -L "$MATCHED105_VIEW" ]] || fatal \
    "exp47a matched common-105 view already exists: $MATCHED105_VIEW"
[[ -s "$LOW_CKPT" ]] || fatal "checkpoint not found: $LOW_CKPT"
[[ -s "$BRAIN_SUPPORT_PCA" ]] || fatal "brain-support PCA not found: $BRAIN_SUPPORT_PCA"
[[ ! -e "$SHARD_DIR" ]] || fatal "retry shard already exists: $SHARD_DIR"
[[ ! -e "$TRANSACTION_PREP" ]] || fatal \
    "recovery transaction staging path already exists: $TRANSACTION_PREP"

validate_case_bin_parity "$EXPECTED_CANONICAL_BEFORE" "canonical dataset before recovery"
if grep -q "^${CANDIDATE_ID}," "$CANONICAL_BINS"; then
    fatal "canonical bins.csv already contains $CANDIDATE_ID"
fi

require_marker_value "$DATASET_MASK_MARKER" threshold 0.05
require_marker_value "$DATASET_MASK_MARKER" fill_holes true
require_marker_value "$DATASET_MASK_MARKER" dilate_pixels 2
require_marker_value "$DATASET_MASK_MARKER" accepted_cases "$EXPECTED_CANONICAL_BEFORE"
require_marker_value "$PANEL_MASK_MARKER" threshold 0.05
require_marker_value "$PANEL_MASK_MARKER" fill_holes true
require_marker_value "$PANEL_MASK_MARKER" dilate_pixels 2
require_marker_value "$PANEL_MASK_MARKER" accepted_volumes "$EXPECTED_PANEL_BEFORE"
! grep -q '^extension_' "$DATASET_MASK_MARKER" || fatal \
    "dataset brain-mask marker already contains extension provenance"
! grep -q '^extension_' "$PANEL_MASK_MARKER" || fatal \
    "panel brain-mask marker already contains extension provenance"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONSAFEPATH=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Recovering ${LABEL}/${CANDIDATE_ID} ==="
echo "Deterministic draws: ${FIRST_RETRY_DRAW}--${LAST_RETRY_DRAW}"
echo "Noise seeds:         ${FIRST_RETRY_SEED}--${LAST_RETRY_SEED}"
echo "Output shard:        ${SHARD_DIR}"

time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=bravo \
    image_size=256 \
    depth=160 \
    trim_slices=10 \
    fov_mm=240.0 \
    image_model="$LOW_CKPT" \
    image_model_high_t=null \
    real_seg_dir="$CANDIDATE_MASKS" \
    expected_real_cases="$NUM_SOURCE_MASKS" \
    expected_real_depth=150 \
    seed="$RETRY_BASE_SEED" \
    num_images="$NUM_IMAGES" \
    current_image="$CANDIDATE_INDEX" \
    num_steps_bravo="$STEPS" \
    ode_solver=euler \
    shift_ratio_bravo=1.0 \
    cfg_scale_bravo=1.0 \
    validate_size_bins=false \
    validate_real_seg_masks=true \
    require_real_bravo_pairs=false \
    validate_brain_mask=true \
    conditioning_brain_qc_mode=reject \
    brain_threshold=0.05 \
    brain_containment_margin_mm="$BRAIN_MARGIN_MM" \
    max_image_attempts_per_mask="$RETRY_DRAWS" \
    skip_failed_fixed_masks=false \
    brain_atlas_path=null \
    brain_support_pca_path="$BRAIN_SUPPORT_PCA" \
    brain_pca_path=null \
    seg_pca_path=null \
    diffrs_checkpoint=null \
    mask_outside_brain=false \
    paths.generated_dir="$EVAL_ROOT" \
    output_subdir="$SHARD_LABEL" \
    verbose=true

[[ -s "$SHARD_CASE/seg.nii.gz" ]] || fatal "retry did not save the fixed mask"
[[ -s "$SHARD_CASE/bravo.nii.gz" ]] || fatal "retry did not save an accepted BRAVO"
[[ -s "$SHARD_DIR/bins.csv" ]] || fatal "retry did not save bins.csv"
[[ "$(head -n 1 "$SHARD_DIR/bins.csv")" == "$BINS_HEADER" ]] || fatal \
    "unexpected retry bins.csv header"
[[ "$(tail -n +2 "$SHARD_DIR/bins.csv")" == "$BINS_ROW" ]] || fatal \
    "retry bins.csv does not contain exactly the expected candidate row"

mapfile -t shard_ids < <(
    find -P "$SHARD_DIR" -mindepth 1 -maxdepth 1 -type d \
        -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort
)
[[ ${#shard_ids[@]} -eq 1 && "${shard_ids[0]}" == "$CANDIDATE_ID" ]] || fatal \
    "retry shard must contain only candidate $CANDIDATE_ID"

python - "$CANDIDATE_MASK" "$SHARD_CASE/seg.nii.gz" <<'PY'
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

source = Path(sys.argv[1])
generated = Path(sys.argv[2])
source_data = np.asarray(nib.load(str(source)).dataobj, dtype=np.float32)
generated_data = np.asarray(nib.load(str(generated)).dataobj, dtype=np.float32)
if source_data.shape != (256, 256, 150):
    raise SystemExit(f"unexpected source-mask shape: {source_data.shape}")
if not np.array_equal(source_data, generated_data):
    raise SystemExit("generated segmentation differs from fixed candidate 00400")
PY

# Apply exactly the same posthoc background masking used by the existing panel.
python -m medgen.scripts.brain_mask_existing \
    --src "$SHARD_DIR" \
    --in-place \
    --threshold 0.05 \
    --dilate-pixels 2 \
    --expected-shape 256 256 150

mask_sha256="$(sha256sum "$CANDIDATE_MASK" | awk '{print $1}')"
bravo_sha256="$(sha256sum "$SHARD_CASE/bravo.nii.gz" | awk '{print $1}')"
repository_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
{
    echo "label=$LABEL"
    echo "candidate_id=$CANDIDATE_ID"
    echo "original_base_seed=$ORIGINAL_BASE_SEED"
    echo "retry_base_seed=$RETRY_BASE_SEED"
    echo "first_retry_draw=$FIRST_RETRY_DRAW"
    echo "last_retry_draw=$LAST_RETRY_DRAW"
    echo "first_retry_seed=$FIRST_RETRY_SEED"
    echo "last_retry_seed=$LAST_RETRY_SEED"
    echo "selection_rule=first_passing_draw_in_4_to_50"
    echo "mask_sha256=$mask_sha256"
    echo "masked_bravo_sha256=$bravo_sha256"
    echo "brain_threshold=0.05"
    echo "brain_fill_holes=true"
    echo "brain_dilate_pixels=2"
    echo "repository_commit=$repository_commit"
    echo "job_id=$SLURM_JOB_ID"
    echo "canonical_cases_before=$EXPECTED_CANONICAL_BEFORE"
    echo "canonical_cases_after=$EXPECTED_CANONICAL_AFTER"
} > "${SHARD_MARKER}.tmp"
mv -- "${SHARD_MARKER}.tmp" "$SHARD_MARKER"

# Prepare a durable transaction before touching the canonical dataset. Its
# backups allow a later allocation to roll back safely after node loss.
transaction_active=false
transaction_committed=false
cleanup_recovery() {
    local status=$?
    trap - EXIT HUP INT TERM
    rm -rf -- "$TRANSACTION_PREP"
    if [[ "$transaction_active" == true && "$transaction_committed" == false ]]; then
        rollback_incomplete_transaction
    fi
    exit "$status"
}
trap cleanup_recovery EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir "$TRANSACTION_PREP"
cp -- "$CANONICAL_BINS" "$TRANSACTION_PREP/bins.before"
cp -- "$DATASET_MASK_MARKER" "$TRANSACTION_PREP/dataset_marker.before"
cp -- "$PANEL_MASK_MARKER" "$TRANSACTION_PREP/panel_marker.before"
mkdir "$TRANSACTION_PREP/case.after"
ln -- "$CANDIDATE_MASK" "$TRANSACTION_PREP/case.after/seg.nii.gz"
ln -- "$SHARD_CASE/bravo.nii.gz" "$TRANSACTION_PREP/case.after/bravo.nii.gz"
cp -- "$SHARD_MARKER" "$TRANSACTION_PREP/case.after/.recovery_provenance"
mkdir "$TRANSACTION_PREP/matched105.after"
cp -- "$SHARD_MARKER" "$TRANSACTION_PREP/matched105.after/.recovery_provenance"
while IFS= read -r candidate_id; do
    [[ "$candidate_id" =~ ^[0-9]{5}$ ]] || fatal \
        "invalid frozen common-105 candidate ID: $candidate_id"
    [[ -s "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" ]] || fatal \
        "frozen common-105 mask is missing: $candidate_id"
    if [[ "$candidate_id" != "$CANDIDATE_ID" ]]; then
        [[ -s "$CANONICAL_DIR/$candidate_id/bravo.nii.gz" ]] || fatal \
            "exp47a is missing another common-105 candidate: $candidate_id"
    fi
    mkdir "$TRANSACTION_PREP/matched105.after/$candidate_id"
    ln -s -- \
        "$CANDIDATE_MASKS/$candidate_id/seg.nii.gz" \
        "$TRANSACTION_PREP/matched105.after/$candidate_id/seg.nii.gz"
    ln -s -- \
        "$CANONICAL_DIR/$candidate_id/bravo.nii.gz" \
        "$TRANSACTION_PREP/matched105.after/$candidate_id/bravo.nii.gz"
done < "$COMMON105_IDS"

{
    head -n 1 "$CANONICAL_BINS"
    {
        tail -n +2 "$CANONICAL_BINS"
        echo "$BINS_ROW"
    } | LC_ALL=C sort -t, -k1,1
} > "$TRANSACTION_PREP/bins.after"

awk -F= -v replacement="$EXPECTED_CANONICAL_AFTER" '
    $1 == "accepted_cases" { print "accepted_cases=" replacement; next }
    { print }
' "$DATASET_MASK_MARKER" > "$TRANSACTION_PREP/dataset_marker.after"
{
    echo "extension_candidate=$CANDIDATE_ID"
    echo "extension_job_id=$SLURM_JOB_ID"
} >> "$TRANSACTION_PREP/dataset_marker.after"

awk -F= -v replacement="$EXPECTED_PANEL_AFTER" '
    $1 == "accepted_volumes" { print "accepted_volumes=" replacement; next }
    { print }
' "$PANEL_MASK_MARKER" > "$TRANSACTION_PREP/panel_marker.after"
{
    echo "extension_label=$LABEL"
    echo "extension_candidate=$CANDIDATE_ID"
    echo "extension_job_id=$SLURM_JOB_ID"
} >> "$TRANSACTION_PREP/panel_marker.after"

staged_bins_count="$(( $(wc -l < "$TRANSACTION_PREP/bins.after") - 1 ))"
[[ "$staged_bins_count" -eq $EXPECTED_CANONICAL_AFTER ]] || fatal \
    "staged bins.csv has $staged_bins_count rows; expected $EXPECTED_CANONICAL_AFTER"
[[ "$(grep -c "^${CANDIDATE_ID}," "$TRANSACTION_PREP/bins.after")" -eq 1 ]] || fatal \
    "staged bins.csv must contain candidate $CANDIDATE_ID exactly once"
duplicate_bins_ids="$(
    tail -n +2 "$TRANSACTION_PREP/bins.after" | cut -d, -f1 | sort | uniq -d | wc -l
)"
[[ "$duplicate_bins_ids" -eq 0 ]] || fatal "staged bins.csv contains duplicate IDs"
require_marker_value \
    "$TRANSACTION_PREP/dataset_marker.after" accepted_cases "$EXPECTED_CANONICAL_AFTER"
require_marker_value \
    "$TRANSACTION_PREP/panel_marker.after" accepted_volumes "$EXPECTED_PANEL_AFTER"
require_marker_value \
    "$TRANSACTION_PREP/dataset_marker.after" extension_candidate "$CANDIDATE_ID"
require_marker_value \
    "$TRANSACTION_PREP/panel_marker.after" extension_candidate "$CANDIDATE_ID"

# Publishing the prepared directory makes the transaction recoverable before
# the first canonical rename. EXIT and signal traps restore every backup until
# COMMITTED exists.
mv -- "$TRANSACTION_PREP" "$TRANSACTION_DIR"
transaction_active=true
mv -- "$TRANSACTION_DIR/case.after" "$CANONICAL_CASE"
atomic_copy_replace "$TRANSACTION_DIR/bins.after" "$CANONICAL_BINS"
atomic_copy_replace \
    "$TRANSACTION_DIR/dataset_marker.after" "$DATASET_MASK_MARKER"
atomic_copy_replace "$TRANSACTION_DIR/panel_marker.after" "$PANEL_MASK_MARKER"
mv -- "$TRANSACTION_DIR/matched105.after" "$MATCHED105_VIEW"

validate_completed_recovery
[[ "$CANONICAL_CASE/bravo.nii.gz" -ef "$SHARD_CASE/bravo.nii.gz" ]] || fatal \
    "canonical BRAVO is not the validated masked retry"
touch "$TRANSACTION_DIR/COMMITTED"
transaction_committed=true
rm -rf -- "$TRANSACTION_DIR"
transaction_active=false
trap - EXIT HUP INT TERM

echo "Recovered ${LABEL}/${CANDIDATE_ID} with deterministic draws ${FIRST_RETRY_DRAW}--${LAST_RETRY_DRAW}."
echo "Canonical ${LABEL} now contains ${EXPECTED_CANONICAL_AFTER} accepted candidates."
echo "Matched common-105 view: $MATCHED105_VIEW"
echo "Recovery provenance: $CASE_RECOVERY_MARKER"
echo "Retained retry shard: $SHARD_DIR"
