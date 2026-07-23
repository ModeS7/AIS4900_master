#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_whh_to210"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/gen_whh_to210_%j.out
#SBATCH --error=IDUN/output/generate/gen_whh_to210_%j.err

# Extend only the validation-locked Weighted Huber handoff source.
#
# The canonical source contains 151 accepted pairs from candidates 00000--00524.
# This job creates a separate continuation tranche of 300 masks, exposes them as
# candidates 00525--00824, and evaluates fixed masks in order until exactly 59
# additional BRAVOs pass the unchanged three-attempt PCA-containment gate. It
# then creates an immutable 210-pair training view without modifying the frozen
# canonical source, common-96 view, or common-105 view.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly TARGET_PAIRS=210
readonly EXPECTED_ORIGINAL_PAIRS=151
readonly NEEDED_NEW_PAIRS=$((TARGET_PAIRS - EXPECTED_ORIGINAL_PAIRS))
readonly ORIGINAL_CANDIDATES=525
readonly NEW_MASK_CANDIDATES=300
readonly GLOBAL_START=$ORIGINAL_CANDIDATES
readonly GLOBAL_STOP=$((GLOBAL_START + NEW_MASK_CANDIDATES))
readonly BASE_SEED=42
readonly NEW_MASK_SEED=$((BASE_SEED + GLOBAL_START))
readonly STEPS=100
readonly MAX_MASK_ATTEMPTS=50
readonly MAX_IMAGE_ATTEMPTS=3
readonly BRAIN_MARGIN_MM=2.323
readonly BINS_HEADER='id,bin_0,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,total_tumors'
readonly EVAL_ID='source_selection_synthmask525_seed42_euler100_pca2p323mm_try3'
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly BRAVO_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"
readonly MASK_PANEL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/source_selection_train105_seed42_euler100"
readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly ORIGINAL_MASKS="${EVAL_ROOT}/ordered_common_seg_masks_seed42/candidates525"
readonly COMMON105_IDS="${EVAL_ROOT}/matched_common105/common_candidate_ids.txt"
readonly ORIGINAL_PAIRS="${EVAL_ROOT}/exp1_to_exp48c_t025"
readonly ORIGINAL_MASK_MARKER="${ORIGINAL_PAIRS}/.brain_mask_stage3_applied"

readonly NEW_MASK_DIR="${MASK_PANEL_ROOT}/common_seg_masks_extension_00525_00824"
readonly NEW_MASK_MARKER="${NEW_MASK_DIR}/.extension_complete"
readonly EXTENSION_ROOT="${EVAL_ROOT}/weighted_huber_handoff_extension_to210"
readonly EXPANDED_MASKS="${EXTENSION_ROOT}/candidates825"
readonly EXPANDED_MASKS_MARKER="${EXPANDED_MASKS}/.extension_view_complete"
readonly PAIR_LABEL='exp1_to_exp48c_t025__extension_00525_00824'
readonly NEW_PAIRS="${EXTENSION_ROOT}/${PAIR_LABEL}"
readonly NEW_PAIRS_MARKER="${NEW_PAIRS}/.extension_complete"
readonly HYBRID_VIEW="${EXTENSION_ROOT}/hybrid_pairs_210"
readonly HYBRID_IDS="${EXTENSION_ROOT}/hybrid_candidate_ids_210.txt"
readonly HYBRID_MARKER="${HYBRID_VIEW}/.hybrid_view_complete"

readonly SEG_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
readonly LOW_CKPT="${BRAVO_ROOT}/exp48c_lowt_only_lpips_huber_20260425-162615/checkpoint_latest.pt"
readonly HIGH_CKPT="${BRAVO_ROOT}/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
readonly ATLAS_PATH="${REPO_ROOT}/data/brain_atlas_256x256x160.nii.gz"
readonly BRAIN_SUPPORT_PCA="${REPO_ROOT}/data/brain_support_pca_train105_var95_256x256x160.npz"

[[ -f "$REPO_ROOT/pyproject.toml" ]] || fatal "repository root not found: $REPO_ROOT"
[[ -d "$ORIGINAL_MASKS" ]] || fatal "frozen 525-mask view not found: $ORIGINAL_MASKS"
[[ -s "$COMMON105_IDS" ]] || fatal "common-105 ID list not found: $COMMON105_IDS"
[[ -d "$ORIGINAL_PAIRS" ]] || fatal "canonical winner not found: $ORIGINAL_PAIRS"
[[ -s "$ORIGINAL_MASK_MARKER" ]] || fatal "canonical winner is not brain-masked"
[[ -f "$SEG_CKPT" ]] || fatal "segmentation checkpoint not found: $SEG_CKPT"
[[ -f "$LOW_CKPT" ]] || fatal "low-t checkpoint not found: $LOW_CKPT"
[[ -f "$HIGH_CKPT" ]] || fatal "high-t checkpoint not found: $HIGH_CKPT"
[[ -f "$ATLAS_PATH" ]] || fatal "brain atlas not found: $ATLAS_PATH"
[[ -s "$BRAIN_SUPPORT_PCA" ]] || fatal "brain-support PCA not found: $BRAIN_SUPPORT_PCA"

mapfile -t ORIGINAL_IDS < <(
    find -P "$ORIGINAL_PAIRS" -mindepth 1 -maxdepth 1 -type d \
        -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort
)
[[ ${#ORIGINAL_IDS[@]} -eq $EXPECTED_ORIGINAL_PAIRS ]] || fatal \
    "canonical winner has ${#ORIGINAL_IDS[@]} pairs; expected $EXPECTED_ORIGINAL_PAIRS"
[[ "$(wc -l < "$COMMON105_IDS")" -eq 105 ]] || fatal \
    "common-105 ID list does not contain 105 rows"
for candidate_id in "${ORIGINAL_IDS[@]}"; do
    [[ -s "$ORIGINAL_PAIRS/$candidate_id/seg.nii.gz" ]] || fatal \
        "canonical winner is missing mask $candidate_id"
    [[ -s "$ORIGINAL_PAIRS/$candidate_id/bravo.nii.gz" ]] || fatal \
        "canonical winner is missing BRAVO $candidate_id"
done
while IFS= read -r candidate_id; do
    [[ -d "$ORIGINAL_PAIRS/$candidate_id" ]] || fatal \
        "common-105 candidate is absent from canonical winner: $candidate_id"
done < "$COMMON105_IDS"

mkdir -p "$EXTENSION_ROOT"
exec 9>"${EXTENSION_ROOT}/.generation.lock"
flock -n 9 || fatal "another winner-extension job is already running"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Weighted Huber handoff extension to exactly 210 pairs ==="
echo "canonical accepted: $EXPECTED_ORIGINAL_PAIRS"
echo "new pairs required: $NEEDED_NEW_PAIRS"
echo "new mask tranche:   ${GLOBAL_START}..$((GLOBAL_STOP - 1))"
echo "handoff:            high-t exp1 -> low-t exp48c at t=0.25"

# Stage 1: generate deterministic mask draws that continue the original seed
# stream. Local mask j uses seed 567+j, exactly matching global candidate
# 525+j under the original base seed 42.
if [[ -s "$NEW_MASK_MARKER" ]]; then
    echo "Reusing completed continuation mask tranche: $NEW_MASK_DIR"
else
    [[ ! -e "$NEW_MASK_DIR" ]] || fatal \
        "incomplete continuation mask directory already exists: $NEW_MASK_DIR"

    time python -m medgen.scripts.generate \
        paths=cluster \
        spatial_dims=3 \
        gen_mode=seg_conditioned \
        image_size=256 \
        depth=160 \
        trim_slices=10 \
        fov_mm=240.0 \
        seg_model="$SEG_CKPT" \
        image_model=null \
        seed="$NEW_MASK_SEED" \
        batch_size=1 \
        num_images="$NEW_MASK_CANDIDATES" \
        current_image=0 \
        num_steps="$STEPS" \
        num_steps_seg="$STEPS" \
        ode_solver=euler \
        shift_ratio=1.0 \
        shift_ratio_seg=1.0 \
        cfg_scale_seg=1.0 \
        cfg_scale_seg_end=null \
        cfg_mode=standard \
        validate_size_bins=false \
        seg_component_connectivity=26 \
        max_retries="$MAX_MASK_ATTEMPTS" \
        max_white_percentage=null \
        brain_atlas_path="$ATLAS_PATH" \
        brain_tolerance=0.0 \
        brain_dilate_pixels=0 \
        brain_pca_path=null \
        seg_pca_path=null \
        validate_brain_mask=false \
        mask_outside_brain=false \
        diffrs_checkpoint=null \
        paths.generated_dir="$MASK_PANEL_ROOT" \
        output_subdir="$(basename "$NEW_MASK_DIR")" \
        verbose=true

    mask_count="$(find -P "$NEW_MASK_DIR" -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz | wc -l)"
    [[ "$mask_count" -eq $NEW_MASK_CANDIDATES ]] || fatal \
        "continuation tranche contains $mask_count masks; expected $NEW_MASK_CANDIDATES"
    [[ -s "$NEW_MASK_DIR/bins.csv" ]] || fatal "continuation mask bins.csv is missing"
    [[ "$(( $(wc -l < "$NEW_MASK_DIR/bins.csv") - 1 ))" -eq $NEW_MASK_CANDIDATES ]] || fatal \
        "continuation mask bins.csv has the wrong row count"

    {
        echo "global_start=$GLOBAL_START"
        echo "global_stop=$GLOBAL_STOP"
        echo "local_seed=$NEW_MASK_SEED"
        echo "base_seed=$BASE_SEED"
        echo "num_masks=$NEW_MASK_CANDIDATES"
        echo "seg_checkpoint_sha256=$(sha256sum "$SEG_CKPT" | awk '{print $1}')"
        echo "job_id=$SLURM_JOB_ID"
    } > "${NEW_MASK_MARKER}.tmp"
    mv -- "${NEW_MASK_MARKER}.tmp" "$NEW_MASK_MARKER"
fi

# Stage 2: expose one immutable 825-mask view. Existing candidate mappings are
# preserved exactly and the continuation tranche is appended without reshuffling.
if [[ ! -s "$EXPANDED_MASKS_MARKER" ]]; then
    [[ ! -e "$EXPANDED_MASKS" ]] || fatal \
        "incomplete expanded mask view already exists: $EXPANDED_MASKS"
    expanded_tmp="${EXPANDED_MASKS}.tmp_${SLURM_JOB_ID}"
    [[ ! -e "$expanded_tmp" ]] || fatal "temporary expanded view already exists: $expanded_tmp"
    mkdir -p "$expanded_tmp"

    for ((index = 0; index < ORIGINAL_CANDIDATES; index++)); do
        printf -v candidate_id '%05d' "$index"
        mkdir -p "$expanded_tmp/$candidate_id"
        ln -s -- "$ORIGINAL_MASKS/$candidate_id/seg.nii.gz" \
            "$expanded_tmp/$candidate_id/seg.nii.gz"
    done
    for ((local_index = 0; local_index < NEW_MASK_CANDIDATES; local_index++)); do
        global_index=$((GLOBAL_START + local_index))
        printf -v local_id '%05d' "$local_index"
        printf -v candidate_id '%05d' "$global_index"
        mkdir -p "$expanded_tmp/$candidate_id"
        ln -s -- "$NEW_MASK_DIR/$local_id/seg.nii.gz" \
            "$expanded_tmp/$candidate_id/seg.nii.gz"
    done

    {
        echo "original_candidates=$ORIGINAL_CANDIDATES"
        echo "extension_start=$GLOBAL_START"
        echo "extension_stop=$GLOBAL_STOP"
        echo "total_candidates=$GLOBAL_STOP"
        echo "job_id=$SLURM_JOB_ID"
    } > "$expanded_tmp/.extension_view_complete"
    mv -- "$expanded_tmp" "$EXPANDED_MASKS"
fi

for ((index = 0; index < GLOBAL_STOP; index++)); do
    printf -v candidate_id '%05d' "$index"
    [[ -s "$EXPANDED_MASKS/$candidate_id/seg.nii.gz" ]] || fatal \
        "expanded view is missing candidate $candidate_id"
done

# Stage 3: evaluate fixed masks in global candidate order and stop immediately
# after 59 new images have passed the unchanged QC gate.
if [[ -s "$NEW_PAIRS_MARKER" ]]; then
    echo "Reusing completed BRAVO extension: $NEW_PAIRS"
else
    [[ ! -e "$NEW_PAIRS" ]] || fatal \
        "incomplete BRAVO extension already exists: $NEW_PAIRS"

    time python -m medgen.scripts.generate \
        paths=cluster \
        spatial_dims=3 \
        gen_mode=bravo \
        image_size=256 \
        depth=160 \
        trim_slices=10 \
        fov_mm=240.0 \
        seg_model=null \
        image_model="$LOW_CKPT" \
        image_model_high_t="$HIGH_CKPT" \
        handoff_t=0.25 \
        real_seg_dir="$EXPANDED_MASKS" \
        expected_real_cases="$GLOBAL_STOP" \
        expected_real_depth=150 \
        seed="$BASE_SEED" \
        num_images="$GLOBAL_STOP" \
        current_image="$GLOBAL_START" \
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
        max_image_attempts_per_mask="$MAX_IMAGE_ATTEMPTS" \
        skip_failed_fixed_masks=true \
        stop_after_accepted_images="$NEEDED_NEW_PAIRS" \
        brain_atlas_path=null \
        brain_support_pca_path="$BRAIN_SUPPORT_PCA" \
        brain_pca_path=null \
        seg_pca_path=null \
        diffrs_checkpoint=null \
        mask_outside_brain=false \
        paths.generated_dir="$EXTENSION_ROOT" \
        output_subdir="$PAIR_LABEL" \
        verbose=true

    [[ -s "$NEW_PAIRS/bins.csv" ]] || fatal "BRAVO extension bins.csv is missing"
    [[ "$(head -n 1 "$NEW_PAIRS/bins.csv")" == "$BINS_HEADER" ]] || fatal \
        "BRAVO extension bins.csv has an unexpected header"
    seg_count="$(find -P "$NEW_PAIRS" -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz | wc -l)"
    bravo_count="$(find -P "$NEW_PAIRS" -mindepth 2 -maxdepth 2 -type f -name bravo.nii.gz | wc -l)"
    bins_count="$(( $(wc -l < "$NEW_PAIRS/bins.csv") - 1 ))"
    [[ "$seg_count" -eq $NEEDED_NEW_PAIRS ]] || fatal \
        "accepted $seg_count new masks; required $NEEDED_NEW_PAIRS"
    [[ "$bravo_count" -eq $NEEDED_NEW_PAIRS ]] || fatal \
        "accepted $bravo_count new BRAVOs; required $NEEDED_NEW_PAIRS"
    [[ "$bins_count" -eq $NEEDED_NEW_PAIRS ]] || fatal \
        "BRAVO extension bins.csv has $bins_count rows; required $NEEDED_NEW_PAIRS"

    python -m medgen.scripts.brain_mask_existing \
        --src "$NEW_PAIRS" \
        --in-place \
        --threshold 0.05 \
        --dilate-pixels 2 \
        --expected-shape 256 256 150

    {
        echo "source=weighted_huber_handoff"
        echo "candidate_start=$GLOBAL_START"
        echo "candidate_stop=$GLOBAL_STOP"
        echo "accepted_pairs=$NEEDED_NEW_PAIRS"
        echo "max_image_attempts=$MAX_IMAGE_ATTEMPTS"
        echo "brain_margin_mm=$BRAIN_MARGIN_MM"
        echo "brain_mask_threshold=0.05"
        echo "brain_mask_dilate_pixels=2"
        echo "low_checkpoint_sha256=$(sha256sum "$LOW_CKPT" | awk '{print $1}')"
        echo "high_checkpoint_sha256=$(sha256sum "$HIGH_CKPT" | awk '{print $1}')"
        echo "job_id=$SLURM_JOB_ID"
    } > "${NEW_PAIRS_MARKER}.tmp"
    mv -- "${NEW_PAIRS_MARKER}.tmp" "$NEW_PAIRS_MARKER"
fi

mapfile -t NEW_IDS < <(
    find -P "$NEW_PAIRS" -mindepth 1 -maxdepth 1 -type d \
        -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort
)
[[ ${#NEW_IDS[@]} -eq $NEEDED_NEW_PAIRS ]] || fatal \
    "BRAVO extension has ${#NEW_IDS[@]} pairs; expected $NEEDED_NEW_PAIRS"
for candidate_id in "${NEW_IDS[@]}"; do
    candidate_number=$((10#$candidate_id))
    (( candidate_number >= GLOBAL_START && candidate_number < GLOBAL_STOP )) || fatal \
        "extension candidate is outside the continuation range: $candidate_id"
    [[ -s "$NEW_PAIRS/$candidate_id/seg.nii.gz" ]] || fatal \
        "extension is missing mask $candidate_id"
    [[ -s "$NEW_PAIRS/$candidate_id/bravo.nii.gz" ]] || fatal \
        "extension is missing BRAVO $candidate_id"
done

# Stage 4: freeze one exact nested order. The first 105 entries are unchanged
# from the matched common-105 experiment, followed by the other 46 canonical
# winner cases, then the 59 continuation cases.
ids_tmp="${HYBRID_IDS}.tmp_${SLURM_JOB_ID}"
: > "$ids_tmp"
cat "$COMMON105_IDS" >> "$ids_tmp"
for candidate_id in "${ORIGINAL_IDS[@]}"; do
    if ! grep -Fxq "$candidate_id" "$COMMON105_IDS"; then
        echo "$candidate_id" >> "$ids_tmp"
    fi
done
printf '%s\n' "${NEW_IDS[@]}" >> "$ids_tmp"
[[ "$(wc -l < "$ids_tmp")" -eq $TARGET_PAIRS ]] || fatal \
    "hybrid ID manifest does not contain $TARGET_PAIRS rows"
[[ "$(LC_ALL=C sort -u "$ids_tmp" | wc -l)" -eq $TARGET_PAIRS ]] || fatal \
    "hybrid ID manifest contains duplicate IDs"
mv -- "$ids_tmp" "$HYBRID_IDS"

if [[ ! -s "$HYBRID_MARKER" ]]; then
    [[ ! -e "$HYBRID_VIEW" ]] || fatal \
        "incomplete hybrid view already exists: $HYBRID_VIEW"
    hybrid_tmp="${HYBRID_VIEW}.tmp_${SLURM_JOB_ID}"
    mkdir -p "$hybrid_tmp"
    {
        echo "$BINS_HEADER"
        while IFS= read -r candidate_id; do
            candidate_number=$((10#$candidate_id))
            if (( candidate_number < GLOBAL_START )); then
                source_dir="$ORIGINAL_PAIRS/$candidate_id"
            else
                source_dir="$NEW_PAIRS/$candidate_id"
            fi
            mkdir -p "$hybrid_tmp/$candidate_id"
            ln -s -- "$source_dir/seg.nii.gz" "$hybrid_tmp/$candidate_id/seg.nii.gz"
            ln -s -- "$source_dir/bravo.nii.gz" "$hybrid_tmp/$candidate_id/bravo.nii.gz"
            echo "$candidate_id,0,0,0,0,0,0,0,0"
        done < "$HYBRID_IDS"
    } > "$hybrid_tmp/bins.csv"
    {
        echo "target_pairs=$TARGET_PAIRS"
        echo "original_pairs=$EXPECTED_ORIGINAL_PAIRS"
        echo "extension_pairs=$NEEDED_NEW_PAIRS"
        echo "ordered_ids=$HYBRID_IDS"
        echo "job_id=$SLURM_JOB_ID"
    } > "$hybrid_tmp/.hybrid_view_complete"
    mv -- "$hybrid_tmp" "$HYBRID_VIEW"
fi

hybrid_count="$(find -P "$HYBRID_VIEW" -mindepth 1 -maxdepth 1 -type d \
    -name '[0-9][0-9][0-9][0-9][0-9]' | wc -l)"
[[ "$hybrid_count" -eq $TARGET_PAIRS ]] || fatal \
    "final hybrid view has $hybrid_count pairs; expected $TARGET_PAIRS"

echo ""
echo "Completed Weighted Huber handoff extension."
echo "New accepted pairs: $NEW_PAIRS"
echo "Frozen 210-pair view: $HYBRID_VIEW"
echo "Nested ID order: $HYBRID_IDS"
