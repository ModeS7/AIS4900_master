#!/usr/bin/env bash
# Evaluate one BRAVO generator on 525 shared synthetic-mask candidates.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

if [[ $# -ne 2 && $# -ne 4 ]]; then
    fatal "usage: $0 LABEL LOW_RUN [HIGH_RUN HANDOFF_T]"
fi

readonly LABEL="$1"
readonly LOW_RUN="$2"
readonly HIGH_RUN="${3:-}"
readonly HANDOFF_T="${4:-}"
[[ "$LABEL" =~ ^[a-zA-Z0-9_.-]+$ ]] || fatal "unsafe label: $LABEL"
[[ "$LOW_RUN" =~ ^[a-zA-Z0-9_.-]+$ ]] || fatal "unsafe low-t run name: $LOW_RUN"
if [[ -n "$HIGH_RUN" ]]; then
    [[ "$HIGH_RUN" =~ ^[a-zA-Z0-9_.-]+$ ]] || fatal "unsafe high-t run name: $HIGH_RUN"
    [[ "$HANDOFF_T" == "0.25" ]] || fatal "handoff must be 0.25"
else
    [[ -z "$HANDOFF_T" ]] || fatal "handoff_t requires a high-t checkpoint"
fi

readonly NUM_CANDIDATES=525
readonly SEED=42
readonly STEPS=100
readonly MAX_IMAGE_ATTEMPTS=3
# Maximum observed training-pair distance was 2.322613 mm. Round upward to 0.001 mm.
readonly BRAIN_MARGIN_MM=2.323
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly BRAVO_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"
readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly CANDIDATE_MASKS="${EVAL_ROOT}/ordered_common_seg_masks_seed42/candidates525"
readonly SELECTION_FILE="${EVAL_ROOT}/ordered_common_seg_masks_seed42/selection.json"
readonly LOW_CKPT="${BRAVO_ROOT}/${LOW_RUN}/checkpoint_latest.pt"
readonly BRAIN_SUPPORT_PCA="${REPO_ROOT}/data/brain_support_pca_train105_var95_256x256x160.npz"
readonly FINAL_DIR="${EVAL_ROOT}/${LABEL}"
HIGH_CKPT=""
if [[ -n "$HIGH_RUN" ]]; then
    HIGH_CKPT="${BRAVO_ROOT}/${HIGH_RUN}/checkpoint_latest.pt"
fi
readonly HIGH_CKPT

[[ -f "$REPO_ROOT/pyproject.toml" ]] || fatal "repository root not found: $REPO_ROOT"
[[ -d "$CANDIDATE_MASKS" ]] || fatal "prepared $NUM_CANDIDATES-mask view not found: $CANDIDATE_MASKS"
[[ -s "$SELECTION_FILE" ]] || fatal "mask selection file not found: $SELECTION_FILE"
[[ -f "$LOW_CKPT" ]] || fatal "low-t checkpoint not found: $LOW_CKPT"
[[ -s "$BRAIN_SUPPORT_PCA" ]] || fatal "brain-support PCA not found: $BRAIN_SUPPORT_PCA"
if [[ -n "$HIGH_CKPT" ]]; then
    [[ -f "$HIGH_CKPT" ]] || fatal "high-t checkpoint not found: $HIGH_CKPT"
fi
[[ ! -e "$FINAL_DIR" ]] || fatal "final output already exists: $FINAL_DIR"

mapfile -t MASK_IDS < <(find "$CANDIDATE_MASKS" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' | LC_ALL=C sort)
[[ ${#MASK_IDS[@]} -eq $NUM_CANDIDATES ]] || fatal \
    "expected $NUM_CANDIDATES prepared masks, found ${#MASK_IDS[@]}"
for ((index = 0; index < NUM_CANDIDATES; index++)); do
    printf -v expected_id '%05d' "$index"
    [[ "${MASK_IDS[index]}" == "$expected_id" ]] || fatal "unexpected mask ID ${MASK_IDS[index]}"
    [[ -s "$CANDIDATE_MASKS/$expected_id/seg.nii.gz" ]] || fatal "missing mask $expected_id"
done

echo "Preflight OK [${LABEL}]: $NUM_CANDIDATES shared synthetic-mask candidates and checkpoint_latest.pt"
cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"

GEN_ARGS=(
    paths=cluster
    spatial_dims=3
    gen_mode=bravo
    image_size=256
    depth=160
    trim_slices=10
    fov_mm=240.0
    image_model="$LOW_CKPT"
    image_model_high_t=null
    real_seg_dir="$CANDIDATE_MASKS"
    expected_real_cases="$NUM_CANDIDATES"
    expected_real_depth=150
    seed="$SEED"
    num_images="$NUM_CANDIDATES"
    current_image=0
    num_steps_bravo="$STEPS"
    ode_solver=euler
    shift_ratio_bravo=1.0
    cfg_scale_bravo=1.0
    validate_size_bins=false
    validate_real_seg_masks=true
    require_real_bravo_pairs=false
    validate_brain_mask=true
    conditioning_brain_qc_mode=reject
    brain_threshold=0.05
    brain_containment_margin_mm="$BRAIN_MARGIN_MM"
    max_image_attempts_per_mask="$MAX_IMAGE_ATTEMPTS"
    skip_failed_fixed_masks=true
    brain_atlas_path=null
    brain_support_pca_path="$BRAIN_SUPPORT_PCA"
    brain_pca_path=null
    seg_pca_path=null
    diffrs_checkpoint=null
    mask_outside_brain=false
    paths.generated_dir="$EVAL_ROOT"
    output_subdir="$LABEL"
    verbose=true
)
if [[ -n "$HIGH_CKPT" ]]; then
    GEN_ARGS+=(image_model_high_t="$HIGH_CKPT" handoff_t="$HANDOFF_T")
fi

echo "=== Evaluating ${LABEL}: $NUM_CANDIDATES mask candidates, up to ${MAX_IMAGE_ATTEMPTS} image attempts each ==="
echo "masks:            $CANDIDATE_MASKS"
echo "brain support:    $BRAIN_SUPPORT_PCA (train105 PCA, 0.5 threshold, ${BRAIN_MARGIN_MM} mm margin)"
echo "low-t checkpoint: $LOW_CKPT"
if [[ -n "$HIGH_CKPT" ]]; then
    echo "high-t checkpoint: $HIGH_CKPT (t > $HANDOFF_T)"
fi
time python -m medgen.scripts.generate "${GEN_ARGS[@]}"

[[ -d "$FINAL_DIR" ]] || fatal "generator did not create output: $FINAL_DIR"
[[ -s "$FINAL_DIR/bins.csv" ]] || fatal "generator did not create bins.csv"
SEG_COUNT="$(find "$FINAL_DIR" -mindepth 2 -maxdepth 2 -type f -name seg.nii.gz | wc -l)"
BRAVO_COUNT="$(find "$FINAL_DIR" -mindepth 2 -maxdepth 2 -type f -name bravo.nii.gz | wc -l)"
BINS_COUNT="$(( $(wc -l < "$FINAL_DIR/bins.csv") - 1 ))"
[[ "$SEG_COUNT" -eq "$BRAVO_COUNT" ]] || fatal \
    "generated dataset contains $SEG_COUNT masks but $BRAVO_COUNT BRAVO volumes"
[[ "$BINS_COUNT" -eq "$BRAVO_COUNT" ]] || fatal \
    "bins.csv contains $BINS_COUNT samples but the dataset contains $BRAVO_COUNT BRAVO volumes"
[[ "$BRAVO_COUNT" -le "$NUM_CANDIDATES" ]] || fatal \
    "generated dataset contains $BRAVO_COUNT BRAVO volumes for only $NUM_CANDIDATES candidates"
echo "Completed candidate panel: accepted $BRAVO_COUNT/$NUM_CANDIDATES, skipped $((NUM_CANDIDATES - BRAVO_COUNT))"
echo "Output: $FINAL_DIR"
