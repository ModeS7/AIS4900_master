#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="recover_rej_exp48c"
#SBATCH --output=IDUN/output/generate/recover_rejected_exp48c_%j.out
#SBATCH --error=IDUN/output/generate/recover_rejected_exp48c_%j.err
# =============================================================================
# Recover BRAVO draws that the acceptance protocol rejected, for a figure.
#
# WHY THIS EXISTS
#   The protocol discards a draw whose conditioning lesion falls outside the
#   generated brain support and retries, writing nothing. So the paper can describe
#   the filter but cannot show it working, because no rejected volume survives
#   anywhere. This regenerates a few of them.
#
#   Reproduction is exact, not approximate: the noise seed is derived from the
#   candidate index and the attempt number (`_derive_sample_seed(..., stream='bravo',
#   attempt=...)` in generate.py), so re-running a skipped candidate reproduces the
#   same three draws the production run made and threw away.
#
# WHAT IT TOUCHES
#   Nothing existing. Each candidate writes to its own SYNTHMASK_OUTPUT_LABEL, and
#   the runner already refuses a partial range that would land in the production
#   directory. Acceptance behaviour is identical with or without the save.
#
#   sbatch IDUN/generate/recover_rejected_draws_exp48c.slurm
#   REJECT_COUNT=5 sbatch IDUN/generate/recover_rejected_draws_exp48c.slurm
# =============================================================================

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

# C handoff -- the validation-selected source, so the figure shows the filter acting
# on the generator whose data was actually used downstream.
readonly LABEL="exp1_to_exp48c_t025"
readonly LOW_RUN="exp48c_lowt_only_lpips_huber_20260425-162615"
readonly HIGH_RUN="exp1_1_1000_pixel_bravo_20260402-121556"
readonly HANDOFF_T="0.25"

readonly NUM_CANDIDATES=525
readonly REJECT_COUNT="${REJECT_COUNT:-3}"
[[ "$REJECT_COUNT" =~ ^[1-9][0-9]*$ ]] || fatal "REJECT_COUNT must be a positive integer"

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly ACCEPTED_DIR="${EVAL_ROOT}/${LABEL}"
readonly REJECTED_ROOT="${EVAL_ROOT}/rejected_draws_${LABEL}"

[[ -d "$ACCEPTED_DIR" ]] || fatal "production output not found: $ACCEPTED_DIR"

# --- which candidates were skipped -------------------------------------------------
# `generated` advances on a skip as well as on an accept (generate.py), so a directory
# name IS the candidate index and the two sets are directly comparable.
mapfile -t ACCEPTED < <(find "$ACCEPTED_DIR" -mindepth 1 -maxdepth 1 -type d \
    -printf '%f\n' | grep -E '^[0-9]{5}$' | LC_ALL=C sort)
(( ${#ACCEPTED[@]} > 0 )) || fatal "no accepted candidates found in $ACCEPTED_DIR"

declare -A IS_ACCEPTED=()
for id in "${ACCEPTED[@]}"; do IS_ACCEPTED["$id"]=1; done

SKIPPED=()
for (( i = 0; i < NUM_CANDIDATES; i++ )); do
    printf -v id '%05d' "$i"
    [[ -n "${IS_ACCEPTED[$id]:-}" ]] || SKIPPED+=("$i")
done

echo "=== ${LABEL} ==="
echo "  accepted: ${#ACCEPTED[@]}/${NUM_CANDIDATES}"
echo "  skipped:  ${#SKIPPED[@]}"
(( ${#SKIPPED[@]} > 0 )) || fatal "no skipped candidates: nothing to recover"

readonly TAKE=$(( REJECT_COUNT < ${#SKIPPED[@]} ? REJECT_COUNT : ${#SKIPPED[@]} ))
echo "  recovering the first ${TAKE}: ${SKIPPED[*]:0:TAKE}"
echo ""

# --- regenerate each one, keeping the rejected draws -------------------------------
for (( n = 0; n < TAKE; n++ )); do
    idx="${SKIPPED[n]}"
    printf -v idstr '%05d' "$idx"
    echo "=== candidate ${idstr} ($((n + 1))/${TAKE}) ==="
    SYNTHMASK_START_CANDIDATE="$idx" \
    SYNTHMASK_END_CANDIDATE="$((idx + 1))" \
    SYNTHMASK_OUTPUT_LABEL="${LABEL}_rejfig_${idstr}" \
    SYNTHMASK_SAVE_REJECTED_DIR="${REJECTED_ROOT}/${idstr}" \
        bash "$REPO_ROOT/IDUN/generate/run_eval_generator_synthmask.sh" \
            "$LABEL" "$LOW_RUN" "$HIGH_RUN" "$HANDOFF_T" \
        || echo "  WARNING: candidate ${idstr} returned non-zero, continuing"
    echo ""
done

echo "=== recovered draws ==="
find "$REJECTED_ROOT" -name containment.json -printf '%h\n' | LC_ALL=C sort | while read -r d; do
    echo "  ${d#$REJECTED_ROOT/}  $(tr -d ' \n' < "$d/containment.json")"
done

echo ""
echo "Rejected draws:  $REJECTED_ROOT"
echo "Accepted volumes for the comparison column are already in: $ACCEPTED_DIR"
echo "Pull both down, then build the figure locally."
