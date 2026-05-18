#!/bin/bash
# Run the 51 test cases through each of the 5 fold models LOCALLY.
# Produces per-fold softmax .npz files at:
#   runs/exp3_baseline_v2_d600/per_fold_test/fold_N/predictions/*.npz
#
# Prereqs:
#   - .venv_nnunet with pytorch 2.4 + nnunetv2 installed
#   - data/nnunet_local/nnUNet_raw/Dataset600_BrainMet/{imagesTs,labelsTs}
#   - runs/exp3_baseline_v2_d600/Dataset600_BrainMet/.../fold_{0..4}/checkpoint_best.pth
#
# Usage:  ./misc/run_per_fold_local.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# Activate the nnunet venv (separate from the main project venv).
source .venv_nnunet/bin/activate
# Make medgen importable inside the new venv (sibling .venv has the package
# installed via pip -e; here we point PYTHONPATH at src/ directly).
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

NNUNET_BASE="${REPO_ROOT}/data/nnunet_local"
NNUNET_RESULTS="${REPO_ROOT}/runs"
EXPERIMENT_NAME="exp3_baseline_v2_d600"
DATASET_ID=600
EVAL_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/eval_${EXPERIMENT_NAME}"
PER_FOLD_DIR="${EVAL_DIR}/per_fold_test"
mkdir -p "${PER_FOLD_DIR}"

echo "=== Per-fold local inference: ${EXPERIMENT_NAME} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

for FOLD in 0 1 2 3 4; do
    OUT_DIR="${PER_FOLD_DIR}/fold_${FOLD}"
    if [ -d "${OUT_DIR}/predictions" ] && [ "$(ls "${OUT_DIR}/predictions"/*.npz 2>/dev/null | wc -l)" -eq 51 ]; then
        echo "=== Fold ${FOLD}: already complete (51 .npz), skipping ==="
        continue
    fi
    echo "=== Fold ${FOLD} → ${OUT_DIR} ==="
    time python -m medgen.scripts.eval_nnunet \
        --experiment baseline \
        --dataset-id "${DATASET_ID}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        --folds "${FOLD}" \
        --nnunet-base "${NNUNET_BASE}" \
        --nnunet-results "${NNUNET_RESULTS}" \
        --trainer nnUNetTrainerBrainMets \
        --plans nnUNetResEncUNetLPlans \
        --save-probabilities \
        --output-dir "${OUT_DIR}"
    echo ""
done

echo "=== All folds complete ==="
for FOLD in 0 1 2 3 4; do
    OUT="${PER_FOLD_DIR}/fold_${FOLD}/predictions"
    N=$(ls "${OUT}"/*.npz 2>/dev/null | wc -l)
    SZ=$(du -sh "${PER_FOLD_DIR}/fold_${FOLD}" 2>/dev/null | cut -f1)
    echo "  fold_${FOLD}: ${N} .npz (${SZ})"
done
