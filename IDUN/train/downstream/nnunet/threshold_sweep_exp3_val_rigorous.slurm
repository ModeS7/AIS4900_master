#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_gen_d600"
#SBATCH --output=IDUN/output/train/downstream/nnunet/threshold_sweep/exp3_gen_probs_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/threshold_sweep/exp3_gen_probs_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Generate ALL softmax probabilities for exp3 (5-fold val + ensemble test).
# Cluster job is pure data generation; download the outputs and analyse locally
# with `medgen.scripts.threshold_sweep` (sweep curves, tune-eval, custom
# strategies, anything you want without re-queueing GPU time).
#
# Pipeline:
#   Step 1: Re-validate each of 5 folds with --export-validation-probabilities.
#           Writes per-case softmax .npz into each fold's `validation/` subdir.
#           ~5-10 min per fold (skips training, only_run_validation).
#   Step 2: Run 5-fold ensemble inference on TEST with --save-probabilities.
#           Writes per-case ensemble softmax .npz into eval/predictions/.
#
# After this job finishes the cluster holds:
#   - 5 × (per-fold val) softmax dirs (~21 cases each, ~50 MB per fold)
#   - 1 × (ensemble test) softmax dir (51 cases, ~250 MB)
# Total to download: ~500 MB.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

set -e
cd ${SLURM_SUBMIT_DIR}

echo "=== Generate probabilities for exp3_baseline_v2_d600 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
EXPERIMENT_NAME="exp3_baseline_v2_d600"
DATASET_ID=600

MODEL_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/Dataset${DATASET_ID}_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres"
EVAL_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/eval_${EXPERIMENT_NAME}"

# ── Step 1: per-fold validation with prob export ───────────────────────────
echo ""
echo "=== Step 1: re-validate folds 0..4 with --export-validation-probabilities ==="
for FOLD in 0 1 2 3 4; do
    echo ""
    echo "--- Fold ${FOLD} ---"
    time python -m medgen.scripts.train_nnunet \
        --experiment baseline \
        --dataset-id ${DATASET_ID} \
        --experiment-name "${EXPERIMENT_NAME}" \
        --fold ${FOLD} \
        --nnunet-base "${NNUNET_BASE}" \
        --nnunet-results "${NNUNET_RESULTS}" \
        --trainer nnUNetTrainerBrainMets \
        --plans nnUNetResEncUNetLPlans \
        --validation-only \
        --export-validation-probabilities
done

# ── Step 2: ensemble inference on TEST with probs ──────────────────────────
echo ""
echo "=== Step 2: 5-fold ensemble inference on TEST with --save-probabilities ==="
time python -m medgen.scripts.eval_nnunet \
    --experiment baseline \
    --dataset-id ${DATASET_ID} \
    --experiment-name "${EXPERIMENT_NAME}" \
    --folds 0 1 2 3 4 \
    --nnunet-base "${NNUNET_BASE}" \
    --nnunet-results "${NNUNET_RESULTS}" \
    --trainer nnUNetTrainerBrainMets \
    --plans nnUNetResEncUNetLPlans \
    --save-probabilities

# ── Summary & download instructions ────────────────────────────────────────
echo ""
echo "=== Outputs (for download) ==="
echo ""
echo "Val probs (per fold):"
for FOLD in 0 1 2 3 4; do
    VAL_DIR="${MODEL_DIR}/fold_${FOLD}/validation"
    N=$(ls "${VAL_DIR}"/*.npz 2>/dev/null | wc -l)
    SZ=$(du -sh "${VAL_DIR}" 2>/dev/null | cut -f1)
    echo "  fold_${FOLD}: ${VAL_DIR}  (${N} .npz, ${SZ})"
done
TEST_PROBS_DIR="${EVAL_DIR}/predictions"
N_TEST=$(ls "${TEST_PROBS_DIR}"/*.npz 2>/dev/null | wc -l)
SZ_TEST=$(du -sh "${TEST_PROBS_DIR}" 2>/dev/null | cut -f1)
echo ""
echo "Test probs (ensemble):"
echo "  ${TEST_PROBS_DIR}  (${N_TEST} .npz, ${SZ_TEST})"
echo ""
echo "=== Suggested rsync (run on your local machine) ==="
echo ""
echo "# All val probs (folds 0..4)"
echo "rsync -av --include='*/' --include='*.npz' --exclude='*' \\"
echo "    idun:${MODEL_DIR}/ ~/NTNU/AIS4900_master/runs/exp3_baseline_v2_d600/val_probs/"
echo ""
echo "# Ensemble test probs"
echo "rsync -av --include='*.npz' --exclude='*' \\"
echo "    idun:${TEST_PROBS_DIR}/ ~/NTNU/AIS4900_master/runs/exp3_baseline_v2_d600/test_probs/"
echo ""
echo "# Test GT (if not already local)"
echo "rsync -av idun:${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet/labelsTs/ \\"
echo "    ~/NTNU/AIS4900_master/runs/exp3_baseline_v2_d600/labelsTs/"
echo ""
echo "# Train GT (for val tuning)"
echo "rsync -av idun:${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet/labelsTr/ \\"
echo "    ~/NTNU/AIS4900_master/runs/exp3_baseline_v2_d600/labelsTr/"
