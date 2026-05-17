#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_thr_q_d600"
#SBATCH --output=IDUN/output/train/downstream/nnunet/threshold_sweep/exp3_test_quick_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/threshold_sweep/exp3_test_quick_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Quick threshold sweep on exp3 TEST set (upper-bound estimate; biased).
#
# Step 1: re-run 5-fold ensemble inference WITH save_probabilities=true so
#         we get .npz softmax outputs alongside the binary predictions.
# Step 2: sweep thresholds 0.10..0.80 on the saved probs, report optimal.
#
# This is the UPPER-BOUND number: selecting the optimum on test labels is
# biased and not publishable. Use the val-tuned slurm for the thesis number.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== Threshold sweep (quick, test-set upper bound): exp3_baseline_v2_d600 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
EXPERIMENT_NAME="exp3_baseline_v2_d600"
EVAL_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/eval_${EXPERIMENT_NAME}"
PROBS_DIR="${EVAL_DIR}/predictions"

# Step 1: inference with probabilities saved
echo ""
echo "=== Step 1: re-run 5-fold ensemble inference (with --save-probabilities) ==="
time python -m medgen.scripts.eval_nnunet \
    --experiment baseline \
    --dataset-id 600 \
    --experiment-name "${EXPERIMENT_NAME}" \
    --folds 0 1 2 3 4 \
    --nnunet-base "${NNUNET_BASE}" \
    --nnunet-results "${NNUNET_RESULTS}" \
    --trainer nnUNetTrainerBrainMets \
    --plans nnUNetResEncUNetLPlans \
    --save-probabilities

# Step 2: threshold sweep on the saved probs
GT_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset600_BrainMet/labelsTs"
OUTPUT_JSON="${EVAL_DIR}/threshold_sweep_test.json"

echo ""
echo "=== Step 2: threshold sweep over saved softmax probabilities ==="
echo "  probs dir: ${PROBS_DIR}"
echo "  gt    dir: ${GT_DIR}"
echo "  output:    ${OUTPUT_JSON}"

time python -m medgen.scripts.threshold_sweep \
    --mode sweep \
    --probs-dir "${PROBS_DIR}" \
    --gt-dir    "${GT_DIR}" \
    --output    "${OUTPUT_JSON}"

echo ""
echo "=== Done. Inspect ${OUTPUT_JSON} for the per-threshold Dice curve. ==="
