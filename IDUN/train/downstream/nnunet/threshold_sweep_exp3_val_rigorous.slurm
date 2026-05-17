#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_thr_r_d600"
#SBATCH --output=IDUN/output/train/downstream/nnunet/threshold_sweep/exp3_val_rigorous_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/threshold_sweep/exp3_val_rigorous_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Rigorous val-tuned threshold sweep on exp3 (publishable workflow).
#
# Step 1: re-run validation for each of the 5 folds with
#         export_validation_probabilities=true. Writes per-case softmax
#         .npz files into each fold's `validation/` subdir.
# Step 2: pool all 5 folds' val probs + GTs into one tune set.
# Step 3: re-run 5-fold ensemble inference on TEST with save_probabilities.
# Step 4: threshold_sweep --mode tune-eval: pick t* on val, apply to test.
#
# Matches Ottesen 2025's "thresholded with separate values that maximized
# the DSC on the validation dataset for each fold" — we go one step
# simpler and average all 5 folds' vals to pick a single t*.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

set -e
cd ${SLURM_SUBMIT_DIR}

echo "=== Threshold sweep (rigorous, val-tuned): exp3_baseline_v2_d600 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
EXPERIMENT_NAME="exp3_baseline_v2_d600"
DATASET_ID=600

# Locate model and labels
MODEL_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/Dataset${DATASET_ID}_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres"
GT_TR_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet/labelsTr"
GT_TS_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet/labelsTs"
EVAL_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/eval_${EXPERIMENT_NAME}"

# Step 1: re-run validation for all 5 folds with prob export
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

# Step 2: pool 5 folds' val probabilities and GTs into shared tune dirs
TUNE_PROBS_DIR="${EVAL_DIR}/val_probs_pooled"
TUNE_GT_DIR="${EVAL_DIR}/val_gt_pooled"
echo ""
echo "=== Step 2: pool fold val outputs into ${TUNE_PROBS_DIR} ==="
mkdir -p "${TUNE_PROBS_DIR}" "${TUNE_GT_DIR}"
for FOLD in 0 1 2 3 4; do
    VAL_DIR="${MODEL_DIR}/fold_${FOLD}/validation"
    if [ ! -d "${VAL_DIR}" ]; then
        echo "ERROR: ${VAL_DIR} missing — Step 1 failed for fold ${FOLD}"
        exit 1
    fi
    # nnU-Net's validation/ contains .npz (probs), .nii.gz (binary preds),
    # and other files. We symlink only the .npz files.
    for NPZ in "${VAL_DIR}"/*.npz; do
        [ -e "${NPZ}" ] || continue
        BN=$(basename "${NPZ}")
        ln -sf "${NPZ}" "${TUNE_PROBS_DIR}/${BN}"
        # The corresponding GT is in labelsTr (val cases are a subset of train)
        CID="${BN%.npz}"
        SRC_GT="${GT_TR_DIR}/${CID}.nii.gz"
        if [ -e "${SRC_GT}" ]; then
            ln -sf "${SRC_GT}" "${TUNE_GT_DIR}/${CID}.nii.gz"
        else
            echo "WARN: no GT for val case ${CID}"
        fi
    done
done

N_TUNE=$(ls "${TUNE_PROBS_DIR}"/*.npz 2>/dev/null | wc -l)
echo "Pooled ${N_TUNE} val probability files"

# Step 3: ensemble inference on test with probabilities saved
echo ""
echo "=== Step 3: 5-fold ensemble inference on TEST with --save-probabilities ==="
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

# Step 4: tune on pooled val, apply chosen threshold to test
TEST_PROBS_DIR="${EVAL_DIR}/predictions"
OUTPUT_JSON="${EVAL_DIR}/threshold_sweep_val_tuned.json"

echo ""
echo "=== Step 4: tune-eval threshold sweep ==="
echo "  tune probs: ${TUNE_PROBS_DIR}  (n=${N_TUNE})"
echo "  test probs: ${TEST_PROBS_DIR}"
echo "  output:     ${OUTPUT_JSON}"

time python -m medgen.scripts.threshold_sweep \
    --mode tune-eval \
    --tune-probs-dir "${TUNE_PROBS_DIR}" \
    --tune-gt-dir    "${TUNE_GT_DIR}" \
    --eval-probs-dir "${TEST_PROBS_DIR}" \
    --eval-gt-dir    "${GT_TS_DIR}" \
    --output         "${OUTPUT_JSON}"

echo ""
echo "=== Done. Publishable val-tuned threshold result in ${OUTPUT_JSON}. ==="
