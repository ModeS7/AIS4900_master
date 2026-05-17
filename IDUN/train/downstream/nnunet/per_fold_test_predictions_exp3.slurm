#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_pf_d600"
#SBATCH --output=IDUN/output/train/downstream/nnunet/threshold_sweep/exp3_per_fold_test_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/threshold_sweep/exp3_per_fold_test_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Run the 51 test cases through EACH of the 5 fold models individually.
# Produces 5 × (51 test predictions + 51 softmax .npz files), so we can
# pull everything down and play with ensembling / threshold strategies
# entirely locally.
#
# Output layout (per fold, isolated dirs so they don't overwrite):
#   <EVAL_DIR>/per_fold_test/fold_0/predictions/*.nii.gz  (51 binary preds)
#   <EVAL_DIR>/per_fold_test/fold_0/predictions/*.npz     (51 softmax)
#   <EVAL_DIR>/per_fold_test/fold_0/eval.json             (per-fold Dice etc.)
#   <EVAL_DIR>/per_fold_test/fold_{1,2,3,4}/...           (same per fold)
#
# 5 single-fold inferences × ~3 min each ≈ 15 min total. Wall time set
# to 30 min as safety.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

set -e
cd ${SLURM_SUBMIT_DIR}

echo "=== Per-fold test predictions: exp3_baseline_v2_d600 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
EXPERIMENT_NAME="exp3_baseline_v2_d600"
DATASET_ID=600
EVAL_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/eval_${EXPERIMENT_NAME}"
PER_FOLD_DIR="${EVAL_DIR}/per_fold_test"
mkdir -p "${PER_FOLD_DIR}"

# Run inference 5 times, one fold at a time, each into its own dir.
for FOLD in 0 1 2 3 4; do
    OUT_DIR="${PER_FOLD_DIR}/fold_${FOLD}"
    echo ""
    echo "=== Fold ${FOLD} → ${OUT_DIR} ==="
    time python -m medgen.scripts.eval_nnunet \
        --experiment baseline \
        --dataset-id ${DATASET_ID} \
        --experiment-name "${EXPERIMENT_NAME}" \
        --folds ${FOLD} \
        --nnunet-base "${NNUNET_BASE}" \
        --nnunet-results "${NNUNET_RESULTS}" \
        --trainer nnUNetTrainerBrainMets \
        --plans nnUNetResEncUNetLPlans \
        --save-probabilities \
        --output-dir "${OUT_DIR}"
done

# ── Summary & download hints ──────────────────────────────────────────────
echo ""
echo "=== Done. Per-fold outputs: ==="
for FOLD in 0 1 2 3 4; do
    OUT_DIR="${PER_FOLD_DIR}/fold_${FOLD}"
    N_NPZ=$(ls "${OUT_DIR}/predictions"/*.npz 2>/dev/null | wc -l)
    N_NII=$(ls "${OUT_DIR}/predictions"/*.nii.gz 2>/dev/null | wc -l)
    SZ=$(du -sh "${OUT_DIR}" 2>/dev/null | cut -f1)
    echo "  fold_${FOLD}: ${N_NPZ} .npz + ${N_NII} .nii.gz (${SZ}) — ${OUT_DIR}"
done
echo ""
echo "=== Suggested rsync (run on your local machine) ==="
echo ""
echo "rsync -av idun:${PER_FOLD_DIR}/ \\"
echo "    ~/NTNU/AIS4900_master/runs/exp3_baseline_v2_d600/per_fold_test/"
echo ""
echo "# Also pull test GT if not already local"
echo "rsync -av idun:${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet/labelsTs/ \\"
echo "    ~/NTNU/AIS4900_master/runs/exp3_baseline_v2_d600/labelsTs/"
