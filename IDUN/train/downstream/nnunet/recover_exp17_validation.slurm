#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="recover_exp17_val"
#SBATCH --array=0-19
#SBATCH --output=IDUN/output/train/downstream/nnunet/recover_exp17_val_%A_%a.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/recover_exp17_val_%A_%a.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --no-requeue

# Regenerate the lost validation/ (predicted val segs + summary.json) AND debug.json
# for every exp17 fold, by re-running nnU-Net validation on the surviving
# checkpoint_final.pth. Deterministic inference -> identical to the originals.
#
# 20-task array: idx = experiment(0-3) * 5 + fold(0-4). Each task reuses that
# experiment's exact training setup (same --n-synthetic + manifest => same
# isolated preprocessed dataset + splits => same 21 real validation cases).
#
# Submit:  sbatch IDUN/train/downstream/nnunet/recover_exp17_validation.slurm

set -Eeuo pipefail
fatal() { echo "FATAL: $*" >&2; exit 1; }

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
readonly DATASET_ID=663
readonly PLANS_NAME="nnUNetResEncUNetLPlansD600"
readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
readonly NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly RAW_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset${DATASET_ID}_BrainMet"
readonly SYNTHETIC_MANIFEST="${RAW_DIR}/synthetic_candidate_order.txt"

# idx -> (experiment, n_synthetic, fold)
readonly IDX="${SLURM_ARRAY_TASK_ID:?submit as a 0-19 array}"
EXP_LIST=(exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663 \
          exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663 \
          exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663 \
          exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663)
NSYN_LIST=(25 50 105 210)
readonly EXP_I=$(( IDX / 5 ))
readonly FOLD=$(( IDX % 5 ))
readonly EXPERIMENT_NAME="${EXP_LIST[$EXP_I]}"
readonly N_SYNTHETIC="${NSYN_LIST[$EXP_I]}"

readonly MODEL_NAME="nnUNetTrainerBrainMets__${PLANS_NAME}__3d_fullres"
readonly FOLD_DIR="${NNUNET_RESULTS}/${EXPERIMENT_NAME}/Dataset${DATASET_ID}_BrainMet/${MODEL_NAME}/fold_${FOLD}"
[[ -f "${FOLD_DIR}/checkpoint_final.pth" ]] || fatal "missing checkpoint: ${FOLD_DIR}/checkpoint_final.pth"

echo "=== recover validation: ${EXPERIMENT_NAME} fold ${FOLD} (n_synthetic=${N_SYNTHETIC}) ==="
echo "  checkpoint: ${FOLD_DIR}/checkpoint_final.pth"
nvidia-smi

cd "${REPO_ROOT}"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m medgen.scripts.train_nnunet \
    --experiment mixed \
    --n-synthetic "${N_SYNTHETIC}" \
    --synthetic-manifest "${SYNTHETIC_MANIFEST}" \
    --dataset-id "${DATASET_ID}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --fold "${FOLD}" \
    --nnunet-base "${NNUNET_BASE}" \
    --nnunet-results "${NNUNET_RESULTS}" \
    --trainer nnUNetTrainerBrainMets \
    --plans "${PLANS_NAME}" \
    --validation-only

echo "=== done: validation/ + summary.json + debug.json regenerated in ${FOLD_DIR} ==="
