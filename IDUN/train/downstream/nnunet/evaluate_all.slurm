#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --job-name="nn_eval"
#SBATCH --output=IDUN/output/train/downstream/nnunet/eval/%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/eval/%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Evaluate all trained nnU-Net experiments on the 51-volume test set.
# Runs 5-fold ensemble inference (checkpoint_best.pth) + MedGen metrics.
#
# Dataset 501: real + ImageNet-optimized synthetic
# Dataset 502: real + RadImageNet-optimized synthetic

cd ${SLURM_SUBMIT_DIR}

echo "=== nnU-Net: Evaluate all experiments ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
COMMON="--nnunet-base ${NNUNET_BASE} --nnunet-results ${NNUNET_RESULTS} --trainer nnUNetTrainerBrainMets --plans nnUNetResEncUNetLPlans --folds 0 1 2 3 4"

evaluate() {
    local name=$1
    local args=$2

    echo ""
    echo "=============================================="
    echo "  Evaluating: ${name}"
    echo "=============================================="

    python -m medgen.scripts.eval_nnunet \
        --experiment-name "${name}" \
        ${args} \
        ${COMMON} \
        || echo "WARNING: Evaluation failed for ${name}"
}

# ── Baselines ────────────────────────────────────────────────────
evaluate "exp3_baseline" \
    "--experiment baseline --dataset-id 501"

evaluate "exp4_baseline_dual" \
    "--experiment baseline_dual --dataset-id 503"

# ── Synthetic-only 525 ───────────────────────────────────────────
evaluate "exp6_synthetic_imagenet" \
    "--experiment synthetic --dataset-id 501"

evaluate "exp6_synthetic_radimagenet" \
    "--experiment synthetic --dataset-id 502"

# ── Synthetic-only 105 ───────────────────────────────────────────
evaluate "exp6_1_synthetic_105_imagenet" \
    "--experiment synthetic --n-synthetic 105 --dataset-id 501"

evaluate "exp6_1_synthetic_105_radimagenet" \
    "--experiment synthetic --n-synthetic 105 --dataset-id 502"

echo ""
echo "=== All evaluations complete ==="
