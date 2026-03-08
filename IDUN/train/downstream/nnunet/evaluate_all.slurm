#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_eval"
#SBATCH --output=IDUN/output/train/downstream/nnunet/eval/%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/eval/%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Evaluate all trained nnU-Net experiments on the 51-volume test set.
# Runs inference (checkpoint_best.pth) + MedGen metrics for each experiment.
#
# Usage:
#   sbatch evaluate_all.slurm                    # evaluate all experiments
#   FOLDS="0" sbatch evaluate_all.slurm          # single fold only
#   EXPERIMENTS="baseline" sbatch evaluate_all.slurm  # single experiment

cd ${SLURM_SUBMIT_DIR}

echo "=== nnU-Net: Evaluate all experiments ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="/cluster/work/modestas/MedicalDataSets/nnunet"
NNUNET_RESULTS="/cluster/work/modestas/AIS4900_master/runs/downstream/nnunet"
FOLDS=${FOLDS:-"0 1 2 3 4"}

# Default: all experiments
if [ -z "${EXPERIMENTS:-}" ]; then
    EXPERIMENTS="baseline baseline_dual synthetic mixed_25 mixed_50 mixed_75 mixed_105 mixed_210 mixed_315 mixed_525"
fi

evaluate_experiment() {
    local experiment=$1
    local dataset_id=$2
    local n_synthetic=$3  # empty string if not applicable

    local syn_flag=""
    if [ -n "$n_synthetic" ]; then
        syn_flag="--n-synthetic $n_synthetic"
    fi

    echo ""
    echo "=============================================="
    echo "=== Evaluating: $experiment (dataset $dataset_id, folds: $FOLDS) ==="
    echo "=============================================="

    python -m medgen.scripts.eval_nnunet \
        --experiment $experiment \
        --dataset-id $dataset_id \
        --folds $FOLDS \
        --nnunet-base "${NNUNET_BASE}" \
        --nnunet-results "${NNUNET_RESULTS}" \
        --trainer nnUNetTrainerBrainMets \
        --plans nnUNetResEncUNetLPlans \
        $syn_flag \
        || echo "WARNING: Evaluation failed for $experiment"
}

for exp in $EXPERIMENTS; do
    case $exp in
        baseline)
            evaluate_experiment baseline 501 ""
            ;;
        baseline_dual)
            evaluate_experiment baseline_dual 502 ""
            ;;
        synthetic)
            evaluate_experiment synthetic 501 ""
            ;;
        synthetic_105)
            evaluate_experiment synthetic 501 "105"
            ;;
        mixed_25)
            evaluate_experiment mixed 501 "25"
            ;;
        mixed_50)
            evaluate_experiment mixed 501 "50"
            ;;
        mixed_75)
            evaluate_experiment mixed 501 "75"
            ;;
        mixed_105)
            evaluate_experiment mixed 501 "105"
            ;;
        mixed_210)
            evaluate_experiment mixed 501 "210"
            ;;
        mixed_315)
            evaluate_experiment mixed 501 "315"
            ;;
        mixed_525)
            evaluate_experiment mixed 501 "525"
            ;;
        *)
            echo "Unknown experiment: $exp"
            ;;
    esac
done

echo ""
echo "=== All evaluations complete ==="
