#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --job-name="nn_eval_lw"
#SBATCH --output=IDUN/output/train/downstream/nnunet/eval/%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/eval/%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# ---------------------------------------------------------------------------
# Re-evaluate the PAPER's downstream conditions to emit BraTS-Mets LESION-WISE
# Dice. evaluate.py already computes 'dice_lesionwise_bratsmets' per case +
# aggregate (evaluate.py:255,262,324); the thesis-era eval JSONs predate it, so
# this just re-runs the SAME 5-fold ensemble inference + MedGen metrics with the
# current code. Volumetric Dice is unchanged (validated locally: real baseline
# = 0.3209, matching the thesis) -- the only *new* number is the lesion-wise one.
#
# Each condition writes:
#   ${NNUNET_RESULTS}/<name>/eval_<name>.json
# with a 'per_case' block containing per-patient 'dice' (volumetric) AND
# 'dice_lesionwise_bratsmets', plus aggregate 'dice_variants'.
# Pull those JSONs back to analysis/lesionwise/ for the 4.2 table + the
# per-patient paired Wilcoxon (real vs synth-only, real vs hybrid).
#
# Conditions + args copied VERBATIM from the per-condition eval_5fold_*.slurm
# files (so they hit the exact trained checkpoints that produced the thesis).
# Each run() is fail-safe: a failure warns and the loop continues, and every
# finished condition writes its JSON, so a timeout still leaves partial results.
# Runtime ~ 10 conditions x (5-fold ensemble inference on 51 vols); 12h is ample.
# ---------------------------------------------------------------------------

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}
echo "=== nnU-Net lesion-wise re-eval | Job ${SLURM_JOB_ID} | ${SLURM_JOB_NODELIST} ==="
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
COMMON="--nnunet-base ${NNUNET_BASE} --nnunet-results ${NNUNET_RESULTS} --trainer nnUNetTrainerBrainMets --plans nnUNetResEncUNetLPlans --folds 0 1 2 3 4"

run() {
    echo ""; echo "==================== eval: $1 ===================="
    time python -m medgen.scripts.eval_nnunet $1 ${COMMON} \
        || echo "WARNING: eval failed for: $1"
}

# ── Real-only baseline (official 51-patient hold-out) ─────────────────────────
run "--experiment baseline --dataset-id 600 --experiment-name exp3_baseline_v2_d600"

# ── Synthetic-only (handoff exp48c -> exp32 base = syn-handoffC), 105 and 525 ──
run "--experiment synthetic --n-synthetic 105 --dataset-id 614 --experiment-name exp6_2_synthetic_105_exp48c_handoff_exp32"
run "--experiment synthetic --dataset-id 614 --experiment-name exp6_2_synthetic_525_exp48c_handoff_exp32"

# ── Real + synthetic mixup: 105 real + N synthetic (the ratio sweep) ──────────
run "--experiment mixed --n-synthetic 25  --dataset-id 614 --experiment-name exp7_4_mixed_25syn_d614"
run "--experiment mixed --n-synthetic 50  --dataset-id 614 --experiment-name exp7_5_mixed_50syn_d614"
run "--experiment mixed --n-synthetic 75  --dataset-id 614 --experiment-name exp7_6_mixed_75syn_d614"
run "--experiment mixed --n-synthetic 105 --dataset-id 614 --experiment-name exp7_1_mixed_105syn_d614"
run "--experiment mixed --n-synthetic 210 --dataset-id 614 --experiment-name exp7_2_mixed_210syn_d614"
run "--experiment mixed --n-synthetic 315 --dataset-id 614 --experiment-name exp7_3_mixed_315syn_d614"
run "--experiment mixed --n-synthetic 525 --dataset-id 614 --experiment-name exp7_7_mixed_525syn_d614"

# ── OPTIONAL extras (uncomment after copying exact args from their eval_5fold_*.slurm) ──
# Split-protocol (random split -> the ~0.57 that proves the official cohort is hard):
#   see eval_5fold_exp3_random_split_d640.slurm   (likely: --experiment baseline --dataset-id 640 --experiment-name exp3_random_split_d640)
# Size-targeted mixup (per-tumour-size table): eval_5fold_exp8_{1..4}_size_targeted_*.slurm

echo ""; echo "=== DONE. Pull ${NNUNET_RESULTS}/*/eval_*.json for per-case dice_lesionwise_bratsmets ==="
