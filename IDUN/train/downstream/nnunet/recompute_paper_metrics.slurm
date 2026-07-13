#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="paper_metrics"
#SBATCH --output=IDUN/output/train/downstream/nnunet/recompute_paper_metrics_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/recompute_paper_metrics_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Prediction-only recomputation for every condition used by the MIA paper.
#
# This job DOES NOT load nnU-Net checkpoints, run a predictor, or regenerate
# masks. It reads the existing five-fold ensemble predictions from:
#   runs/downstream/nnunet/<experiment>/eval_<experiment>/predictions/
# The canonical inference log for every pool is checked for folds 0--4 and
# fingerprinted in the output manifest.
#
# It recomputes, at full precision:
#   - per-patient complete-volume Dice and sagittal slice-wise Dice;
#   - pooled voxel precision/recall;
#   - spacing-aware HD95 in mm with explicit empty-mask accounting;
#   - one-to-one 26-connected lesion matching using unsmoothed full-component
#     Dice > 0.1, including sensitivity, precision/F1, FP per patient, matched
#     Dice, and the existing study-defined size strata;
#   - the predefined nine paired Wilcoxon comparisons against real-only Dice,
#     with Holm/Bonferroni adjustment, effects, and bootstrap confidence limits.
#
# Historical eval JSONs are never overwritten. Each run writes a new,
# job-specific result directory.

set -euo pipefail

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd "${SLURM_SUBMIT_DIR}"

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
NNUNET_RAW="${CLUSTER_BASE}/MedicalDataSets/nnunet/nnUNet_raw"
OUTPUT_DIR="${NNUNET_RESULTS}/paper_metric_recompute/job_${SLURM_JOB_ID}"

PROVENANCE_ROOT="IDUN/output/train/downstream/nnunet/eval_5fold"

# label | experiment directory | original nnU-Net dataset ID | canonical stdout
CONDITIONS=(
  "real_official|exp3_baseline_v2_d600|600|${PROVENANCE_ROOT}/exp3_baseline_v2_d600_24512664.out"
  "real_random_seed42|exp3_random_split_d640|640|${PROVENANCE_ROOT}/exp3_random_split_d640_24578359.out"
  "A_105|exp6_2_synthetic_105_exp1_1_1000_RIN|616|${PROVENANCE_ROOT}/exp6_2_synthetic_105_exp1_1_1000_RIN_24512668.out"
  "B_105|exp6_2_synthetic_105_exp1_1_1000plus|611|${PROVENANCE_ROOT}/exp1_1_1000plus_24512669.out"
  "C_105|exp6_2_synthetic_105_exp32_2_1000|612|${PROVENANCE_ROOT}/exp32_2_1000_24512670.out"
  "D_105|exp6_2_synthetic_105_exp48c_standalone|613|${PROVENANCE_ROOT}/exp48c_standalone_24512672.out"
  "E_105|exp6_2_synthetic_105_exp48d_handoff_exp1|615|${PROVENANCE_ROOT}/exp48d_handoff_exp1_24512673.out"
  "F_105|exp6_2_synthetic_105_exp48c_handoff_exp32|614|${PROVENANCE_ROOT}/exp48c_handoff_exp32_24512671.out"
  "F_525|exp6_2_synthetic_525_exp48c_handoff_exp32|614|${PROVENANCE_ROOT}/exp6_2_synthetic_525_exp48c_handoff_exp32_24512674.out"
  "hybrid_25|exp7_4_mixed_25syn_d614|614|${PROVENANCE_ROOT}/exp7_4_mixed_25syn_d614_24512675.out"
  "hybrid_50|exp7_5_mixed_50syn_d614|614|${PROVENANCE_ROOT}/exp7_5_mixed_50syn_d614_24512676.out"
  "hybrid_75|exp7_6_mixed_75syn_d614|614|${PROVENANCE_ROOT}/exp7_6_mixed_75syn_d614_24512677.out"
  "hybrid_105|exp7_1_mixed_105syn_d614|614|${PROVENANCE_ROOT}/exp7_1_mixed_105syn_d614_24512678.out"
  "hybrid_210|exp7_2_mixed_210syn_d614|614|${PROVENANCE_ROOT}/exp7_2_mixed_210syn_d614_24512679.out"
  "hybrid_315|exp7_3_mixed_315syn_d614|614|${PROVENANCE_ROOT}/exp7_3_mixed_315syn_d614_24771143.out"
  "hybrid_525|exp7_7_mixed_525syn_d614|614|${PROVENANCE_ROOT}/exp7_7_mixed_525syn_d614_24512681.out"
)

# This is the same nine-comparison downstream family used in the paper audit.
COMPARE=(
  "F_105"
  "F_525"
  "hybrid_25"
  "hybrid_50"
  "hybrid_75"
  "hybrid_105"
  "hybrid_210"
  "hybrid_315"
  "hybrid_525"
)

ARGS=()
for condition in "${CONDITIONS[@]}"; do
  ARGS+=(--condition "${condition}")
done
for condition in "${COMPARE[@]}"; do
  ARGS+=(--compare "${condition}")
done

echo "=== Prediction-only MIA paper metric recomputation ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
echo "Predictions: ${NNUNET_RESULTS}"
echo "Ground truth: ${NNUNET_RAW}"
echo "Output: ${OUTPUT_DIR}"
echo "Conditions: ${#CONDITIONS[@]} | paired family: ${#COMPARE[@]}"

time python -m medgen.scripts.recompute_nnunet_metrics \
  --nnunet-results "${NNUNET_RESULTS}" \
  --nnunet-raw "${NNUNET_RAW}" \
  --output-dir "${OUTPUT_DIR}" \
  --baseline real_official \
  --expected-cases 51 \
  --min-component-voxels 5 \
  --detection-threshold 0.1 \
  --bootstrap-draws 10000 \
  --seed 0 \
  "${ARGS[@]}"

echo "=== Completed prediction-only recomputation ==="
echo "Summary JSON: ${OUTPUT_DIR}/summary.json"
echo "Table CSV:    ${OUTPUT_DIR}/summary.csv"
echo "Per case:     ${OUTPUT_DIR}/per_case.csv"
echo "Per lesion:   ${OUTPUT_DIR}/per_lesion.csv"
echo "Statistics:   ${OUTPUT_DIR}/paired_statistics.json"
