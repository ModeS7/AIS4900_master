#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="blur_cmp"
#SBATCH --output=IDUN/output/eval/blur_attack_cmp_%j.out
#SBATCH --error=IDUN/output/eval/blur_attack_cmp_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Final comparison across all blur-attack methods.
# Run AFTER all individual experiments have produced refined volumes.
# Edit the METHOD paths below to match the actual run directories.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== Blur attack comparison ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/test_new"
OUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/blur_attack_compare_$(date +%Y%m%d-%H%M%S)"

# EDIT THESE — paths to refined volumes from each experiment.
# baseline  → original synthetic outputs (no refinement)
# exp42b    → exp42b best-checkpoint applied to baseline volumes
# sdedit    → SDEdit at chosen best t₀
# spectral  → spectral EQ at chosen best clip-max
# exp43_irsde / exp44_fm → respective trained model outputs
BASELINE_DIR="${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"
EXP42B_DIR="<PLACEHOLDER — exp42b refined dir>"
SDEDIT_DIR="<PLACEHOLDER — choose best t0 dir from sdedit run>"
SPECTRAL_DIR="<PLACEHOLDER — choose best clip-max from spectral_eq run>"
EXP43_DIR="<PLACEHOLDER — exp43 IR-SDE inference output dir>"
EXP44_DIR="<PLACEHOLDER — exp44 inference output dir>"

mkdir -p "${OUT_DIR}"

time python -m medgen.scripts.eval_blur_attack_compare \
    --methods \
        "baseline:${BASELINE_DIR}" \
        "exp42b:${EXP42B_DIR}" \
        "sdedit:${SDEDIT_DIR}" \
        "spectral:${SPECTRAL_DIR}" \
        "exp43_irsde:${EXP43_DIR}" \
        "exp44_fm:${EXP44_DIR}" \
    --real-dir   "${REAL_DIR}" \
    --output-dir "${OUT_DIR}" \
    --num-volumes 25 \
    --num-real 25 \
    --num-real-lpips 5 \
    --grid-subjects 5

echo "=== Done ==="
ls -la "${OUT_DIR}"
echo "--- results.csv ---"
cat "${OUT_DIR}/results.csv"
