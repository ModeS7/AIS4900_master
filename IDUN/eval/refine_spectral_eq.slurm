#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="spec_eq"
#SBATCH --output=IDUN/output/eval/spectral_eq_%j.out
#SBATCH --error=IDUN/output/eval/spectral_eq_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# T1B: Spectral equalization probe (no training).
# Apply gain(f)=sqrt(P_real/P_synth) per radial frequency to each synth volume.
# Sweep gain-clip-max to control noise amplification.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== T1B: Spectral EQ refinement ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

DIR_525="${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"
DIR_OPT="${CLUSTER_BASE}/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp1_1_1000"
REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/test_new"
OUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/spectral_eq_$(date +%Y%m%d-%H%M%S)"

mkdir -p "${OUT_DIR}"

time python -m medgen.scripts.refine_spectral_eq \
    --synth-dirs "${DIR_525}" "${DIR_OPT}" \
    --real-dir   "${REAL_DIR}" \
    --output-dir "${OUT_DIR}" \
    --gain-clip-maxes 1.5 2.0 3.0 5.0 \
    --num-synth 25 \
    --num-real 25 \
    --num-real-lpips 5

echo "=== Done ==="
ls -la "${OUT_DIR}"
