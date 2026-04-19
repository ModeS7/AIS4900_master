#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name="fft_cmp"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/fft_comparison/%j.out
#SBATCH --error=IDUN/output/eval/fft_comparison/%j.err

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# FFT amplitude comparison: real vs generated brain MRI
# CPU-only — no GPU needed

cd ${SLURM_SUBMIT_DIR}

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/fft_comparison_${TIMESTAMP}"

python -m medgen.scripts.visualize_fft_comparison \
    --real-dir ${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/train \
    --generated-dir ${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525 \
    --output-dir ${OUTPUT_DIR} \
    --num-volumes 20 \
    --depth 160

echo "Results saved to: ${OUTPUT_DIR}"
