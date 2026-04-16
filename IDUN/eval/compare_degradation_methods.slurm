#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name="deg_cmp"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/compare_degradation/%j.out
#SBATCH --error=IDUN/output/eval/compare_degradation/%j.err

# Compare empirical vs Wiener filter degradation methods
# CPU-only job — no GPU needed (just FFT operations)
# ~10 min for 20 volumes

cd ${SLURM_SUBMIT_DIR}

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="/cluster/work/modestas/AIS4900_master/runs/eval/degradation_comparison_${TIMESTAMP}"

python -m medgen.scripts.compare_degradation_methods \
    --real-dir /cluster/work/modestas/MedicalDataSets/brainmetshare-3/train \
    --generated-dir /cluster/work/modestas/MedicalDataSets/generated/exp1_1_bravo_imagenet_525 \
    --output-dir ${OUTPUT_DIR} \
    --num-volumes 20 \
    --depth 160

echo "Results saved to: ${OUTPUT_DIR}"
