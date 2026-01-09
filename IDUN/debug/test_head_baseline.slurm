#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="test_head_baseline"
#SBATCH --output=IDUN/output/debug/test_head_baseline_%j.out
#SBATCH --error=IDUN/output/debug/test_head_baseline_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# HEAD baseline - all defaults enabled
# Compare loss trajectory to known good (baadcdd)

cd ${SLURM_SUBMIT_DIR}

echo "=== Test: HEAD baseline (all defaults) ==="
echo "Commit: $(git rev-parse HEAD)"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model.image_size=64 \
    training.epochs=500 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=test_head_baseline_

conda deactivate
