#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="test_no_compile"
#SBATCH --output=IDUN/output/debug/test_no_compile_%j.out
#SBATCH --error=IDUN/output/debug/test_no_compile_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Test hypothesis: compiled forward functions cause divergence
# Disable torch.compile to test

cd ${SLURM_SUBMIT_DIR}

echo "=== Test: HEAD with use_compile=false ==="
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
    training.use_compile=false \
    training.name=test_no_compile_

conda deactivate
