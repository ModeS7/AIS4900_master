#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="test_no_compile_no_msssim"
#SBATCH --output=IDUN/output/debug/test_no_compile_no_msssim_%j.out
#SBATCH --error=IDUN/output/debug/test_no_compile_no_msssim_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Test hypothesis: both compile and msssim cause divergence
# Disable both to see if that matches baadcdd baseline

cd ${SLURM_SUBMIT_DIR}

echo "=== Test: HEAD with use_compile=false AND msssim=false ==="
echo "Commit: $(git rev-parse HEAD)"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

pip uninstall medgen -y 2>/dev/null || true
pip install -e . --quiet

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
    training.logging.msssim=false \
    training.name=test_no_compile_no_msssim_

conda deactivate
