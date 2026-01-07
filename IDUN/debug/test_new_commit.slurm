#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="test_new_HEAD"
#SBATCH --output=IDUN/output/debug/test_new_%j.out
#SBATCH --error=IDUN/output/debug/test_new_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Testing NEW commit (HEAD) ==="
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

# Reinstall package to pick up any module changes
pip uninstall medgen -y 2>/dev/null || true
pip install -e . --quiet --no-build-isolation

# Run training on current code
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model.image_size=64 \
    training.epochs=500 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=debug_new_HEAD_

conda deactivate
