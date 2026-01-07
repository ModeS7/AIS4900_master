#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="test_old_d2839b9"
#SBATCH --output=IDUN/output/debug/test_old_%j.out
#SBATCH --error=IDUN/output/debug/test_old_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Test OLD commit d2839b9 (Dec 26 baseline) using isolated worktree
COMMIT="d2839b9"
WORKTREE_DIR="/tmp/test_old_${COMMIT}_${SLURM_JOB_ID}"

cd ${SLURM_SUBMIT_DIR}

echo "=== Testing OLD commit d2839b9 (Dec 26) ==="
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

# Create isolated worktree for this commit
echo "Creating worktree at ${WORKTREE_DIR}..."
git worktree add ${WORKTREE_DIR} ${COMMIT}
cd ${WORKTREE_DIR}

# Install package from this commit
pip uninstall medgen -y 2>/dev/null || true
pip install -e . --quiet --no-build-isolation

# Run training
echo "Starting training at commit ${COMMIT}..."
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model.image_size=64 \
    training.epochs=500 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=debug_old_d2839b9_

# Cleanup worktree
cd ${SLURM_SUBMIT_DIR}
git worktree remove ${WORKTREE_DIR} --force

conda deactivate
