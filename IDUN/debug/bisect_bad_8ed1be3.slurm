#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="bisect_BAD_8ed1be3"
#SBATCH --output=IDUN/output/debug/bisect_BAD_8ed1be3_%j.out
#SBATCH --error=IDUN/output/debug/bisect_BAD_8ed1be3_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# FIRST BAD COMMIT - removed .float() from perceptual inputs
COMMIT="8ed1be3"
WORKTREE_DIR="/tmp/bisect_${COMMIT}_${SLURM_JOB_ID}"

cd ${SLURM_SUBMIT_DIR}

echo "=== Bisect test: FIRST BAD commit ${COMMIT} ==="
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
pip install -e . --quiet

# Run training
echo "Starting training at commit ${COMMIT}..."
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model.image_size=64 \
    training.epochs=100 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=bisect_BAD_${COMMIT}_

# Cleanup worktree
cd ${SLURM_SUBMIT_DIR}
git worktree remove ${WORKTREE_DIR} --force

conda deactivate
