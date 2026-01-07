#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="bisect_60dc0af"
#SBATCH --output=IDUN/output/debug/bisect_60dc0af_%j.out
#SBATCH --error=IDUN/output/debug/bisect_60dc0af_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# 60dc0af - Right before 8ed1be3 refactor (Jan 3)
COMMIT="60dc0af"
WORKTREE_DIR="/tmp/bisect_${COMMIT}_${SLURM_JOB_ID}"

cd ${SLURM_SUBMIT_DIR}

echo "=== Bisect: ${COMMIT} (before big refactor) ==="
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

git worktree add ${WORKTREE_DIR} ${COMMIT}
cd ${WORKTREE_DIR}
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
    training.name=bisect_60dc0af_

cd ${SLURM_SUBMIT_DIR}
git worktree remove ${WORKTREE_DIR} --force

conda deactivate
