#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp2_rflow_100steps"
#SBATCH --output=IDUN/output/train/exp2_rflow_100steps_%j.out
#SBATCH --error=IDUN/output/train/exp2_rflow_100steps_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Experiment 2: RFlow with 100 training steps
# RFlow 128 bravo, 100 steps instead of 1000

cd ${SLURM_SUBMIT_DIR}

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4005

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100

conda deactivate
