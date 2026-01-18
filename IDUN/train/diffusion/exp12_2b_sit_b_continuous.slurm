#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp12_2b_sit_b"
#SBATCH --output=IDUN/output/train/diffusion/exp12_2b_sit_b_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp12_2b_sit_b_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# exp12_2b: SiT-B (Base) - 130M params with CONTINUOUS timesteps
# Compare with exp12_2 (discrete timesteps)
# Reference: https://arxiv.org/abs/2401.08740
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=sit \
    model.variant=B \
    model.patch_size=2 \
    model.image_size=128 \
    training.name=exp12_2b_

conda deactivate
