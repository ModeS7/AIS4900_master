#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp12_1c_sit_s"
#SBATCH --output=IDUN/output/train/diffusion/exp12_1c_sit_s_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp12_1c_sit_s_%j.err
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

# exp12_1c: SiT-S (Small) - 33M params, 256x256 with patch_size=4
# Same computation as 128x128 with patch_size=2 (4096 tokens)
# Reference: https://arxiv.org/abs/2401.08740
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=sit \
    model.variant=S \
    model.patch_size=4 \
    model.image_size=256 \
    training.name=exp12_1c_

conda deactivate
