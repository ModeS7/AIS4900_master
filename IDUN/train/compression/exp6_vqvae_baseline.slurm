#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp6_vqvae"
#SBATCH --output=IDUN/output/train/compression/exp6_vqvae_%j.out
#SBATCH --error=IDUN/output/train/compression/exp6_vqvae_%j.err
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

# VQ-VAE baseline: dual mode (t1_pre + t1_gd), 256 resolution
# Codebook: 512 embeddings x 64 dim
# GAN enabled for sharper reconstructions
time python -m medgen.scripts.train_vqvae \
    paths=cluster \
    mode=dual \
    model.image_size=256 \
    training.epochs=100 \
    training.batch_size=16 \
    training.use_ema=true \
    training.name=exp5_

conda deactivate
