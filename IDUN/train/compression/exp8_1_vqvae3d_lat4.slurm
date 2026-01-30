#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp8_1_vqvae3d_lat4"
#SBATCH --output=IDUN/output/train/compression/exp8_1_vqvae3d_lat4_%j.out
#SBATCH --error=IDUN/output/train/compression/exp8_1_vqvae3d_lat4_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp8.1 3D VQ-VAE 4 latent channels ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Exp8.1: 3D VQ-VAE with 4 latent channels (vs default 3)
time python -m medgen.scripts.train_compression --config-name=vqvae_3d \
    paths=cluster \
    mode=multi_modality \
    training.epochs=125 \
    training.batch_size=1 \
    vqvae_3d.embedding_dim=4 \
    training.name=exp8_1_

conda deactivate
