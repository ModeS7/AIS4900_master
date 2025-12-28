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
#SBATCH --job-name="exp6_1_vqvae_4x"
#SBATCH --output=IDUN/output/train/compression/exp6_1_vqvae_4x_%j.out
#SBATCH --error=IDUN/output/train/compression/exp6_1_vqvae_4x_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp6.1 VQ-VAE Multi-Modality 256x256 (4x compression, 64x64 latent) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# VQ-VAE 4x compression (matches exp4 VAE latent size)
time python -m medgen.scripts.train_vqvae \
    paths=cluster \
    mode=multi_modality \
    model.image_size=256 \
    'vqvae.channels=[64,128]' \
    'vqvae.num_res_channels=[64,128]' \
    'vqvae.downsample_parameters=[[2,4,1,1],[2,4,1,1]]' \
    'vqvae.upsample_parameters=[[2,4,1,1,0],[2,4,1,1,0]]' \
    training.epochs=125 \
    training.batch_size=16 \
    training.augment=false \
    training.name=exp6_1_

conda deactivate
