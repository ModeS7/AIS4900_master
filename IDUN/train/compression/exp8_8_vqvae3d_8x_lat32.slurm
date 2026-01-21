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
#SBATCH --job-name="exp8_8_8x_lat32"
#SBATCH --output=IDUN/output/train/compression/exp8_8_8x_lat32_%j.out
#SBATCH --error=IDUN/output/train/compression/exp8_8_8x_lat32_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp8.8: 3D VQ-VAE 8x compression, 32 latent channels ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# 8x compression: 3 levels (256->128->64->32)
time python -m medgen.scripts.train_vqvae_3d \
    paths=cluster \
    mode=multi_modality \
    training.epochs=125 \
    training.batch_size=1 \
    vqvae_3d.embedding_dim=32 \
    'vqvae_3d.channels=[64,128,256]' \
    'vqvae_3d.num_res_channels=[64,128,256]' \
    'vqvae_3d.downsample_parameters=[[2,4,1,1],[2,4,1,1],[2,4,1,1]]' \
    'vqvae_3d.upsample_parameters=[[2,4,1,1,0],[2,4,1,1,0],[2,4,1,1,0]]' \
    training.name=exp8_8_

conda deactivate
