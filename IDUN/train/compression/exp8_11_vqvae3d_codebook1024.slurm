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
#SBATCH --job-name="exp8_11_cb1024"
#SBATCH --output=IDUN/output/train/compression/exp8_11_vqvae3d_cb1024_%j.out
#SBATCH --error=IDUN/output/train/compression/exp8_11_vqvae3d_cb1024_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp8.11: 3D VQ-VAE LARGER codebook (1024 vs 512) ==="
echo "Hypothesis: More codebook entries = finer-grained discrete representation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Exp8.11: Larger codebook (1024 entries instead of 512)
# Monitor codebook utilization to check for dead codes
time python -m medgen.scripts.train_vqvae_3d \
    paths=cluster \
    mode=multi_modality \
    training.epochs=125 \
    training.batch_size=1 \
    vqvae_3d.num_embeddings=1024 \
    training.name=exp8_11_

conda deactivate
