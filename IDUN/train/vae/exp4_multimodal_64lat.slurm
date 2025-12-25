#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp4_vae_mm64lat"
#SBATCH --output=IDUN/output/train/vae/exp4_multimodal_64lat_%j.out
#SBATCH --error=IDUN/output/train/vae/exp4_multimodal_64lat_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp4 VAE Multi-Modality 256x256 → 64x64 latent (4x compression) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Exp4: Multi-modality VAE 256x256 with 4x compression (64x64 latent)
# - Pools all modalities (bravo, flair, t1_pre, t1_gd) = 4x training data
# - Per-modality validation metrics + regional tumor metrics
# - 3-level VAE: channels=[64,128,256] for 4x compression instead of 8x
# - Single H100 GPU with torch.compile
# - 500 epochs
time python -m medgen.scripts.train_vae \
    paths=cluster \
    mode=multi_modality \
    model.image_size=256 \
    'vae.channels=[64,128,256]' \
    'vae.attention_levels=[false,false,true]' \
    training.epochs=500 \
    training.batch_size=16 \
    training.val_interval=30 \
    training.augment=false \
    training.name=exp4_

conda deactivate
