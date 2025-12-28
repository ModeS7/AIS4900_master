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
#SBATCH --job-name="exp7_vae3d_mm"
#SBATCH --output=IDUN/output/train/compression/exp7_vae3d_multimodal_%j.out
#SBATCH --error=IDUN/output/train/compression/exp7_vae3d_multimodal_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp7 3D VAE Multi-Modality 256x256x160 (8x compression) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Exp7: 3D VAE multi-modality (bravo, flair, t1_pre, t1_gd)
# - 256x256x160 volumes (8x compression -> 32x32x20 latent)
# - Pools all 4 modalities = 4x training data
# - MONAI proven config: channels=[32,64,128,128], no attention
# - batch_size=2 (memory constraint for 3D)
# - 100 epochs
time python -m medgen.scripts.train_vae_3d \
    paths=cluster \
    mode=multi_modality \
    training.epochs=125 \
    training.batch_size=2 \
    training.name=exp7_

conda deactivate
