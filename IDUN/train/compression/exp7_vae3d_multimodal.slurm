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

echo "=== Exp7 3D VAE Multi-Modality (train 128x128, val 256x256) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Memory: reduce fragmentation, disable discriminator (saves ~15GB)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Exp7: 3D VAE train at 128x128x160, validate at 256x256x160
# - Training at lower resolution for speed
# - Validation at full resolution for accurate metrics
# - disable_gan=true skips discriminator creation entirely (saves ~15GB VRAM)
time python -m medgen.scripts.train_compression --config-name=vae_3d \
    paths=cluster \
    mode=multi_modality \
    training.epochs=125 \
    training.batch_size=1 \
    training.gradient_checkpointing=true \
    vae_3d.disable_gan=true \
    volume.train_height=128 \
    volume.train_width=128 \
    training.name=exp7_train128_val256_

conda deactivate
