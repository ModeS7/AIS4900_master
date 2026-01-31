#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="profile_unet"
#SBATCH --output=IDUN/output/profiling/profile_latent_3d_unet_%j.out
#SBATCH --error=IDUN/output/profiling/profile_latent_3d_unet_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END

cd ${SLURM_SUBMIT_DIR}

# Create output directory if it doesn't exist
mkdir -p IDUN/output/profiling

echo "=== UNet VRAM Profiling for Latent Diffusion ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo ""
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

echo ""
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
echo ""

python misc/profiling/profile_latent_3d_unet.py

conda deactivate
