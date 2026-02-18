#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="profile_pixel"
#SBATCH --output=IDUN/output/profiling/profile_pixel_3d_unet_%j.out
#SBATCH --error=IDUN/output/profiling/profile_pixel_3d_unet_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END

cd ${SLURM_SUBMIT_DIR}

mkdir -p IDUN/output/profiling

echo "=== Pixel-Space 3D UNet VRAM Profiling ==="
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

export PYTORCH_ALLOC_CONF=expandable_segments:True

python misc/profiling/profile_pixel_3d_unet.py

conda deactivate
