#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="profile_small"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/profile_small_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/profile_small_%j.err

# Profile small 3D UNet architectures — VRAM sweep
# ~30 configs x 2 (with/without GC) = ~60 runs, ~1 min each

cd ${SLURM_SUBMIT_DIR}

echo "=== Small 3D UNet VRAM Profiler ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

time python misc/profiling/profile_small_3d_unet.py
