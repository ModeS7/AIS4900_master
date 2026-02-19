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
#SBATCH --job-name="prof_hdit_uvit"
#SBATCH --output=IDUN/output/profiling/profile_hdit_uvit_%j.out
#SBATCH --error=IDUN/output/profiling/profile_hdit_uvit_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END

cd ${SLURM_SUBMIT_DIR}

mkdir -p IDUN/output/profiling

echo "=== HDiT / U-ViT 3D VRAM Profiling @ 256x256x160 ==="
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python misc/profiling/profile_hdit_uvit_memory.py --grad-ckpt

conda deactivate
