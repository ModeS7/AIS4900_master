#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="profile_mamba"
#SBATCH --output=IDUN/output/profiling/profile_mamba_%j.out
#SBATCH --error=IDUN/output/profiling/profile_mamba_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END

cd ${SLURM_SUBMIT_DIR}
mkdir -p IDUN/output/profiling

echo "=== MambaDiff Architecture Profiling ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo ""
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

echo ""
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba_ssm CUDA: OK')" 2>/dev/null || echo "WARNING: mamba_ssm CUDA not available"
echo ""

echo "=== Without gradient checkpointing ==="
python misc/profile_mamba_memory.py

echo ""
echo "=== With gradient checkpointing ==="
python misc/profile_mamba_memory.py --grad-ckpt

conda deactivate
