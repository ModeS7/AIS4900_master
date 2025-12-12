#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="medgen_train"
#SBATCH --output=IDUN/output/train_%j.out
#SBATCH --error=IDUN/output/train_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Change to submission directory
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

# Print job information
echo "Job Name:          $SLURM_JOB_NAME"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodes used:        $SLURM_JOB_NODELIST"
echo "GPUs:              $CUDA_VISIBLE_DEVICES"

# Check GPU
nvidia-smi

# Load modules and activate environment
module purge
module load Anaconda3/2024.02-1
conda activate AIS4005

# Verify PyTorch can see GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# ============================================================================
# TRAINING EXAMPLES (uncomment one):
# ============================================================================

# DDPM 128 bravo (default)
time python -m medgen.scripts.train paths=cluster

# RFlow 128 bravo
# time python -m medgen.scripts.train paths=cluster strategy=rflow

# DDPM 128 seg (unconditional mask generation)
# time python -m medgen.scripts.train paths=cluster mode=seg

# DDPM 256 dual (T1 pre + post)
# time python -m medgen.scripts.train paths=cluster mode=dual model=unet_256

# Custom overrides
# time python -m medgen.scripts.train paths=cluster training.epochs=1000 training.batch_size=32

# ============================================================================
# MULTI-GPU (change SBATCH to: --gres=gpu:4, --ntasks-per-node=4, --mem=128G)
# ============================================================================
# time srun python -m medgen.scripts.train paths=cluster training.use_multi_gpu=true model=unet_256

conda deactivate
