#!/bin/sh
#SBATCH --partition=GPUQ                    # GPU partition
#SBATCH --account=share-ie-idi              # High priority account
#SBATCH --time=3-00:00:00                   # Time
#SBATCH --nodes=1                           # Single node
#SBATCH --cpus-per-task=16                  # CPU cores
#SBATCH --mem=32G                           # Allocate RAM
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --constraint=gpu80g                 # Need gpu with 80GB memory
#SBATCH --job-name="base_256_images"        # Job name
#SBATCH --output=IDUN/output/base_256_%j.out# Output file (%j = job ID)
#SBATCH --error=IDUN/output/base_256_%j.err # Error file
#SBATCH --mail-user=modestas@stud.ntnu.no   # Email
#SBATCH --mail-type=ALL                     # Email notifications

# Change to submission directory
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

# Print job information
echo "Job Name:          $SLURM_JOB_NAME"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodes used:        $SLURM_JOB_NODELIST"
echo "Number of nodes:   $SLURM_JOB_NUM_NODES"
echo "Cores per node:    $SLURM_CPUS_ON_NODE"
echo "Total cores:       $SLURM_NTASKS"
echo "GPUs:              $CUDA_VISIBLE_DEVICES"

# Check GPU
nvidia-smi

# Load modules and activate environment
module purge
module load Anaconda3/2024.02-1

# Activate conda environment
conda activate AIS4005

# Verify PyTorch can see GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current GPU: {torch.cuda.current_device()}'); print(f'GPU name: {torch.cuda.get_device_name(0)}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

# Set PYTHONPATH if needed
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

# Run training script
time python Generation/Diffusion/diffusion.py --epochs 500 --val_interval 30 --compute 'cluster' --patience 90 --image_size 256 --image_type seg

# Deactivate conda environment
conda deactivate