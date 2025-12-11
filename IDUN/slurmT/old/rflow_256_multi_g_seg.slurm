#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu40g|gpu80g"
#SBATCH --job-name="rflow_256_images_multi_g_seg"
#SBATCH --output=IDUN/output/train/rflow_256_multi_seg_%j.out
#SBATCH --error=IDUN/output/train/rflow_256_multi_seg_%j.err
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
echo "Number of nodes:   $SLURM_JOB_NUM_NODES"
echo "Cores per node:    $SLURM_CPUS_ON_NODE"
echo "Total tasks:       $SLURM_NTASKS"
echo "Tasks per node:    $SLURM_NTASKS_PER_NODE"
echo "GPUs:              $CUDA_VISIBLE_DEVICES"

# Check GPU
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 nvidia-smi

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
time srun python Generation/RFlow/rflow_multi_gpu.py --epochs 500 --val_interval 30 --compute 'cluster' --image_size 256 --image_type seg --patience 90

# Deactivate conda environment
conda deactivate
