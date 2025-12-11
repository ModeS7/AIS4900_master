#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40g
#SBATCH --job-name="rflow_test"
#SBATCH --output=IDUN/output/gen/rflow_test_%j.out
#SBATCH --error=IDUN/output/gen/rflow_test_%j.err
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
time python Generation/RFlow/rflow_gen.py --num_steps 10 --batch_size 512 --compute cluster --seg_model RFlow_seg_128_20250919-132722/Epoch499_of_500 --bravo_model RFlow_bravo_128_20250919-013949/Epoch499_of_500 --output_dir test

# Deactivate conda environment
conda deactivate