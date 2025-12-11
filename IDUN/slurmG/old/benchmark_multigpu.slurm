#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --constraint=h100
#SBATCH --job-name="benchmark_multigpu"
#SBATCH --output=IDUN/output/benchmark_multigpu_%j.out
#SBATCH --error=IDUN/output/benchmark_multigpu_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Change to submission directory
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

# Print job information
echo "============================================================"
echo "Multi-GPU Performance Benchmark Job"
echo "============================================================"
echo "Job Name:          $SLURM_JOB_NAME"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodes used:        $SLURM_JOB_NODELIST"
echo "Num tasks:         $SLURM_NTASKS"
echo "GPUs per node:     $SLURM_GPUS_PER_NODE"
echo "GPUs:              $CUDA_VISIBLE_DEVICES"
echo "============================================================"
echo ""

# Check GPUs
nvidia-smi

# Load modules and activate environment
module purge
module load Anaconda3/2024.02-1

# Activate conda environment
conda activate AIS4005

# Verify PyTorch can see GPUs
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Number of GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)') for i in range(torch.cuda.device_count())]"

# Set PYTHONPATH
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

# Set up distributed training environment
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$SLURM_NTASKS

echo ""
echo "============================================================"
echo "Distributed Training Configuration"
echo "============================================================"
echo "MASTER_ADDR:       $MASTER_ADDR"
echo "MASTER_PORT:       $MASTER_PORT"
echo "WORLD_SIZE:        $WORLD_SIZE"
echo "============================================================"
echo ""

echo "============================================================"
echo "Starting Multi-GPU Benchmark - 128x128"
echo "============================================================"
echo ""

# Run benchmark for 128x128 with srun for proper task distribution
time srun python misc/bench/benchmark.py \
    --mode both \
    --image_size 128 \
    --batch_size 16 \
    --compute cluster \
    --train_epochs 10 \
    --gen_iterations 10 \
    --gen_batch_size 200 \
    --gen_steps 1000 \
    --multi_gpu \
    --output misc/bench/benchmark_results || exit 1

echo ""
echo "============================================================"
echo "Starting Multi-GPU Benchmark - 256x256"
echo "============================================================"
echo ""

# Run benchmark for 256x256 with srun for proper task distribution
time srun python misc/bench/benchmark.py \
    --mode both \
    --image_size 256 \
    --batch_size 16 \
    --compute cluster \
    --train_epochs 10 \
    --gen_iterations 10 \
    --gen_batch_size 32 \
    --gen_steps 1000 \
    --multi_gpu \
    --output misc/bench/benchmark_results || exit 1

echo ""
echo "============================================================"
echo "All Multi-GPU Benchmarks Complete (2 total)"
echo "============================================================"

# Deactivate conda environment
conda deactivate
