#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --job-name="performance_benchmark"
#SBATCH --output=IDUN/output/benchmark_%j.out
#SBATCH --error=IDUN/output/benchmark_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Change to submission directory
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

# Print job information
echo "============================================================"
echo "Performance Benchmark Job"
echo "============================================================"
echo "Job Name:          $SLURM_JOB_NAME"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodes used:        $SLURM_JOB_NODELIST"
echo "GPUs:              $CUDA_VISIBLE_DEVICES"
echo "============================================================"
echo ""

# Check GPU
nvidia-smi

# Load modules and activate environment
module purge
module load Anaconda3/2024.02-1

# Activate conda environment
conda activate AIS4005

# Verify PyTorch can see GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

# Set PYTHONPATH
export PYTHONPATH=$SLURM_SUBMIT_DIR:$PYTHONPATH

echo ""
echo "============================================================"
echo "Starting Benchmark - 128x128"
echo "============================================================"
echo ""

# Run benchmark for 128x128 (both training and generation)
time python misc/bench/benchmark.py \
    --mode both \
    --image_size 128 \
    --batch_size 16 \
    --compute cluster \
    --train_epochs 10 \
    --gen_iterations 10 \
    --gen_batch_size 200 \
    --gen_steps 1000 \
    --output misc/bench/benchmark_results || exit 1

echo ""
echo "============================================================"
echo "Starting Benchmark - 256x256"
echo "============================================================"
echo ""

# Run benchmark for 256x256 (both training and generation)
time python misc/bench/benchmark.py \
    --mode both \
    --image_size 256 \
    --batch_size 16 \
    --compute cluster \
    --train_epochs 10 \
    --gen_iterations 10 \
    --gen_batch_size 32 \
    --gen_steps 1000 \
    --output misc/bench/benchmark_results || exit 1

echo ""
echo "============================================================"
echo "All Benchmarks Complete (2 total)"
echo "============================================================"

# Deactivate conda environment
conda deactivate
