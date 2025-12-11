#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="eval_rflow"
#SBATCH --output=IDUN/output/eval_rflow_%j.out
#SBATCH --error=IDUN/output/eval_rflow_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
## Uncomment to exclude bad nodes after identifying them:
#SBATCH --exclude=idun-06-[01-07]

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

# Set PYTHONPATH to include both project root and evaluations directory
export PYTHONPATH=$SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR/evaluations:$PYTHONPATH

# Change to evaluations directory
cd eval/obj_eval

# Run evaluation script with error handling
echo "============================================================"
echo "Evaluating bravo images (128x128)"
echo "============================================================"
time python run_all.py --datasets rflow_256_50 --image-type bravo --image-size 128 --use-cluster --real-data-name brainmetshare-3 || exit 1
time python run_all.py --datasets rflow_256_50 --image-type bravo --image-size 128 --use-cluster --no-positive-only --real-data-name brainmetshare-3 || exit 1

echo ""
echo "============================================================"
echo "Evaluating segmentation masks (128x128)"
echo "============================================================"
time python run_all.py --datasets rflow_256_50 --image-type seg --image-size 128 --use-cluster --real-data-name brainmetshare-3 || exit 1
time python run_all.py --datasets rflow_256_50 --image-type seg --image-size 128 --use-cluster --no-positive-only --real-data-name brainmetshare-3 || exit 1

echo ""
echo "============================================================"
echo "Evaluating bravo images (256x256)"
echo "============================================================"
time python run_all.py --datasets rflow_256_50 --image-type bravo --image-size 256 --use-cluster --real-data-name brainmetshare-3 || exit 1
time python run_all.py --datasets rflow_256_50 --image-type bravo --image-size 256 --use-cluster --no-positive-only --real-data-name brainmetshare-3 || exit 1

echo ""
echo "============================================================"
echo "Evaluating segmentation masks (256x256)"
echo "============================================================"
time python run_all.py --datasets rflow_256_50 --image-type seg --image-size 256 --use-cluster --real-data-name brainmetshare-3 || exit 1
time python run_all.py --datasets rflow_256_50 --image-type seg --image-size 256 --use-cluster --no-positive-only --real-data-name brainmetshare-3 || exit 1

echo ""
echo "============================================================"
echo "SUCCESS: All evaluations completed (8 total)"
echo "============================================================"

# Deactivate conda environment
conda deactivate
