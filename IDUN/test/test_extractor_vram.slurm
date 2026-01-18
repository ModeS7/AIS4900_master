#!/bin/bash
#SBATCH --job-name=vram_test
#SBATCH --output=misc/vram_test_%j.out
#SBATCH --error=misc/vram_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load modules
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0

# Activate virtual environment
source ~/.virtualenvs/medgen/bin/activate

# Change to project directory
cd /home/mode/NTNU/AIS4900_master

# Print environment info
echo "=========================================="
echo "VRAM Test on Cluster"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
nvidia-smi
echo "=========================================="

# Run the basic test
echo ""
echo "Running basic extractor test..."
python misc/test_extractor_vram.py

# Run the full test (includes LPIPS)
echo ""
echo "Running full test (all compiled models)..."
python misc/test_extractor_vram.py full

# Run the 3D workload test
echo ""
echo "Running 3D workload test..."
python misc/test_extractor_vram.py 3d

echo ""
echo "=========================================="
echo "Test completed at $(date)"
echo "=========================================="
