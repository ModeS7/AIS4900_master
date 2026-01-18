#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp1_mem_profile"
#SBATCH --output=IDUN/output/debug/exp1_memory_profile_%j.out
#SBATCH --error=IDUN/output/debug/exp1_memory_profile_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Memory Profiling: Exp1 Bravo (finding 61GB VRAM usage) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Memory Profiling Run
# - Same config as exp1_pixel_bravo but with memory_profiling=true
# - Only 5 epochs (enough to see the memory pattern)
# ============================================================================

time python -m medgen.scripts.train_diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=default_3d \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.batch_size=1 \
    training.epochs=5 \
    training.learning_rate=1e-4 \
    training.gradient_checkpointing=true \
    training.memory_profiling=true \
    training.name=exp1_memory_profile_

conda deactivate
