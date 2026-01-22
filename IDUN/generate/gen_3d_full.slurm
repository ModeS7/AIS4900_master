#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_3d_full"
#SBATCH --output=IDUN/output/generate/gen_3d_full_%j.out
#SBATCH --error=IDUN/output/generate/gen_3d_full_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== 3D Volume Generation: Full Pipeline ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Full Pipeline: size_bins -> seg_mask (exp2) -> bravo (exp1)
# ============================================================================
#
# IMPORTANT: Update checkpoint paths to your trained models!
#
# exp1 (bravo): Generates bravo images conditioned on seg masks
# exp2 (seg_conditioned): Generates seg masks conditioned on size bins
#
# ============================================================================

# === UPDATE THESE PATHS ===
SEG_CHECKPOINT="runs/exp2_pixel_seg_sizebin_.../checkpoint_best.pt"
BRAVO_CHECKPOINT="runs/exp1_pixel_bravo_.../checkpoint_best.pt"
OUTPUT_DIR="generated/3d_full_pipeline"
# ==========================

NUM_SAMPLES=100
NUM_STEPS=50

# Volume dimensions (should match training)
HEIGHT=128
WIDTH=128
DEPTH=160

time python -m medgen.scripts.generate \
    --spatial_dims 3 \
    --mode bravo \
    --seg_model ${SEG_CHECKPOINT} \
    --image_model ${BRAVO_CHECKPOINT} \
    --output_dir ${OUTPUT_DIR} \
    --num_images ${NUM_SAMPLES} \
    --num_steps ${NUM_STEPS} \
    --image_size ${HEIGHT} \
    --depth ${DEPTH} \
    --min_tumors 1 \
    --max_tumors 5 \
    --no_progress

conda deactivate
