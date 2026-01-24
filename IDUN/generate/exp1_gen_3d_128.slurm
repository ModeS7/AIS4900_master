#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp1_gen_3d_128"
#SBATCH --output=IDUN/output/generate/exp1_gen_3d_128_%j.out
#SBATCH --error=IDUN/output/generate/exp1_gen_3d_128_%j.err
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

# === UPDATE THESE PATHS ===
SEG_CHECKPOINT="runs/diffusion_3d/seg_conditioned/exp2_pixel_seg_sizebin_rflow_128x160_20260121-163331/checkpoint_latest.pt"
BRAVO_CHECKPOINT="runs/diffusion_3d/bravo/exp1_pixel_bravo_rflow_128x160_20260119-212710/checkpoint_latest.pt"
# ==========================

time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=bravo \
    seg_model=${SEG_CHECKPOINT} \
    image_model=${BRAVO_CHECKPOINT} \
    output_subdir=exp1_3d_128_100_baseline \
    num_images=100 \
    num_steps=100 \
    image_size=128 \
    depth=160 \
    cfg_scale_seg=2.0 \
    cfg_scale_bravo=2.0 \
    min_tumors=1 \
    max_tumors=5 \
    verbose=false

conda deactivate
