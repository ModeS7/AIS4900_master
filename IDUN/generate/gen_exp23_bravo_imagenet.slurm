#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_exp23_IN"
#SBATCH --output=IDUN/output/generate/gen_exp23_bravo_imagenet_%j.out
#SBATCH --error=IDUN/output/generate/gen_exp23_bravo_imagenet_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Generate 525 bravo volumes using exp23 (ScoreAug 1000ep)
# ImageNet FID-optimized: 27 steps, shift 1.22
# Seg model: exp14_1 (unconditional seg), 25 steps, shift 3.47

cd ${SLURM_SUBMIT_DIR}

echo "=== Generate: exp23 bravo (ImageNet-optimized) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

SEG_CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
BRAVO_CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt"

time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=bravo \
    strategy=rflow \
    image_size=256 \
    depth=160 \
    seg_model="${SEG_CKPT}" \
    image_model="${BRAVO_CKPT}" \
    num_images=525 \
    num_steps_seg=25 \
    num_steps_bravo=27 \
    shift_ratio_seg=3.47 \
    shift_ratio_bravo=1.22 \
    output_subdir="exp23_bravo_imagenet_525" \
    verbose=true

conda deactivate
