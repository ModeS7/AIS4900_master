#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-14:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp9_1_ldm_4x"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp9_1_ldm_4x_bravo_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp9_1_ldm_4x_bravo_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp9.1 LDM_4x BRAVO: Mid-size UNet (500 epochs in ~13h) ==="
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

# VQ-VAE 4x compression checkpoint (exp8_1)
COMPRESSION_CHECKPOINT=/cluster/work/modestas/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt

echo "Using compression checkpoint: $COMPRESSION_CHECKPOINT"
echo ""
echo "Mode: bravo_seg_cond (Latent BRAVO conditioned on Latent SEG)"
echo "  in_channels:  8 = [bravo_latent(4), seg_latent(4)]"
echo "  out_channels: 4 = noise prediction for bravo_latent"
echo ""
echo "UNet Architecture: 4-level MID-SIZE (for 4x latent, depth=40)"
echo "  channels:         [128, 256, 512, 1024]"
echo "  attention_levels: [False, False, True, True]"
echo "  num_res_blocks:   [2, 2, 2, 2]"
echo "  num_head_channels: 32"
echo ""

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    mode=bravo_seg_cond \
    training.gradient_clip_norm=0.5 \
    training.warmup_epochs=10 \
    latent.enabled=true \
    latent.compression_checkpoint=${COMPRESSION_CHECKPOINT} \
    latent.compression_type=vqvae \
    latent.auto_encode=true \
    latent.validate_cache=true \
    'model.channels=[128, 256, 512, 1024]' \
    'model.attention_levels=[false, false, true, true]' \
    'model.num_res_blocks=[2, 2, 2, 2]' \
    model.num_head_channels=32 \
    model.norm_num_groups=32 \
    training.name=exp9_1_ldm_4x_bravo_

conda deactivate
