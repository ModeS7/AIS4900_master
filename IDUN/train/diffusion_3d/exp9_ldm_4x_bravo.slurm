#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp9_ldm_4x_bravo"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp9_ldm_4x_bravo_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp9_ldm_4x_bravo_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== LDM_4x BRAVO: Latent BRAVO Generation Conditioned on Latent Seg ==="
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
echo "UNet Architecture: LDM_4x"
echo "  channels:         [256, 512, 1024, 2048]"
echo "  attention_levels: [False, False, True, True]"
echo "  num_res_blocks:   [3, 3, 3, 3]"
echo "  num_head_channels: 64"
echo ""

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo_seg_cond \
    training.epochs=1000 \
    training.learning_rate=1e-4 \
    training.batch_size=1 \
    training.gradient_checkpointing=true \
    latent.enabled=true \
    latent.compression_checkpoint=${COMPRESSION_CHECKPOINT} \
    latent.compression_type=vqvae \
    latent.auto_encode=true \
    latent.validate_cache=true \
    'model.channels=[256, 512, 1024, 2048]' \
    'model.attention_levels=[false, false, true, true]' \
    'model.num_res_blocks=[3, 3, 3, 3]' \
    model.num_head_channels=64 \
    model.norm_num_groups=32 \
    training.logging.regional_losses=true \
    training.generation_metrics.enabled=true \
    training.generation_metrics.samples_per_epoch=1 \
    training.generation_metrics.samples_extended=4 \
    training.generation_metrics.samples_test=10 \
    training.name=exp9_ldm_4x_bravo_

conda deactivate
