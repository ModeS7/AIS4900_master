#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp10_3_sit_dcae_2x2"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp10_3_sit_dcae_2x2_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp10_3_sit_dcae_2x2_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== EXP10_3: 3D-SiT with DC-AE 2x2x512 Latent Space ==="
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

# DC-AE checkpoints (128x spatial compression, 2x2 latent, 512 channels)
BRAVO_ENCODER=/cluster/work/modestas/AIS4900_master/runs/compression_2d/multi_modality/exp9_f128_20260106-165531/checkpoint_latest.pt
SEG_ENCODER=/cluster/work/modestas/AIS4900_master/runs/compression_2d/seg/exp13_3_seg_f128_20260201-025617/latest.pt

echo "DC-AE Configuration:"
echo "  Compression: 128x spatial (256x256 -> 2x2)"
echo "  Latent channels: 512"
echo "  Latent volume: 2x2xD with 512 channels"
echo ""
echo "Encoders:"
echo "  BRAVO: $BRAVO_ENCODER"
echo "  SEG:   $SEG_ENCODER"
echo ""
echo "Mode: bravo_seg_cond_dcae"
echo "  in_channels:  1024 = [bravo_latent(512), seg_latent(512)]"
echo "  out_channels: 512 = velocity prediction for bravo_latent"
echo ""
echo "Model: 3D-SiT-B (Scalable Interpolant Transformer)"
echo "  input_size: 2 (spatial)"
echo "  depth_size: D (preserved from input)"
echo "  patch_size: 1"
echo ""

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo_seg_cond \
    mode.in_channels=1024 \
    mode.out_channels=512 \
    mode.latent_channels=512 \
    model=sit_3d \
    model.variant=B \
    model.patch_size=1 \
    latent.enabled=true \
    latent.compression_type=dcae \
    latent.scale_factor=128 \
    latent.compression_checkpoint=${BRAVO_ENCODER} \
    latent.seg_compression_checkpoint=${SEG_ENCODER} \
    latent.slicewise_encoding=true \
    training.name=exp10_3_sit_b_p1_dcae_2x2x512_

conda deactivate
