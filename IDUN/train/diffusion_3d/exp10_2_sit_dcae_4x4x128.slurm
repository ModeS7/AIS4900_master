#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp10_2_sit_dcae_4x4"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp10_2_sit_dcae_4x4_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp10_2_sit_dcae_4x4_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== EXP10_2: 3D-SiT with DC-AE 4x4x128 Latent Space ==="
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

# DC-AE checkpoints (64x spatial compression, 4x4 latent, 128 channels)
BRAVO_ENCODER=/cluster/work/modestas/AIS4900_master/runs/compression_2d/multi_modality/exp9_f64_20251231-175024/latest.pt
SEG_ENCODER=/cluster/work/modestas/AIS4900_master/runs/compression_2d/seg/exp13_2_seg_f64_20260201-024544/checkpoint_latest.pt

echo "DC-AE Configuration:"
echo "  Compression: 64x spatial (256x256 -> 4x4)"
echo "  Latent channels: 128"
echo "  Latent volume: 4x4xD with 128 channels"
echo ""
echo "Encoders:"
echo "  BRAVO: $BRAVO_ENCODER"
echo "  SEG:   $SEG_ENCODER"
echo ""
echo "Mode: bravo_seg_cond_dcae"
echo "  in_channels:  256 = [bravo_latent(128), seg_latent(128)]"
echo "  out_channels: 128 = velocity prediction for bravo_latent"
echo ""
echo "Model: 3D-SiT-B (Scalable Interpolant Transformer)"
echo "  input_size: 4 (spatial)"
echo "  depth_size: D (preserved from input)"
echo "  patch_size: 1"
echo ""

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo_seg_cond \
    mode.in_channels=256 \
    mode.out_channels=128 \
    mode.latent_channels=128 \
    model=sit_3d \
    model.variant=B \
    model.patch_size=1 \
    latent.enabled=true \
    latent.compression_type=dcae \
    latent.scale_factor=64 \
    latent.compression_checkpoint=${BRAVO_ENCODER} \
    latent.seg_compression_checkpoint=${SEG_ENCODER} \
    latent.slicewise_encoding=true \
    training.name=exp10_2_sit_b_p1_dcae_4x4x128_

conda deactivate
