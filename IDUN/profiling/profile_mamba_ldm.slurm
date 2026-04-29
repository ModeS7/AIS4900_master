#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="prof_mamba_ldm"
#SBATCH --output=IDUN/output/profiling/profile_mamba_ldm_%j.out
#SBATCH --error=IDUN/output/profiling/profile_mamba_ldm_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END

# Profile Mamba {S, B, L, XL} across 5 latent configs:
#   WDM        :    8 × 128 × 128 × 80
#   VQ-VAE     :    4 ×  64 ×  64 × 40
#   DC-AE 32×  :  128 ×   8 ×   8 × 160
#   DC-AE 64×  :  512 ×   4 ×   4 × 160
#   DC-AE 128× : 2048 ×   2 ×   2 × 160
#
# Reports: params, peak VRAM (fwd+bwd), fwd/bwd latency, OOM-safe.

cd ${SLURM_SUBMIT_DIR}
mkdir -p IDUN/output/profiling

echo "=== Mamba LDM Profiling (S/B/L/XL × 5 latent configs) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba_ssm CUDA: OK')" 2>/dev/null \
    || echo "WARNING: mamba_ssm CUDA not available"

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo ""
echo "=== Without gradient checkpointing ==="
python misc/profile_mamba_ldm.py

echo ""
echo "=== With gradient checkpointing ==="
python misc/profile_mamba_ldm.py --grad-ckpt

conda deactivate
