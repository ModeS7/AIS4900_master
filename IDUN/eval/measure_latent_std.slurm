#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="latent_std"
#SBATCH --output=IDUN/output/eval/measure_latent_std_%j.out
#SBATCH --error=IDUN/output/eval/measure_latent_std_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Measure VAE Latent Statistics ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# === CHECKPOINTS TO MEASURE ===
# Edit these paths to point to your VAE checkpoints

EXP14_0="runs/compression_3d/multi_modality/exp14_0_vae3d_4x_20260214-233543/checkpoint_best.pt"
EXP14_3="runs/compression_3d/multi_modality/exp14_3_vae3d_4x_lowkl_20260217-023208/checkpoint_best.pt"

echo ""
echo "=========================================="
echo "exp14_0 (kl_weight=1e-6)"
echo "=========================================="
python -m medgen.scripts.measure_latent_std \
    +checkpoint="${EXP14_0}" \
    paths=cluster \
    mode=multi_modality \
    +max_samples=50

echo ""
echo "=========================================="
echo "exp14_3 (kl_weight=1e-8)"
echo "=========================================="
python -m medgen.scripts.measure_latent_std \
    +checkpoint="${EXP14_3}" \
    paths=cluster \
    mode=multi_modality \
    +max_samples=50

echo "=== Done ==="
conda deactivate
