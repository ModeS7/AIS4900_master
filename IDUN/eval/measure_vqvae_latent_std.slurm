#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="vqvae_std"
#SBATCH --output=IDUN/output/eval/measure_vqvae_latent_std_%j.out
#SBATCH --error=IDUN/output/eval/measure_vqvae_latent_std_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Measure VQ-VAE Latent Statistics (all exp8) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Find all exp8 VQ-VAE runs and measure each
FOUND=0
for dir in runs/compression_3d/multi_modality/exp8_*/ runs/compression_3d/multi_modality/exp8_2*/; do
    CKPT="${dir}checkpoint_latest.pt"
    if [ ! -f "${CKPT}" ]; then
        echo "SKIP: ${dir} (no checkpoint_latest.pt)"
        continue
    fi

    FOUND=$((FOUND + 1))
    NAME=$(basename "${dir}")

    echo ""
    echo "=========================================="
    echo "${NAME}"
    echo "=========================================="
    python -m medgen.scripts.measure_latent_std \
        +checkpoint="${CKPT}" \
        paths=cluster \
        mode=multi_modality \
        +max_samples=50
done

if [ ${FOUND} -eq 0 ]; then
    echo "ERROR: No exp8 checkpoints found in runs/compression_3d/multi_modality/"
    exit 1
fi

echo ""
echo "=== Done (${FOUND} checkpoints measured) ==="
conda deactivate
