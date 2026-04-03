#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="test_pca"
#SBATCH --output=IDUN/output/eval/test_pca_wdm_ldm_%j.out
#SBATCH --error=IDUN/output/eval/test_pca_wdm_ldm_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Quick test: 1 WDM + 1 LDM model, 5 volumes, 10 steps
cd ${SLURM_SUBMIT_DIR}
echo "=== Quick PCA Test: WDM + LDM ==="
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
PCA_MODEL="/cluster/work/modestas/AIS4900_master/data/brain_pca_256x256x160.npz"
VQVAE_CKPT="/cluster/work/modestas/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt"
OUT="/cluster/work/modestas/MedicalDataSets/test_pca_wdm_ldm"

echo ""
echo "=== WDM: exp26_1 (5 vol, 10 steps) ==="
time python -m medgen.scripts.eval_steps_pca \
    --checkpoint "/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_1_wdm_1000ep_20260327-024744/checkpoint_latest.pt" \
    --data-root "${DATA_ROOT}" \
    --pca-model "${PCA_MODEL}" \
    --output-dir "${OUT}/exp26_1" \
    --image-size 256 --depth 160 \
    --num-volumes 5 \
    --step-list 10 \
    --space wavelet \
    --seed 42

echo ""
echo "=== LDM: exp21_2 (5 vol, 10 steps) ==="
time python -m medgen.scripts.eval_steps_pca \
    --checkpoint "/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp21_2_ldm_4x_unet_maisi_20260304-083519/checkpoint_latest.pt" \
    --data-root "${DATA_ROOT}" \
    --pca-model "${PCA_MODEL}" \
    --output-dir "${OUT}/exp21_2" \
    --image-size 256 --depth 160 \
    --num-volumes 5 \
    --step-list 10 \
    --space latent \
    --compression-checkpoint "${VQVAE_CKPT}" \
    --seed 42

echo ""
echo "=== DONE ==="

conda deactivate
