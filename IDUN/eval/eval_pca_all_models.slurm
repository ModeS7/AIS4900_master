#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="pca_all"
#SBATCH --output=IDUN/output/eval/eval_pca_all_models_%j.out
#SBATCH --error=IDUN/output/eval/eval_pca_all_models_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Evaluate PCA brain shape quality for all candidate models
# 25 volumes, 20 steps, real seg masks from train split
# Compares pixel-space models directly

cd ${SLURM_SUBMIT_DIR}

echo "=== PCA Brain Shape Evaluation: All Models ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
PCA_MODEL="/cluster/work/modestas/AIS4900_master/data/brain_pca_256x256x160.npz"
OUT_BASE="/cluster/work/modestas/MedicalDataSets/eval_pca_all"
VQVAE_CKPT="/cluster/work/modestas/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt"
STEPS=20
VOLS=25

# === Pixel-space models (in=2 [bravo+seg], out=1) ===
PIXEL_MODELS=(
    "exp1_1_1000:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt"
    "exp23:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt"
    "exp24:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239/checkpoint_latest.pt"
    "exp25:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp25_pixel_bravo_combined_20m_20260315-193139/checkpoint_latest.pt"
)

for entry in "${PIXEL_MODELS[@]}"; do
    NAME="${entry%%:*}"
    CKPT="${entry#*:}"

    if [ ! -f "$CKPT" ]; then
        echo "SKIP $NAME: checkpoint not found"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  Pixel: $NAME (20 steps)"
    echo "=========================================="

    time python -m medgen.scripts.eval_steps_pca \
        --checkpoint "${CKPT}" \
        --data-root "${DATA_ROOT}" \
        --pca-model "${PCA_MODEL}" \
        --output-dir "${OUT_BASE}/${NAME}" \
        --image-size 256 --depth 160 \
        --num-volumes ${VOLS} \
        --step-list ${STEPS} \
        --shift-ratio 1.0 \
        --seed 42 \
        --space pixel \
        --compute-fid --ref-split test1
done

# === WDM models (wavelet space) ===
WDM_MODELS=(
    "exp26:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_wdm_scoreaug_20260325-013334/checkpoint_latest.pt"
    "exp26_1:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_1_wdm_1000ep_20260327-024744/checkpoint_latest.pt"
)

for entry in "${WDM_MODELS[@]}"; do
    NAME="${entry%%:*}"
    CKPT="${entry#*:}"

    if [ ! -f "$CKPT" ]; then
        echo "SKIP $NAME: checkpoint not found"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  WDM: $NAME (20 steps)"
    echo "=========================================="

    time python -m medgen.scripts.eval_steps_pca \
        --checkpoint "${CKPT}" \
        --data-root "${DATA_ROOT}" \
        --pca-model "${PCA_MODEL}" \
        --output-dir "${OUT_BASE}/${NAME}" \
        --image-size 256 --depth 160 \
        --num-volumes ${VOLS} \
        --step-list ${STEPS} \
        --shift-ratio 1.0 \
        --seed 42 \
        --space wavelet \
        --compute-fid --ref-split test1
done

# === LDM models (latent space, need VQ-VAE decoding) ===
LDM_MODELS=(
    "exp21_2:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp21_2_ldm_4x_unet_maisi_20260304-083519/checkpoint_latest.pt"
    "exp22_2:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_2_ldm_4x_dit_l_20260304-100612/checkpoint_latest.pt"
    "exp27:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp27_ldm_4x_dit_l_scoreaug_20260325-190518/checkpoint_latest.pt"
    "exp28:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp28_ldm_4x_unet_maisi_scoreaug_20260325-013810/checkpoint_latest.pt"
    "exp28_1:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp28_1_ldm_4x_unet_maisi_scoreaug_mixup_20260325-211501/checkpoint_latest.pt"
    "exp28_2:/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp28_2_ldm_4x_unet_maisi_scoreaug_v2_20260326-144538/checkpoint_latest.pt"
)

for entry in "${LDM_MODELS[@]}"; do
    NAME="${entry%%:*}"
    CKPT="${entry#*:}"

    if [ ! -f "$CKPT" ]; then
        echo "SKIP $NAME: checkpoint not found"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  LDM: $NAME (20 steps)"
    echo "=========================================="

    time python -m medgen.scripts.eval_steps_pca \
        --checkpoint "${CKPT}" \
        --data-root "${DATA_ROOT}" \
        --pca-model "${PCA_MODEL}" \
        --output-dir "${OUT_BASE}/${NAME}" \
        --image-size 256 --depth 160 \
        --num-volumes ${VOLS} \
        --step-list ${STEPS} \
        --shift-ratio 1.0 \
        --seed 42 \
        --space latent \
        --compression-checkpoint "${VQVAE_CKPT}" \
        --compute-fid --ref-split test1
done

# === Summary ===
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY"
echo "=========================================="
for dir in ${OUT_BASE}/*/; do
    NAME=$(basename "$dir")
    JSON="${dir}pca_step_results.json"
    if [ -f "$JSON" ]; then
        MEAN=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['results'][0]['mean_error']:.6f}\")")
        PASS=$(python3 -c "import json; d=json.load(open('$JSON')); print(f\"{d['results'][0]['pass_rate']:.0%}\")")
        echo "  ${NAME}: mean_error=${MEAN}, pass_rate=${PASS}"
    fi
done

conda deactivate
