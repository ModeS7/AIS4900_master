#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_21_2"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp21_2_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp21_2_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp21_2: LDM 4x MAISI UNet 167M, bravo_seg_cond (256x256x160), RFlow
# Multi-metric search: FID + RadImageNet FID + PCA in one run

cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp21_2 (LDM 4x MAISI UNet) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo_latent/exp21_2_ldm_4x_unet_maisi_20260304-083519/checkpoint_latest.pt"
COMP_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp21_2"

if [ ! -f "$CKPT" ]; then
    CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo_latent/exp21_2_ldm_4x_unet_maisi_20260304-083519/checkpoint_best.pt"
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found for exp21_2"
    exit 1
fi

echo "Checkpoint: $CKPT"
echo "Compression: $COMP_CKPT"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_DIR}" \
    --metric fid,fid_radimagenet,pca \
    --data-root "${DATA_ROOT}" \
    --image-size 256 \
    --num-volumes 25 \
    --lo 10 --hi 100 \
    --ref-split test \
    --seed 42 \
    --space latent \
    --compression-checkpoint "${COMP_CKPT}"

echo ""
echo "=== Results ==="
cat "${OUT_DIR}/search_results.json" 2>/dev/null

conda deactivate
