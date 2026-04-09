#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_26_1"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp26_1_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp26_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp26_1: WDM 270M vanilla 1000ep (256x256x160), DDPM x0
# Multi-metric search: FID + RadImageNet FID + PCA in one run
# Wavelet space: Haar 3D, per-subband N(0,1) normalized

cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp26_1 (WDM 1000ep) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_1_wdm_1000ep_20260327-024744/checkpoint_latest.pt"
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUT_DIR="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp26_1"

if [ ! -f "$CKPT" ]; then
    CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp26_1_wdm_1000ep_20260327-024744/checkpoint_best.pt"
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found for exp26_1"
    exit 1
fi

echo "Checkpoint: $CKPT"
echo "Space: wavelet (Haar 3D, per-subband N(0,1))"
echo "Strategy: DDPM x0 prediction"

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
    --space wavelet \
    --strategy ddpm \
    --prediction-type sample \
    --schedule linear_beta

echo ""
echo "=== Results ==="
cat "${OUT_DIR}/search_results.json" 2>/dev/null

conda deactivate
