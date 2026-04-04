#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_exp22_2"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp22_2_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp22_2_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp22_2: LDM 4x DiT-L 478M, bravo (256x256x160), RFlow
# Architecture: DiT-L 478M in 4x compressed latent space
# Mode: bravo_seg_cond, latent diffusion (normalize=false)
# Strategy: RFlow (continuous timesteps)
# Runs: ImageNet FID sweep, then RadImageNet FID sweep

cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp22_2 (LDM 4x DiT-L bravo 256x256x160) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_2_ldm_4x_dit_l_20260304-100612/checkpoint_latest.pt"
COMP_CKPT="/cluster/work/modestas/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt"

BASE_ARGS="--data-root ${DATA_ROOT} --image-size 256 --num-volumes 25 --lo 10 --hi 50 --ref-split test --seed 42 --space latent --compression-checkpoint ${COMP_CKPT}"

if [ ! -f "$CKPT" ]; then
    CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp22_2_ldm_4x_dit_l_20260304-100612/checkpoint_best.pt"
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found for exp22_2, aborting"
    exit 1
fi

echo "Checkpoint: $CKPT"
echo "Compression: $COMP_CKPT"

# ============================================================
# 1) Optimize by ImageNet FID
# ============================================================
echo ""
echo "=========================================="
echo "  exp22_2: Optimal steps by ImageNet FID"
echo "=========================================="

OUT_FID="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp22_2_fid"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_FID}" \
    --metric fid \
    ${BASE_ARGS}

echo "--- ImageNet FID results ---"
cat "${OUT_FID}/search_results.json"

# ============================================================
# 2) Optimize by RadImageNet FID
# ============================================================
echo ""
echo "=========================================="
echo "  exp22_2: Optimal steps by RadImageNet FID"
echo "=========================================="

OUT_FID_RIN="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp22_2_fid_rin"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_FID_RIN}" \
    --metric fid_radimagenet \
    ${BASE_ARGS}

echo "--- RadImageNet FID results ---"
cat "${OUT_FID_RIN}/search_results.json"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: exp22_2"
echo "=========================================="
echo "--- ImageNet FID ---"
cat "${OUT_FID}/search_results.json" 2>/dev/null
echo ""
echo "--- RadImageNet FID ---"
cat "${OUT_FID_RIN}/search_results.json" 2>/dev/null
echo ""

conda deactivate
