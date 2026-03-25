#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="tshift_exp14_1"
#SBATCH --output=IDUN/output/eval/eval_time_shift_exp14_1_%j.out
#SBATCH --error=IDUN/output/eval/eval_time_shift_exp14_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp14_1: 3D Pixel Seg (unconditional seg generation, 256x256x160)
# Golden section search for optimal time-shift ratio
# Two runs: morphological metric (25 steps) and FID metric (37 steps)

cd ${SLURM_SUBMIT_DIR}

echo "=== Time-Shift Search: exp14_1 (3D Pixel Seg) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUT_DIR="/cluster/work/modestas/MedicalDataSets/eval_time_shift_exp14_1"
COMMON_ARGS="--checkpoint ${CKPT} --data-root ${DATA_ROOT} --mode seg --ref-modality seg --image-size 256 --num-volumes 25 --seed 42 --search --search-lo 1.0 --search-hi 5.0 --search-tol 0.1"

echo ""
echo "=========================================="
echo "  exp14_1: Optimal shift by Morphological (25 steps)"
echo "=========================================="
time python -m medgen.scripts.eval_time_shift \
    --output-dir "${OUT_DIR}_morph" \
    --metric morphological \
    --ref-split all \
    --num-steps 25 \
    ${COMMON_ARGS}
echo "--- Morphological results ---"
cat "${OUT_DIR}_morph/time_shift_results.json"

echo ""
echo "=========================================="
echo "  exp14_1: Optimal shift by FID (37 steps)"
echo "=========================================="
time python -m medgen.scripts.eval_time_shift \
    --output-dir "${OUT_DIR}_fid" \
    --metric fid \
    --ref-split all \
    --num-steps 37 \
    ${COMMON_ARGS}
echo "--- FID results ---"
cat "${OUT_DIR}_fid/time_shift_results.json"

echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: exp14_1"
echo "=========================================="
echo "--- Morphological ---"
cat "${OUT_DIR}_morph/time_shift_results.json"
echo ""
echo "--- FID ---"
cat "${OUT_DIR}_fid/time_shift_results.json"

conda deactivate
