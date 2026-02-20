#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="optim_exp14_1"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp14_1_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp14_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Euler Steps: exp14_1 (3D Pixel Seg) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# === PATHS ===
CHECKPOINT="runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUTPUT_DIR="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp14_1"

# exp14_1 is mode=seg (unconditional 3D seg generation, in_ch=1, out_ch=1)
# Auto-detected from checkpoint; --mode seg as explicit fallback
time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --mode seg \
    --num-volumes 25 \
    --lo 10 --hi 50 \
    --metric fid \
    --seed 42

echo "=== Search complete ==="
cat "${OUTPUT_DIR}/search_results.json"

conda deactivate
