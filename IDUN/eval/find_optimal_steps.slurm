#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="find_optimal_steps"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Euler Steps (Golden Section Search) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# === PATHS ===
BRAVO_CHECKPOINT="runs/diffusion_3d/bravo/exp1_1_pixel_bravo_rflow_256x160_20260119-212710/checkpoint_latest.pt"
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUTPUT_DIR="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps"

# Golden section search over [10, 50] — ~8-10 evaluations, ~2-3 hours
time python -m medgen.scripts.find_optimal_steps \
    --bravo-model "${BRAVO_CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-volumes 25 \
    --lo 10 --hi 50 \
    --metric fid \
    --seed 42

echo "=== Search complete ==="
cat "${OUTPUT_DIR}/search_results.json"

conda deactivate
