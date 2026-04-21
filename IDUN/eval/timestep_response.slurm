#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="t_response"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/timestep_response_%j.out
#SBATCH --error=IDUN/output/eval/timestep_response_%j.err

# Timestep-response diagnostic on exp1_1_1000.
# Sweeps 1000 t values × 10 real volumes, plots velocity MSE vs t and its
# derivative. Used to detect timestep-embedding pathologies, sudden
# discontinuities, or narrow "training gap" regions of t-space.

cd ${SLURM_SUBMIT_DIR}

echo "=== Timestep response diagnostic ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE_MIB" -lt 60000 ]; then
    echo "ERROR: Only ${FREE_MIB} MiB free on GPU. Aborting."
    exit 78
fi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_DIR="/cluster/work/modestas/AIS4900_master/runs/eval/timestep_response_${TIMESTAMP}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

time python -m medgen.scripts.analyze_timestep_response \
    --checkpoint "${CKPT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --seg-split test1 \
    --num-volumes 10 \
    --num-t 1000 \
    --t-min 0.001 \
    --t-max 0.999 \
    --depth 160

echo "Results saved to: ${OUT_DIR}"
