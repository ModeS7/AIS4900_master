#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="stoch_samp"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/stochastic_sampling_%j.out
#SBATCH --error=IDUN/output/eval/stochastic_sampling_%j.err

# Stochastic-Euler ablation on exp1_1_1000.
# Does adding per-step Gaussian noise during Euler sampling (same model,
# no retraining) break the MSE posterior-mean collapse we diagnosed?
#
# Grid: σ ∈ {0.0, 0.01, 0.03, 0.05, 0.1}, 5 paired seeds, 32 steps.
# σ=0.0 is the deterministic baseline (matches production).

cd ${SLURM_SUBMIT_DIR}

echo "=== Stochastic Sampling ablation on exp1_1_1000 ==="
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
OUT_DIR="/cluster/work/modestas/AIS4900_master/runs/eval/stochastic_sampling_${TIMESTAMP}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

time python -m medgen.scripts.analyze_stochastic_sampling \
    --checkpoint "${CKPT}" \
    --data-root "${DATA_ROOT}" \
    --real-dir "${DATA_ROOT}/test1" \
    --output-dir "${OUT_DIR}" \
    --num-seeds 5 \
    --num-steps 32 \
    --sigmas 0.0 0.01 0.03 0.05 0.1 \
    --depth 160 \
    --seg-split test1 \
    --seed-base 42

echo "Results saved to: ${OUT_DIR}"
