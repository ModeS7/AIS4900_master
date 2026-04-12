#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_deg"
#SBATCH --output=IDUN/output/generate/gen_degradation_pairs_%j.out
#SBATCH --error=IDUN/output/generate/gen_degradation_pairs_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

# Generate SDEdit degradation pairs for restoration training.
# Creates 4 degraded variants per volume across train/val/test splits.
# Uses --resume to continue after timeout/resubmission.
#
# NOTE: Update --t0-min and --t0-max after calibration results are in!

CHAIN_MAX=5
CHAIN_COUNT=${CHAIN_COUNT:-0}

# Calibrated (2026-04-12): best t₀=0.50, range ±0.15
T0_MIN=${T0_MIN:-0.35}
T0_MAX=${T0_MAX:-0.65}

handle_timeout() {
    echo "=== Wall time approaching (SIGUSR1). Resubmitting... ==="
    kill $PID 2>/dev/null
    wait $PID 2>/dev/null

    NEXT_COUNT=$((CHAIN_COUNT + 1))
    if [ "$NEXT_COUNT" -ge "$CHAIN_MAX" ]; then
        echo "Chain limit reached (${CHAIN_MAX}). Stopping."
        exit 1
    fi

    sbatch --export=ALL,CHAIN_COUNT=${NEXT_COUNT},T0_MIN=${T0_MIN},T0_MAX=${T0_MAX} "$0"
    echo "Resubmitted chain segment ${NEXT_COUNT}/${CHAIN_MAX}"
    exit 0
}

trap handle_timeout SIGUSR1

cd ${SLURM_SUBMIT_DIR}

echo "=== Generate Degradation Pairs for Restoration Training ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
echo "t0 range: [${T0_MIN}, ${T0_MAX}]"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

BRAVO_CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUT_DIR="/cluster/work/modestas/MedicalDataSets/restoration_pairs"

time python -m medgen.scripts.generate_degradation_pairs \
    --bravo-model "${BRAVO_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --t0-min ${T0_MIN} \
    --t0-max ${T0_MAX} \
    --num-degradations 4 \
    --num-steps 32 \
    --splits "train,val,test_new" \
    --depth 160 \
    --image-size 256 \
    --seed 42 \
    --resume &

PID=$!
wait $PID

conda deactivate
