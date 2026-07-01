#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp51_tada"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp51/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp51/%j.err
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --exclude=idun-06-02

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# exp51: TADA prior-art comparison arm (Park et al., NeurIPS 2023).
#
# Same exp1 baseline + same augmentation set (medium_no_brightness) as exp49/exp50,
# but augmentation is gated by TADA's parabola-in-log10(SNR) instead of our one-sided
# detail window: STRONG at the near-clean (t<0.5) and near-noise (t>0.97) ends,
# SUPPRESSED in the intermediate 'content' band t in [0.5, 0.97] (min at t~0.9).
#   augmentation_tada_schedule=[-3.0, 0.0, 0.5]  -> [r_rough, r_fine, delta] in log10(SNR)
# Matched to dose 0.49 (== exp49c cutoff-0.5 / exp50c uniform) via max_prob=0.924, so the
# 3-way comparison at dose 0.49 is pure placement/shape: our detail-gate (exp49c) vs
# uniform (exp50c) vs TADA (this). Isolates whether TADA's two-sided log-SNR schedule
# beats our one-sided detail window at equal augmentation amount.
#
# CAVEAT (see docstring of _compute_tada_logsnr_weight): the [-3, 0] band edges are
# TADA's DDPM/2D-face calibration. At 128^3 the SNR scale is resolution-shifted
# (Hoogeboom 2023) — this is the 'as-published port', not a re-tuned optimum. Re-pin
# the dose to the observed exp49 pocket once results are in.

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp51_tada_p049"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/${EXP_NAME}_${TIMESTAMP}"
fi

CHAIN_ARGS="hydra.run.dir=${CHAIN_RUN_DIR}"
CKPT="${CHAIN_RUN_DIR}/checkpoint_latest.pt"
if [ -f "${CKPT}" ] && [ -s "${CKPT}" ]; then
    CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CKPT}"
    echo "Resuming from: ${CKPT}"
fi

handle_timeout() {
    echo "=== Wall time approaching (SIGUSR1). Saving and resubmitting... ==="
    kill -USR1 $PID 2>/dev/null
    wait $PID 2>/dev/null

    if [ -f "${CHAIN_RUN_DIR}/.training_complete" ]; then
        echo "Training complete. Not resubmitting."
        exit 0
    fi

    NEXT_COUNT=$((CHAIN_COUNT + 1))
    if [ "$NEXT_COUNT" -ge "$CHAIN_MAX" ]; then
        echo "Chain limit reached (${CHAIN_MAX}). Stopping."
        exit 1
    fi

    sbatch --export=ALL,CHAIN_RUN_DIR="${CHAIN_RUN_DIR}",CHAIN_COUNT=${NEXT_COUNT} "$0"
    echo "Resubmitted chain segment ${NEXT_COUNT}/${CHAIN_MAX}"
    exit 0
}

trap handle_timeout SIGUSR1

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp51: TADA parabola-in-log-SNR augmentation (dose-matched to 0.49) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=default_3d \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.use_compile=true \
    training.gradient_checkpointing=false \
    training.augment=false \
    training.augmentation_level=medium_no_brightness \
    'training.augmentation_tada_schedule=[-3.0,0.0,0.5]' \
    training.augmentation_max_prob=0.924 \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
