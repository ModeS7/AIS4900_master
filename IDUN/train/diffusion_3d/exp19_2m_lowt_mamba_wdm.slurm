#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp19_2m_lt"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp19_2m_lowt/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp19_2m_lowt/%j.err
#SBATCH --signal=B:SIGUSR1@120

# exp19_2m_lowt: exp19_2m (Mamba-XL WDM wavelet bravo) trained ONLY on the
# last 250 timesteps (low-t curriculum, t ∈ [0, 0.25] of the 1000-step DDPM
# schedule). Same low-t-only setup as the exp48 family but in wavelet space
# with Mamba backbone instead of pixel-space UNet.
#
# IMPORTANT: trained from scratch — there is no pretrained Mamba wavelet
# checkpoint to fine-tune from (unlike exp48 which started from
# exp32_2_1000). This means the model never learns high-t structure
# formation; expect generation from pure noise to fail. Useful as a
# refinement specialist for direct comparison against full-range exp19_2m.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp19_2m_lowt"

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

echo "=== Exp19_2m_lowt WDM Wavelet DDPM Mamba-XL low-t (t∈[0,0.25]): Bravo ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba_ssm CUDA: OK')" 2>/dev/null \
    || echo "WARNING: mamba_ssm CUDA not available — will fall back to slow path"

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=ddpm \
    strategy.prediction_type=sample \
    strategy.schedule=linear_beta \
    mode=bravo \
    model=mamba_3d \
    model.variant=XL \
    model.patch_size=4 \
    model.image_size=128 \
    model.depth_size=80 \
    model.window_size=8 \
    wavelet.enabled=true \
    wavelet.normalize=true \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.batch_size=1 \
    training.gradient_checkpointing=true \
    training.learning_rate=1e-4 \
    training.gradient_clip_norm=0.5 \
    training.warmup_epochs=10 \
    training.curriculum.enabled=true \
    training.curriculum.warmup_epochs=1 \
    training.curriculum.min_t_start=0.0 \
    training.curriculum.min_t_end=0.0 \
    training.curriculum.max_t_start=0.25 \
    training.curriculum.max_t_end=0.25 \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
