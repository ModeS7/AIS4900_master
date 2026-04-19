#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="e33_5"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp33_5/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp33_5/%j.err
#SBATCH --signal=B:SIGUSR1@120

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# exp33_5: Diffusion Bridge restoration — 17M UNet, full volumes
# Zhang et al. "Diffusion Bridge Models for 3D Medical Image Translation" (2025)
# Forward: x_t = (1-t)*clean + t*degraded + γ_max*sqrt(4t(1-t))*ε
# γ_max = 0.125 (validated on 3D brain MRI), 40 sampling steps
# Model predicts x̂_0 (clean image), MSE loss (unweighted)
# Key advantage: noise bubble prevents memorization of degraded→clean mapping

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp33_5_bridge_restoration_17m"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/restoration/${EXP_NAME}_${TIMESTAMP}"
fi

CHAIN_ARGS="hydra.run.dir=${CHAIN_RUN_DIR}"
CKPT="${CHAIN_RUN_DIR}/checkpoint_latest.pt"
if [ -f "${CKPT}" ] && [ -s "${CKPT}" ]; then
    CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CKPT}"
    echo "Resuming from own checkpoint: ${CKPT}"
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

echo "=== Exp33_5: Diffusion Bridge 17M full-volume restoration ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=bridge \
    mode=restoration \
    model=default_3d \
    'model.channels=[8, 16, 32, 64, 128, 128]' \
    'model.attention_levels=[false, false, false, false, true, true]' \
    'model.num_res_blocks=[1, 1, 1, 2, 2, 2]' \
    model.num_head_channels=8 \
    model.norm_num_groups=8 \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.augment=false \
    training.generation_metrics.enabled=false \
    training.gradient_checkpointing=false \
    +paths.restoration_data_dir=${CLUSTER_BASE}/MedicalDataSets/restoration_pairs \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.warmup_epochs=10 \
    training.gradient_clip_norm=0.5 \
    +restoration.eval_gen_imagenet_dir=${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525 \
    +restoration.eval_gen_radimagenet_dir=${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_radimagenet_525 \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
