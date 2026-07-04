#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp52a"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp52a/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp52a/%j.err
#SBATCH --signal=B:SIGUSR1@120

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# exp52a (was exp32_2_3d_1000): copy of exp32_2_1000 with the perceptual loss swapped from the 2D
#   RadImageNet-ResNet50 (center-slice) to a TRUE-3D whole-volume MedicalNet-ResNet10
#   (medicalnet_resnet10_23datasets). Everything else identical: fine-tune exp1_1_1000,
#   LPIPS gated at t<250, 1000ep.
# perceptual_weight bumped 0.1 -> 2.0: the MedicalNet loss is ~20x smaller in scale than
#   RadImageNet (benchmarked: ~0.087 vs ~1.88), so 2.0 matches the SAME effective
#   contribution as exp32_2 (0.1*1.88 ~= 2.0*0.087 ~= 0.19). Same perceptual strength,
#   3D features -- not an overscale.
# NOTE: MedicalNet weights auto-download from HuggingFace — pre-cache on the LOGIN node
#   (one warm-up with internet) before submitting, like BiomedCLIP, or compute nodes
#   fail offline.

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp52a_exp32_2_medicalnet3d"
PRETRAINED="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/${EXP_NAME}_${TIMESTAMP}"
fi

CHAIN_ARGS="hydra.run.dir=${CHAIN_RUN_DIR}"
CKPT="${CHAIN_RUN_DIR}/checkpoint_latest.pt"
if [ -f "${CKPT}" ] && [ -s "${CKPT}" ]; then
    CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CKPT}"
    echo "Resuming from own checkpoint: ${CKPT}"
else
    CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${PRETRAINED} +training.reset_scheduler=true"
    echo "Continuing from: ${PRETRAINED} (reset scheduler for new cosine cycle)"
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

echo "=== Exp52a (exp32_2 + 3D MedicalNet perceptual): LPIPS low-t fine-tune with TRUE-3D MedicalNet perceptual loss ==="
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
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=1000 \
    training.learning_rate=1e-5 \
    training.warmup_epochs=10 \
    training.gradient_clip_norm=0.5 \
    training.perceptual_weight=2.0 \
    training.perceptual_max_timestep=250 \
    training.perceptual_network_type=medicalnet_resnet10_23datasets \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
