#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp43b"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp43b/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp43b/%j.err
#SBATCH --signal=B:SIGUSR1@120

# exp43b: VQ-VAE deblur refinement with extra blur+noise augmentation.
# Same data + arch as exp43, FROM-SCRATCH 200-epoch training (not a continuation
# of exp43). The augmentation widens the corruption distribution at training
# time so the model learns to handle inputs spanning real_rt → synth_rt.

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp43b_vqvae_deblur_aug"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/refinement/${EXP_NAME}_${TIMESTAMP}"
fi

RESUME_ARG=""
CKPT="${CHAIN_RUN_DIR}/checkpoint_latest.pt"
if [ -f "${CKPT}" ] && [ -s "${CKPT}" ]; then
    RESUME_ARG="--resume-from ${CKPT}"
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

echo "=== Exp43b: VQ-VAE deblur + blur/noise aug, fresh 200-epoch run ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/vqvae_pairs"

if [ ! -d "${DATA_ROOT}/train" ]; then
    echo "ERROR: paired training data not found at ${DATA_ROOT}/train"
    echo "Run precompute_vqvae_pairs.slurm first."
    exit 1
fi
N_TRAIN=$(find "${DATA_ROOT}/train" -maxdepth 1 -type d -mindepth 1 | wc -l)
N_VAL=$(find "${DATA_ROOT}/val" -maxdepth 1 -type d -mindepth 1 2>/dev/null | wc -l)
echo "Train subjects: ${N_TRAIN}  |  Val subjects: ${N_VAL}"

mkdir -p "${CHAIN_RUN_DIR}"

time python -m medgen.scripts.train_refinement_gan \
    --data-root "${DATA_ROOT}" \
    --output-dir "${CHAIN_RUN_DIR}" \
    --image-size 256 \
    --depth 160 \
    --augmentation-level heavy \
    --extra-blur-sigma-max 0.4 \
    --extra-noise-std-max 0.015 \
    --extra-aug-prob 0.5 \
    --epochs 200 \
    --batch-size 1 \
    --lr-g 1e-4 \
    --lr-d 5e-5 \
    --warmup-epochs 3 \
    --gradient-clip-norm 1.0 \
    --lambda-l1 100.0 \
    --lambda-perc 10.0 \
    --adv-weight 0.1 \
    --d-warmup-steps 500 \
    --adv-ramp-steps 2000 \
    --disc-num-layers 2 \
    --disc-num-channels 16 \
    --r1-weight 0.1 \
    --r1-every 4 \
    --ema-beta 0.9999 \
    --log-every 20 \
    --val-every-epoch 1 \
    --save-every-epoch 5 \
    --seed 42 \
    ${RESUME_ARG} &

PID=$!
wait $PID
