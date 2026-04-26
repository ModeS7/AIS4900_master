#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp44b"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp44b/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp44b/%j.err
#SBATCH --signal=B:SIGUSR1@120

# exp44b: Pix2Pix refinement on synthetic-blur pairs + STRONG extra augmentation.
#
# Same premise as exp44 (synthetic Gaussian-blur pairs, raw_synth at inference)
# but:
#   - REMOVED feature-matching loss (non-stationary D features → unstable G).
#   - REMOVED residual learning (amplifies OOD artifacts on raw_synth input).
#   - ADDED strong on-the-fly aug to widen the training corruption distribution
#     so it covers raw_synth's actual statistics (σ_max=0.6, noise_std=0.025,
#     prob=0.8 — heavier than exp43b's 0.4/0.015/0.5).
#   - LOWERED adv_weight 0.3 → 0.05 and increased d-warmup-steps 500 → 2000
#     to fix the GAN instability that plagued exp44.
#
# Seeds from exp42b's last checkpoint (clean Pix2Pix ancestor, same data,
# no residual/FM bad weights). Continues +100 epochs.
#
# Differentiator from exp43b in the thesis 3-way:
#   exp43b: real VQ-VAE artifact pairs, synth_rt input,    aug σ=0.4
#   exp44b: synthetic Gauss-blur pairs,  raw_synth input,  aug σ=0.6 (this one)

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp44b_pix2pix_aug"

# Seed: most recent exp42b run dir, checkpoint_latest.pt
SEED_RUN=$(ls -td /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/refinement/exp42b_pix2pix_refinement_* 2>/dev/null | head -1)
SEED_CKPT="${SEED_RUN}/checkpoint_latest.pt"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/refinement/${EXP_NAME}_${TIMESTAMP}"
fi

RESUME_ARG=""
OWN_CKPT="${CHAIN_RUN_DIR}/checkpoint_latest.pt"
if [ -f "${OWN_CKPT}" ] && [ -s "${OWN_CKPT}" ]; then
    RESUME_ARG="--resume-from ${OWN_CKPT}"
    echo "Resuming from own checkpoint: ${OWN_CKPT}"
elif [ -n "${SEED_RUN}" ] && [ -f "${SEED_CKPT}" ] && [ -s "${SEED_CKPT}" ]; then
    RESUME_ARG="--resume-from ${SEED_CKPT}"
    echo "First launch: seeding from exp42b checkpoint: ${SEED_CKPT}"
else
    echo "ERROR: neither own checkpoint nor exp42b seed checkpoint found."
    echo "  own:  ${OWN_CKPT}"
    echo "  seed: ${SEED_CKPT}"
    exit 1
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

echo "=== Exp44b: Pix2Pix on synth-blur pairs + strong extra aug ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/restoration_pairs_real"

if [ ! -d "${DATA_ROOT}/train" ]; then
    echo "ERROR: paired training data not found at ${DATA_ROOT}/train"
    echo "Run exp41b_pregen_real_pairs.slurm first to generate the pairs."
    exit 1
fi
N_TRAIN=$(find "${DATA_ROOT}/train" -maxdepth 1 -type d -mindepth 1 | wc -l)
echo "Found ${N_TRAIN} train subjects in pair store"

mkdir -p "${CHAIN_RUN_DIR}"

time python -m medgen.scripts.train_refinement_gan \
    --data-root "${DATA_ROOT}" \
    --output-dir "${CHAIN_RUN_DIR}" \
    --image-size 256 \
    --depth 160 \
    --augmentation-level heavy \
    --extra-blur-sigma-max 0.6 \
    --extra-noise-std-max 0.025 \
    --extra-aug-prob 0.8 \
    --epochs 200 \
    --batch-size 1 \
    --lr-g 1e-4 \
    --lr-d 5e-5 \
    --warmup-epochs 3 \
    --gradient-clip-norm 1.0 \
    --lambda-l1 100.0 \
    --lambda-perc 10.0 \
    --adv-weight 0.05 \
    --d-warmup-steps 2000 \
    --adv-ramp-steps 2000 \
    --disc-num-layers 2 \
    --disc-num-channels 16 \
    --r1-weight 0.1 \
    --r1-every 8 \
    --ema-beta 0.9999 \
    --log-every 20 \
    --val-every-epoch 1 \
    --save-every-epoch 5 \
    --seed 42 \
    ${RESUME_ARG} &

PID=$!
wait $PID
