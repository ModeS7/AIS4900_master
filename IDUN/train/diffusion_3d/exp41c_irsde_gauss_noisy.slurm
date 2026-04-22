#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp41c"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp41c/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp41c/%j.err
#SBATCH --signal=B:SIGUSR1@120

# exp41c: IR-SDE restoration — augmented synthetic degradation.
# Same base idea as exp41a but adds randomization:
#   - blur_sigma widened to [0.35, 0.50] from [0.37, 0.47] — covers exp37 family
#     (σ≈0.29) through exp1_1_1000 (σ=0.41) for cross-model transferability
#   - noise_sigma ∈ [0.003, 0.008] — approximates texture-noise artifacts that
#     pure Gaussian misses (low-freq overshoot residual from spectrum fit)
#
# Hypothesis: exp41a vs exp41c isolates whether degradation randomization
# improves SR generalization. If exp41c markedly outperforms, augmentation
# matters. If they're similar, pure spectrum-match is sufficient.
#
# Paired experiments:
#   exp41a — pure Gaussian (narrow σ, no noise), same IR-SDE config
#   exp41b — real pre-generated pairs (most expensive, most realistic)

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp41c_irsde_gauss_noisy"

if [ -z "${CHAIN_RUN_DIR:-}" ]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    CHAIN_RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/restoration/${EXP_NAME}_${TIMESTAMP}"
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

echo "=== Exp41c: IR-SDE restoration (Gaussian σ=0.42 synthetic, from scratch) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Spectrum-matched Gaussian degradation: σ ∈ [0.37, 0.47] bracket around fitted σ=0.42.
# No additive noise, no downsample — pure Gaussian to match the fitted transfer function.
time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=irsde \
    mode=restoration \
    model=default_3d \
    model.spatial_dims=3 \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=80 \
    training.learning_rate=3e-5 \
    training.warmup_epochs=5 \
    training.eta_min=1e-7 \
    training.gradient_clip_norm=0.5 \
    '+restoration.degradation.type=synthetic' \
    '+restoration.degradation.blur_sigma=[0.35,0.50]' \
    '+restoration.degradation.noise_sigma=[0.003,0.008]' \
    +restoration.degradation.downsample_upsample_prob=0.0 \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
