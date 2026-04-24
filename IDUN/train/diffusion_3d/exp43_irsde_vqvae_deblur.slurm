#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp43i"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp43_irsde/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp43_irsde/%j.err
#SBATCH --signal=B:SIGUSR1@120

# exp43 IR-SDE variant: diffusion-based restoration on (real, real_rt) VQ-VAE
# pairs. Different family from the existing exp43_vqvae_deblur.slurm (Pix2Pix
# GAN). Trained on the same vqvae_pairs/ data.
#
# Why a diffusion-restoration approach: stochastic SDE sampling avoids the
# posterior-mean collapse (mean-blur) inherent to deterministic regression
# (L1/MSE/Pix2Pix). Each inference run produces a different sample from
# p(real | real_rt), not the mean.
#
# Mirrors exp41b_irsde_real_pairs.slurm config but points at vqvae_pairs/
# instead of restoration_pairs_real/. Depends on precompute_vqvae_pairs.slurm
# completing first.

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp43_irsde_vqvae_deblur"

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

echo "=== Exp43 IR-SDE: VQ-VAE pair deblur (diffusion family) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

RESTORATION_ROOT="/cluster/work/modestas/MedicalDataSets/vqvae_pairs"
if [ ! -d "${RESTORATION_ROOT}/train" ]; then
    echo "ERROR: ${RESTORATION_ROOT}/train does not exist."
    echo "Run precompute_vqvae_pairs.slurm first."
    exit 1
fi
N_TRAIN=$(find ${RESTORATION_ROOT}/train -maxdepth 1 -type d -mindepth 1 | wc -l)
echo "Found ${N_TRAIN} train subjects in vqvae pair store"
if [ "${N_TRAIN}" -lt 50 ]; then
    echo "ERROR: too few paired training subjects (${N_TRAIN})"
    exit 1
fi

# Loader auto-uses precomputed mode when degraded_*.nii.gz files exist
# (no degradation override needed). Points at vqvae_pairs/.
time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=irsde \
    mode=restoration \
    model=default_3d \
    model.spatial_dims=3 \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=100 \
    training.learning_rate=3e-5 \
    training.warmup_epochs=5 \
    training.eta_min=1e-7 \
    training.gradient_clip_norm=0.5 \
    +paths.restoration_data_dir=${RESTORATION_ROOT} \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
