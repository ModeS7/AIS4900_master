#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp41b"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp41b/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp41b/%j.err
#SBATCH --signal=B:SIGUSR1@120

# exp41b Phase 2: IR-SDE restoration trained on REAL pre-generated pairs.
# Depends on exp41b_pregen_real_pairs.slurm completing first.
#
# Training set: restoration_pairs_real/{train,val}/<subj>/
#   {clean_bravo.nii.gz, seg.nii.gz, degraded_{000,001,002}.nii.gz}
#
# Unlike exp41a (synthetic Gaussian σ=0.42 degradation), this version sees the
# ACTUAL exp1_1_1000 output distribution — so both spectrum AND phase match the
# inference-time task. Addresses the "spectrum match ≠ phase match" risk
# identified in the SR feasibility analysis.

CHAIN_MAX=20
CHAIN_COUNT=${CHAIN_COUNT:-0}
EXP_NAME="exp41b_irsde_real_pairs"

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

echo "=== Exp41b: IR-SDE restoration (REAL precomputed pairs from exp1_1_1000) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST} | Chain: ${CHAIN_COUNT}/${CHAIN_MAX}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Sanity check: pair data must exist (run pregen SLURM first)
RESTORATION_ROOT="/cluster/work/modestas/MedicalDataSets/restoration_pairs_real"
if [ ! -d "${RESTORATION_ROOT}/train" ]; then
    echo "ERROR: ${RESTORATION_ROOT}/train does not exist."
    echo "Run exp41b_pregen_real_pairs.slurm first to generate the pairs."
    exit 1
fi
N_TRAIN=$(find ${RESTORATION_ROOT}/train -maxdepth 1 -type d -mindepth 1 | wc -l)
echo "Found ${N_TRAIN} train subjects in pair store"
if [ "${N_TRAIN}" -lt 50 ]; then
    echo "ERROR: too few paired training subjects (${N_TRAIN})"
    exit 1
fi

# NOTE: no degradation CLI override — the loader defaults to precomputed mode
# when no restoration.degradation override is set. Points paths.restoration_data_dir
# at the real pair store.
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
    +paths.restoration_data_dir=${RESTORATION_ROOT} \
    training.name=${EXP_NAME}_ \
    ${CHAIN_ARGS} &

PID=$!
wait $PID
