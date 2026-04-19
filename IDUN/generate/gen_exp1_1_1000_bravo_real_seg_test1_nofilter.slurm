#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen11_rs_nf"
#SBATCH --output=IDUN/output/generate/gen_exp1_1_1000_bravo_real_seg_test1_nofilter_%j.out
#SBATCH --error=IDUN/output/generate/gen_exp1_1_1000_bravo_real_seg_test1_nofilter_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# Generate exp1_1_1000 bravo on REAL test1 seg masks, NO PCA filter.
# Companion to gen_exp23_bravo_real_seg_test1_nofilter.slurm — same seg input,
# so the only varying factor is the bravo model. Together they form the
# controlled ablation dataset for the PCA filter.

CHAIN_MAX=5
CHAIN_COUNT=${CHAIN_COUNT:-0}
OUTPUT_SUBDIR="exp1_1_1000_bravo_real_seg_test1_nofilter"

handle_timeout() {
    echo "=== Wall time approaching. Resubmitting... ==="
    kill $PID 2>/dev/null
    wait $PID 2>/dev/null
    NEXT_COUNT=$((CHAIN_COUNT + 1))
    if [ "$NEXT_COUNT" -ge "$CHAIN_MAX" ]; then
        echo "Chain limit reached (${CHAIN_MAX}). Stopping."
        exit 1
    fi
    sbatch --export=ALL,CHAIN_COUNT=${NEXT_COUNT} "$0"
    echo "Resubmitted chain segment ${NEXT_COUNT}/${CHAIN_MAX}"
    exit 0
}

trap handle_timeout SIGUSR1

cd ${SLURM_SUBMIT_DIR}

echo "=== Generate: exp1_1_1000 bravo | test1 real segs | NO PCA filter | Chain: ${CHAIN_COUNT}/${CHAIN_MAX} ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

BRAVO_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"

GEN_DIR="${CLUSTER_BASE}/MedicalDataSets/generated/${OUTPUT_SUBDIR}"
if [ -d "${GEN_DIR}" ]; then
    CURRENT=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
else
    CURRENT=0
fi
echo "Resuming from sample ${CURRENT}/150"

# NO brain_pca_path — PCA filter disabled. Atlas + single-component kept.
# 32 steps = current ImageNet-FID optimum for exp1_1_1000 (repeat run, FID 20.84).
# Matches exp23's ImageNet-FID optimum (27/1.22) as closely as possible so the
# only meaningful difference between the two nofilter runs is the bravo model.
time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=bravo \
    image_size=256 \
    depth=160 \
    image_model="${BRAVO_CKPT}" \
    +real_seg_dir="${DATA_ROOT}/test1" \
    num_images=150 \
    current_image=${CURRENT} \
    num_steps_bravo=32 \
    validate_size_bins=false \
    validate_brain_mask=true \
    brain_atlas_path=auto \
    output_subdir="${OUTPUT_SUBDIR}" \
    verbose=true &

PID=$!
wait $PID

if [ -d "${GEN_DIR}" ]; then
    FINAL=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
    echo "Generated ${FINAL}/150 volumes"
    if [ "$FINAL" -ge 150 ]; then
        echo "Generation complete!"
        exit 0
    fi
fi

conda deactivate
