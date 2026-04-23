#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_1v2_525"
#SBATCH --output=IDUN/output/generate/gen_exp1v2_2_1000_dual_525_%j.out
#SBATCH --error=IDUN/output/generate/gen_exp1v2_2_1000_dual_525_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# Dataset-scale generation: 525 dual volumes (T1pre + T1gd) with exp1v2_2_1000
# at ImageNet-FID-optimum step count (69). Generated seg masks (via exp14_1)
# with full pipeline filtering enabled — atlas, PCA (brain + seg), brain-mask
# validation, single-component check.
#
# Chain-resubmission pattern: on SIGUSR1 (wall limit approaching) the handler
# resubmits this SLURM; the script resumes from `current_image` based on
# existing output directories so no volume is regenerated.

CHAIN_MAX=10
CHAIN_COUNT=${CHAIN_COUNT:-0}
OUTPUT_SUBDIR="exp1v2_2_1000_dual_525"

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

echo "=== Generate: exp1v2_2_1000 dual (525 volumes) | Chain: ${CHAIN_COUNT}/${CHAIN_MAX} ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

SEG_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
DUAL_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt"

if [ ! -f "${SEG_CKPT}" ]; then
    echo "ERROR: seg checkpoint not found: ${SEG_CKPT}"
    exit 1
fi
if [ ! -f "${DUAL_CKPT}" ]; then
    echo "ERROR: dual checkpoint not found: ${DUAL_CKPT}"
    exit 1
fi

# Resume: count existing volumes
GEN_DIR="${CLUSTER_BASE}/MedicalDataSets/generated/${OUTPUT_SUBDIR}"
if [ -d "${GEN_DIR}" ]; then
    CURRENT=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
else
    CURRENT=0
fi
echo "Resuming from sample ${CURRENT}/525"

time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=dual \
    image_size=256 \
    depth=160 \
    seg_model="${SEG_CKPT}" \
    image_model="${DUAL_CKPT}" \
    num_images=525 \
    current_image=${CURRENT} \
    num_steps_seg=40 \
    num_steps_bravo=69 \
    validate_size_bins=false \
    brain_atlas_path=auto \
    +brain_pca_path=auto \
    +seg_pca_path=auto \
    validate_brain_mask=true \
    output_subdir="${OUTPUT_SUBDIR}" \
    verbose=true &

PID=$!
wait $PID

# Check if complete
if [ -d "${GEN_DIR}" ]; then
    FINAL=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
    echo "Generated ${FINAL}/525 volumes"
    if [ "$FINAL" -ge 525 ]; then
        echo "Generation complete!"
        exit 0
    fi
fi

conda deactivate
