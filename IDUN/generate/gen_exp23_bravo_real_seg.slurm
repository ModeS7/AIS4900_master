#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_exp23_rs"
#SBATCH --output=IDUN/output/generate/gen_exp23_bravo_real_seg_%j.out
#SBATCH --error=IDUN/output/generate/gen_exp23_bravo_real_seg_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

# Generate bravo volumes using exp23 (ScoreAug 1000ep) conditioned on REAL seg masks
# RadImageNet FID-optimized: 48 steps, shift 1.14
# No seg generation — uses real masks from training + test1 data

CHAIN_MAX=10
CHAIN_COUNT=${CHAIN_COUNT:-0}
OUTPUT_SUBDIR="exp23_bravo_real_seg"

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

echo "=== Generate: exp23 bravo with REAL seg masks | Chain: ${CHAIN_COUNT}/${CHAIN_MAX} ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

BRAVO_CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt"
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"

# Count existing generated volumes for resume
GEN_DIR="/cluster/work/modestas/MedicalDataSets/generated/${OUTPUT_SUBDIR}"
if [ -d "${GEN_DIR}" ]; then
    CURRENT=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
else
    CURRENT=0
fi
echo "Resuming from sample ${CURRENT}"

# Use real seg masks from train split (105 volumes)
# num_images = number of real masks available
time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=bravo \
    image_size=256 \
    depth=160 \
    image_model="${BRAVO_CKPT}" \
    real_seg_dir="${DATA_ROOT}/train" \
    num_images=525 \
    current_image=${CURRENT} \
    num_steps_bravo=48 \
    shift_ratio_bravo=1.14 \
    validate_size_bins=false \
    validate_brain_mask=true \
    brain_pca_path=auto \
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
