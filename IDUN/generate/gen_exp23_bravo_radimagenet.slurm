#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_exp23_RIN"
#SBATCH --output=IDUN/output/generate/gen_exp23_bravo_radimagenet_%j.out
#SBATCH --error=IDUN/output/generate/gen_exp23_bravo_radimagenet_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

# Generate 525 bravo volumes using exp23 (ScoreAug 1000ep)
# RadImageNet FID-optimized: 48 steps, shift 1.14
# Seg model: exp14_1 (unconditional seg), 25 steps, shift 3.47

CHAIN_MAX=10
CHAIN_COUNT=${CHAIN_COUNT:-0}
OUTPUT_SUBDIR="exp23_bravo_radimagenet_525"

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

echo "=== Generate: exp23 bravo (RadImageNet-optimized) | Chain: ${CHAIN_COUNT}/${CHAIN_MAX} ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

SEG_CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
BRAVO_CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt"

# Count existing generated volumes for resume
GEN_DIR="/cluster/work/modestas/MedicalDataSets/generated/${OUTPUT_SUBDIR}"
if [ -d "${GEN_DIR}" ]; then
    CURRENT=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
else
    CURRENT=0
fi
echo "Resuming from sample ${CURRENT}/525"

time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=bravo \
    image_size=256 \
    depth=160 \
    seg_model="${SEG_CKPT}" \
    image_model="${BRAVO_CKPT}" \
    num_images=525 \
    current_image=${CURRENT} \
    num_steps_seg=25 \
    num_steps_bravo=48 \
    shift_ratio_seg=3.47 \
    shift_ratio_bravo=1.14 \
    validate_size_bins=false \
    brain_atlas_path=auto \
    +brain_pca_path=auto \
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
