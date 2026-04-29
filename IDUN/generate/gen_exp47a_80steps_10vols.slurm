#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_47a"
#SBATCH --output=IDUN/output/generate/gen_exp47a_80steps_%j.out
#SBATCH --error=IDUN/output/generate/gen_exp47a_80steps_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Generate 10 bravo volumes using exp47a at 80 steps.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

OUTPUT_SUBDIR="exp47a_bravo_80steps_10"

echo "=== Generate: exp47a bravo, 80 steps, 10 vols ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

SEG_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
BRAVO_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp47a_lpips_strong_20260425-055252/checkpoint_latest.pt"

[ ! -f "$SEG_CKPT" ]   && { echo "ERROR: seg ckpt not found: $SEG_CKPT";   exit 1; }
[ ! -f "$BRAVO_CKPT" ] && { echo "ERROR: bravo ckpt not found: $BRAVO_CKPT"; exit 1; }
echo "seg:   $SEG_CKPT"
echo "bravo: $BRAVO_CKPT"

GEN_DIR="${CLUSTER_BASE}/MedicalDataSets/generated/${OUTPUT_SUBDIR}"
if [ -d "${GEN_DIR}" ]; then
    CURRENT=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
else
    CURRENT=0
fi
echo "Resuming from sample ${CURRENT}/10"

time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=bravo \
    image_size=256 \
    depth=160 \
    seg_model="${SEG_CKPT}" \
    image_model="${BRAVO_CKPT}" \
    num_images=10 \
    current_image=${CURRENT} \
    num_steps_seg=25 \
    num_steps_bravo=80 \
    shift_ratio_seg=3.47 \
    shift_ratio_bravo=1.0 \
    validate_size_bins=false \
    brain_atlas_path=auto \
    +brain_pca_path=auto \
    +seg_pca_path=auto \
    validate_brain_mask=true \
    output_subdir="${OUTPUT_SUBDIR}" \
    verbose=true

if [ -d "${GEN_DIR}" ]; then
    FINAL=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
    echo "Generated ${FINAL}/10 volumes at: ${GEN_DIR}"
fi
