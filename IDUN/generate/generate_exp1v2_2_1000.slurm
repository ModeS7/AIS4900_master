#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_1v2_2"
#SBATCH --output=IDUN/output/generate/generate_exp1v2_2_1000_%j.out
#SBATCH --error=IDUN/output/generate/generate_exp1v2_2_1000_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Generate 10 volumes with exp1v2_2_1000 (dual mode: T1pre + T1gd) at the
# ImageNet-FID-optimum step count (69 steps, FID=65.67 from find_optimal_steps).
# No time-shift (shift tuning showed no improvement).
# Uses real test1 seg masks for conditioning (same as compare_imagenet_optima
# to keep generations comparable with the existing 7-model set).

cd ${SLURM_SUBMIT_DIR}

echo "=== Generate exp1v2_2_1000 (dual, 69 steps, ImageNet FID optimum) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE_MIB" -lt 60000 ]; then
    echo "ERROR: Only ${FREE_MIB} MiB free on GPU. Expected >=60GB. Aborting."
    exit 78
fi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
GEN_ROOT="/cluster/work/modestas/MedicalDataSets/generated"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_SUBDIR="exp1v2_2_1000_dual_${TIMESTAMP}"
GEN_DIR="${GEN_ROOT}/${OUT_SUBDIR}"

CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_2_1000_pixel_dual_20260411-024806/checkpoint_latest.pt"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT}"
    exit 1
fi
mkdir -p "${GEN_DIR}"

# Dual generation: 2-channel output (T1pre + T1gd), conditioned on real seg
# from test1 split. num_steps_bravo is the Euler step count (69 = ImageNet FID optimum).
time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=dual \
    image_size=256 \
    depth=160 \
    image_model="${CKPT}" \
    +real_seg_dir="${DATA_ROOT}/test1" \
    num_images=10 \
    num_steps_bravo=69 \
    validate_size_bins=false \
    validate_brain_mask=true \
    brain_atlas_path=auto \
    output_subdir="${OUT_SUBDIR}" \
    verbose=true

cnt=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | wc -l)
echo ""
echo "=== Done ==="
echo "Generated ${cnt}/10 volumes at: ${GEN_DIR}"
echo ""
echo "Sync to local:"
echo "  rsync -avh cluster:${GEN_DIR}/ /home/mode/NTNU/MedicalDataSets/generated/${OUT_SUBDIR}/"
