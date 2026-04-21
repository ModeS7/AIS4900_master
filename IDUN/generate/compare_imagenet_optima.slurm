#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="cmp_optima"
#SBATCH --output=IDUN/output/generate/compare_imagenet_optima_%j.out
#SBATCH --error=IDUN/output/generate/compare_imagenet_optima_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Visual comparison: generate 10 volumes with each of 7 models — this time
# at each model's ImageNet FID-optimal step count (from find_optimal_steps
# searches). Removes the step-count confound present in compare_exp37_*:
# each model is evaluated where it was found to score best on ImageNet FID.
#
# ImageNet FID optima (from IDUN/output/eval/find_optimal_steps_*):
#   exp1_1_1000   =  32
#   exp32_1_1000  =  55
#   exp32_2_1000  =  80
#   exp32_3_1000  =  56
#   exp37_1       =  79
#   exp37_2       =  79
#   exp37_3       =  80
#
# Total compute: sum of steps ≈ 461 per seed × 10 seeds × ≈1.3s/step ≈ 100 min.

cd ${SLURM_SUBMIT_DIR}

echo "=== Compare 7 models @ ImageNet FID optima: 10 volumes each ==="
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
COMPARE_ROOT="${GEN_ROOT}/compare_imagenet_optima_${TIMESTAMP}"
mkdir -p "${COMPARE_ROOT}"

# Seven checkpoints
CKPT_EXP1_1_1000="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
CKPT_EXP32_1_1000="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt"
CKPT_EXP32_2_1000="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt"
CKPT_EXP32_3_1000="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt"
CKPT_EXP37_1="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_1_pixel_bravo_lpips_hight_20260418-002202/checkpoint_latest.pt"
CKPT_EXP37_2="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt"
CKPT_EXP37_3="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042/checkpoint_latest.pt"

# Per-model ImageNet FID optima
STEPS_EXP1_1_1000=32
STEPS_EXP32_1_1000=55
STEPS_EXP32_2_1000=80
STEPS_EXP32_3_1000=56
STEPS_EXP37_1=79
STEPS_EXP37_2=79
STEPS_EXP37_3=80

run_model() {
    local label="$1"
    local ckpt="$2"
    local steps="$3"
    local out_subdir="compare_imagenet_optima_${TIMESTAMP}/${label}"
    local gen_dir="${GEN_ROOT}/${out_subdir}"

    echo ""
    echo "=========================================================="
    echo "  [${label}] Generating 10 volumes @ ${steps} steps"
    echo "  checkpoint: ${ckpt}"
    echo "=========================================================="

    if [ ! -f "${ckpt}" ]; then
        echo "  SKIP: checkpoint not found"
        return
    fi

    mkdir -p "${gen_dir}"

    time python -m medgen.scripts.generate \
        paths=cluster \
        spatial_dims=3 \
        gen_mode=bravo \
        image_size=256 \
        depth=160 \
        image_model="${ckpt}" \
        +real_seg_dir="${DATA_ROOT}/test1" \
        num_images=10 \
        num_steps_bravo=${steps} \
        validate_size_bins=false \
        validate_brain_mask=true \
        brain_atlas_path=auto \
        output_subdir="${out_subdir}" \
        verbose=true

    local final=$(find "${gen_dir}" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | wc -l)
    echo "  [${label}] done: ${final}/10 volumes at ${steps} steps"
}

# All seven — sequential on the same GPU
run_model "exp1_1_1000"  "${CKPT_EXP1_1_1000}"  "${STEPS_EXP1_1_1000}"
run_model "exp32_1_1000" "${CKPT_EXP32_1_1000}" "${STEPS_EXP32_1_1000}"
run_model "exp32_2_1000" "${CKPT_EXP32_2_1000}" "${STEPS_EXP32_2_1000}"
run_model "exp32_3_1000" "${CKPT_EXP32_3_1000}" "${STEPS_EXP32_3_1000}"
run_model "exp37_1"      "${CKPT_EXP37_1}"      "${STEPS_EXP37_1}"
run_model "exp37_2"      "${CKPT_EXP37_2}"      "${STEPS_EXP37_2}"
run_model "exp37_3"      "${CKPT_EXP37_3}"      "${STEPS_EXP37_3}"

echo ""
echo "=== All comparisons complete ==="
echo "Output root: ${COMPARE_ROOT}"
for spec in \
    "exp1_1_1000=${STEPS_EXP1_1_1000}" \
    "exp32_1_1000=${STEPS_EXP32_1_1000}" \
    "exp32_2_1000=${STEPS_EXP32_2_1000}" \
    "exp32_3_1000=${STEPS_EXP32_3_1000}" \
    "exp37_1=${STEPS_EXP37_1}" \
    "exp37_2=${STEPS_EXP37_2}" \
    "exp37_3=${STEPS_EXP37_3}"; do
    label="${spec%%=*}"
    steps="${spec#*=}"
    cnt=$(find "${GEN_ROOT}/compare_imagenet_optima_${TIMESTAMP}/${label}" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | wc -l)
    echo "  ${label} @ ${steps} steps: ${cnt}/10 volumes"
done

echo ""
echo "Sync to local for viewing:"
echo "  rsync -avh cluster:${COMPARE_ROOT}/ /home/mode/NTNU/MedicalDataSets/generated/compare_imagenet_optima_${TIMESTAMP}/"
