#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="cmp_37"
#SBATCH --output=IDUN/output/generate/compare_exp1_1_vs_exp37_%j.out
#SBATCH --error=IDUN/output/generate/compare_exp1_1_vs_exp37_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Visual comparison: generate 10 volumes with each of
#   exp1_1_1000     (baseline, MSE only)
#   exp32_1_1000    (FFL, low-t)
#   exp32_2_1000    (LPIPS, low-t)
#   exp32_3_1000    (Pseudo-Huber, low-t)
#   exp37_1         (LPIPS, high-t)
#   exp37_2         (LPIPS + FFL, high-t)
#   exp37_3         (LPIPS + FFL, high-t, short + heavier)
#
# All on identical real test1 seg masks (first 10 available), same step count,
# no PCA filter during generation — raw outputs from each model on the same input.
#
# Step count: 20 for all models. Identical step count + identical seg masks ensures
# the ONLY varying factor is the fine-tune objective. Seg masks are the first 10
# alphabetically from test1 (deterministic: all 7 runs get the same 10 masks).

cd ${SLURM_SUBMIT_DIR}

echo "=== Compare exp1_1_1000 vs exp32_{1,2,3} vs exp37_{1,2,3}: 10 volumes each ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

# Fail fast if GPU is occupied by a zombie process (see idun-01-05 VLLM issue)
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
COMPARE_ROOT="${GEN_ROOT}/compare_exp37_${TIMESTAMP}"
mkdir -p "${COMPARE_ROOT}"

# Seven checkpoints to compare
CKPT_EXP1_1_1000="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
CKPT_EXP32_1_1000="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt"
CKPT_EXP32_2_1000="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt"
CKPT_EXP32_3_1000="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt"
CKPT_EXP37_1="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_1_pixel_bravo_lpips_hight_20260418-002202/checkpoint_latest.pt"
CKPT_EXP37_2="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt"
CKPT_EXP37_3="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042/checkpoint_latest.pt"

run_model() {
    local label="$1"
    local ckpt="$2"
    local out_subdir="compare_exp37_${TIMESTAMP}/${label}"
    local gen_dir="${GEN_ROOT}/${out_subdir}"

    echo ""
    echo "=========================================================="
    echo "  [${label}] Generating 10 volumes"
    echo "  checkpoint: ${ckpt}"
    echo "=========================================================="

    if [ ! -f "${ckpt}" ]; then
        echo "  SKIP: checkpoint not found"
        return
    fi

    mkdir -p "${gen_dir}"

    # No +brain_pca_path — raw output, no shape filtering during generation.
    # Atlas + single-component validation kept (prevents infinite retry on total garbage).
    time python -m medgen.scripts.generate \
        paths=cluster \
        spatial_dims=3 \
        gen_mode=bravo \
        image_size=256 \
        depth=160 \
        image_model="${ckpt}" \
        +real_seg_dir="${DATA_ROOT}/test1" \
        num_images=10 \
        num_steps_bravo=20 \
        validate_size_bins=false \
        validate_brain_mask=true \
        brain_atlas_path=auto \
        output_subdir="${out_subdir}" \
        verbose=true

    local final=$(find "${gen_dir}" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | wc -l)
    echo "  [${label}] done: ${final}/10 volumes"
}

# Run all seven — sequential so they share the same GPU with model unload between
run_model "exp1_1_1000"  "${CKPT_EXP1_1_1000}"
run_model "exp32_1_1000" "${CKPT_EXP32_1_1000}"
run_model "exp32_2_1000" "${CKPT_EXP32_2_1000}"
run_model "exp32_3_1000" "${CKPT_EXP32_3_1000}"
run_model "exp37_1"      "${CKPT_EXP37_1}"
run_model "exp37_2"      "${CKPT_EXP37_2}"
run_model "exp37_3"      "${CKPT_EXP37_3}"

echo ""
echo "=== All comparisons complete ==="
echo "Output root: ${COMPARE_ROOT}"
for label in exp1_1_1000 exp32_1_1000 exp32_2_1000 exp32_3_1000 exp37_1 exp37_2 exp37_3; do
    cnt=$(find "${GEN_ROOT}/compare_exp37_${TIMESTAMP}/${label}" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | wc -l)
    echo "  ${label}: ${cnt}/10 volumes -> ${GEN_ROOT}/compare_exp37_${TIMESTAMP}/${label}"
done

echo ""
echo "Sync to local for viewing:"
echo "  rsync -avh cluster:${COMPARE_ROOT}/ /home/mode/NTNU/MedicalDataSets/generated/compare_exp37_${TIMESTAMP}/"
