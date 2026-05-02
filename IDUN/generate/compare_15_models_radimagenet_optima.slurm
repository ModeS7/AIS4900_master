#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="cmp15rin"
#SBATCH --output=IDUN/output/generate/compare_15_models_radimagenet_%j.out
#SBATCH --error=IDUN/output/generate/compare_15_models_radimagenet_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Visual comparison: 10 volumes × 15 models, all using the SAME real seg
# masks from test_new (first 10 subjects). Each model uses its own
# RadImageNet-FID optimum step count from the dense-grid sweep.
# Includes handoff variants for the curriculum (exp48) family.

cd ${SLURM_SUBMIT_DIR}

echo "=== Compare 15 models @ RadImageNet optima — 10 vols, same real seg masks ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$FREE_MIB" -lt 60000 ] && { echo "ERROR: Only ${FREE_MIB} MiB free. Aborting."; exit 78; }

module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN_ROOT="${CLUSTER_BASE}/MedicalDataSets/generated"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
COMPARE_ROOT="${GEN_ROOT}/compare_15_models_radimagenet_${TIMESTAMP}"
mkdir -p "${COMPARE_ROOT}"

SEG_DIR="${DATA_ROOT}/test_new"

# Args: <label> <low-t-glob> <steps> [<high-t-glob>]
run_model () {
    local label="$1"
    local glob="$2"
    local steps="$3"
    local high_t_glob="${4:-}"
    local out_subdir="compare_15_models_radimagenet_${TIMESTAMP}/${label}"
    local gen_dir="${GEN_ROOT}/${out_subdir}"

    local run_dir
    run_dir=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/${glob} 2>/dev/null | head -1)
    local ckpt="${run_dir}/checkpoint_latest.pt"
    [ ! -f "${ckpt}" ] && ckpt="${run_dir}/checkpoint_best.pt"

    local handoff_args=""
    if [ -n "${high_t_glob}" ]; then
        local high_run_dir
        high_run_dir=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/${high_t_glob} 2>/dev/null | head -1)
        local high_ckpt="${high_run_dir}/checkpoint_latest.pt"
        [ ! -f "${high_ckpt}" ] && high_ckpt="${high_run_dir}/checkpoint_best.pt"
        if [ ! -f "${high_ckpt}" ]; then
            echo "  SKIP [${label}]: high-t ckpt not found at ${high_run_dir}"
            return
        fi
        handoff_args="+image_model_high_t=${high_ckpt} +handoff_t=0.25"
    fi

    echo ""
    echo "=========================================================="
    echo "  [${label}] 10 vols @ ${steps} steps"
    echo "  ckpt: ${ckpt}"
    [ -n "${high_t_glob}" ] && echo "  high-t handoff @ 0.25 ← ${high_run_dir}"
    echo "=========================================================="

    if [ ! -f "${ckpt}" ]; then
        echo "  SKIP: low-t ckpt not found at ${run_dir}"
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
        ${handoff_args} \
        +real_seg_dir="${SEG_DIR}" \
        num_images=10 \
        num_steps_bravo=${steps} \
        validate_size_bins=false \
        validate_brain_mask=true \
        brain_atlas_path=auto \
        output_subdir="${out_subdir}" \
        verbose=true

    local final
    final=$(find "${gen_dir}" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | wc -l)
    echo "  [${label}] done: ${final}/10 volumes"
}

# === RadImageNet FID optima ===
# label                              low-t glob                                            steps   high-t glob (optional)
run_model exp1_1_1000                "exp1_1_1000_pixel_bravo_*"                            49
run_model exp1_1_1000plus            "exp1_1_1000plus_pixel_bravo_*"                        79
run_model exp32_1_1000               "exp32_1_1000_pixel_bravo_ffl_*"                      100
run_model exp32_2_1000               "exp32_2_1000_pixel_bravo_lpips_lowt_*"               100
run_model exp32_3_1000               "exp32_3_1000_pixel_bravo_pseudo_huber_*"              50
run_model exp32_2_1000_17M           "exp32_2_1000_exp20_6_pixel_bravo_lpips_lowt_*"        40
run_model exp35_1                    "exp35_1_scoreaug_detail_*"                           100
run_model exp37_3                    "exp37_3_pixel_bravo_lpips_ffl_short_*"                55
run_model exp45                      "exp45_stack_lpips_scoreaug_*"                         65
run_model exp47a                     "exp47a_lpips_strong_*"                                80
run_model exp47e                     "exp47e_perceptual_only_lowt_*"                       100
run_model exp48c_standalone          "exp48c_lowt_only_lpips_huber_*"                      100
run_model exp48c_handoff_exp32       "exp48c_lowt_only_lpips_huber_*"                      100  "exp32_2_1000_pixel_bravo_lpips_lowt_*"
run_model exp48d_handoff_exp32       "exp48d_lowt_only_huber_lpips_*"                      100  "exp32_2_1000_pixel_bravo_lpips_lowt_*"
run_model exp48d_handoff_exp1        "exp48d_lowt_only_huber_lpips_*"                      100  "exp1_1_1000_pixel_bravo_*"

echo ""
echo "All done. Outputs at: ${COMPARE_ROOT}"
ls "${COMPARE_ROOT}"
