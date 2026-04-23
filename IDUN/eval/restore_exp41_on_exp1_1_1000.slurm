#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="restore_41"
#SBATCH --output=IDUN/output/eval/restore_exp41_%j.out
#SBATCH --error=IDUN/output/eval/restore_exp41_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Apply each exp41 restoration variant (a/b/c) to existing exp1_1_1000 outputs
# from compare_imagenet_optima_20260421-165759/. Produces restored volume sets
# for spectrum / Frangi / visual comparison.
#
# Output structure per variant:
#   runs/eval/restoration_eval_<variant>/<subject>/bravo.nii.gz
#
# After this job: run the standard diagnostic scripts (analyze_generation_spectrum,
# analyze_vessel_prominence, analyze_cortical_vessels) on each output dir.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== Apply exp41 restoration variants to exp1_1_1000 outputs ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE_MIB" -lt 60000 ]; then
    echo "ERROR: Only ${FREE_MIB} MiB free on GPU. Aborting."
    exit 78
fi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
# Source of blurry volumes to restore — exp1_1_1000 outputs at ImageNet-FID-optimum
# step count from compare_imagenet_optima_20260421-165759
INPUT_DIR="${CLUSTER_BASE}/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp1_1_1000"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input dir not found: $INPUT_DIR"
    exit 1
fi

restore_variant() {
    local variant="$1"
    local ckpt_dir=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/restoration/exp41${variant}_* 2>/dev/null | head -1)
    local ckpt="${ckpt_dir}/checkpoint_latest.pt"
    local out_dir="${CLUSTER_BASE}/AIS4900_master/runs/eval/restoration_eval_exp41${variant}"

    echo ""
    echo "=========================================================="
    echo "  [exp41${variant}] Restoring from ${INPUT_DIR}"
    echo "  checkpoint: ${ckpt}"
    echo "  output:     ${out_dir}"
    echo "=========================================================="

    if [ ! -f "${ckpt}" ]; then
        echo "  SKIP: checkpoint not found — did exp41${variant} finish training?"
        return
    fi

    mkdir -p "${out_dir}"

    time python -m medgen.scripts.restore_volumes \
        --restoration-model "${ckpt}" \
        --strategy irsde \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${out_dir}" \
        --num-steps 25 \
        --num-timesteps 100 \
        --depth 160 \
        --image-size 256 \
        --eval \
        --data-root "${DATA_ROOT}"

    local count=$(find "${out_dir}" -name "*.nii.gz" 2>/dev/null | wc -l)
    echo "  [exp41${variant}] done: ${count} restored volumes"
}

restore_variant "a_irsde_gauss042"
restore_variant "b_irsde_real_pairs"
restore_variant "c_irsde_gauss_noisy"

echo ""
echo "=== All exp41 restorations complete ==="
for v in a b c; do
    suffix=""
    case $v in
        a) suffix="a_irsde_gauss042";;
        b) suffix="b_irsde_real_pairs";;
        c) suffix="c_irsde_gauss_noisy";;
    esac
    dir="${CLUSTER_BASE}/AIS4900_master/runs/eval/restoration_eval_exp41${suffix}"
    cnt=$(find "${dir}" -name "*.nii.gz" 2>/dev/null | wc -l)
    echo "  exp41${v}: ${cnt} volumes -> ${dir}"
done
