#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="perc_dist"
#SBATCH --output=IDUN/output/eval/perception_distortion_%j.out
#SBATCH --error=IDUN/output/eval/perception_distortion_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Perception-Distortion proof of the MSE mean-blur failure mode.
# Single-step denoising of real test volumes at low-t for each model:
#   MSE baseline (exp1_1_1000) -> best PSNR/SSIM, worst FID (distortion-optimal, perceptually poor)
#   LPIPS-low-t  (exp32_2_1000) -> trades fidelity for realism
#   specialist   (exp48d)       -> trades fidelity for realism
# Each model writes <OUT_ROOT>/<label>/perception_distortion.json; assemble the
# perception-distortion plane across the JSONs. The generation-FID payoff is
# already covered by the existing find_optimal_steps results (baseline 0.71,
# LPIPS-low-t 0.53, handoff 0.48) -- no new run needed for that.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

echo "=== Perception-Distortion proof ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
OUT_ROOT="${CLUSTER_BASE}/MedicalDataSets/eval_perception_distortion"
RUNS="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"

# Models -> checkpoints (paths match the find_optimal_steps_*.slurm conventions).
CKPT_BASELINE="${RUNS}/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt"
CKPT_LPIPSLOWT="${RUNS}/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt"
_SPEC_D_RUN=$(ls -td ${RUNS}/exp48d_lowt_only_huber_lpips_* 2>/dev/null | head -1)
CKPT_SPEC_D="${_SPEC_D_RUN}/checkpoint_latest.pt"

run_model () {
    label="$1"
    ckpt="$2"
    if [ ! -f "$ckpt" ]; then
        echo "SKIP ${label}: checkpoint not found: $ckpt"
        return
    fi
    echo ""
    echo "==================== ${label} ===================="
    echo "Checkpoint: ${ckpt}"
    time python -m medgen.scripts.eval_perception_distortion \
        --checkpoint "${ckpt}" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${OUT_ROOT}/${label}" \
        --t-values 0.05 0.1 0.2 \
        --num-volumes 25 \
        --n-noise 8 \
        --pred-std-volumes 5 \
        --image-size 256 \
        --depth 160 \
        --ref-split test \
        --seed 42
}

run_model baseline    "${CKPT_BASELINE}"
run_model lpips_lowt  "${CKPT_LPIPSLOWT}"
run_model spec_exp48d "${CKPT_SPEC_D}"

echo ""
echo "=========================================="
echo "  Perception-Distortion points per model"
echo "=========================================="
for d in baseline lpips_lowt spec_exp48d; do
    echo "--- ${d} ---"
    cat "${OUT_ROOT}/${d}/perception_distortion.json" 2>/dev/null
    echo ""
done
