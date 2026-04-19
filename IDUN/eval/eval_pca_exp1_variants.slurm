#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="pca_exp1"
#SBATCH --output=IDUN/output/eval/eval_pca_exp1_variants_%j.out
#SBATCH --error=IDUN/output/eval/eval_pca_exp1_variants_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# Evaluate PCA + FID for all exp1 technique ablation variants (256x256x160)
# 25 volumes, 20 steps

cd ${SLURM_SUBMIT_DIR}

echo "=== PCA + FID Evaluation: exp1 Variants ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
PCA_MODEL="${CLUSTER_BASE}/AIS4900_master/data/brain_pca_256x256x160.npz"
OUT_BASE="${CLUSTER_BASE}/MedicalDataSets/eval_pca_exp1_variants"
RUN_BASE="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"

MODELS=(
    "exp1_1:${RUN_BASE}/exp1_1_pixel_bravo_20260301-180232/checkpoint_latest.pt"
    "exp1_1_1000:${RUN_BASE}/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt"
    "exp1b_1:${RUN_BASE}/exp1b_1_20260310-123920/checkpoint_latest.pt"
    "exp1c_1:${RUN_BASE}/exp1c_1_20260223-004121/checkpoint_latest.pt"
    "exp1e_1:${RUN_BASE}/exp1e_1_pixel_bravo_snr_gamma_rflow_256x160_20260225-031917/checkpoint_latest.pt"
    "exp1f_1:${RUN_BASE}/exp1f_1_pixel_bravo_edm_precond_rflow_256x160_20260225-033959/checkpoint_latest.pt"
    "exp1g_1:${RUN_BASE}/exp1g_1_pixel_bravo_pseudo_huber_20260226-034319/checkpoint_latest.pt"
    "exp1h_1:${RUN_BASE}/exp1h_1_pixel_bravo_lpips_huber_20260226-202457/checkpoint_latest.pt"
    "exp1i_1:${RUN_BASE}/exp1i_1_pixel_bravo_scoreaug_ema_20260301-015714/checkpoint_latest.pt"
    "exp1j_1:${RUN_BASE}/exp1j_1_pixel_bravo_grad_accum_20260301-015744/checkpoint_latest.pt"
    "exp1k_1:${RUN_BASE}/exp1k_1_pixel_bravo_offset_noise_20260311-201606/checkpoint_latest.pt"
    "exp1l_1:${RUN_BASE}/exp1l_1_pixel_bravo_adjusted_offset_20260309-202039/checkpoint_latest.pt"
    "exp1m_1:${RUN_BASE}/exp1m_1_pixel_bravo_global_norm_20260306-215802/checkpoint_latest.pt"
    "exp1o_1:${RUN_BASE}/exp1o_1_pixel_bravo_20260310-002926/checkpoint_latest.pt"
    "exp1p_1:${RUN_BASE}/exp1p_1_pixel_bravo_uniform_timestep_20260312-014123/checkpoint_latest.pt"
)

for entry in "${MODELS[@]}"; do
    NAME="${entry%%:*}"
    CKPT="${entry#*:}"

    if [ ! -f "$CKPT" ]; then
        echo "SKIP $NAME: checkpoint not found"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  $NAME (20 steps)"
    echo "=========================================="

    time python -m medgen.scripts.eval_steps_pca \
        --checkpoint "${CKPT}" \
        --data-root "${DATA_ROOT}" \
        --pca-model "${PCA_MODEL}" \
        --output-dir "${OUT_BASE}/${NAME}" \
        --image-size 256 --depth 160 \
        --num-volumes 25 \
        --step-list 20 \
        --shift-ratio 1.0 \
        --seed 42 \
        --compute-fid --ref-split test1

    echo "--- $NAME done ---"
done

# === Summary ===
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: exp1 Variants"
echo "=========================================="
printf "%-12s %10s %7s %8s %8s %8s\n" "Model" "PCA_err" "Pass%" "FID" "KID" "CMMD"
echo "--------------------------------------------------------------"
for dir in ${OUT_BASE}/*/; do
    NAME=$(basename "$dir")
    JSON="${dir}pca_step_results.json"
    if [ -f "$JSON" ]; then
        python3 -c "
import json
d = json.load(open('$JSON'))
r = d['results'][0]
pca = f\"{r['mean_error']:.6f}\"
pr = f\"{r['pass_rate']:.0%}\"
fid = f\"{r.get('FID', 0):.2f}\" if 'FID' in r else '—'
kid = f\"{r.get('KID_mean', 0):.4f}\" if 'KID_mean' in r else '—'
cmmd = f\"{r.get('CMMD', 0):.4f}\" if 'CMMD' in r else '—'
print(f'$NAME'.ljust(12) + pca.rjust(10) + pr.rjust(7) + fid.rjust(8) + kid.rjust(8) + cmmd.rjust(8))
"
    fi
done

conda deactivate
