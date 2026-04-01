#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="pca_128"
#SBATCH --output=IDUN/output/eval/eval_pca_exp1_128_variants_%j.out
#SBATCH --error=IDUN/output/eval/eval_pca_exp1_128_variants_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Evaluate PCA + FID for all exp1 128x128x160 variants
# 25 volumes, 20 steps

cd ${SLURM_SUBMIT_DIR}

echo "=== PCA + FID Evaluation: exp1 128x128 Variants ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
PCA_MODEL="/cluster/work/modestas/AIS4900_master/data/brain_pca_256x256x160.npz"
OUT_BASE="/cluster/work/modestas/MedicalDataSets/eval_pca_exp1_128_variants"
RUN_BASE="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo"

MODELS=(
    "exp1:${RUN_BASE}/exp1_pixel_bravo_20260224-163719/checkpoint_latest.pt"
    "exp1b:${RUN_BASE}/exp1b_20260221-172156/checkpoint_latest.pt"
    "exp1c:${RUN_BASE}/exp1c_20260224-172410/checkpoint_latest.pt"
    "exp1e:${RUN_BASE}/exp1e_pixel_bravo_snr_gamma_rflow_128x160_20260225-015538/checkpoint_latest.pt"
    "exp1f:${RUN_BASE}/exp1f_pixel_bravo_edm_precond_rflow_128x160_20260225-022154/checkpoint_latest.pt"
    "exp1g:${RUN_BASE}/exp1g_pixel_bravo_pseudo_huber_20260226-093915/checkpoint_latest.pt"
    "exp1h:${RUN_BASE}/exp1h_pixel_bravo_lpips_huber_20260226-221951/checkpoint_latest.pt"
    "exp1k:${RUN_BASE}/exp1k_pixel_bravo_offset_noise_20260306-113529/checkpoint_latest.pt"
    "exp1l:${RUN_BASE}/exp1l_pixel_bravo_adjusted_offset_20260306-114524/checkpoint_latest.pt"
    "exp1m:${RUN_BASE}/exp1m_pixel_bravo_global_norm_20260306-230725/checkpoint_latest.pt"
    "exp1n:${RUN_BASE}/exp1n_pixel_bravo_cfg_zero_star_20260309-191558/checkpoint_latest.pt"
    "exp1r:${RUN_BASE}/exp1r_pixel_bravo_attn_dropout_20260313-142342/checkpoint_latest.pt"
    "exp1s:${RUN_BASE}/exp1s_pixel_bravo_weight_decay_20260313-141301/checkpoint_latest.pt"
    "exp4:${RUN_BASE}/exp4_pixel_bravo_sda_rflow_128x160_20260130-025210/checkpoint_latest.pt"
    "exp5_1:${RUN_BASE}/exp5_1_pixel_bravo_scoreaug_rflow_128x160_20260130-030053/checkpoint_latest.pt"
    "exp8:${RUN_BASE}/exp8_pixel_bravo_ema_rflow_128x160_20260131-061549/checkpoint_latest.pt"
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
    echo "  $NAME (20 steps, 128x128x160)"
    echo "=========================================="

    time python -m medgen.scripts.eval_steps_pca \
        --checkpoint "${CKPT}" \
        --data-root "${DATA_ROOT}" \
        --pca-model "${PCA_MODEL}" \
        --output-dir "${OUT_BASE}/${NAME}" \
        --image-size 128 --depth 160 \
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
echo "  RESULTS SUMMARY: exp1 128x128 Variants"
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
