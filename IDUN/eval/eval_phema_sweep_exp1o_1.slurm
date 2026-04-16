#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="phema_1o1"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/phema_sweep_exp1o_1_%j.out
#SBATCH --error=IDUN/output/eval/phema_sweep_exp1o_1_%j.err

# Post-hoc EMA sweep for exp1o_1 (270M bravo, sigma_rels=[0.05, 0.10])
# Evaluates: raw model + EMA at sigma_rel=0.05 + EMA at sigma_rel=0.10
# 25 volumes × 25 steps each, FID/KID/CMMD/FWD/PCA metrics

cd ${SLURM_SUBMIT_DIR}

RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260413-023304"

echo "=== Post-hoc EMA Sweep: exp1o_1 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
echo "Run dir: ${RUN_DIR}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.synthesize_phema \
    --run-dir "${RUN_DIR}" \
    --data-root /cluster/work/modestas/MedicalDataSets/brainmetshare-3 \
    --training-sigma-rels 0.05 0.10 \
    --num-volumes 25 \
    --num-steps 25 \
    --depth 160 \
    --image-size 256 \
    --seed 42 \
    --include-raw

echo ""
echo "=== Results ==="
cat "${RUN_DIR}/phema_sweep/phema_sweep_results.json" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for name, r in data.items():
        if 'error' in r:
            print(f'{name}: ERROR - {r[\"error\"]}')
        else:
            fid = r.get('fid', 0)
            fwd = r.get('fwd', 0)
            pca = r.get('pca_pass_rate', 0) * 100
            print(f'{name}: FID={fid:.2f}  FWD={fwd:.4f}  PCA={pca:.0f}%')
except: pass
" 2>/dev/null
