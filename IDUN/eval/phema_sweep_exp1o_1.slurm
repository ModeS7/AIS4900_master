#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="ph_1o1"
#SBATCH --output=IDUN/output/eval/phema_sweep_exp1o_1_%j.out
#SBATCH --error=IDUN/output/eval/phema_sweep_exp1o_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# PostHocEMA sweep: exp1o_1 (bravo 270M, 500ep)

cd ${SLURM_SUBMIT_DIR}
module purge
module load Anaconda3/2024.02-1
# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== PostHocEMA Sweep: exp1o_1 ==="
nvidia-smi

time python -m medgen.scripts.synthesize_phema \
    --run-dir ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1o_1_pixel_bravo_20260310-002926 \
    --data-root ${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3 \
    --num-volumes 25 --num-steps 25 --include-raw
