#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="ph_31_0"
#SBATCH --output=IDUN/output/eval/phema_sweep_exp31_0_%j.out
#SBATCH --error=IDUN/output/eval/phema_sweep_exp31_0_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# PostHocEMA sweep: exp31_0 (bravo 270M, combined no ScoreAug, 1000ep)

cd ${SLURM_SUBMIT_DIR}
module purge
module load Anaconda3/2024.02-1
# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== PostHocEMA Sweep: exp31_0 ==="
nvidia-smi

time python -m medgen.scripts.synthesize_phema \
    --run-dir ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp31_0_pixel_bravo_combined_no_sa_20260406-041427 \
    --data-root ${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3 \
    --num-volumes 25 --num-steps 25 --include-raw
