#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="ph_31_1"
#SBATCH --output=IDUN/output/eval/phema_sweep_exp31_1_%j.out
#SBATCH --error=IDUN/output/eval/phema_sweep_exp31_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# PostHocEMA sweep: exp31_1 (bravo 67M, combined no ScoreAug, 1000ep)

cd ${SLURM_SUBMIT_DIR}
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== PostHocEMA Sweep: exp31_1 ==="
nvidia-smi

time python -m medgen.scripts.synthesize_phema \
    --run-dir /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp31_1_pixel_bravo_combined_no_sa_67m_20260405-211125 \
    --data-root /cluster/work/modestas/MedicalDataSets/brainmetshare-3 \
    --num-volumes 25 --num-steps 32 --include-raw
