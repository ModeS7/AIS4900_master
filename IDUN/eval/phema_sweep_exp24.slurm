#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="ph_24"
#SBATCH --output=IDUN/output/eval/phema_sweep_exp24_%j.out
#SBATCH --error=IDUN/output/eval/phema_sweep_exp24_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# PostHocEMA sweep: exp24 (bravo 270M, combined techniques, 500ep)

cd ${SLURM_SUBMIT_DIR}
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== PostHocEMA Sweep: exp24 ==="
nvidia-smi

time python -m medgen.scripts.synthesize_phema \
    --run-dir /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp24_pixel_bravo_combined_20260315-193239 \
    --data-root /cluster/work/modestas/MedicalDataSets/brainmetshare-3 \
    --num-volumes 25 --num-steps 25 --include-raw
