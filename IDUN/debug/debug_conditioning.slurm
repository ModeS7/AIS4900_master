#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="debug_cond"
#SBATCH --output=IDUN/output/debug/debug_conditioning_%j.out
#SBATCH --error=IDUN/output/debug/debug_conditioning_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Debug: Does conditioning change output? ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT_EXP2C="runs/diffusion_3d/seg/exp2c_1_pixel_seg_improved_20260218-183155/checkpoint_latest.pt"

echo ""
echo "=== exp2c_1 (Improved FiLM) ==="
time python misc/debug_conditioning.py --checkpoint "${CKPT_EXP2C}" --label "exp2c_1 Improved FiLM" --num_steps "40,100,200"

echo ""
echo "=== Done ==="
conda deactivate
