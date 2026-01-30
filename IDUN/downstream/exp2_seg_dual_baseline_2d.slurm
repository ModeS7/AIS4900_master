#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp2_seg_dual_2d"
#SBATCH --output=IDUN/output/downstream/exp2_seg_dual_2d_%j.out
#SBATCH --error=IDUN/output/downstream/exp2_seg_dual_2d_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Downstream Segmentation: Dual Input (T1 pre + T1 gd) 2D ==="
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

time python -m medgen.scripts.train_segmentation \
    paths=cluster \
    scenario=baseline \
    model.spatial_dims=2 \
    model.in_channels=2 \
    "data.modality=[t1_pre,t1_gd]" \
    training.epochs=500 \
    training.batch_size=16 \
    training.name=exp2_dual_

conda deactivate
