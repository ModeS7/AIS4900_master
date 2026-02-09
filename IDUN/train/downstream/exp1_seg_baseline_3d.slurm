#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="exp1_seg_baseline_3d"
#SBATCH --output=IDUN/output/downstream/exp1_seg_baseline_3d_%j.out
#SBATCH --error=IDUN/output/downstream/exp1_seg_baseline_3d_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Downstream Segmentation: Baseline 3D ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Train 3D segmentation on real data only (baseline)
time python -m medgen.scripts.train_segmentation \
    paths=cluster \
    scenario=baseline \
    model.spatial_dims=3 \
    training.epochs=500 \
    training.batch_size_3d=2 \
    training.name=exp1_baseline_3d_

conda deactivate
