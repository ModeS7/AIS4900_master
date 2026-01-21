#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp9_1_dcae_phase3"
#SBATCH --output=IDUN/output/train/compression/exp9_1_dcae_f32_phase3_%j.out
#SBATCH --error=IDUN/output/train/compression/exp9_1_dcae_f32_phase3_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp9.1 Phase 3: DC-AE f32 GAN Refinement ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Exp9.1 Phase 3: DC-AE f32 GAN local refinement
# Paper: "we only tune the head layers of the decoder"
# Trainable: ~1.4K params (decoder head only)
# Frozen: encoder + decoder body (~362M params)
# Constant LR: 5.4e-5 (no warmup, eta_min=LR)
# NOTE: epochs = Phase1_epochs + Phase3_epochs (checkpoint resumes epoch counter)
#
CHECKPOINT="runs/compression_2d/multi_modality/exp9_f32_20251231-175024/latest.pt"

time python -m medgen.scripts.train_dcae \
    paths=cluster \
    mode=multi_modality \
    dcae=f32 \
    training.phase=3 \
    dcae.adv_weight=0.1 \
    dcae.disc_lr=5.4e-5 \
    training.learning_rate=5.4e-5 \
    training.warmup_epochs=0 \
    training.eta_min=5.4e-5 \
    'training.optimizer.betas=[0.5,0.9]' \
    training.epochs=175 \
    training.batch_size=16 \
    training.use_compile=false \
    training.name=exp9_1_f32_phase3_ \
    pretrained_checkpoint=${CHECKPOINT}

conda deactivate
