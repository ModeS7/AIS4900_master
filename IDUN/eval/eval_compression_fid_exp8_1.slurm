#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="comp_fid_exp8_1"
#SBATCH --output=IDUN/output/eval/eval_compression_fid_exp8_1_%j.out
#SBATCH --error=IDUN/output/eval/eval_compression_fid_exp8_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# exp8_1: VQ-VAE 4x compression model (3D, multi_modality)
# Measures reconstruction FID — the quality floor for any LDM using this model

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Compression Reconstruction FID: exp8_1 (VQ-VAE 4x) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === PATHS ===
COMPRESSION_CHECKPOINT=/cluster/work/modestas/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUTPUT_DIR="/cluster/work/modestas/MedicalDataSets/eval_compression_fid_exp8_1"

echo "Compression checkpoint: $COMPRESSION_CHECKPOINT"
echo "Data root: $DATA_ROOT"
echo ""

time python -m medgen.scripts.eval_compression_fid \
    --compression-checkpoint "${COMPRESSION_CHECKPOINT}" \
    --compression-type vqvae \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --modality bravo

echo "=== Evaluation complete ==="
cat "${OUTPUT_DIR}/reconstruction_fid.json"

conda deactivate
