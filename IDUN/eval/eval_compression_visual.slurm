#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --job-name="cmp_visual"
#SBATCH --output=IDUN/output/eval/eval_compression_visual_%j.out
#SBATCH --error=IDUN/output/eval/eval_compression_visual_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Reconstruct N real volumes through every compression model in our zoo.
# Outputs: per-volume NIfTIs + a thesis-ready PDF figure with PSNR/MS-SSIM.
# Edit the --model lines below to point to YOUR best checkpoints.

cd ${SLURM_SUBMIT_DIR}
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

echo "=== Compression model visual reconstruction ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
OUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/compression_visual_$(date +%Y%m%d-%H%M%S)"

# === Edit these to point to your best checkpoints ============================
DCAE_2D_F32=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/compression_2d/dcae_f32_*/checkpoint_best.pt 2>/dev/null | head -1)
DCAE_2D_F128=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/compression_2d/dcae_f128_*/checkpoint_best.pt 2>/dev/null | head -1)
VAE_2D=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/compression_2d/vae_*/checkpoint_best.pt 2>/dev/null | head -1)
VQVAE_2D=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/compression_2d/vqvae_*/checkpoint_best.pt 2>/dev/null | head -1)

VAE_3D=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/compression_3d/vae_*/checkpoint_best.pt 2>/dev/null | head -1)
VQVAE_3D=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/compression_3d/vqvae_*/checkpoint_best.pt 2>/dev/null | head -1)
DCAE_3D=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/compression_3d/dcae_*/checkpoint_best.pt 2>/dev/null | head -1)

MODEL_ARGS=""
[ -n "${DCAE_2D_F32}"  ] && MODEL_ARGS="${MODEL_ARGS} --model 'DC-AE 2D f32:${DCAE_2D_F32}'"
[ -n "${DCAE_2D_F128}" ] && MODEL_ARGS="${MODEL_ARGS} --model 'DC-AE 2D f128:${DCAE_2D_F128}'"
[ -n "${VAE_2D}"       ] && MODEL_ARGS="${MODEL_ARGS} --model 'VAE 2D:${VAE_2D}'"
[ -n "${VQVAE_2D}"     ] && MODEL_ARGS="${MODEL_ARGS} --model 'VQ-VAE 2D:${VQVAE_2D}'"
[ -n "${VAE_3D}"       ] && MODEL_ARGS="${MODEL_ARGS} --model 'VAE 3D:${VAE_3D}'"
[ -n "${VQVAE_3D}"     ] && MODEL_ARGS="${MODEL_ARGS} --model 'VQ-VAE 3D:${VQVAE_3D}'"
[ -n "${DCAE_3D}"      ] && MODEL_ARGS="${MODEL_ARGS} --model 'DC-AE 3D:${DCAE_3D}'"

if [ -z "${MODEL_ARGS}" ]; then
    echo "ERROR: no checkpoints resolved. Edit the glob paths in the SLURM."
    exit 2
fi

echo "Resolved checkpoints:"
echo "  DCAE_2D_F32  = ${DCAE_2D_F32}"
echo "  DCAE_2D_F128 = ${DCAE_2D_F128}"
echo "  VAE_2D       = ${VAE_2D}"
echo "  VQVAE_2D     = ${VQVAE_2D}"
echo "  VAE_3D       = ${VAE_3D}"
echo "  VQVAE_3D     = ${VQVAE_3D}"
echo "  DCAE_3D      = ${DCAE_3D}"
echo ""
echo "Output: ${OUT_DIR}"

# Use eval to handle the quoted spaces in model labels
eval time python -m medgen.scripts.eval_compression_visual \
    --data-root "${DATA_ROOT}" \
    --split test_new \
    --modality bravo \
    --num-volumes 4 \
    --depth 160 \
    --image-size 256 \
    --output-dir "${OUT_DIR}" \
    --seed 42 \
    ${MODEL_ARGS}

echo ""
echo "=== Done. Outputs at: ${OUT_DIR} ==="
ls -la "${OUT_DIR}"
