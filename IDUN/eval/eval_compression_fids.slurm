#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --job-name="cmp_fids"
#SBATCH --output=IDUN/output/eval/eval_compression_fids_%j.out
#SBATCH --error=IDUN/output/eval/eval_compression_fids_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Compute ImageNet + RadImageNet FID across the full compression-model zoo,
# plus a real-vs-real baseline. Companion to eval_compression_visual.slurm.
#
# Same checkpoint-resolution conventions:
#   2D models (slice-by-slice through 3D volumes):
#     DC-AE 2D f32  v1.5 → exp9_dcae_f32_1_5
#     DC-AE 2D f64  v1.5 → exp9_dcae_f64_1_5
#     DC-AE 2D f128 v1.5 → exp9_dcae_f128_1_5
#   3D models (full-volume):
#     VAE 3D    → exp14_4_vae3d_4x_nokl
#     VQ-VAE 3D → exp8_1_256x160
#     MAISI VAE 3D → bundles/maisi_ct_generative
#
# Time estimate: ~3-4 hours wall-clock for the full zoo on the test_new split
# (51 volumes × 6 models × 2 backbones).

cd ${SLURM_SUBMIT_DIR}
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

echo "=== Compression model FID zoo ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
OUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/compression_fids_$(date +%Y%m%d-%H%M%S)"
RUNS_2D="${CLUSTER_BASE}/AIS4900_master/runs/compression_2d"
RUNS_3D="${CLUSTER_BASE}/AIS4900_master/runs/compression_3d"

# Find freshest checkpoint of given basename whose path contains $token.
find_ckpt() {
    [ -d "$1" ] || return 0
    find "$1" -maxdepth 4 -name "$3" -path "*$2*" \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}'
}

# Prefer best, fall back to latest (training still chaining).
pick_ckpt() {
    local hit
    hit=$(find_ckpt "$1" "$2" 'checkpoint_best.pt')
    if [ -z "$hit" ]; then
        hit=$(find_ckpt "$1" "$2" 'checkpoint_latest.pt')
    fi
    echo "$hit"
}

DCAE_2D_F32=$(pick_ckpt  "$RUNS_2D" 'exp9_dcae_f32_1_5')
DCAE_2D_F64=$(pick_ckpt  "$RUNS_2D" 'exp9_dcae_f64_1_5')
DCAE_2D_F128=$(pick_ckpt "$RUNS_2D" 'exp9_dcae_f128_1_5')
VAE_3D=$(pick_ckpt       "$RUNS_3D" 'exp14_4_vae3d_4x_nokl')
VQVAE_3D=$(pick_ckpt     "$RUNS_3D" 'exp8_1_256x160')

# MAISI VAE — pretrained NVIDIA model, lives in the project's bundles/ dir.
MAISI_BUNDLE="${SLURM_SUBMIT_DIR}/bundles/maisi_ct_generative"
MAISI_CKPT=""
if [ -f "${MAISI_BUNDLE}/models/autoencoder.pt" ]; then
    MAISI_CKPT="${MAISI_BUNDLE}/models/autoencoder.pt"
fi

echo ""
echo "=== Selected checkpoints ==="
printf '  %-15s = %s\n' \
    'DCAE_2D_F32'  "${DCAE_2D_F32:-(none)}" \
    'DCAE_2D_F64'  "${DCAE_2D_F64:-(none)}" \
    'DCAE_2D_F128' "${DCAE_2D_F128:-(none)}" \
    'VAE_3D'       "${VAE_3D:-(none)}" \
    'VQVAE_3D'     "${VQVAE_3D:-(none)}" \
    'MAISI_CKPT'   "${MAISI_CKPT:-(none)}"

MODEL_ARGS=""
[ -n "${DCAE_2D_F32}"  ] && MODEL_ARGS="${MODEL_ARGS} --model 'DC-AE 2D f32:${DCAE_2D_F32}'"
[ -n "${DCAE_2D_F64}"  ] && MODEL_ARGS="${MODEL_ARGS} --model 'DC-AE 2D f64:${DCAE_2D_F64}'"
[ -n "${DCAE_2D_F128}" ] && MODEL_ARGS="${MODEL_ARGS} --model 'DC-AE 2D f128:${DCAE_2D_F128}'"
[ -n "${VAE_3D}"       ] && MODEL_ARGS="${MODEL_ARGS} --model 'VAE 3D:${VAE_3D}'"
[ -n "${VQVAE_3D}"     ] && MODEL_ARGS="${MODEL_ARGS} --model 'VQ-VAE 3D:${VQVAE_3D}'"
[ -n "${MAISI_CKPT}"   ] && MODEL_ARGS="${MODEL_ARGS} --model 'MAISI VAE 3D:${MAISI_CKPT}'"

if [ -z "${MODEL_ARGS}" ]; then
    echo "ERROR: no checkpoints resolved."
    exit 2
fi

echo ""
echo "Output: ${OUT_DIR}"

# eval handles the quoted spaces in model labels
eval time python -m medgen.scripts.eval_compression_fids \
    --data-root "${DATA_ROOT}" \
    --split test_new \
    --modality bravo \
    --depth 160 \
    --image-size 256 \
    --output-dir "${OUT_DIR}" \
    --seed 42 \
    ${MODEL_ARGS}

echo ""
echo "=== Done. Outputs at: ${OUT_DIR} ==="
ls -la "${OUT_DIR}"
