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
#
# Run-dir layout (per training SLURMs in IDUN/train/compression/):
#   runs/compression_2d/multi_modality/<exp_name>_<timestamp>/checkpoint_best.pt   (newer)
#   runs/compression_2d/<exp_name>_<timestamp>/checkpoint_best.pt                  (older)
#   runs/compression_3d/multi_modality/<exp_name>_<timestamp>/checkpoint_best.pt
#
# Token map (path-substring used to disambiguate which family a checkpoint is):
#   exp9_dcae_f32_1_5               → DC-AE 2D f32  v1.5 (32x compression, 8x8x32 latent)
#   exp9_dcae_f128_1_5              → DC-AE 2D f128 v1.5 (128x compression, 2x2x128 latent)
#   exp3_                           → VAE 2D (multi-modality 256x256, 32x32x4 latent)
#   exp6_1_ / exp6_                 → VQ-VAE 2D (4x compression)
#   exp14_3_vae3d_4x_lowkl          → VAE 3D (4x, low KL — best loss tradeoff)
#   exp8_vqvae3d_4x                 → VQ-VAE 3D (4x baseline, lat=4)
#   exp10_1                         → DC-AE 3D (default 4x)

cd ${SLURM_SUBMIT_DIR}
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

echo "=== Compression model visual reconstruction ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
OUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/compression_visual_$(date +%Y%m%d-%H%M%S)"
RUNS_2D="${CLUSTER_BASE}/AIS4900_master/runs/compression_2d"
RUNS_3D="${CLUSTER_BASE}/AIS4900_master/runs/compression_3d"

# Find the freshest checkpoint_best.pt under $1 whose path contains $2 (or $3 as fallback).
find_ckpt() {
    local root="$1"
    local primary="$2"
    local fallback="$3"
    local hit
    if [ -d "$root" ]; then
        hit=$(find "$root" -maxdepth 4 -name 'checkpoint_best.pt' -path "*${primary}*" \
                -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
        if [ -z "$hit" ] && [ -n "$fallback" ]; then
            hit=$(find "$root" -maxdepth 4 -name 'checkpoint_best.pt' -path "*${fallback}*" \
                    -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
        fi
    fi
    echo "$hit"
}

DCAE_2D_F32=$(find_ckpt  "$RUNS_2D" 'exp9_dcae_f32_1_5'  '')
DCAE_2D_F128=$(find_ckpt "$RUNS_2D" 'exp9_dcae_f128_1_5' '')
VAE_2D=$(find_ckpt       "$RUNS_2D" 'exp3_'          '')
VQVAE_2D=$(find_ckpt     "$RUNS_2D" 'exp6_1_'        'exp6_')

VAE_3D=$(find_ckpt   "$RUNS_3D" 'exp14_3_vae3d_4x_lowkl' 'exp14_0_vae3d_4x')
VQVAE_3D=$(find_ckpt "$RUNS_3D" 'exp8_vqvae3d_4x'        'exp8_1_vqvae3d_lat4')
DCAE_3D=$(find_ckpt  "$RUNS_3D" 'exp10_1'                '')

# Diagnostic: list ALL discovered checkpoints, even ones we didn't pick.
echo ""
echo "=== All discovered checkpoint_best.pt under runs/compression_{2d,3d} ==="
find "$RUNS_2D" "$RUNS_3D" -maxdepth 4 -name 'checkpoint_best.pt' \
    -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort

echo ""
echo "=== Selected checkpoints ==="
printf '  %-15s = %s\n' \
    'DCAE_2D_F32'  "${DCAE_2D_F32:-(none)}" \
    'DCAE_2D_F128' "${DCAE_2D_F128:-(none)}" \
    'VAE_2D'       "${VAE_2D:-(none)}" \
    'VQVAE_2D'     "${VQVAE_2D:-(none)}" \
    'VAE_3D'       "${VAE_3D:-(none)}" \
    'VQVAE_3D'     "${VQVAE_3D:-(none)}" \
    'DCAE_3D'      "${DCAE_3D:-(none)}"

MODEL_ARGS=""
[ -n "${DCAE_2D_F32}"  ] && MODEL_ARGS="${MODEL_ARGS} --model 'DC-AE 2D f32:${DCAE_2D_F32}'"
[ -n "${DCAE_2D_F128}" ] && MODEL_ARGS="${MODEL_ARGS} --model 'DC-AE 2D f128:${DCAE_2D_F128}'"
[ -n "${VAE_2D}"       ] && MODEL_ARGS="${MODEL_ARGS} --model 'VAE 2D:${VAE_2D}'"
[ -n "${VQVAE_2D}"     ] && MODEL_ARGS="${MODEL_ARGS} --model 'VQ-VAE 2D:${VQVAE_2D}'"
[ -n "${VAE_3D}"       ] && MODEL_ARGS="${MODEL_ARGS} --model 'VAE 3D:${VAE_3D}'"
[ -n "${VQVAE_3D}"     ] && MODEL_ARGS="${MODEL_ARGS} --model 'VQ-VAE 3D:${VQVAE_3D}'"
[ -n "${DCAE_3D}"      ] && MODEL_ARGS="${MODEL_ARGS} --model 'DC-AE 3D:${DCAE_3D}'"

if [ -z "${MODEL_ARGS}" ]; then
    echo ""
    echo "ERROR: no checkpoints resolved."
    echo "  Either (a) no checkpoint_best.pt exists under runs/compression_{2d,3d},"
    echo "  or (b) the path tokens above don't match your run-dir names."
    echo "  See the 'All discovered' listing above and edit the find_ckpt tokens"
    echo "  in this SLURM, or pass explicit --model 'Label:Path' lines below."
    exit 2
fi

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
