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

# Reconstruct N real volumes through the compression-model zoo for thesis
# section "Compression Model Evaluation". Only models actually evaluated:
#
#   2D models (slice-by-slice on 3D volumes):
#     DC-AE 2D f32  v1.5 → exp9_dcae_f32_1_5
#     DC-AE 2D f128 v1.5 → exp9_dcae_f128_1_5
#
#   3D models (full-volume):
#     VAE 3D    → exp14_3_vae3d_4x_lowkl  (low-KL gives best reconstruction)
#     VQ-VAE 3D → exp8_3_vqvae3d_lat16    (highest-capacity 4x VQ-VAE)
#
# Hydra run-dir resolution (from configs/{dcae,vae_3d,vqvae_3d}.yaml):
#   compression_2d/${mode.name}/${training.name}${now}                     (DC-AE 2D)
#   compression_3d/${mode.name}/${training.name}${volume.height}x${volume.depth}_${now}  (3D)
#   The _1_5 SLURMs additionally pin hydra.run.dir to .../multi_modality/<exp>_<ts>/.
#
# Override any pick: pass --model "Label:explicit/path/checkpoint.pt" CLI flag.

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

# Find freshest checkpoint of given basename whose path contains $token, under $root.
# $1 root, $2 token, $3 basename (e.g. checkpoint_best.pt)
find_ckpt() {
    [ -d "$1" ] || return 0
    find "$1" -maxdepth 4 -name "$3" -path "*$2*" \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}'
}

# Prefer best, fall back to latest if best doesn't exist (training still chaining).
pick_ckpt() {
    local hit
    hit=$(find_ckpt "$1" "$2" 'checkpoint_best.pt')
    if [ -z "$hit" ]; then
        hit=$(find_ckpt "$1" "$2" 'checkpoint_latest.pt')
    fi
    echo "$hit"
}

DCAE_2D_F32=$(pick_ckpt  "$RUNS_2D" 'exp9_dcae_f32_1_5')
DCAE_2D_F128=$(pick_ckpt "$RUNS_2D" 'exp9_dcae_f128_1_5')
VAE_3D=$(pick_ckpt       "$RUNS_3D" 'exp14_3_vae3d_4x_lowkl')
VQVAE_3D=$(pick_ckpt     "$RUNS_3D" 'exp8_3_vqvae3d_lat16')

# Diagnostic: list every compression checkpoint (best AND latest) so the user
# can see what's actually on disk if a pick doesn't match expectations.
echo ""
echo "=== All compression checkpoints on disk ==="
find "$RUNS_2D" "$RUNS_3D" -maxdepth 4 \
    \( -name 'checkpoint_best.pt' -o -name 'checkpoint_latest.pt' \) \
    -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort

echo ""
echo "=== Selected checkpoints (token → freshest match) ==="
printf '  %-15s = %s\n' \
    'DCAE_2D_F32'  "${DCAE_2D_F32:-(none — exp9_dcae_f32_1_5 not found)}" \
    'DCAE_2D_F128' "${DCAE_2D_F128:-(none — exp9_dcae_f128_1_5 not found)}" \
    'VAE_3D'       "${VAE_3D:-(none — exp14_3_vae3d_4x_lowkl not found)}" \
    'VQVAE_3D'     "${VQVAE_3D:-(none — exp8_3_vqvae3d_lat16 not found)}"

MODEL_ARGS=""
[ -n "${DCAE_2D_F32}"  ] && MODEL_ARGS="${MODEL_ARGS} --model 'DC-AE 2D f32:${DCAE_2D_F32}'"
[ -n "${DCAE_2D_F128}" ] && MODEL_ARGS="${MODEL_ARGS} --model 'DC-AE 2D f128:${DCAE_2D_F128}'"
[ -n "${VAE_3D}"       ] && MODEL_ARGS="${MODEL_ARGS} --model 'VAE 3D:${VAE_3D}'"
[ -n "${VQVAE_3D}"     ] && MODEL_ARGS="${MODEL_ARGS} --model 'VQ-VAE 3D:${VQVAE_3D}'"

if [ -z "${MODEL_ARGS}" ]; then
    echo ""
    echo "ERROR: no checkpoints resolved. Either training hasn't produced a"
    echo "checkpoint_best.pt yet, or the path tokens above don't match the"
    echo "actual run-dir names. See the 'All compression checkpoints' listing"
    echo "above and either wait for training to checkpoint, or override with"
    echo "explicit paths by editing the four pick_ckpt lines."
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
