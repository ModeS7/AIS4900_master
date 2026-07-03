#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="p3d_bench"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/train/diffusion_3d/bench_perceptual3d/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/bench_perceptual3d/%j.err

# Benchmark: TRUE-3D MedicalNet perceptual loss vs the previous 2D RadImageNet one,
# and torch.compile ON vs OFF for each. Times a short 30-batch run per config and
# records peak GPU memory. UNet compile is DISABLED (training.use_compile=false) to
# isolate the perceptual-loss cost; perceptual is applied EVERY step (no low-t gate)
# for a clean worst-case per-step comparison. Trains from scratch (timing only).
#
# NOTE: MedicalNet weights auto-download from HuggingFace. Compute nodes are offline,
# so PRE-CACHE on the login node first (the pre-warm step below will fail loudly if not).

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

SCRATCH="${CLUSTER_BASE}/AIS4900_master/runs/_bench/perceptual3d_${SLURM_JOB_ID}"
SUMMARY="${SCRATCH}/SUMMARY.txt"
mkdir -p "${SCRATCH}"

# Pre-warm the MedicalNet download so the timed 3D runs don't pay it (and to fail
# loudly here if the weights aren't cached on an offline node).
echo "=== Pre-warming MedicalNet weights ==="
python -c "from monai.losses import PerceptualLoss; PerceptualLoss(spatial_dims=3, network_type='medicalnet_resnet10_23datasets', is_fake_3d=False, pretrained=True); print('MedicalNet weights OK')" \
    || { echo 'FATAL: MedicalNet weights not available (pre-cache on login node).'; exit 1; }

COMMON="--config-name=diffusion_3d paths=cluster strategy=rflow mode=bravo model=default_3d \
    volume.height=256 volume.width=256 volume.pad_depth_to=160 \
    training.epochs=1 training.limit_train_batches=30 training.warmup_epochs=0 \
    training.perceptual_weight=0.1 training.use_compile=false \
    training.generation_metrics.enabled=false training.verbose=true"

sample_gpu() {
    local out="$1"; echo 0 > "$out"
    while :; do
        u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        m=$(cat "$out" 2>/dev/null || echo 0)
        if [ -n "$u" ] && [ "$u" -gt "$m" ] 2>/dev/null; then echo "$u" > "$out"; fi
        sleep 2
    done
}

run_bench() {
    label="$1"; shift
    echo ""; echo "########## BENCH: ${label} ##########"
    maxf="/tmp/gpumax_${SLURM_JOB_ID}_${label}"
    sample_gpu "$maxf" & sampler=$!
    start=$(date +%s)
    python -m medgen.scripts.train ${COMMON} "$@" hydra.run.dir="${SCRATCH}/${label}"
    rc=$?
    end=$(date +%s)
    kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
    peak=$(cat "$maxf" 2>/dev/null || echo "?")
    line=">>> ${label}: exit=${rc}  wall=$((end - start))s  peak_gpu=${peak}MB"
    echo "$line"; echo "$line" >> "$SUMMARY"
}

# Previous implementation (2D RadImageNet) vs new TRUE-3D MedicalNet, compile on/off.
run_bench "2d_compile"   training.perceptual_network_type=radimagenet_resnet50          training.perceptual_compile=true
run_bench "2d_nocompile" training.perceptual_network_type=radimagenet_resnet50          training.perceptual_compile=false
run_bench "3d_nocompile" training.perceptual_network_type=medicalnet_resnet10_23datasets training.perceptual_compile=false
run_bench "3d_compile"   training.perceptual_network_type=medicalnet_resnet10_23datasets training.perceptual_compile=true

echo ""; echo "########## SUMMARY ##########"
cat "$SUMMARY"
echo ""
echo "Per-step rate: read the tqdm 'it/s' in each ########## BENCH block above"
echo "(wall includes model/data setup; it/s is the clean per-step signal)."
