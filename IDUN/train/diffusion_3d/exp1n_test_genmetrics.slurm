#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp1n_test_genmetrics_v6"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp1n/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp1n/%j.err

# Test v6: Fix OOM in diversity LPIPS + CUDA Graph pool accumulation
# v5 fixed generation (Phase 1 succeeded) but diversity LPIPS created 9.59 GiB
# of CUDA Graph private pools via torch.compile(mode="reduce-overhead").
# Fix: use_compile=False for 3D diversity LPIPS + clear_metric_caches before defrag
# IMPORTANT: Must sync code to cluster before submitting!

cd ${SLURM_SUBMIT_DIR}

echo "=== Test v6: Diversity LPIPS + CUDA Graph Fix ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Verify fixes are deployed
python -c "
import inspect
# Check two-phase streaming
import medgen.metrics.generation_sampling as gs
src = inspect.getsource(gs.generate_and_extract_features_3d_streaming)
assert 'Phase 1' in src and 'Phase 2' in src, 'ERROR: Two-phase fix not deployed!'
# Check diversity LPIPS use_compile=False default
import medgen.metrics.quality as q
src = inspect.getsource(q.compute_lpips_diversity_3d)
assert 'use_compile: bool = False' in src, 'ERROR: Diversity LPIPS use_compile=False not deployed!'
# Check clear_metric_caches in validation
import medgen.pipeline.validation as v
src = inspect.getsource(v.run_validation)
assert 'clear_metric_caches' in src, 'ERROR: clear_metric_caches not deployed!'
print('OK: All v6 fixes verified')
"

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=default_3d \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=3 \
    training.num_figures=3 \
    training.learning_rate=1e-4 \
    training.warmup_epochs=1 \
    training.gradient_clip_norm=0.5 \
    training.conditioning_dropout.prob=0.15 \
    training.generation_metrics.cfg_scale=2.0 \
    training.generation_metrics.cfg_mode=zero_star \
    training.name=exp1n_test_genmetrics_v6_
