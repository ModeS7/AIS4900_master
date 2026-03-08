#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp1n_test_genmetrics_v5"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/train/diffusion_3d/exp1n/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp1n/%j.err

# Test v5: OOM fix — two-phase generation/extraction
# Root cause: 3 feature extractors (~2.2 GB) stayed loaded on GPU during
# strategy.generate(), eating memory needed for Conv3d workspace (~18 GiB).
# Fix: Phase 1 generates all samples with extractors unloaded (max GPU memory),
#      Phase 2 loads extractors one at a time for feature extraction.
# Also: unload extractors after cache_reference_features + before defrag.
# IMPORTANT: Must sync code to cluster before submitting!

cd ${SLURM_SUBMIT_DIR}

echo "=== Test v5: Two-Phase 3D Gen Metrics ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Verify two-phase fix is deployed: check for Phase 1 log message
python -c "
import inspect, medgen.metrics.generation_sampling as m
src = inspect.getsource(m.generate_and_extract_features_3d_streaming)
assert 'Phase 1' in src and 'Phase 2' in src, 'ERROR: Two-phase fix not deployed! Sync code first.'
print('OK: Two-phase generation/extraction verified')
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
    training.name=exp1n_test_genmetrics_v5_
