#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp1_chained"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp1_chained/%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp1_chained/%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp1 3D Pixel Diffusion: Bravo (chained 3h segments) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# === Job chaining support (set by misc/chain_submit.sh) ===
CHAIN_ARGS=""
if [ -n "${CHAIN_RUN_DIR:-}" ]; then
    CHAIN_ARGS="hydra.run.dir=${CHAIN_RUN_DIR}"
    echo "Chain mode: writing to ${CHAIN_RUN_DIR}"
fi
if [ -n "${CHAIN_RESUME:-}" ]; then
    if [ -f "${CHAIN_RESUME}" ]; then
        # Diagnostic: show file details to help debug corruption
        echo "Chain mode: found checkpoint at ${CHAIN_RESUME}"
        ls -la "${CHAIN_RESUME}"
        echo "File header (hex):" && xxd -l 32 "${CHAIN_RESUME}" 2>/dev/null || true
        # Lightweight validation: check zip header (PyTorch saves are zip files)
        if python -c "import zipfile; zipfile.ZipFile('${CHAIN_RESUME}').close()"; then
            CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CHAIN_RESUME}"
            echo "Chain mode: resuming from ${CHAIN_RESUME}"
        else
            echo "WARNING: checkpoint_latest.pt failed validation, trying checkpoint_best.pt"
            BEST_CKPT="$(dirname "${CHAIN_RESUME}")/checkpoint_best.pt"
            if [ -f "$BEST_CKPT" ]; then
                ls -la "$BEST_CKPT"
                echo "File header (hex):" && xxd -l 32 "$BEST_CKPT" 2>/dev/null || true
            fi
            if [ -f "$BEST_CKPT" ] && python -c "import zipfile; zipfile.ZipFile('${BEST_CKPT}').close()"; then
                CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${BEST_CKPT}"
                echo "Chain mode: resuming from ${BEST_CKPT} (fallback)"
            else
                echo "WARNING: no valid checkpoint found, starting fresh."
                echo "Directory listing:"
                ls -la "$(dirname "${CHAIN_RESUME}")"/*.pt* 2>/dev/null || echo "  (no .pt files found)"
            fi
        fi
    else
        echo "Chain mode: no checkpoint at ${CHAIN_RESUME}, starting fresh."
    fi
fi

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=default_3d \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.epochs=100 \
    training.learning_rate=1e-4 \
    training.name=exp1_chained_ \
    ${CHAIN_ARGS}

conda deactivate
