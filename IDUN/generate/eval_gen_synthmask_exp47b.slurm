#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_sm_exp47b"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/gen_sm_exp47b_%j.out
#SBATCH --error=IDUN/output/generate/gen_sm_exp47b_%j.err

set -Eeuo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec bash "$REPO_ROOT/IDUN/generate/run_eval_generator_synthmask.sh" \
    exp47b \
    exp47b_huber_lpips_lowt_20260425-212907
