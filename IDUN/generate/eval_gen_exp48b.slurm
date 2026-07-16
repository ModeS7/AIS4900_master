#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_exp1_to_exp48b"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/gen_exp1_to_exp48b_%j.out
#SBATCH --error=IDUN/output/generate/gen_exp1_to_exp48b_%j.err

set -Eeuo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec bash "$REPO_ROOT/IDUN/generate/run_eval_generator.sh" \
    exp1_to_exp48b_t025 \
    exp48b_lowt_only_l1_lpips_20260425-202256 \
    exp1_1_1000_pixel_bravo_20260402-121556 \
    0.25
