#!/usr/bin/env bash
# Submit 14 independent generators and one afterok combined-metric job.

set -Eeuo pipefail

usage() {
    echo "usage: $0 [--dry-run]"
}

DRY_RUN=0
if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    [[ "$1" == "--dry-run" ]] || { usage >&2; exit 2; }
    DRY_RUN=1
fi

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PANEL_ID="${PANEL_ID:-source_selection_train105_seed42_euler100}"
readonly METRIC_JOB="IDUN/generate/eval_generator_panel_metrics.slurm"
readonly WRAPPERS=(
    IDUN/generate/eval_gen_exp1_1_1000.slurm
    IDUN/generate/eval_gen_exp1_1_1000plus.slurm
    IDUN/generate/eval_gen_exp32_2_1000.slurm
    IDUN/generate/eval_gen_exp32_3_1000.slurm
    IDUN/generate/eval_gen_exp47a.slurm
    IDUN/generate/eval_gen_exp47b.slurm
    IDUN/generate/eval_gen_exp47c.slurm
    IDUN/generate/eval_gen_exp47d.slurm
    IDUN/generate/eval_gen_exp47e.slurm
    IDUN/generate/eval_gen_exp48a.slurm
    IDUN/generate/eval_gen_exp48b.slurm
    IDUN/generate/eval_gen_exp48c.slurm
    IDUN/generate/eval_gen_exp48d.slurm
    IDUN/generate/eval_gen_exp48e.slurm
)

for script in "${WRAPPERS[@]}" "$METRIC_JOB" IDUN/generate/run_eval_generator.sh; do
    [[ -f "$script" ]] || { echo "FATAL: missing $script" >&2; exit 1; }
    bash -n "$script"
done

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run: validated shell syntax for 14 wrappers, shared runner, and metric job."
    for wrapper in "${WRAPPERS[@]}"; do
        echo "sbatch --parsable $wrapper"
    done
    echo "sbatch --parsable --dependency=afterok:<14-colon-separated-job-ids> $METRIC_JOB"
    exit 0
fi

command -v sbatch >/dev/null || { echo "FATAL: sbatch is not available" >&2; exit 1; }
PANEL_SOURCE_COMMIT="$(git rev-parse --verify HEAD)" || { echo "FATAL: could not resolve the repository commit" >&2; exit 1; }
export PANEL_SOURCE_COMMIT
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "FATAL: repository worktree is not clean; commit or remove every tracked and untracked change before submission" >&2
    exit 1
fi
mkdir -p IDUN/output/generate

echo "=== Preflighting exact checkpoints, train cases, and fresh final outputs ==="
echo "Pinned source commit: $PANEL_SOURCE_COMMIT"
for wrapper in "${WRAPPERS[@]}"; do
    PREFLIGHT_ONLY=1 SLURM_SUBMIT_DIR="$REPO_ROOT" bash "$wrapper"
done

job_ids=()
for wrapper in "${WRAPPERS[@]}"; do
    if ! raw_job_id="$(sbatch --parsable "$wrapper")"; then
        echo "FATAL: submission failed after these generator jobs were submitted: ${job_ids[*]:-none}" >&2
        exit 1
    fi
    job_id="${raw_job_id%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "FATAL: unexpected sbatch response: $raw_job_id" >&2; exit 1; }
    job_ids+=("$job_id")
    echo "Submitted $(basename "$wrapper"): $job_id"
done

dependency="$(IFS=:; printf '%s' "${job_ids[*]}")"
metric_raw="$(sbatch --parsable --dependency="afterok:${dependency}" "$METRIC_JOB")"
metric_job_id="${metric_raw%%;*}"
[[ "$metric_job_id" =~ ^[0-9]+$ ]] || { echo "FATAL: unexpected metric sbatch response: $metric_raw" >&2; exit 1; }

echo "Submitted combined metric job: $metric_job_id"
echo "Dependency: afterok:${dependency}"
echo "All 14 generators may run independently; metrics begin only if every generator succeeds."
