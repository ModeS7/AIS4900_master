#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name="merge_sm_48_tail300"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/merge_sm_exp48_tail300_%j.out
#SBATCH --error=IDUN/output/generate/merge_sm_exp48_tail300_%j.err

# Run after the five original exp48 jobs have passed 300 and stopped, and after
# all five tail300 jobs have completed:
#   sbatch IDUN/generate/merge_synthmask_exp48_tail300.slurm

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly LABELS=(
    exp1_to_exp48a_t025
    exp1_to_exp48b_t025
    exp1_to_exp48c_t025
    exp1_to_exp48d_t025
    exp1_to_exp48e_t025
)
readonly LOG_STEMS=(exp48a exp48b exp48c exp48d exp48e)
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
readonly LOG_DIR="$REPO_ROOT/IDUN/output/generate"

command -v sacct >/dev/null 2>&1 || fatal "sacct is unavailable"

ORIGINAL_LOGS=()
ORIGINAL_JOB_IDS=()
for ((index = 0; index < ${#LABELS[@]}; index++)); do
    log_stem="${LOG_STEMS[index]}"
    mapfile -t matching_logs < <(
        find "$LOG_DIR" -maxdepth 1 -type f \
            -name "gen_sm_${log_stem}_[0-9]*.out" -printf '%p\n' | LC_ALL=C sort -V
    )
    (( ${#matching_logs[@]} > 0 )) || fatal "no original $log_stem stdout log found"

    head_log="${matching_logs[$((${#matching_logs[@]} - 1))]}"
    log_name="${head_log##*/}"
    [[ "$log_name" =~ ^gen_sm_${log_stem}_([0-9]+)\.out$ ]] || fatal \
        "could not read original job ID from $log_name"
    head_job_id="${BASH_REMATCH[1]}"

    if ! sacct_output="$(
        sacct -X -n -P -j "$head_job_id" --format=JobIDRaw,State 2>&1
    )"; then
        fatal "could not query original job $head_job_id: $sacct_output"
    fi
    head_state="$(
        awk -F'|' -v job_id="$head_job_id" '
            $1 == job_id {
                state = $2
                sub(/[ +].*$/, "", state)
                print state
                exit
            }
        ' <<< "$sacct_output"
    )"
    case "$head_state" in
        BOOT_FAIL|CANCELLED|COMPLETED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|TIMEOUT)
            ;;
        "")
            fatal "could not confirm terminal state for original job $head_job_id"
            ;;
        *)
            fatal "original job $head_job_id is not stopped ($head_state)"
            ;;
    esac

    head_progress="$(
        sed -n 's/.*Progress: \([0-9][0-9]*\)\/525.*/\1/p' "$head_log" | tail -n 1
    )"
    [[ "$head_progress" =~ ^[0-9]+$ ]] || fatal "no progress found in $head_log"
    (( head_progress >= 300 )) || fatal \
        "$log_stem reached only $head_progress/525; it must reach at least 300/525"

    ORIGINAL_LOGS+=("$head_log")
    ORIGINAL_JOB_IDS+=("$head_job_id")
done

for ((index = 0; index < ${#LABELS[@]}; index++)); do
    MERGE_TAG="head${ORIGINAL_JOB_IDS[index]}_merge${SLURM_JOB_ID}" \
    SYNTHMASK_HEAD_STOP_CONFIRMED=true \
        bash "$REPO_ROOT/IDUN/generate/merge_synthmask_exp48_tail300.sh" \
            "${LABELS[index]}" \
            "${ORIGINAL_LOGS[index]}"
done

echo "Merged all five exp48 head/tail datasets"
