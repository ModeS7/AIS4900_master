#!/usr/bin/env bash
# Nightly test runner for cron
# Logs output to REPO_ROOT/logs/nightly/
#
# Install:  crontab -e  then add:
#   0 3 * * * /home/mode/NTNU/AIS4900_master/misc/nightly_cron.sh
#
# Or run: ./misc/nightly_cron.sh --install

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/nightly"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/nightly_${DATE}.log"
CRON_ENTRY="0 3 * * * $REPO_ROOT/misc/nightly_cron.sh"

# ── Install/uninstall ────────────────────────────────────────────────────────

if [ "${1:-}" = "--install" ]; then
    { crontab -l 2>/dev/null | grep -v "nightly_cron.sh" || true; echo "$CRON_ENTRY"; } | crontab -
    echo "Installed cron job: runs daily at 3 AM"
    echo "  $CRON_ENTRY"
    echo "Logs: $LOG_DIR/"
    exit 0
fi

if [ "${1:-}" = "--uninstall" ]; then
    { crontab -l 2>/dev/null | grep -v "nightly_cron.sh" || true; } | crontab -
    echo "Removed nightly cron job"
    exit 0
fi

# ── Run tests ────────────────────────────────────────────────────────────────

cd "$REPO_ROOT"

# Activate venv
source "$REPO_ROOT/.venv/bin/activate"

echo "=== Nightly Tests: $DATE ===" | tee "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

"$REPO_ROOT/misc/local_ci.sh" --nightly >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "" >> "$LOG_FILE"
echo "Finished: $(date)" >> "$LOG_FILE"
echo "Exit code: $EXIT_CODE" >> "$LOG_FILE"

# Keep only last 30 days of logs
find "$LOG_DIR" -name "nightly_*.log" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE
