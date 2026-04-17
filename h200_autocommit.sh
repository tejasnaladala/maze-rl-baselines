#!/bin/bash
# Auto-commit + push results to GitHub every N minutes during H200 run.
# Run this in a separate tmux pane or background.
#
# Usage:
#   bash h200_autocommit.sh
#
# Stops when no python processes are running (i.e. all jobs done).

set -uo pipefail

INTERVAL_SECONDS=${1:-600}  # default: 10 minutes

cd "$(dirname "$0")"

# Configure git for autonomous commits
git config user.email "h200-autobot@engram.local"
git config user.name "H200 Autobot"

while true; do
    # Stop if no python launchers running
    if ! pgrep -f "launch_.*\.py" > /dev/null; then
        echo "$(date) -- no launchers running, exiting autocommit loop"
        break
    fi

    # Commit progress
    cd "$(git rev-parse --show-toplevel)"
    git add raw_results/ logs/ analysis_output/ paper_figures/ 2>/dev/null || true
    if ! git diff --cached --quiet; then
        n_runs=$(find raw_results -name '*.json' ! -name 'checkpoint.json' 2>/dev/null | wc -l)
        msg="h200 autocommit: $(date -u +%Y-%m-%dT%H:%M:%SZ) -- $n_runs result files"
        git commit -m "$msg" --quiet
        if git push origin main --quiet 2>/dev/null; then
            echo "$(date) -- pushed: $msg"
        else
            echo "$(date) -- push failed (will retry next loop)"
        fi
    else
        echo "$(date) -- no changes to commit"
    fi

    sleep "$INTERVAL_SECONDS"
done

echo "$(date) -- final push"
git add raw_results/ logs/ analysis_output/ paper_figures/ 2>/dev/null || true
if ! git diff --cached --quiet; then
    git commit -m "h200 final autocommit: $(date -u +%Y-%m-%dT%H:%M:%SZ)" --quiet
    git push origin main || echo "final push failed"
fi
