#!/usr/bin/env bash
# Run a Python script under a hard memory cap so it can never freeze the laptop.
#
# Usage (from the repo root):
#   code/business_entity_resolution/experiments/run_capped.sh [--max-gb N] script.py [args...]
#
# Why this exists: a fixed cap is not enough.  On 2026-09-25 a 10 GB cap was set
# while only ~9 GB was actually free (VS Code, mysqld, GNOME use the rest), so the
# whole SYSTEM ran out first and the kernel killed VS Code and the desktop session.
#
# So the cap is computed at launch time:
#   cap = min(--max-gb, currently-available memory - HEADROOM)
# and the job runs as a systemd service unit with OOMPolicy=stop, which kills the
# script AND all its worker processes together when the cap is hit (a plain
# --scope can leave orphaned workers holding memory).
#
# No MemoryHigh (soft limit): on 2026-09-25 it throttled a run at 90% of the cap
# for 20 minutes at ~1% CPU instead of failing.  A clean stop is better -- the
# pipeline resumes from its saved stages.
set -euo pipefail

MAX_GB=8
HEADROOM_GB=2.5      # left for the desktop, editor and browser
MIN_GB=2             # refuse to start below this

if [[ "${1:-}" == "--max-gb" ]]; then MAX_GB="$2"; shift 2; fi
[[ $# -ge 1 ]] || { echo "usage: $0 [--max-gb N] script.py [args...]"; exit 2; }

avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
cap_mb=$(awk -v a="$avail_kb" -v h="$HEADROOM_GB" -v m="$MAX_GB" \
    'BEGIN { c = a/1024 - h*1024; if (c > m*1024) c = m*1024; printf "%d", c }')
if (( cap_mb < MIN_GB * 1024 )); then
    echo "Only $((avail_kb / 1024)) MB free -- close some apps first (need cap >= ${MIN_GB} GB)." >&2
    exit 3
fi
echo "[run_capped] free now: $((avail_kb / 1024)) MB  ->  memory cap: ${cap_mb} MB" >&2

exec systemd-run --user --wait --pipe --collect --quiet \
    -p MemoryMax="${cap_mb}M" \
    -p MemorySwapMax=0 -p OOMPolicy=stop \
    -p WorkingDirectory="$PWD" \
    -E PYTHONHASHSEED=0 -E OPENBLAS_NUM_THREADS=1 -E OMP_NUM_THREADS=1 -E MKL_NUM_THREADS=1 \
    -E PATH="$PATH" -E PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -u "$@"
