#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<EOF
Usage: $0 <WATCHER_CONFIG> -- <STEP1_COMMAND...>
Example:
  $0 scripts/step2_watcher_example.cfg -- python3 -u my_step1.py --flag

This starts the STEP1 command in background, launches the watcher pointed
at STEP1's PID, and waits for both to finish.
EOF
}

if [ "$#" -lt 3 ]; then
  show_usage
  exit 2
fi

CFG="$1"
shift
if [ "$1" != "--" ]; then
  echo "Missing separator '--' before step1 command"
  show_usage
  exit 2
fi
shift

if [ ! -f "$CFG" ]; then
  echo "Watcher config not found: $CFG"
  exit 3
fi

STEP1_CMD="$*"

echo "Starting step1: $STEP1_CMD"
bash -lc "$STEP1_CMD" &
STEP1_PID=$!
echo "Started step1 PID=$STEP1_PID"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/step2_watcher.sh" "$STEP1_PID" "$CFG" &
WATCHER_PID=$!
echo "Started watcher PID=$WATCHER_PID for step1 PID=$STEP1_PID"

wait "$STEP1_PID"
STEP1_EXIT=$?
echo "Step1 (PID $STEP1_PID) exited with code $STEP1_EXIT"

# watcher will detect that the PID is gone and exit; wait for it to finish
wait "$WATCHER_PID" || true
echo "Watcher exited; pipeline finished."
exit $STEP1_EXIT
