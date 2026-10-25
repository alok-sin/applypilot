#!/usr/bin/env bash
set -uo pipefail

show_usage() {
  cat <<EOF
Usage: $0 <STEP1_PID> <CONFIG_FILE>
CONFIG_FILE format:
  Each non-empty, non-comment line has:
    <interval_seconds> <command...>
  Example:
    10 /usr/local/bin/step2_quick.sh arg1
    60 /usr/bin/python3 /path/to/step2_medium.py --opt
Note: Commands are executed with 'sh -c' and logs are written to the watcher log dir.
EOF
}

if [ "$#" -ne 2 ]; then
  show_usage
  exit 2
fi

PID="$1"
CFG="$2"

if ! kill -0 "$PID" 2>/dev/null; then
  echo "Error: no process with PID $PID"
  exit 3
fi

if [ ! -f "$CFG" ]; then
  echo "Error: config file not found: $CFG"
  exit 4
fi

declare -a INTERVALS
declare -a CMDS
while IFS= read -r line || [ -n "$line" ]; do
  # strip inline comments and surrounding whitespace
  line="${line%%#*}"
  line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [ -z "$line" ] && continue
  interval="${line%%[[:space:]]*}"
  cmd="${line#${interval}}"
  cmd="$(echo "$cmd" | sed -e 's/^[[:space:]]*//')"
  if ! [[ "$interval" =~ ^[0-9]+$ ]]; then
    echo "Invalid interval in line: $line"
    exit 5
  fi
  INTERVALS+=("$interval")
  CMDS+=("$cmd")
done <"$CFG"

N=${#INTERVALS[@]}
if [ "$N" -eq 0 ]; then
  echo "No tasks defined in $CFG"
  exit 6
fi

declare -a LAST_RUN
for ((i=0;i<N;i++)); do
  LAST_RUN[$i]=0
done

LOG_DIR="/tmp/step2_watcher_${PID}_$(date +%Y%m%d%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Watcher started for step1 PID=$PID; log: $LOG_DIR"
echo "Tasks:"
for ((i=0;i<N;i++)); do
  echo "  [$i] every ${INTERVALS[$i]}s -> ${CMDS[$i]}"
done

trap 'echo "Terminated"; exit 0' SIGINT SIGTERM

while kill -0 "$PID" 2>/dev/null; do
  now=$(date +%s)
  for ((i=0;i<N;i++)); do
    last=${LAST_RUN[$i]}
    interval=${INTERVALS[$i]}
    if [ "$last" -eq 0 ] || [ $((now - last)) -ge "$interval" ]; then
      ts=$(date -u +%FT%TZ)
      echo "$ts running task $i: ${CMDS[$i]}" >>"$LOG_DIR/watcher.log"
      sh -c "${CMDS[$i]}" >>"$LOG_DIR/task_$i.out" 2>>"$LOG_DIR/task_$i.err" &
      child_pid=$!
      echo "$ts started pid=$child_pid" >>"$LOG_DIR/task_${i}.pid"
      LAST_RUN[$i]=$now
    fi
  done
  sleep 1
done

echo "Step1 (PID $PID) no longer running; exiting; logs in $LOG_DIR"
exit 0
