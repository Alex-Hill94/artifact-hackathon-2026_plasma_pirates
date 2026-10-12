#!/usr/bin/env bash
# Stop the local HSDS instance started by scripts/hsds_start.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_DIR="$PWD/hsds_run"
PID_FILE="$RUN_DIR/hsds.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "no pid file at $PID_FILE; HSDS does not appear to be running"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  # hsds_start.sh used setsid, so the service and data nodes share a process
  # group with the launcher and one signal takes down the whole tree.
  PGID="$(ps -o pgid= "$PID" | tr -d ' ')"
  kill -TERM "-$PGID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$PID" 2>/dev/null; then
    kill -KILL "-$PGID" 2>/dev/null || true
  fi
  echo "stopped HSDS (pid $PID)"
else
  echo "pid $PID is not running"
fi
rm -f "$PID_FILE"
