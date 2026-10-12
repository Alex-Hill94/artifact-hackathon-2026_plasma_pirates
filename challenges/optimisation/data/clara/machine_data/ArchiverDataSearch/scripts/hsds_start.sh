#!/usr/bin/env bash
# Start a local HSDS instance backed by a POSIX directory.
#
# State lives entirely under this checkout: object data in ./hsds_data, the
# password file and logs in ./hsds_run. Stop it again with scripts/hsds_stop.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT_DIR="${HSDS_ROOT_DIR:-$PWD/hsds_data}"
RUN_DIR="$PWD/hsds_run"
PORT="${HSDS_PORT:-5101}"
USERNAME="${HSDS_USERNAME:-admin}"
PASSWORD="${HSDS_PASSWORD:-admin}"

mkdir -p "$ROOT_DIR" "$RUN_DIR"

# HSDS needs its own password file; the format is user:password per line.
PASSWD_FILE="$RUN_DIR/passwd.txt"
if [[ ! -f "$PASSWD_FILE" ]]; then
  printf '%s:%s\n' "$USERNAME" "$PASSWORD" > "$PASSWD_FILE"
  chmod 600 "$PASSWD_FILE"
fi

if [[ -f "$RUN_DIR/hsds.pid" ]] && kill -0 "$(cat "$RUN_DIR/hsds.pid")" 2>/dev/null; then
  echo "HSDS already running (pid $(cat "$RUN_DIR/hsds.pid")) on port $PORT"
  exit 0
fi

echo "starting HSDS on http://localhost:$PORT (data in $ROOT_DIR)"
# setsid puts HSDS and the service/data nodes it spawns into their own process
# group, so hsds_stop.sh can take the whole tree down in one signal. Note that
# passing --socket_dir would make HSDS listen on unix sockets and ignore --port.
setsid hsds \
  --root_dir "$ROOT_DIR" \
  --host localhost \
  --port "$PORT" \
  --password_file "$PASSWD_FILE" \
  --logfile "$RUN_DIR/hsds.log" \
  --loglevel WARNING \
  --count "${HSDS_DN_COUNT:-4}" \
  > "$RUN_DIR/hsds.out" 2>&1 &
echo $! > "$RUN_DIR/hsds.pid"

for _ in $(seq 1 60); do
  if curl -fs -m 2 "http://localhost:$PORT/about" > /dev/null 2>&1; then
    echo "HSDS is up (pid $(cat "$RUN_DIR/hsds.pid"))"
    exit 0
  fi
  sleep 1
done

echo "HSDS did not come up; see $RUN_DIR/hsds.out and $RUN_DIR/hsds.log" >&2
exit 1
