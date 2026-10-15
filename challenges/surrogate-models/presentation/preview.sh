#!/usr/bin/env bash
# Live preview of the whole deck at http://localhost:$PORT/.
#
# Quarto does not watch files pulled in with {{< include >}}, so this polls
# sections/ and re-renders the deck whenever a slide file changes; the running
# preview server then reloads the browser.
#
# Usage: ./preview.sh [extra quarto preview flags, e.g. --no-browser]
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-4848}"
quarto preview index.md --port "$PORT" "$@" &
quarto_pid=$!

stamp="$(mktemp)"
trap 'kill "$quarto_pid" 2>/dev/null; rm -f "$stamp"' EXIT INT TERM

while kill -0 "$quarto_pid" 2>/dev/null; do
  sleep 1
  if [ -n "$(find sections -type f -newer "$stamp" -print -quit)" ]; then
    touch "$stamp"
    quarto render index.md --quiet || true
  fi
done
