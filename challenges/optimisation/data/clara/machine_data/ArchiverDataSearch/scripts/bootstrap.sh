#!/usr/bin/env bash
# One-time setup: install dependencies into ./.venv and write an .hscfg so the
# h5pyd command-line tools point at the local HSDS instance.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${HSDS_PORT:-5101}"
USERNAME="${HSDS_USERNAME:-admin}"
PASSWORD="${HSDS_PASSWORD:-admin}"

if [[ ! -d .venv ]]; then
  echo "creating .venv"
  if command -v uv > /dev/null; then uv venv .venv; else python3 -m venv .venv; fi
fi

echo "installing dependencies into .venv"
if command -v uv > /dev/null; then
  uv pip install --python .venv/bin/python -e '.[server,dev]'
else
  .venv/bin/python -m pip install -e '.[server,dev]'
fi

# .hscfg is what hsls/hstouch and bare h5pyd read. The API itself is configured
# through ARCHIVER_* environment variables (see .env.example).
cat > .hscfg <<EOF
hs_endpoint = http://localhost:$PORT
hs_username = $USERNAME
hs_password = $PASSWORD
EOF
echo "wrote .hscfg"

echo
echo "next:"
echo "  scripts/hsds_start.sh                        # start HSDS"
echo "  .venv/bin/python -m archiver_search.ingest sample   # creates the domain"
echo "  .venv/bin/python -m archiver_search"
