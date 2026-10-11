#!/usr/bin/env bash
# Create the Python environment for the forecasting challenge and register it in Jupyter.
#
# Usage (from anywhere):   ./setup_venv.sh
# Pick another Python:     PYTHON=python3.12 ./setup_venv.sh
# Then open kickoff_notebook.ipynb and choose the kernel "Python (forecasting)".
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is needed"'

# Create python environment
"$PYTHON" -m venv venv

# Activate it
source venv/bin/activate

# Keep packaging tools current
python -m pip install --upgrade pip setuptools wheel

# Install packages
python -m pip install -r requirements.txt ipykernel

# Register a kernel for this env
python -m ipykernel install --user --name forecasting-venv --display-name "Python (forecasting)"

echo
echo "Done. Activate the environment with:  source venv/bin/activate"
echo "In Jupyter, pick the kernel:           Python (forecasting)"
