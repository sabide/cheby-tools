#!/usr/bin/env bash
set -euo pipefail

: "${CHEBY_PYTHON_ENV:?Source env.sh before running this script}"

if [[ "${VIRTUAL_ENV:-}" != "$CHEBY_PYTHON_ENV" ]]; then
    echo "Expected active Python environment: $CHEBY_PYTHON_ENV" >&2
    echo "Run: source env.sh" >&2
    exit 1
fi

export CC=mpicc
export HDF5_MPI=ON
export HDF5_DIR=$HDF5_ROOT

python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install --no-binary=h5py --no-cache-dir --force-reinstall h5py==3.12.1
python "$(dirname "$0")/check_python_env.py"
