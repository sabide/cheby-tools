#!/usr/bin/env bash
set -euo pipefail

: "${CHEBY_BOOST_INCLUDE_DIR:?Source env.sh before running this script}"
: "${CHEBY_PYTHON_ENV:?Source env.sh before running this script}"

if [[ "${VIRTUAL_ENV:-}" != "$CHEBY_PYTHON_ENV" ]]; then
  echo "Expected active Python environment: $CHEBY_PYTHON_ENV" >&2
  echo "Run: source env.sh" >&2
  exit 1
fi

if [[ ! -f "${CHEBY_BOOST_INCLUDE_DIR}/boost/version.hpp" ]]; then
  echo "Boost headers not found in ${CHEBY_BOOST_INCLUDE_DIR}" >&2
  exit 1
fi

SITEPKG=$(python -c "import site; print(site.getsitepackages()[0])")
PREFIX=$(python -c "import sys; print(sys.prefix)")
REL_SITEPKG=$(python -c "import os, site, sys; print(os.path.relpath(site.getsitepackages()[0], sys.prefix))")

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCHEBY_PYTHON_INSTALL_DIR="$REL_SITEPKG" \
  -DCHEBY_BOOST_INCLUDE_DIR="$CHEBY_BOOST_INCLUDE_DIR" \
  -DPython_EXECUTABLE="$CHEBY_PYTHON_ENV/bin/python"

cmake --build . -j
cmake --install .

"$CHEBY_PYTHON_ENV/bin/python" "$(dirname "$0")/check_python_env.py"
