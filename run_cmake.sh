#!/usr/bin/env bash
set -euo pipefail

: "${CHEBY_PYTHON_ENV:?Source env.sh before running this script}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${CHEBY_BUILD_DIR:-${PROJECT_ROOT}/build}"
: "${CHEBY_BOOST_INCLUDE_DIR:=${PROJECT_ROOT}/external/boost}"

if [[ "${VIRTUAL_ENV:-}" != "$CHEBY_PYTHON_ENV" ]]; then
  echo "Expected active Python environment: $CHEBY_PYTHON_ENV" >&2
  echo "Run: source env.sh" >&2
  exit 1
fi

if [[ ! -f "${CHEBY_BOOST_INCLUDE_DIR}/boost/version.hpp" ]]; then
  echo "Boost headers not found in ${CHEBY_BOOST_INCLUDE_DIR}" >&2
  exit 1
fi

REL_SITEPKG=$(python -c "import os, site, sys; print(os.path.relpath(site.getsitepackages()[0], sys.prefix))")

cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_INSTALL_PREFIX="${CHEBY_PYTHON_ENV}" \
  -DCHEBY_PYTHON_INSTALL_DIR="$REL_SITEPKG" \
  -DCHEBY_BOOST_INCLUDE_DIR="$CHEBY_BOOST_INCLUDE_DIR" \
  -DPython_EXECUTABLE="$CHEBY_PYTHON_ENV/bin/python"

cmake --build "${BUILD_DIR}" -j
cmake --install "${BUILD_DIR}"

"$CHEBY_PYTHON_ENV/bin/python" "${PROJECT_ROOT}/check_python_env.py"
