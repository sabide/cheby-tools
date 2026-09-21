#!/usr/bin/env bash

module purge
module load cpe/24.07
module load PrgEnv-gnu/8.5.0
module load cmake/4.0.3
module load python/3.12.1

: "${CHEBY_PYTHON_ENV:?Set CHEBY_PYTHON_ENV to a writable Python environment path}"

if [[ ! -x "${CHEBY_PYTHON_ENV}/bin/python" ]]; then
    echo "Python environment not found: ${CHEBY_PYTHON_ENV}" >&2
    return 1 2>/dev/null || exit 1
fi

source "${CHEBY_PYTHON_ENV}/bin/activate"

if ! python -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
    echo "cheby-tools requires Python 3.11 or newer" >&2
    return 1 2>/dev/null || exit 1
fi

# TecIO only needs the repository-local Boost headers. Do not load/link a
# Boost library module.
CHEBY_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CHEBY_BOOST_INCLUDE_DIR="${CHEBY_PROJECT_ROOT}/external/boost"

if [[ ! -f "${CHEBY_BOOST_INCLUDE_DIR}/boost/version.hpp" ]]; then
    echo "Boost headers not found in ${CHEBY_BOOST_INCLUDE_DIR}" >&2
    return 1 2>/dev/null || exit 1
fi

export CC=cc
export CXX=CC
