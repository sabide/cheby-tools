#!/usr/bin/env bash

module purge
module load cpe/24.07
module load PrgEnv-gnu/8.5.0
module load cray-mpich/8.1.30
module load cray-hdf5-parallel/1.14.3.1
module load cmake/4.0.3
module load python/3.12.1

# One Python environment for cheby-tools (Python packages and C++ extension).
export CHEBY_PYTHON_ENV=/lus/work/CT2A/c1916929/SHARED/opt/post

if [[ ! -x "${CHEBY_PYTHON_ENV}/bin/python" ]]; then
    echo "Python environment not found in ${CHEBY_PYTHON_ENV}" >&2
    return 1 2>/dev/null || exit 1
fi

source "${CHEBY_PYTHON_ENV}/bin/activate"

if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
    echo "cheby-tools requires Python 3.12" >&2
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
export FC=ftn
export HDF5_MPI=ON
export HDF5_DIR=$HDF5_ROOT
