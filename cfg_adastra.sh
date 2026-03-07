#!/usr/bin/env bash
#set -euo pipefail

module purge
module load GCC-GPU-5.0.0
module load boost/1.88.0-mpi
module load cmake
module load python

# Optional overrides from user environment:
#   export CHEBY_BOOST_INCLUDE_DIR=/path/to/boost/include
#   export CHEBY_TECIO_INCLUDE_DIR=/path/to/tecio/teciosrc
#   export CHEBY_TECIO_LIBRARY=/path/to/libtecio.a

CHEBY_ADASTRA_CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release"

if [[ -n "${CHEBY_BOOST_INCLUDE_DIR:-}" ]]; then
  CHEBY_ADASTRA_CMAKE_FLAGS="${CHEBY_ADASTRA_CMAKE_FLAGS} -DCHEBY_BOOST_INCLUDE_DIR=${CHEBY_BOOST_INCLUDE_DIR}"
fi

if [[ -n "${CHEBY_TECIO_INCLUDE_DIR:-}" && -n "${CHEBY_TECIO_LIBRARY:-}" ]]; then
  CHEBY_ADASTRA_CMAKE_FLAGS="${CHEBY_ADASTRA_CMAKE_FLAGS} -DCHEBY_USE_BUNDLED_TECIO=OFF -DCHEBY_TECIO_INCLUDE_DIR=${CHEBY_TECIO_INCLUDE_DIR} -DCHEBY_TECIO_LIBRARY=${CHEBY_TECIO_LIBRARY}"
else
  CHEBY_ADASTRA_CMAKE_FLAGS="${CHEBY_ADASTRA_CMAKE_FLAGS} -DCHEBY_USE_BUNDLED_TECIO=ON"
fi

export CC=gcc
export CXX=g++
export CHEBY_ADASTRA_CMAKE_FLAGS
