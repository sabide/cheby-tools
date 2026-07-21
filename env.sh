#!/usr/bin/env bash

module purge
module load GCC-GPU-5.0.0
module load boost/1.88.0-mpi
module load hdf5/1.14.6-mpi
module load cmake/4.0.3
module load python/3.12.1

export CC=mpicc
export CXX=mpicxx
export FC=mpif90
export HDF5_MPI=ON
export HDF5_DIR=$HDF5_ROOT
