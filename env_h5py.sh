module purge
module load cpe/24.07
module load PrgEnv-gnu/8.5.0
module load cray-mpich/8.1.30
module load cray-python/3.11.7
module load cray-hdf5-parallel/1.14.3.1

export CC=cc
export HDF5_MPI=ON
export HDF5_DIR=$HDF5_ROOT
