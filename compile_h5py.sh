module purge
module load GCC-CPU-5.0.0
module load hdf5/1.14.6-mpi
module load python/3.12.1

export CC=mpicc
export HDF5_MPI=ON
export HDF5_DIR=$HDF5_ROOT

python3 -m pip install --user --upgrade pip setuptools wheel packaging
python3 -m pip install --user --no-binary=h5py --no-cache-dir --force-reinstall h5py
python3 -c "import h5py; print(h5py.__file__); print('MPI =', h5py.get_config().mpi)"
