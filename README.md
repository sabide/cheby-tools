cmake -Wno-dev .. -DTECIO_INCLUDE_DIR=/Users/abides/workdir/git/H5toStar/tecio/teciosrc/ -DTECIO_LIBRARY=/Users/abides/workdir/git/H5toStar/tecio/teciosrc/build/libtecio.a
conda activate PyTecplot

export PYTHONPATH=/Users/abides/workdir/git/cheby-tools:$PYTHONPATH


module load python
CC=gcc CXX=g++ cmake ..   -Wno-dev -DBoost_INCLUDE_DIR=/lus/work/CT2A/c1916929/SHARED/local/boost-1.90.0/include/ -DTECIO_INCLUDE_DIR=../external/tecio/teciosrc/ -DTECIO_LIBRARY=/lus/scratch/CT2A/c1916929/sabide/cheby-tools/external/tecio/teciosrc/build/libtecio.a

export PYTHONPATH=/lus/scratch/CT2A/c1916929/sabide/cheby-tools/:$PYTHONPATH

module load GCC-GPU-5.0.0
module load boost/1.88.0-mpi




