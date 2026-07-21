source /lus/work/CT2A/c1916929/SHARED/opt/post/bin/activate

SITEPKG=$(python -c "import site; print(site.getsitepackages()[0])")
PREFIX=$(python -c "import sys; print(sys.prefix)")
REL_SITEPKG=$(python -c "import os, site, sys; print(os.path.relpath(site.getsitepackages()[0], sys.prefix))")

cmake .. \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCHEBY_PYTHON_INSTALL_DIR="$REL_SITEPKG" \
  -DCHEBY_BOOST_INCLUDE_DIR=/lus/work/CT2A/c1916929/SHARED/opt/boost-1.88.0/include \
  -DPython_EXECUTABLE="$(which python)"

cmake --build . -j
cmake --install .
