# cheby-tools-adastra

Install:
- `tecio_wrapper`
- `discr`
- `stats`
- `spec_forge`

## 1) Clone with submodules

```bash
git clone <repo-url>
cd cheby-tools-adastra
git submodule update --init --recursive
```

`pybind11` is taken from the repository submodule at `external/pybind11`.

## 2) Build and install

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix build/install
```

Set Python path to the installed package:

```bash
export PYTHONPATH="$PWD/build/install/lib/python:$PYTHONPATH"
```

You can then use:

```python
import stats, tecio_wrapper
from spec_forge import SpectralDiscretization, SpectralInterpolate

ops = SpectralDiscretization(
    xmin=[0, 0],
    xmax=[1, 1],
    n=[16, 16],
    bases=["chebyshev", "chebyshev"],
)
```

## 3) Useful options

Install only the post-processing tools (`stats`, `spec_forge`, plus the
deprecated `discr` compatibility facade):

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCHEBY_INSTALL_TECIO_WRAPPER=OFF \
  -DCHEBY_INSTALL_POSTPROCESSING_TOOLS=ON
cmake --install build --prefix build/install
```

Install only `tecio_wrapper`:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCHEBY_INSTALL_POSTPROCESSING_TOOLS=OFF
cmake --build build -j
cmake --install build --prefix build/install
```

Use an external/prebuilt TecIO:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCHEBY_USE_BUNDLED_TECIO=OFF \
  -DCHEBY_TECIO_INCLUDE_DIR=/path/to/tecio/teciosrc \
  -DCHEBY_TECIO_LIBRARY=/path/to/libtecio.a
cmake --build build -j
cmake --install build --prefix build/install
```

## 4) Notes

- Prerequisites: `cmake`, `python`, compiler toolchain, and `boost`.
- The default build uses bundled TecIO from the repository and system Boost.
- If Boost is installed in a non-standard location, pass its include directory explicitly:

```bash
cmake -S . -B build \
  -DCHEBY_BOOST_INCLUDE_DIR=/opt/homebrew/include
```

Only the Boost headers are required; no compiled Boost library is linked.
`CHEBY_BOOST_INCLUDE_DIR` must point to the directory containing
`boost/version.hpp`.

With a modern system Boost, no option is needed: CMake uses `Boost::headers`
(or `Boost::boost` on older CMake/Boost configurations).

Boost 1.88 headers required by TecIO are vendored under:

```text
external/boost/boost/version.hpp
```

They are detected automatically, so the default build needs no system Boost
installation and no Boost-related CMake option. Alternative headers can still
be selected explicitly:

```bash
cmake -S . -B build \
  -DCHEBY_BOOST_INCLUDE_DIR=/path/to/boost-root
```

On Adastra, as on macOS, the project deliberately uses the repository-local
header-only copy and does not load or link a Boost library module:

```bash
source env.sh
```

## 5) Adastra notes

See [`cfg_adastra.sh`](./cfg_adastra.sh) for a reproducible build script using environment variables.

Typical workflow:

```bash
source cfg_adastra.sh
cmake -S . -B build-adastra $CHEBY_ADASTRA_CMAKE_FLAGS
cmake --build build-adastra -j
cmake --install build-adastra --prefix build-adastra/install
```

### Project Python environment

`cheby-tools` uses one Python 3.12 environment for NumPy, MPI-enabled h5py,
mpi4py, the pure Python packages, and the compiled `tecio_wrapper` extension:

```text
/lus/work/CT2A/c1916929/SHARED/opt/post
```

Activate and validate it with:

```bash
source env.sh
python check_python_env.py
```

Do not use `pip --user`: packages must be installed inside this environment so
that Python, NumPy, h5py, mpi4py and the extension share the same ABI.

The corresponding HPC runtime is fixed in `env.sh`: GNU programming
environment, Cray MPICH 8.1.30 and parallel HDF5 1.14.3.1. Changing that stack
requires rebuilding both mpi4py and h5py in the same Python environment.

### Spectral API and tests

`spec_forge` is the supported spectral API:

```python
from spec_forge import SpectralDiscretization, SpectralInterpolate
```

The historical `discr.discr_2d` API remains installed as a deprecated
compatibility facade backed by `spec_forge`; new code should not use it.

Run the numerical and compatibility tests from the source tree with:

```bash
python -m unittest discover -s tests -v
```
