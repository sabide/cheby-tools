# cheby-tools-adastra

Install:
- `tecio_wrapper`
- `discr`
- `stats`

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
import discr, stats, tecio_wrapper
ops = discr.discr_2d(xmin=[0, 0], xmax=[1, 1], n=[16, 16])
```

## 3) Useful options

Install only the post-processing tools (`discr`, `stats`):

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

## 5) Adastra notes

See [`cfg_adastra.sh`](./cfg_adastra.sh) for a reproducible build script using environment variables.

Typical workflow:

```bash
source cfg_adastra.sh
cmake -S . -B build-adastra $CHEBY_ADASTRA_CMAKE_FLAGS
cmake --build build-adastra -j
cmake --install build-adastra --prefix build-adastra/install
```
