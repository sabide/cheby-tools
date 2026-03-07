# cheby-tools-adastra

Build and install the Python module `tecio_wrapper` for:
- local macOS development
- Adastra (HPC) builds

## 1) Clone with submodules

```bash
git clone <repo-url>
cd cheby-tools-adastra
git submodule update --init --recursive
```

## 2) Build with bundled TecIO (recommended)

This mode builds `external/tecio/teciosrc` and links the Python extension against it.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCHEBY_USE_BUNDLED_TECIO=ON
cmake --build build -j
cmake --install build --prefix build/install
```

Set Python path to the installed package:

```bash
export PYTHONPATH="$PWD/build/install/lib/python:$PYTHONPATH"
```

You can also use presets:

```bash
cmake --preset macos
cmake --build --preset macos
```

## 3) Build with external/prebuilt TecIO

Use this when TecIO is already compiled elsewhere.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCHEBY_USE_BUNDLED_TECIO=OFF \
  -DCHEBY_TECIO_INCLUDE_DIR=/path/to/tecio/teciosrc \
  -DCHEBY_TECIO_LIBRARY=/path/to/libtecio.a
cmake --build build -j
cmake --install build --prefix build/install
```

## 4) macOS notes

- Install prerequisites (example): `cmake`, `python`, compiler toolchain, and `boost`.
- If Boost headers are not auto-detected when building bundled TecIO:

```bash
cmake -S . -B build \
  -DCHEBY_USE_BUNDLED_TECIO=ON \
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

Preset alternative:

```bash
cmake --preset adastra-bundled
cmake --build --preset adastra-bundled
```


