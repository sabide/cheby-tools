# One-command native installation design

## Objective

Make the editable installation of `cheby-tools` build and install the native
TecIO backend as part of the normal `pip` operation:

```bash
python -m pip install -e ../cheby-tools-update
```

TecIO is enabled and required by default. A deliberate Python-only installation
remains available with:

```bash
python -m pip install -e ../cheby-tools-update \
  -Ccmake.define.CHEBY_INSTALL_TECIO=OFF
```

The initial supported platforms are Linux, including ADASTRA, and macOS.
Windows support is outside this change.

## Source checkout and build prerequisites

The checkout must initialize the pybind11 submodule:

```bash
git clone --recurse-submodules -b update \
  git@github.com:sabide/cheby-tools.git cheby-tools-update
```

The build must use the TecIO and Boost sources stored in the repository and the
checked-out pybind11 submodule. CMake must not download dependencies implicitly.
If the submodule is absent, configuration must fail with an actionable message
containing:

```bash
git submodule update --init --recursive
```

The host environment must provide Python 3.11 or newer, a C++ compiler and
CMake. The Python build frontend must be able to obtain the declared
`scikit-build-core` build requirement from its configured package index or
cache.

## Packaging architecture

Replace the setuptools build backend with `scikit-build-core`. It is responsible
for both parts of the distribution:

1. exposing the `cheby_tools` Python package from the source tree for editable
   installations; and
2. configuring, compiling and installing the CMake `_tecio` extension into the
   `cheby_tools` package.

The Python package must be included through scikit-build-core package discovery.
During a pip build, CMake must not install a second copy of the pure-Python
package. The existing CMake-only installation path may retain that capability
for advanced native development.

The CMake install destination for `_tecio` must adapt to the build context:

- under scikit-build-core, install into `cheby_tools` in the wheel/editable
  staging tree;
- under a direct CMake install, preserve the existing configurable
  `CHEBY_PYTHON_INSTALL_DIR/cheby_tools` destination.

When `CHEBY_INSTALL_TECIO=OFF` is passed through pip, a valid Python-only
distribution must still be produced even though CMake has no native extension
to install. The current direct-CMake guard against an empty installation must
remain effective outside the scikit-build context.

## Editable behavior

Use scikit-build-core's default redirect-style editable installation. Python
source changes must be visible without reinstalling the package. Automatic
native rebuild-on-import is intentionally disabled because it is experimental
and would make imports invoke the compiler unexpectedly.

After changing C++ or native build configuration, the developer reruns:

```bash
python -m pip install -e ../cheby-tools-update
```

The same CMake integration must support regular native wheels. Wheels produced
with TecIO enabled are platform- and Python-specific, not pure-Python wheels.

## Public API and failure behavior

The public imports remain unchanged:

```python
from cheby_tools import Field, SpectralDiscretization
from cheby_tools.tecio import write_plt
```

`Field` and `SpectralDiscretization` must work in both installation modes. In a
Python-only installation, importing `cheby_tools.tecio` remains valid. Calling
`write_plt` must raise `ImportError` with an actionable message explaining that
the TecIO backend is absent and that reinstalling without the opt-out enables
it.

With TecIO enabled by default, missing compiler tools, missing sources, CMake
configuration failures and native compilation failures must fail the pip
installation. They must not silently degrade to a Python-only installation.

## Documentation

The README's primary installation path must become:

```bash
git clone --recurse-submodules -b update \
  git@github.com:sabide/cheby-tools.git cheby-tools-update
python -m venv post-processing/.venv
source post-processing/.venv/bin/activate
python -m pip install -e ./cheby-tools-update
```

The exact relative path may vary with the user's directory layout; the text
must explain that the package is installed into the currently active virtual
environment. The README must also document:

- Linux/ADASTRA and macOS as the supported targets;
- compiler, CMake and submodule prerequisites;
- the explicit `CHEBY_INSTALL_TECIO=OFF` command;
- the need to rerun pip after native source changes; and
- direct CMake installation as an advanced workflow rather than the normal
  user path.

## Verification

Automated verification must cover the following boundaries:

1. In a clean virtual environment, the exact default editable command succeeds,
   `cheby_tools._tecio` imports, and `write_plt` writes a valid classic `.plt`
   file.
2. In another clean virtual environment, the opt-out command succeeds;
   `Field` and `SpectralDiscretization` work; and `write_plt` raises the expected
   actionable `ImportError`.
3. A regular wheel builds, is tagged as native, installs in a clean environment
   and provides the working `_tecio` extension.
4. Existing pure-Python, TecIO API and native TecIO tests pass.
5. The direct CMake configure/build/install workflow still works.
6. Build products and virtual-environment files do not become tracked source
   files.
