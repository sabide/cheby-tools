# Field-focused cheby-tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the historical statistics/application packages with one clean `cheby_tools` API centered on spectral `Field` objects and optional multi-field Tecplot `.plt` output.

**Architecture:** Move the validated spectral implementation behind the `cheby_tools` namespace, then layer a small `Field` object over it. Keep TecIO optional through a pure-Python adapter and a private pybind11 extension; package only the Python core while CMake installs the native backend into the same namespace.

**Tech Stack:** Python 3.11+, NumPy 1.23+, `unittest`, setuptools/PEP 517, CMake 3.18+, C++14, pybind11, TecIO 142.

**Spec:** `docs/superpowers/specs/2026-09-21-field-focused-cleanup-design.md`

## Global Constraints

- Public import: `from cheby_tools import Field, SpectralDiscretization`.
- No aliases for `spec_forge`, `discr`, or `stats`.
- Keep one-, two-, and three-dimensional Chebyshev/Fourier grids and the validated Nyquist behavior.
- Require `Field.values.shape == tuple(Field.discretization.n)` without implicit reshape.
- Keep `requires-python = ">=3.11"`, `numpy>=1.23`, and no other runtime dependency.
- TecIO is optional, writes classic `.plt` with `FILEFORMAT_PLT = 0`, and is not imported by `cheby_tools.__init__`.
- Remove HDF5, statistics databases, Sebilleau/ERCOFTAC assets, `.lay` files, and generated figures.
- Implement each behavior test-first and commit each task separately.

## Review Focus

- Reject derivative orders supplied as `True`, a float, zero, or a negative integer instead of coercing them.
- Accept distinct but identical grids in `write_plt`; reject any domain, basis, size, or node mismatch.
- Reject field names colliding with generated coordinate names `x`, `y`, or `z`.
- Accept non-contiguous and non-float64 real fields, converting only at the TecIO boundary without changing the source.
- Keep core imports usable without TecIO and raise an actionable `ImportError` only when output is requested.

## Target Files

```text
cheby_tools/__init__.py
cheby_tools/field.py
cheby_tools/spectral.py
cheby_tools/tecio.py
native/tecio/CMakeLists.txt
native/tecio/tecio.cpp
examples/field_quickstart.py
examples/write_plt.py
tests/test_spectral.py
tests/test_spectral_edge_cases.py
tests/test_field.py
tests/test_tecio.py
tests/test_tecio_native.py
tests/test_distribution.py
```

### Task 1: Move the spectral core into `cheby_tools`

**Files:**
- Create: `cheby_tools/__init__.py`
- Move: `spec_forge/spectral_tools.py` → `cheby_tools/spectral.py`
- Move: `tests/test_spec_forge.py` → `tests/test_spectral.py`
- Modify: `tests/test_spectral.py`
- Modify: `tests/test_spectral_edge_cases.py`
- Delete: `spec_forge/__init__.py`

**Interfaces:**
- Consumes: existing `SpectralDiscretization` and `SpectralInterpolate`.
- Produces: public `cheby_tools.SpectralDiscretization` and internal `cheby_tools.spectral.SpectralInterpolate`.

- [ ] **Step 1: Write imports and public API tests for the new namespace**

Rename the test module before editing it:

```bash
git mv tests/test_spec_forge.py tests/test_spectral.py
```

Use these imports in both spectral test modules:

```python
import cheby_tools
from cheby_tools import SpectralDiscretization
from cheby_tools.spectral import SpectralInterpolate
```

Replace the old API test with:

```python
class PublicApiTests(unittest.TestCase):
    def test_public_api_exports_only_supported_symbols(self):
        self.assertEqual(cheby_tools.__all__, ["SpectralDiscretization"])
        self.assertFalse(hasattr(cheby_tools, "SpectralInterpolate"))
        self.assertFalse(hasattr(cheby_tools, "np"))
```

Remove `DiscrCompatibilityTests`; removal is asserted in Task 5.

- [ ] **Step 2: Verify the tests fail before the namespace exists**

Run:

```bash
python -m unittest tests.test_spectral tests.test_spectral_edge_cases -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cheby_tools'`.

- [ ] **Step 3: Move the source and create the public package**

Run:

```bash
mkdir -p cheby_tools
git mv spec_forge/spectral_tools.py cheby_tools/spectral.py
git rm spec_forge/__init__.py
```

Create `cheby_tools/__init__.py` with only the spectral public export; Task 2 adds `Field` and updates the API assertion:

```python
"""Field-oriented spectral post-processing tools."""

from .spectral import SpectralDiscretization

__all__ = ["SpectralDiscretization"]
```

In `cheby_tools/spectral.py`, remove `_build_aliases()` and its constructor call. Replace `ddx`, `ddy`, and `ddz` with calls to `self.diff(phi, axis=N)`; retain their existing dimensionality errors. Keep `SpectralInterpolate` internal.

- [ ] **Step 4: Run all migrated numerical tests**

```bash
python -m unittest tests.test_spectral tests.test_spectral_edge_cases -v
```

Expected: all migrated spectral and edge-case tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cheby_tools tests/test_spectral.py tests/test_spectral_edge_cases.py
git commit -m "refactor: move spectral core into cheby_tools"
```

### Task 2: Add `Field`

**Files:**
- Create: `cheby_tools/field.py`
- Create: `tests/test_field.py`
- Modify: `cheby_tools/__init__.py`
- Modify: `cheby_tools/spectral.py`
- Modify: `tests/test_spectral.py`

**Interfaces:**
- Consumes: `SpectralDiscretization.diff(phi, axis, order)` and `SpectralInterpolate(source, target) @ values`.
- Produces: `Field(values, discretization, name)`, `Field.derivative(axis, order=1)`, and `Field.interpolate(target)`.

- [ ] **Step 1: Write failing `Field` tests**

Create `tests/test_field.py`. Cover constructor identity and validation, 1D/2D/3D shapes, non-contiguous/complex values, source immutability, first and second derivatives, interpolation, invalid axes, and invalid orders. The key tests are:

```python
class FieldTests(unittest.TestCase):
    def setUp(self):
        self.grid = SpectralDiscretization(
            [0.0, -1.0], [2.0 * np.pi, 1.0], [24, 17],
            ["fourier", "chebyshev"],
        )
        self.x, self.y = self.grid.meshgrid()
        self.values = np.sin(3.0 * self.x) * (1.0 + self.y**2)

    def test_constructor_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, r"expected shape.*24, 17"):
            Field(self.values[:, :-1], self.grid, "temperature")

    def test_derivative_returns_new_field_without_mutating_source(self):
        source = Field(self.values.copy(), self.grid, "u")
        before = source.values.copy()
        result = source.derivative(axis=0)
        expected = 3.0 * np.cos(3.0 * self.x) * (1.0 + self.y**2)
        self.assertIsNot(result, source)
        self.assertIs(result.discretization, self.grid)
        self.assertEqual(result.name, "u")
        np.testing.assert_allclose(result.values, expected, atol=2e-11)
        np.testing.assert_array_equal(source.values, before)

    def test_second_fourier_derivative_handles_nyquist(self):
        grid = SpectralDiscretization(
            [0.0], [2.0 * np.pi], [8], ["fourier"]
        )
        values = np.cos(4.0 * grid.nodes[0])
        result = Field(values, grid, "mode").derivative(0, order=2)
        np.testing.assert_allclose(result.values, -16.0 * values, atol=2e-12)

    def test_invalid_orders_are_rejected(self):
        field = Field(self.values, self.grid, "u")
        for order in (True, 1.5):
            with self.subTest(order=order), self.assertRaises(TypeError):
                field.derivative(0, order)
        for order in (0, -1):
            with self.subTest(order=order), self.assertRaises(ValueError):
                field.derivative(0, order)

    def test_interpolation_returns_field_on_target(self):
        target = SpectralDiscretization(
            [0.0, -1.0], [2.0 * np.pi, 1.0], [48, 25],
            ["fourier", "chebyshev"],
        )
        result = Field(self.values, self.grid, "u").interpolate(target)
        xt, yt = target.meshgrid()
        np.testing.assert_allclose(
            result.values, np.sin(3.0 * xt) * (1.0 + yt**2), atol=2e-11
        )
        self.assertIs(result.discretization, target)

    def test_constructor_rejects_invalid_grid_and_name(self):
        with self.assertRaises(TypeError):
            Field(self.values, object(), "u")
        with self.assertRaises(TypeError):
            Field(self.values, self.grid, 4)
        for name in ("", "   "):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Field(self.values, self.grid, name)

    def test_invalid_axes_are_rejected(self):
        field = Field(self.values, self.grid, "u")
        for axis in (-1, 2):
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                field.derivative(axis)
        for axis in (True, 0.5):
            with self.subTest(axis=axis), self.assertRaises(TypeError):
                field.derivative(axis)

    def test_non_contiguous_complex_values_are_preserved(self):
        values = np.asfortranarray(self.values) * (1.0 + 0.5j)
        field = Field(values, self.grid, "mode")
        self.assertIs(field.values, values)
        self.assertTrue(np.iscomplexobj(field.values))
        np.testing.assert_array_equal(field.values, values)
```

Add a loop constructing valid 1D, 2D, and 3D grids and arrays with
`np.zeros(tuple(grid.n))`; assert construction succeeds and preserves each exact
shape.

- [ ] **Step 2: Verify failure**

```bash
python -m unittest tests.test_field -v
```

Expected: FAIL because `Field` is not exported.

- [ ] **Step 3: Extend `SpectralDiscretization.diff` to positive derivative orders**

Replace `diff` with:

```python
def diff(self, phi, axis, order=1):
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, Integral):
        raise TypeError("axis must be an integer.")
    axis = int(axis)
    if not 0 <= axis < self.dim:
        raise ValueError(f"Invalid axis {axis} for dim={self.dim}.")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral):
        raise TypeError("order must be an integer.")
    order = int(order)
    if order < 1:
        raise ValueError("order must be at least 1.")
    if self.bases[axis] == "fourier":
        operator = FourierDiffOp1D(
            self.n[axis], self.xmin[axis], self.xmax[axis],
            axis=axis, order=order, name=f"d{axis}^{order}",
        )
        return operator @ phi
    out = np.asarray(phi)
    for _ in range(order):
        out = self.diff_ops[axis] @ out
    return out
```

Keep `ddx`, `ddy`, and `ddz` as first-order helpers.

- [ ] **Step 4: Implement and export `Field`**

Create `cheby_tools/field.py`:

```python
from numbers import Integral
import numpy as np
from .spectral import SpectralDiscretization, SpectralInterpolate


class Field:
    def __init__(self, values, discretization, name):
        if not isinstance(discretization, SpectralDiscretization):
            raise TypeError("discretization must be a SpectralDiscretization.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        name = name.strip()
        if not name:
            raise ValueError("name must not be empty.")
        values = np.asarray(values)
        expected = tuple(discretization.n)
        if values.shape != expected:
            raise ValueError(
                f"Field {name!r} expected shape {expected}, got {values.shape}."
            )
        self.values = values
        self.discretization = discretization
        self.name = name

    def derivative(self, axis, order=1):
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, Integral):
            raise TypeError("axis must be an integer.")
        if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral):
            raise TypeError("order must be an integer.")
        axis, order = int(axis), int(order)
        if not 0 <= axis < self.discretization.dim:
            raise ValueError(
                f"Invalid axis {axis} for dim={self.discretization.dim}."
            )
        if order < 1:
            raise ValueError("order must be at least 1.")
        values = self.discretization.diff(self.values, axis, order=order)
        return type(self)(values, self.discretization, self.name)

    def interpolate(self, target):
        if not isinstance(target, SpectralDiscretization):
            raise TypeError("target must be a SpectralDiscretization.")
        values = SpectralInterpolate(self.discretization, target) @ self.values
        return type(self)(values, target, self.name)
```

Update `cheby_tools/__init__.py` to import `Field` and set `__all__ = ["Field", "SpectralDiscretization"]`. Update the Task 1 API test to the same list.

- [ ] **Step 5: Verify field and core behavior**

```bash
python -m unittest tests.test_field tests.test_spectral tests.test_spectral_edge_cases -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add cheby_tools tests/test_field.py tests/test_spectral.py
git commit -m "feat: add spectral field abstraction"
```

### Task 3: Add field-oriented TecIO Python API

**Files:**
- Create: `cheby_tools/tecio.py`
- Create: `tests/test_tecio.py`

**Interfaces:**
- Consumes: `Field`, `meshgrid()`, and optional backend method `write_plt(filename, names, arrays)`.
- Produces: `cheby_tools.tecio.write_plt(path, fields)`.

- [ ] **Step 1: Write failing adapter tests with a recording backend**

Use this fake:

```python
class RecordingBackend:
    def __init__(self):
        self.calls = []

    def write_plt(self, filename, names, arrays):
        self.calls.append((filename, names, arrays))
```

Use these concrete boundary tests (with `setUp` creating a 4×3
Fourier/Chebyshev grid and its mesh):

```python
def test_multiple_fields_use_i_fastest_float64_arrays(self):
    u = Field(self.x + self.y, self.grid, "u")
    v = Field(self.x - self.y, self.grid, "v")
    backend = RecordingBackend()
    with mock.patch.object(tecio, "_backend", backend):
        tecio.write_plt("velocity.plt", [u, v])
    filename, names, arrays = backend.calls[0]
    self.assertEqual(filename, "velocity.plt")
    self.assertEqual(names, ["x", "y", "u", "v"])
    self.assertEqual([array.shape for array in arrays], [(3, 4)] * 4)
    for source, converted in zip((self.x, self.y, u.values, v.values), arrays):
        self.assertTrue(converted.flags.c_contiguous)
        self.assertEqual(converted.dtype, np.float64)
        np.testing.assert_array_equal(converted, source.transpose(1, 0))

def test_equivalent_distinct_grids_are_accepted(self):
    other = self.make_grid()
    fields = [
        Field(self.x, self.grid, "u"),
        Field(other.meshgrid()[1], other, "v"),
    ]
    backend = RecordingBackend()
    with mock.patch.object(tecio, "_backend", backend):
        tecio.write_plt("fields.plt", fields)
    self.assertEqual(len(backend.calls), 1)

def test_invalid_field_collections_are_rejected(self):
    values = np.ones(tuple(self.grid.n))
    other = self.make_grid(xmax=3.0)
    cases = (
        [],
        [object()],
        [Field(values, self.grid, "u"), Field(values, self.grid, "u")],
        [Field(values, self.grid, "x")],
        [Field(values, self.grid, "u"),
         Field(np.ones(tuple(other.n)), other, "v")],
        [Field(values.astype(complex), self.grid, "u")],
    )
    for fields in cases:
        with self.subTest(fields=fields), self.assertRaises((TypeError, ValueError)):
            tecio.write_plt("fields.plt", fields)

def test_suffix_is_exactly_lowercase_plt(self):
    field = Field(np.ones(tuple(self.grid.n)), self.grid, "u")
    for path in ("fields.szplt", "fields.PLT", "fields"):
        with self.subTest(path=path), self.assertRaises(ValueError):
            tecio.write_plt(path, field)

def test_missing_backend_is_actionable(self):
    field = Field(np.ones(tuple(self.grid.n)), self.grid, "u")
    with mock.patch.object(tecio, "_backend", None):
        with self.assertRaisesRegex(ImportError, r"CMake.*TecIO"):
            tecio.write_plt("field.plt", field)
```

Add 1D and 3D cases asserting backend shapes `(n0,)` and `(n2, n1, n0)`.
For a float32 non-contiguous view, save dtype, strides, and a copy before the
call; assert all three source properties remain unchanged afterward.

- [ ] **Step 2: Verify failure**

```bash
python -m unittest tests.test_tecio -v
```

Expected: FAIL because `cheby_tools.tecio` is absent.

- [ ] **Step 3: Implement `cheby_tools.tecio`**

Create `cheby_tools/tecio.py`:

```python
from pathlib import Path
import numpy as np
from .field import Field

try:
    from . import _tecio as _backend
except ImportError:
    _backend = None


def _same_grid(left, right):
    if (
        left.dim != right.dim
        or left.n != right.n
        or left.bases != right.bases
        or left.xmin != right.xmin
        or left.xmax != right.xmax
    ):
        return False
    return all(
        np.array_equal(left_nodes, right_nodes)
        for left_nodes, right_nodes in zip(left.nodes, right.nodes)
    )


def _normalize_fields(fields):
    if isinstance(fields, Field):
        return [fields]
    try:
        fields = list(fields)
    except TypeError as exc:
        raise TypeError(
            "fields must be a Field or an iterable of Field objects."
        ) from exc
    if not fields:
        raise ValueError("fields must contain at least one Field.")
    for index, field in enumerate(fields):
        if not isinstance(field, Field):
            raise TypeError(f"fields[{index}] must be a Field.")
    return fields


def write_plt(path, fields):
    path = Path(path)
    if path.suffix != ".plt":
        raise ValueError(f"TecIO path must end in '.plt', got {path!s}.")
    fields = _normalize_fields(fields)
    grid = fields[0].discretization
    coordinate_names = ["x", "y", "z"][: grid.dim]
    names = [field.name for field in fields]
    if len(set(names)) != len(names):
        raise ValueError("Field names must be unique.")
    collisions = set(names).intersection(coordinate_names)
    if collisions:
        raise ValueError(
            f"Field names conflict with coordinates: {sorted(collisions)}."
        )
    for field in fields:
        if not _same_grid(grid, field.discretization):
            raise ValueError(f"Field {field.name!r} uses an incompatible grid.")
        if np.iscomplexobj(field.values):
            raise ValueError(
                f"Field {field.name!r} is complex and cannot be written."
            )
    if _backend is None:
        raise ImportError(
            "TecIO output requires the native backend; "
            "build and install it with CMake."
        )
    axes = tuple(range(grid.dim - 1, -1, -1))
    source_arrays = [*grid.meshgrid(), *(field.values for field in fields)]
    arrays = [
        np.ascontiguousarray(np.transpose(array, axes=axes), dtype=np.float64)
        for array in source_arrays
    ]
    _backend.write_plt(str(path), coordinate_names + names, arrays)


__all__ = ["write_plt"]
```

- [ ] **Step 4: Verify adapter and core tests**

```bash
python -m unittest tests.test_tecio tests.test_field tests.test_spectral tests.test_spectral_edge_cases -v
```

Expected: all tests PASS without native TecIO.

- [ ] **Step 5: Commit**

```bash
git add cheby_tools/tecio.py tests/test_tecio.py
git commit -m "feat: add field-oriented TecIO API"
```

### Task 4: Replace the TecIO wrapper with a private `.plt` backend

**Files:**
- Create: `native/tecio/CMakeLists.txt`
- Create: `native/tecio/tecio.cpp`
- Create: `tests/test_tecio_native.py`
- Modify: `CMakeLists.txt`
- Delete: `tecio_wrapper/`

**Interfaces:**
- Consumes: C-contiguous float64 arrays in `(k, j, i)` order.
- Produces: private `cheby_tools._tecio.write_plt(filename, names, arrays)`.

- [ ] **Step 1: Write the native integration test**

Create `tests/test_tecio_native.py`:

```python
import tempfile
from pathlib import Path
import unittest
import numpy as np
from cheby_tools import Field, SpectralDiscretization
from cheby_tools import tecio


@unittest.skipIf(tecio._backend is None, "native TecIO backend is not installed")
class NativeTecIOTests(unittest.TestCase):
    def test_multifield_output_is_nonempty(self):
        grid = SpectralDiscretization(
            [0.0, -1.0], [2.0 * np.pi, 1.0], [8, 7],
            ["fourier", "chebyshev"],
        )
        x, y = grid.meshgrid()
        fields = [Field(np.sin(x), grid, "u"), Field(y**2, grid, "v")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fields.plt"
            tecio.write_plt(output, fields)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
```

- [ ] **Step 2: Verify the explicit skip before installation**

```bash
python -m unittest tests.test_tecio_native -v
```

Expected: one SKIP saying the backend is not installed.

- [ ] **Step 3: Implement one generic C++ writer**

Replace the three duplicated writers and time-series demo classes with one function:

```cpp
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
void write_plt(const std::string& filename,
               const std::vector<std::string>& names,
               const std::vector<Array>& arrays);
PYBIND11_MODULE(_tecio, module) {
    module.def("write_plt", &write_plt,
               py::arg("filename"), py::arg("names"), py::arg("arrays"));
}
```

Inside it: reject empty arrays and name-count mismatch; request all buffers; require identical 1D–3D shape; reject non-positive dimensions and dimensions/count exceeding `INTEGER4`; map the last C axis to I, previous to J, and previous to K; join names with spaces; call `TECINI142` with `INTEGER4 file_format = FILEFORMAT_PLT`, grid-and-solution type, double precision; create one ordered block zone with `TECZNE142`; call `TECDAT142` once per array; call `TECEND142` on success and in every post-initialization exception path. Convert every nonzero TecIO return to `RuntimeError` naming the failed routine.

- [ ] **Step 4: Install `_tecio` in the package namespace**

Create `native/tecio/CMakeLists.txt` with `pybind11_add_module(_tecio tecio.cpp)`, C++14, TecIO includes/link target, and:

```cmake
install(TARGETS _tecio
  LIBRARY DESTINATION ${CHEBY_PYTHON_INSTALL_DIR}/cheby_tools
)
```

In top-level CMake: rename `CHEBY_INSTALL_POSTPROCESSING_TOOLS` to `CHEBY_INSTALL_PYTHON_CORE`; install only `cheby_tools`; replace `add_subdirectory(tecio_wrapper)` by `add_subdirectory(native/tecio)`; preserve bundled/external TecIO and Boost selection.

- [ ] **Step 5: Build, install, and run the real integration test**

```bash
cmake -S . -B build-test -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" \
  -DCHEBY_PYTHON_INSTALL_DIR="$(python -c 'import os, site, sys; print(os.path.relpath(site.getsitepackages()[0], sys.prefix))')"
cmake --build build-test -j
cmake --install build-test
python -m unittest tests.test_tecio_native -v
```

Expected: build/install succeed and the test PASSes with a non-empty `.plt`.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt native/tecio tests/test_tecio_native.py
git rm -r tecio_wrapper
git commit -m "refactor: make TecIO a private plt backend"
```

### Task 5: Remove application code and narrow package boundaries

**Files:**
- Modify: `pyproject.toml`, `MANIFEST.in`, `.gitignore`
- Modify: `tests/test_distribution.py`
- Modify: `check_python_env.py`, `env.sh`, `run_cmake.sh`
- Delete: `stats/`, `discr/`, HDF5 scripts, all old examples, `.lay`, EPS/PNG figures, `cfg_adastra.sh`

**Interfaces:**
- Consumes: final package from Tasks 1–4.
- Produces: wheel/sdist containing only `cheby_tools` at this stage, and one portable ADASTRA setup. Task 6 adds the focused examples to the sdist.

- [ ] **Step 1: Rewrite distribution and cleanup tests**

For the wheel, use exact assertions:

```python
for member in (
    "cheby_tools/__init__.py",
    "cheby_tools/field.py",
    "cheby_tools/spectral.py",
    "cheby_tools/tecio.py",
):
    self.assertIn(member, names)
self.assertFalse(any(name.startswith("spec_forge/") for name in names))
self.assertFalse(any(name.startswith("discr/") for name in names))
self.assertFalse(any(name.startswith("stats/") for name in names))
self.assertFalse(any(name.startswith("native/") for name in names))
self.assertEqual(top_level_packages, {"cheby_tools"})
self.assertNotIn("Provides-Extra: io", metadata)
self.assertIn("Provides-Extra: dev", metadata)
```

For the sdist require packaging files and `cheby_tools/*.py`. Reject unwanted
members with:

```python
for fragment in (
    "/tests/", "/native/", "/external/", "/stats/", "/discr/",
    "/spec_forge/", ".lay", ".eps", ".png",
):
    self.assertFalse(any(fragment in name for name in names), fragment)
```

Add source-tree assertions for every explicitly removed path and glob checks
that `examples/**/*.lay`, `examples/**/*.eps`, and `examples/**/*.png` are empty.
Update the installed-wheel smoke code to construct a `Field`, differentiate it,
and require maximum error below `2e-12`.

- [ ] **Step 2: Verify distribution tests fail before cleanup**

```bash
python -m unittest tests.test_distribution -v
```

Expected: FAIL on old packages and application assets.

- [ ] **Step 3: Narrow package discovery and manifest**

Use:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["cheby_tools", "cheby_tools.*"]
exclude = ["external*", "native*", "tests*"]
```

Remove the `io` extra. Set `MANIFEST.in` to include README, pyproject, `cheby_tools/*.py`, and `examples/*.py`; prune external, native, and tests; globally exclude caches, bytecode, and `.DS_Store`.

- [ ] **Step 4: Remove all approved historical/application assets**

```bash
git rm -r stats discr examples/figs
git rm examples/Basic_stats_X_0p5_comparison.lay
git rm examples/Data_midwidth_comparison.lay
git rm examples/sebilleau_2d.lay
git rm examples/build_ercoftac_db.py
git rm examples/discr_example.py examples/tecio_example.py
git rm examples/spectral_quickstart.py
git rm compile_h5py.sh env_h5py.sh cfg_adastra.sh
```

Search for stale runtime/docs references outside design records and negative tests:

```bash
rg -n "stats|h5py|HDF5|Sebilleau|ERCOFTAC|spec_forge|from discr|import discr|szplt|\\.lay" \
  --glob '!external/**' --glob '!docs/superpowers/**' .
```

Expected: no stale positive references.

- [ ] **Step 5: Simplify ADASTRA setup**

In `env.sh`, require caller-provided `CHEBY_PYTHON_ENV`; load compiler, CMake,
and Python 3.12 modules; remove MPI/HDF5 modules and `FC`, `HDF5_MPI`,
`HDF5_DIR`; retain repository Boost and `CC`/`CXX`. The environment guard is:

```bash
: "${CHEBY_PYTHON_ENV:?Set CHEBY_PYTHON_ENV to a writable environment path}"
if [[ ! -x "${CHEBY_PYTHON_ENV}/bin/python" ]]; then
    echo "Python environment not found: ${CHEBY_PYTHON_ENV}" >&2
    return 1 2>/dev/null || exit 1
fi
source "${CHEBY_PYTHON_ENV}/bin/activate"
```

Rewrite `check_python_env.py` to require `sys.version_info >= (3, 11)`, import
NumPy and `cheby_tools`, print their versions/paths, and use
`importlib.util.find_spec("cheby_tools._tecio")` to report native availability;
import no historical/HDF5 module.

Rewrite `run_cmake.sh` around:

```bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${CHEBY_BUILD_DIR:-${PROJECT_ROOT}/build}"
REL_SITEPKG="$(python -c 'import os, site, sys; print(os.path.relpath(site.getsitepackages()[0], sys.prefix))')"
cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_INSTALL_PREFIX="${CHEBY_PYTHON_ENV}" \
  -DCHEBY_PYTHON_INSTALL_DIR="${REL_SITEPKG}" \
  -DCHEBY_BOOST_INCLUDE_DIR="${CHEBY_BOOST_INCLUDE_DIR}" \
  -DPython_EXECUTABLE="${CHEBY_PYTHON_ENV}/bin/python"
cmake --build "${BUILD_DIR}" -j
cmake --install "${BUILD_DIR}"
"${CHEBY_PYTHON_ENV}/bin/python" "${PROJECT_ROOT}/check_python_env.py"
```

- [ ] **Step 6: Verify archives**

```bash
python -m unittest tests.test_distribution -v
python -m build
python -m twine check dist/*
```

Expected: tests PASS and both archives pass Twine.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml MANIFEST.in .gitignore tests/test_distribution.py check_python_env.py env.sh run_cmake.sh
git add -u
git commit -m "chore: remove application-specific tooling"
```

### Task 6: Add focused examples, documentation, and final verification

**Files:**
- Create: `examples/field_quickstart.py`
- Create: `examples/write_plt.py`
- Modify: `README.md`
- Modify: `tests/test_distribution.py`

**Interfaces:**
- Consumes: final public `Field`, `SpectralDiscretization`, and `write_plt`.
- Produces: runnable documentation for workstations and ADASTRA.

- [ ] **Step 1: Make distribution tests execute the new quick-start**

Copy `field_quickstart.py` outside the source tree after wheel and sdist installation; assert output contains `derivative max error:` and `interpolation max error:`. Run the test and expect failure because the example is absent.

- [ ] **Step 2: Create `field_quickstart.py`**

Create the executable example around this body:

```python
coarse = SpectralDiscretization([0.0], [2.0 * np.pi], [24], ["fourier"])
fine = SpectralDiscretization([0.0], [2.0 * np.pi], [48], ["fourier"])
x_coarse, x_fine = coarse.nodes[0], fine.nodes[0]
field = Field(
    np.sin(3.0 * x_coarse) + 0.25 * np.cos(5.0 * x_coarse),
    coarse,
    "temperature",
)
derivative_error = np.max(np.abs(
    field.derivative(0).values
    - (3.0 * np.cos(3.0 * x_coarse) - 1.25 * np.sin(5.0 * x_coarse))
))
interpolation_error = np.max(np.abs(
    field.interpolate(fine).values
    - (np.sin(3.0 * x_fine) + 0.25 * np.cos(5.0 * x_fine))
))
print(f"derivative max error:    {derivative_error:.3e}")
print(f"interpolation max error: {interpolation_error:.3e}")
if derivative_error > 2.0e-12 or interpolation_error > 2.0e-12:
    raise SystemExit("field quick-start validation failed")
```

Wrap it in `main()` and the standard `if __name__ == "__main__": main()` guard.

- [ ] **Step 3: Create `write_plt.py`**

Use this complete example:

```python
import numpy as np
from cheby_tools import Field, SpectralDiscretization
from cheby_tools.tecio import write_plt

grid = SpectralDiscretization(
    [0.0, -1.0], [2.0 * np.pi, 1.0], [64, 33],
    ["fourier", "chebyshev"],
)
x, y = grid.meshgrid()
u = Field(np.sin(x) * (1.0 - y**2), grid, "u")
temperature = Field(np.cos(2.0 * x) + y, grid, "temperature")
write_plt("fields.plt", [u, temperature])
print("wrote fields.plt")
```

- [ ] **Step 4: Rewrite README**

Document: purpose/public API; Python 3.11+ installation; shape convention; field differentiation/interpolation; package build; optional TecIO CMake build; multi-field `write_plt`; ADASTRA with caller-selected `CHEBY_PYTHON_ENV`; and that output is classic `.plt`, never `.szplt`. Include no statistics, database, HDF5, historical package, Sebilleau/ERCOFTAC, or `.lay` instructions.

- [ ] **Step 5: Run full verification**

```bash
python examples/field_quickstart.py
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
git diff --check
```

Expected: errors below `2e-12`, all tests PASS, both archives pass, and diff check is silent.

- [ ] **Step 6: Test clean wheel and sdist installations**

In separate temporary Python 3.11+ environments, install each archive, run the quick-start outside the repository, import `Field` and `SpectralDiscretization`, and assert `importlib.util.find_spec` returns `None` for `stats`, `discr`, and `spec_forge`.

- [ ] **Step 7: Record native verification**

```bash
python -m unittest tests.test_tecio_native -v
```

Expected with the installed backend: PASS and non-empty `.plt`. If the native toolchain is unavailable, retain the explicit SKIP and state that limitation in the handoff.

- [ ] **Step 8: Commit and request review**

```bash
git add README.md examples tests/test_distribution.py
git commit -m "docs: document the field-focused toolbox"
```

Give the reviewer the spec, this plan, `origin/main..HEAD`, test output, archive inspection, clean-install results, and native TecIO result. Resolve all Critical and Important findings before pushing.
