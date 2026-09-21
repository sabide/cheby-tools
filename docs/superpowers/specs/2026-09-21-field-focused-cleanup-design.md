# Field-focused cheby-tools redesign

## Objective

Refocus `cheby-tools` on numerical operations over fields defined on tensor-product
Chebyshev and Fourier grids. Remove database- and statistics-specific code while
retaining TecIO as an optional writer for classic Tecplot `.plt` files.

The redesign intentionally introduces a new, compact public API:

```python
from cheby_tools import Field, SpectralDiscretization
```

There will be no compatibility aliases for the historical `spec_forge`, `discr`,
or `stats` packages.

## Scope

### Retained capabilities

- One-, two-, and three-dimensional tensor-product spectral grids.
- Chebyshev and Fourier differentiation, interpolation, transforms, and
  quadrature already implemented by the numerical core.
- A field abstraction combining numerical values, a discretization, and a
  variable name.
- Optional TecIO support for writing one or more fields to classic Tecplot
  `.plt` files.
- CMake support for compiling the TecIO extension on ADASTRA and other supported
  machines.
- A pure-Python wheel for the numerical core.

### Removed capabilities and assets

- The complete `stats` package and all HDF5/statistics database readers.
- The historical `discr` compatibility package.
- The historical `spec_forge` import path after its implementation has moved.
- HDF5-specific environment, build, and validation scripts.
- Sebilleau and ERCOFTAC application examples.
- Tecplot `.lay` files and generated EPS/PNG figures.
- Demo helpers embedded in the TecIO package.

## Package architecture

The public Python package will have one name and one responsibility boundary:

```text
cheby_tools/
├── __init__.py
├── field.py
├── spectral.py
└── tecio.py
```

- `cheby_tools.__init__` exports `Field` and `SpectralDiscretization`.
- `cheby_tools.spectral` contains the existing spectral implementation, cleaned
  of historical compatibility attributes when they are not required by the new
  API.
- `cheby_tools.field` contains the high-level field abstraction.
- `cheby_tools.tecio` contains the optional Python adapter around a private C++
  extension. Importing `cheby_tools` must not require that extension.

The C++ extension is an implementation detail. User code interacts with
`cheby_tools.tecio`, not with a top-level `tecio_wrapper` package.

## Core data model

### `SpectralDiscretization`

`SpectralDiscretization` continues to describe a one-, two-, or
three-dimensional tensor-product grid. Its canonical field shape is
`tuple(discretization.n)`, with Python array axis `d` corresponding to grid axis
`d`.

The class owns grid geometry and numerical operators. Existing validated
Chebyshev/Fourier behavior, including the Nyquist conventions fixed on the
`update` branch, must be preserved.

### `Field`

A field combines:

- `values`: a NumPy array;
- `discretization`: a `SpectralDiscretization`;
- `name`: a non-empty variable name.

Construction validates that `values.shape == tuple(discretization.n)`. Values
may be real or complex for numerical operations. Construction does not silently
reshape or reorder input data.

The first public operations are:

```python
derived = field.derivative(axis=0, order=1)
resampled = field.interpolate(target_discretization)
```

Both operations return new `Field` instances and leave the source unchanged.
The result keeps the source field name. No general arithmetic, unit system, or
xarray-style metadata model is introduced in this iteration.

## TecIO `.plt` output

The public writer is:

```python
from cheby_tools.tecio import write_plt

write_plt("velocity.plt", [u, v, w])
```

The writer accepts one `Field` or a sequence of fields. It:

1. verifies that at least one field is present;
2. verifies that every item is a `Field`;
3. requires unique, non-empty field names;
4. requires all fields to share the same discretization and nodal coordinates;
5. rejects complex-valued fields rather than silently discarding imaginary
   components;
6. generates coordinate arrays from the discretization;
7. converts Python's canonical axis ordering to TecIO's ordered-zone,
   I-fastest memory layout internally;
8. writes a classic binary Tecplot `.plt` file.

The path must use the `.plt` suffix. A different suffix raises `ValueError` so
the selected format is explicit.

The bundled TecIO implementation already selects `FILEFORMAT_PLT = 0`. Existing
symbols incorrectly named `write_szplt_*` and `Szplt*Writer`, and comments that
describe SZL output, will be renamed or replaced. The low-level extension may
continue to consume NumPy arrays internally, but those functions are private.

Time-series writers and multiple-zone streaming are not part of the new public
API in this iteration. They may be redesigned later around `Field` if a concrete
use case requires them.

## Optional dependency behavior

NumPy is the only runtime dependency of the pure-Python package. The HDF5 `io`
extra is removed.

Importing either of the following must work without TecIO:

```python
import cheby_tools
from cheby_tools import Field, SpectralDiscretization
```

Importing `cheby_tools.tecio` may also succeed without the extension so its API
can be inspected. Calling `write_plt` without the compiled backend raises an
`ImportError` that explains that TecIO must be built with CMake.

## Build and distribution

`pyproject.toml` packages only `cheby_tools` and its pure-Python submodules.
The wheel and source distribution exclude `stats`, `discr`, `spec_forge`, the
native TecIO sources, application examples, generated figures, and tests.

CMake installs the same `cheby_tools` package layout and places the compiled
private TecIO extension where `cheby_tools.tecio` can import it. The ADASTRA
environment scripts are simplified to the dependencies required by Python,
NumPy, CMake, pybind11, Boost headers, and TecIO; MPI HDF5 setup is removed.

## Examples and documentation

The repository keeps only focused examples:

- a numerical example that constructs fields, differentiates them, and
  interpolates them;
- a TecIO example that writes multiple named fields on one grid to `.plt`.

The README documents the new imports, field shape convention, installation,
pure-Python verification, optional TecIO build, ADASTRA setup, and `.plt`
writing. It contains no statistics database, HDF5, Sebilleau, ERCOFTAC, `.lay`,
or SZPLT instructions.

## Error handling

Public validation failures use specific Python exceptions:

- `TypeError` for objects of the wrong type;
- `ValueError` for invalid shapes, axes, derivative orders, names, output
  suffixes, complex TecIO data, and incompatible field grids;
- `ImportError` when `.plt` output is requested without the native backend;
- `RuntimeError` for a failure reported by the TecIO library itself.

Errors identify the offending field, axis, shape, or path where applicable.

## Verification strategy

Implementation follows test-driven development. Verification includes:

- porting all existing spectral regression and edge-case tests to
  `cheby_tools`;
- unit tests for `Field` construction, differentiation, interpolation,
  immutability of operations, and validation errors;
- Python adapter tests for single- and multi-field `.plt` output using a fake
  native backend, including coordinate ordering and rejection cases;
- distribution tests proving that wheel and source archives contain only the
  intended Python package and focused example sources;
- clean-environment installation and execution of the numerical example;
- CMake compilation on ADASTRA and an integration test proving that TecIO
  creates a non-empty `.plt` file.

The full Python test suite and distribution checks must pass before the branch
is considered ready. If the ADASTRA native build cannot run in the current job,
that limitation is reported explicitly rather than treating mocked TecIO tests
as equivalent.

## Completion criteria

The redesign is complete when:

- `from cheby_tools import Field, SpectralDiscretization` works after a clean
  installation;
- field differentiation and interpolation preserve the validated spectral
  behavior;
- `write_plt` writes one or multiple real fields on a shared grid to `.plt`;
- the repository and built distributions contain none of the removed
  application/statistics assets;
- the README describes only the new field-focused toolbox;
- Python tests, package builds, archive checks, and available TecIO integration
  checks pass.
