# One-command Native Installation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `python -m pip install -e <cheby-tools>` compile and install the private TecIO backend by default while retaining an explicit Python-only installation mode.

**Architecture:** Replace setuptools with scikit-build-core so one PEP 517/660 backend owns both editable Python packaging and the CMake build. Teach CMake to install `_tecio` into scikit-build-core's staging package without duplicating the Python sources, while preserving the existing direct-CMake workflow. Validate both modes from clean environments and package the native sources needed to build wheels and source distributions.

**Tech Stack:** Python 3.11+, scikit-build-core, CMake 3.18+, pybind11, bundled TecIO, bundled Boost headers, `unittest`, `build`, `pip`.

**Spec:** `docs/superpowers/specs/2026-09-21-one-command-native-install-design.md`

## Global Constraints

- TecIO is enabled and required by default; native failures must fail installation instead of silently degrading.
- The explicit opt-out is `-Ccmake.define.CHEBY_INSTALL_TECIO=OFF`.
- Supported targets for this change are Linux, including ADASTRA, and macOS; Windows is out of scope.
- Python 3.11 or newer, CMake 3.18 or newer and a C++ compiler are required.
- Use repository-local TecIO and Boost sources and the initialized `external/pybind11` submodule; CMake must not download dependencies.
- The public imports remain `Field`, `SpectralDiscretization` and `cheby_tools.tecio.write_plt`.
- Editable Python changes are immediate; native changes require rerunning `pip install -e`.
- Keep the direct CMake installation path functional as an advanced workflow.
- Continue producing classic `.plt` files only; do not add `.szplt` support.

## Review Focus

- A checkout without `external/pybind11` must fail with the exact corrective submodule command; Task 2 adds this failure-path test.
- A Python-only editable install must import from the installed editable mapping, not accidentally from the test checkout; Task 4 runs its probe from an unrelated temporary directory.
- A default wheel must carry a platform-specific `_tecio` extension and must not be tagged `py3-none-any`; Task 3 inspects both filename and archive members.
- A source archive must contain every native input needed for an offline CMake build, including pybind11 contents rather than only the gitlink; Task 3 builds and installs from the extracted archive.
- A missing native backend must preserve its original import exception as `ImportError.__cause__` while giving the user a pip-based recovery command; Task 2 pins both behaviors.

---

### Task 1: Declare the scikit-build-core packaging contract

**Files:**
- Modify: `pyproject.toml`
- Delete: `MANIFEST.in`
- Modify: `tests/test_distribution.py`

**Interfaces:**
- Consumes: Existing project metadata and the source package at `cheby_tools/`.
- Produces: PEP 517 backend `scikit_build_core.build`, wheel package mapping for `cheby_tools`, CMake definition `CHEBY_INSTALL_PYTHON_CORE=OFF`, and explicit native-source inclusion in sdists.

- [ ] **Step 1: Add a failing metadata-contract test**

Add `tomllib` to the imports and this test class near the top of `tests/test_distribution.py`:

```python
import tomllib


class PackagingConfigurationTests(unittest.TestCase):
    def test_scikit_build_core_owns_python_and_cmake_packaging(self):
        with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
            configuration = tomllib.load(stream)

        build_system = configuration["build-system"]
        self.assertEqual(
            build_system["build-backend"], "scikit_build_core.build"
        )
        self.assertTrue(
            any(
                requirement.startswith("scikit-build-core>=")
                for requirement in build_system["requires"]
            )
        )

        scikit_build = configuration["tool"]["scikit-build"]
        self.assertEqual(scikit_build["wheel"]["packages"], ["cheby_tools"])
        self.assertFalse(
            scikit_build["cmake"]["define"]["CHEBY_INSTALL_PYTHON_CORE"]
        )
        included = scikit_build["sdist"]["include"]
        for required in (
            "CMakeLists.txt",
            "native/tecio/**",
            "external/boost/**",
            "external/tecio/teciosrc/**",
            "external/pybind11/**",
        ):
            self.assertIn(required, included)

    def test_obsolete_setuptools_manifest_is_absent(self):
        self.assertFalse((REPOSITORY_ROOT / "MANIFEST.in").exists())
```

- [ ] **Step 2: Run the test and verify the old backend fails it**

Run:

```bash
python -m unittest tests.test_distribution.PackagingConfigurationTests -v
```

Expected: FAIL because the backend is `setuptools.build_meta` and `MANIFEST.in` exists.

- [ ] **Step 3: Replace the backend and configure package/source inclusion**

Replace the setuptools-specific sections in `pyproject.toml` with this structure, preserving the existing `[project]`, classifiers and URLs:

```toml
[build-system]
requires = ["scikit-build-core>=0.10"]
build-backend = "scikit_build_core.build"

[project.optional-dependencies]
dev = [
    "build>=1.2",
    "scikit-build-core>=0.10",
    "twine>=5",
]

[tool.scikit-build]
minimum-version = "build-system.requires"
wheel.packages = ["cheby_tools"]
sdist.include = [
    "CMakeLists.txt",
    "native/tecio/**",
    "external/boost/**",
    "external/tecio/teciosrc/**",
    "external/pybind11/**",
    "examples/*.py",
]
sdist.exclude = [
    "tests/**",
    "docs/**",
    "build/**",
    "dist/**",
]

[tool.scikit-build.cmake.define]
CHEBY_INSTALL_PYTHON_CORE = false
```

Delete `MANIFEST.in`; its setuptools pruning rules conflict with the new native sdist.

- [ ] **Step 4: Run the focused metadata tests**

Run:

```bash
python -m unittest tests.test_distribution.PackagingConfigurationTests -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit the packaging contract**

```bash
git add pyproject.toml MANIFEST.in tests/test_distribution.py
git commit -m "build: adopt scikit-build-core backend"
```

### Task 2: Make CMake aware of pip staging and improve the opt-out error

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `native/tecio/CMakeLists.txt`
- Modify: `cheby_tools/tecio.py`
- Modify: `tests/test_distribution.py`
- Modify: `tests/test_tecio.py`

**Interfaces:**
- Consumes: The `SKBUILD` CMake variable supplied by scikit-build-core and `CHEBY_INSTALL_TECIO` supplied through pip config settings.
- Produces: `_tecio` installed at `cheby_tools/_tecio<suffix>` for pip builds, unchanged direct-CMake placement, a valid empty native install when pip explicitly disables TecIO, and an installation-oriented `ImportError`.

- [ ] **Step 1: Add failing CMake-context tests**

Add this helper and the two tests to `PackagingConfigurationTests`:

```python
    def run_cmake(self, source, build, *definitions):
        return subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                *definitions,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_scikit_build_context_allows_python_only_install(self):
        with tempfile.TemporaryDirectory(prefix="cheby-cmake-python-only-") as tmp:
            result = self.run_cmake(
                REPOSITORY_ROOT,
                Path(tmp) / "build",
                "-DSKBUILD=ON",
                "-DCHEBY_INSTALL_TECIO=OFF",
                "-DCHEBY_INSTALL_PYTHON_CORE=OFF",
            )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_missing_pybind11_reports_submodule_recovery_command(self):
        with tempfile.TemporaryDirectory(prefix="cheby-missing-pybind11-") as tmp:
            source = Path(tmp) / "project"
            shutil.copytree(
                REPOSITORY_ROOT,
                source,
                ignore=shutil.ignore_patterns(
                    ".git", "build", "dist", "__pycache__", "pybind11"
                ),
            )
            result = self.run_cmake(source, Path(tmp) / "build")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("git submodule update --init --recursive", result.stdout)
```

The execution environment for these tests must have CMake and a C++ compiler loaded.

- [ ] **Step 2: Tighten the existing missing-backend test before changing the message**

Change `test_missing_backend_is_actionable` in `tests/test_tecio.py` to:

```python
    def test_missing_backend_is_actionable(self):
        field = Field(np.ones(tuple(self.grid.n)), self.grid, "u")
        load_error = ImportError("missing TecIO shared library")

        with mock.patch.object(tecio, "_backend", None), mock.patch.object(
            tecio, "_backend_import_error", load_error
        ):
            with self.assertRaisesRegex(
                ImportError,
                r"TecIO backend is not installed.*pip install -e",
            ) as caught:
                tecio.write_plt("field.plt", field)

        self.assertIs(caught.exception.__cause__, load_error)
```

- [ ] **Step 3: Run the focused tests and verify both intended failures**

Run:

```bash
python -m unittest \
  tests.test_distribution.PackagingConfigurationTests.test_scikit_build_context_allows_python_only_install \
  tests.test_distribution.PackagingConfigurationTests.test_missing_pybind11_reports_submodule_recovery_command \
  tests.test_tecio.TecIOAdapterTests.test_missing_backend_is_actionable -v
```

Expected: the Python-only CMake test fails with `Nothing to install`, and the error-message test fails because it still refers only to CMake. The missing-submodule test should already pass and protects its exact recovery command.

- [ ] **Step 4: Separate direct-CMake and scikit-build install destinations**

In the top-level `CMakeLists.txt`, retain the current default `CHEBY_PYTHON_INSTALL_DIR` and change the empty-install guard to apply only outside scikit-build-core:

```cmake
if(NOT CHEBY_INSTALL_TECIO AND NOT CHEBY_INSTALL_PYTHON_CORE AND NOT SKBUILD)
  message(FATAL_ERROR
    "Nothing to install: enable CHEBY_INSTALL_TECIO and/or "
    "CHEBY_INSTALL_PYTHON_CORE.")
endif()
```

In `native/tecio/CMakeLists.txt`, define one context-sensitive destination and use it in the existing install rule:

```cmake
if(SKBUILD)
  set(CHEBY_TECIO_PYTHON_DESTINATION "cheby_tools")
else()
  set(CHEBY_TECIO_PYTHON_DESTINATION
    "${CHEBY_PYTHON_INSTALL_DIR}/cheby_tools")
endif()

install(TARGETS _tecio
  LIBRARY DESTINATION "${CHEBY_TECIO_PYTHON_DESTINATION}"
)
```

- [ ] **Step 5: Replace the missing-backend message**

Change only the message in `cheby_tools/tecio.py`, preserving exception chaining:

```python
    if _backend is None:
        raise ImportError(
            "TecIO backend is not installed. Reinstall with: "
            "python -m pip install -e <cheby-tools-path>"
        ) from _backend_import_error
```

- [ ] **Step 6: Run the focused tests and the TecIO adapter suite**

Run:

```bash
python -m unittest \
  tests.test_distribution.PackagingConfigurationTests \
  tests.test_tecio -v
```

Expected: all tests PASS. Native tests may remain skipped when `_tecio` is not installed in the controlling environment.

- [ ] **Step 7: Commit CMake staging and error behavior**

```bash
git add CMakeLists.txt native/tecio/CMakeLists.txt cheby_tools/tecio.py \
  tests/test_distribution.py tests/test_tecio.py
git commit -m "build: stage TecIO through scikit-build-core"
```

### Task 3: Rebuild distribution tests around native artifacts

**Files:**
- Modify: `tests/test_distribution.py`

**Interfaces:**
- Consumes: The scikit-build-core metadata from Task 1 and CMake install rules from Task 2.
- Produces: Regression tests proving that wheel and sdist builds contain and install the native backend without restoring removed application packages.

- [ ] **Step 1: Make the isolated source fixture copy real native inputs**

Replace `copy_distribution_sources` with a fixture that copies the actual build inputs and excludes transient files:

```python
    def copy_distribution_sources(self, destination):
        destination = Path(destination)
        for filename in ("CMakeLists.txt", "pyproject.toml", "README.md"):
            shutil.copy2(REPOSITORY_ROOT / filename, destination / filename)

        for directory in ("cheby_tools", "examples", "native", "external"):
            shutil.copytree(
                REPOSITORY_ROOT / directory,
                destination / directory,
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "*.pyc", "._*"
                ),
            )

        for directory in ("stats", "discr", "spec_forge"):
            marker_directory = destination / directory
            marker_directory.mkdir()
            (marker_directory / "must_not_ship.py").write_text(
                "raise RuntimeError('distribution boundary failed')\n",
                encoding="utf-8",
            )
```

- [ ] **Step 2: Change the wheel test to require a native wheel and backend**

In `test_wheel_contains_only_cheby_tools_and_runs_when_installed`, add these assertions after selecting the wheel:

```python
            self.assertNotIn("py3-none-any", wheel.name)
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                extension_members = [
                    name
                    for name in names
                    if name.startswith("cheby_tools/_tecio")
                    and name.endswith((".so", ".dylib", ".pyd"))
                ]
                self.assertEqual(len(extension_members), 1, extension_members)
```

Extend `smoke_installed_package` so the subprocess imports the private backend and reports its module file:

```python
from cheby_tools import _tecio

# Add to the JSON result:
"tecio_backend": _tecio.__file__,
```

Then assert:

```python
        self.assertIn("_tecio", result["tecio_backend"])
```

Keep the existing checks excluding `stats`, `discr` and `spec_forge`. Remove the obsolete assertion that every `native/` archive member is forbidden; build sources remain absent from the wheel naturally, while the compiled extension is required.

- [ ] **Step 3: Run the wheel test and verify the old pure wheel fails**

Run in an environment containing `build`, scikit-build-core, CMake and the compiler:

```bash
python -m unittest \
  tests.test_distribution.DistributionTests.test_wheel_contains_only_cheby_tools_and_runs_when_installed -v
```

Expected before Tasks 1 and 2 are complete: FAIL because the wheel is pure or lacks `cheby_tools/_tecio`. After their implementation, expected: PASS.

- [ ] **Step 4: Change the sdist test to require build-complete native sources**

Rename the test to `test_sdist_contains_complete_native_build_inputs_and_installs`. Require these suffixes in the archive:

```python
            for suffix in (
                "/pyproject.toml",
                "/README.md",
                "/CMakeLists.txt",
                "/cheby_tools/__init__.py",
                "/native/tecio/CMakeLists.txt",
                "/native/tecio/tecio.cpp",
                "/external/boost/boost/version.hpp",
                "/external/tecio/teciosrc/CMakeLists.txt",
                "/external/pybind11/CMakeLists.txt",
                "/examples/field_quickstart.py",
                "/examples/write_plt.py",
            ):
                self.assertTrue(any(name.endswith(suffix) for name in names), suffix)
```

Continue rejecting `/tests/`, `/docs/`, removed application packages and generated assets. Keep installation from the produced sdist with `--no-build-isolation`, then run `smoke_installed_package` so the installed sdist must provide `_tecio`.

- [ ] **Step 5: Run both distribution tests**

Run:

```bash
python -m unittest tests.test_distribution.DistributionTests -v
```

Expected: wheel and sdist tests PASS; the wheel is native; the installation probes run outside the source tree and import `_tecio` from the installed target.

- [ ] **Step 6: Commit native distribution coverage**

```bash
git add tests/test_distribution.py
git commit -m "test: verify native wheel and sdist contents"
```

### Task 4: Prove both editable installation modes in clean environments

**Files:**
- Create: `tests/test_editable_install.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: The public pip commands from the specification.
- Produces: Opt-in end-to-end tests selected with `CHEBY_RUN_INSTALL_TESTS=1`; default mode writes `.plt`, and opt-out mode retains the pure-Python API while rejecting TecIO use.

- [ ] **Step 1: Add the editable-install integration test harness**

Create `tests/test_editable_install.py` with a class skipped unless explicitly enabled, so ordinary unit tests stay fast:

```python
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_INSTALL_TESTS = os.environ.get("CHEBY_RUN_INSTALL_TESTS") == "1"


@unittest.skipUnless(
    RUN_INSTALL_TESTS,
    "set CHEBY_RUN_INSTALL_TESTS=1 to exercise clean editable installs",
)
class EditableInstallTests(unittest.TestCase):
    def run_command(self, command, *, cwd):
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def make_venv(self, root):
        environment = root / ".venv"
        self.run_command([sys.executable, "-m", "venv", str(environment)], cwd=root)
        return environment / "bin" / "python"

    def run_probe(self, python, directory, code):
        output = self.run_command([str(python), "-c", code], cwd=directory)
        return json.loads(output.strip().splitlines()[-1])
```

Add `.test-venv/` to `.gitignore` as a defensive exclusion for developers who rerun the commands manually with a persistent environment.

- [ ] **Step 2: Add the failing default editable-install test**

Add this test method:

```python
    def test_default_editable_install_builds_and_writes_plt(self):
        with tempfile.TemporaryDirectory(prefix="cheby-editable-native-") as tmp:
            root = Path(tmp)
            python = self.make_venv(root)
            self.run_command(
                [str(python), "-m", "pip", "install", "-e", str(REPOSITORY_ROOT)],
                cwd=root,
            )
            probe = self.run_probe(
                python,
                root,
                """
import json
from pathlib import Path
import numpy as np
from cheby_tools import Field, SpectralDiscretization, _tecio
from cheby_tools.tecio import write_plt
grid = SpectralDiscretization([0.0], [1.0], [8], ["fourier"])
output = Path("editable.plt")
write_plt(output, Field(np.ones(8), grid, "u"))
print(json.dumps({
    "backend": _tecio.__file__,
    "header": output.read_bytes()[:8].decode("ascii"),
}))
""",
            )
            self.assertIn("_tecio", probe["backend"])
            self.assertEqual(probe["header"], "#!TDV112")
```

- [ ] **Step 3: Add the failing Python-only editable-install test**

Add this second method:

```python
    def test_explicit_opt_out_installs_working_python_core(self):
        with tempfile.TemporaryDirectory(prefix="cheby-editable-python-") as tmp:
            root = Path(tmp)
            python = self.make_venv(root)
            self.run_command(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    str(REPOSITORY_ROOT),
                    "-Ccmake.define.CHEBY_INSTALL_TECIO=OFF",
                ],
                cwd=root,
            )
            probe = self.run_probe(
                python,
                root,
                """
import json
import numpy as np
from cheby_tools import Field, SpectralDiscretization, tecio
grid = SpectralDiscretization([0.0], [1.0], [8], ["fourier"])
field = Field(np.ones(8), grid, "u")
try:
    tecio.write_plt("must-not-exist.plt", field)
except ImportError as exc:
    message = str(exc)
else:
    raise SystemExit("write_plt unexpectedly succeeded")
print(json.dumps({
    "shape": list(field.values.shape),
    "backend_missing": tecio._backend is None,
    "message": message,
}))
""",
            )
            self.assertEqual(probe["shape"], [8])
            self.assertTrue(probe["backend_missing"])
            self.assertIn("pip install -e", probe["message"])
```

- [ ] **Step 4: Run the end-to-end tests with build isolation enabled**

Run the exact public commands through the tests; do not add `--no-build-isolation`:

```bash
CHEBY_RUN_INSTALL_TESTS=1 python -m unittest tests.test_editable_install -v
```

Expected: both tests PASS. The subprocess working directories are temporary directories outside the checkout, preventing source-tree imports from masking packaging errors.

- [ ] **Step 5: Commit editable-install coverage**

```bash
git add .gitignore tests/test_editable_install.py
git commit -m "test: exercise editable native installation"
```

### Task 5: Document the one-command workflow and retain direct CMake as advanced use

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: The exact commands proven by Task 4.
- Produces: Colleague-facing installation instructions for a recursive clone, a project venv, default TecIO, explicit Python-only mode, ADASTRA prerequisites and native rebuilds.

- [ ] **Step 1: Replace the primary installation section**

State that NumPy is the runtime Python dependency and TecIO is built by default. Use this primary workflow, adapting only `<parent-directory>` in prose:

```bash
cd <parent-directory>
git clone --recurse-submodules -b update \
  git@github.com:sabide/cheby-tools.git cheby-tools-update
python3 -m venv post-processing/.venv
source post-processing/.venv/bin/activate
python -m pip install -e ./cheby-tools-update
```

Explain explicitly that the last command installs into the active
`post-processing/.venv` and that rerunning it rebuilds `_tecio` after C++ or CMake changes.

- [ ] **Step 2: Document the explicit Python-only command and behavior**

Add this command immediately after the default workflow:

```bash
python -m pip install -e ./cheby-tools-update \
  -Ccmake.define.CHEBY_INSTALL_TECIO=OFF
```

State that `Field` and `SpectralDiscretization` remain available, while `write_plt` raises an actionable `ImportError`.

- [ ] **Step 3: Rewrite the TecIO and ADASTRA build instructions**

List Python 3.11+, CMake 3.18+, a C++ compiler and initialized submodules as prerequisites. For ADASTRA, retain the module stack from `env.sh`:

```bash
module purge
module load cpe/24.07
module load PrgEnv-gnu/8.5.0
module load cmake/4.0.3
module load python/3.12.1
source post-processing/.venv/bin/activate
python -m pip install -e ./cheby-tools-update
```

Move `CHEBY_PYTHON_ENV`, `env.sh` and `run_cmake.sh` into an “Advanced direct CMake workflow” subsection. Do not present them as prerequisites for normal pip users.

- [ ] **Step 4: Check every documented command against the implementation**

Run:

```bash
rg -n "pip install|recurse-submodules|CHEBY_INSTALL_TECIO|run_cmake" README.md
python -m unittest tests.test_tecio.TecIOAdapterTests.test_missing_backend_is_actionable -v
```

Expected: the default editable command appears first, the opt-out spelling exactly matches the tests, direct CMake appears only in the advanced subsection, and the error-message test PASSes.

- [ ] **Step 5: Commit the documentation**

```bash
git add README.md
git commit -m "docs: simplify editable TecIO installation"
```

### Task 6: Run release-level verification on ADASTRA

**Files:**
- Modify only if a verification failure exposes a defect in a file owned by Tasks 1–5.

**Interfaces:**
- Consumes: All implementation tasks.
- Produces: Evidence that source tests, editable installs, native distributions and direct CMake remain functional without generated source-tree debris.

- [ ] **Step 1: Prepare an isolated ADASTRA verification environment**

Run outside the repository so the environment is not tracked:

```bash
module purge
module load cpe/24.07
module load PrgEnv-gnu/8.5.0
module load cmake/4.0.3
module load python/3.12.1
python -m venv /tmp/cheby-tools-release-venv
source /tmp/cheby-tools-release-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[dev]'
```

Expected: installation succeeds and the active Python reports version 3.12.

- [ ] **Step 2: Run the complete source suite**

```bash
python -m unittest discover -s tests -v
```

Expected: all ordinary tests PASS; only the explicitly gated clean-install tests may be skipped.

- [ ] **Step 3: Run both clean editable-install tests**

```bash
CHEBY_RUN_INSTALL_TESTS=1 python -m unittest tests.test_editable_install -v
```

Expected: two PASS results, including a classic `.plt` header from the default install and a pip-oriented `ImportError` from the opt-out install.

- [ ] **Step 4: Build and inspect release archives**

```bash
python -m build
python -m twine check dist/*
python - <<'PY'
from pathlib import Path
wheels = list(Path("dist").glob("cheby_tools-*.whl"))
assert len(wheels) == 1, wheels
assert "py3-none-any" not in wheels[0].name, wheels[0]
print(wheels[0])
PY
```

Expected: one native wheel and one sdist build successfully; Twine reports both distributions `PASSED`.

- [ ] **Step 5: Verify the advanced direct-CMake path in a separate prefix**

```bash
python -m venv /tmp/cheby-tools-cmake-venv
export CHEBY_PYTHON_ENV=/tmp/cheby-tools-cmake-venv
source env.sh
export CHEBY_BUILD_DIR=/tmp/cheby-tools-cmake-build
./run_cmake.sh
python -c 'from cheby_tools import _tecio; print(_tecio.__file__)'
```

Expected: `run_cmake.sh` completes and `_tecio` resolves inside `/tmp/cheby-tools-cmake-venv`.

- [ ] **Step 6: Confirm repository cleanliness**

```bash
git diff --check
git status --short
git ls-files | rg '(^|/)(__pycache__|build|dist|\.venv)(/|$)' && exit 1 || true
```

Expected: no whitespace errors, no unexpected generated files, and no tracked build/venv/cache paths. Only intentional implementation changes not yet committed may appear in status.

- [ ] **Step 7: Commit any verification-only correction, if one was necessary**

Use the exact affected paths rather than `git add -A`; for example, if only the sdist include list required correction:

```bash
git add pyproject.toml tests/test_distribution.py
git commit -m "fix: include complete native sdist inputs"
```

If every verification step passes without a correction, do not create an empty commit.
