import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PackagingConfigurationTests(unittest.TestCase):
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

    def test_scikit_build_core_owns_python_and_cmake_packaging(self):
        with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
            configuration = tomllib.load(stream)

        build_system = configuration["build-system"]
        self.assertEqual(
            build_system["build-backend"], "scikit_build_core.build"
        )
        self.assertTrue(
            any(
                requirement.startswith("scikit-build-core>=1.0")
                for requirement in build_system["requires"]
            )
        )

        scikit_build = configuration["tool"]["scikit-build"]
        self.assertEqual(scikit_build["wheel"]["packages"], ["cheby_tools"])
        self.assertEqual(scikit_build["sdist"]["inclusion-mode"], "explicit")
        self.assertFalse(
            scikit_build["cmake"]["define"]["CHEBY_INSTALL_PYTHON_CORE"]
        )
        included = scikit_build["sdist"]["include"]
        for required in (
            "pyproject.toml",
            "README.md",
            "cheby_tools/**",
            "CMakeLists.txt",
            "native/tecio/**",
            "external/boost/**",
            "external/tecio/teciosrc/**",
            "external/pybind11/CMakeLists.txt",
            "external/pybind11/LICENSE",
            "external/pybind11/include/**",
            "external/pybind11/tools/**",
        ):
            self.assertIn(required, included)
        self.assertNotIn("external/pybind11/**", included)

    def test_obsolete_setuptools_manifest_is_absent(self):
        self.assertFalse((REPOSITORY_ROOT / "MANIFEST.in").exists())

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


class DistributionTests(unittest.TestCase):
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

    def run_command(self, command, *, cwd, env=None):
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def smoke_installed_package(self, installed, working_directory):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(installed)
        environment["CHEBY_TEST_INSTALL"] = str(installed)
        smoke_code = """
import importlib.util
import json
from importlib.metadata import version
import numpy as np
import os
import sys

installed = os.environ["CHEBY_TEST_INSTALL"]
sys.path[:] = [
    installed,
    *(path for path in sys.path if "site-packages" not in path),
]

from cheby_tools import Field, SpectralDiscretization
from cheby_tools import _tecio

grid = SpectralDiscretization([0.0], [2.0 * np.pi], [24], ["fourier"])
field = Field(np.sin(3.0 * grid.nodes[0]), grid, "u")
error = float(np.max(np.abs(
    field.derivative(0).values - 3.0 * np.cos(3.0 * grid.nodes[0])
)))
removed_modules = {
    name: importlib.util.find_spec(name) is None
    for name in ("stats", "discr", "spec_forge")
}
print(json.dumps({
    "error": error,
    "removed_modules": removed_modules,
    "tecio_backend": _tecio.__file__,
    "version": version("cheby-tools"),
}))
"""
        output = self.run_command(
            [sys.executable, "-c", smoke_code],
            cwd=working_directory,
            env=environment,
        )
        result = json.loads(output.strip().splitlines()[-1])
        self.assertLess(result["error"], 2.0e-12)
        self.assertTrue(all(result["removed_modules"].values()))
        self.assertIn("_tecio", result["tecio_backend"])
        self.assertEqual(result["version"], "0.1.0")

    def run_quickstart(self, example, installed, working_directory):
        copied_example = working_directory / "field_quickstart.py"
        shutil.copy2(example, copied_example)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(installed)
        output = self.run_command(
            [sys.executable, str(copied_example)],
            cwd=working_directory,
            env=environment,
        )
        self.assertIn("derivative max error:", output)
        self.assertIn("interpolation max error:", output)

    def test_wheel_contains_only_cheby_tools_and_runs_when_installed(self):
        with tempfile.TemporaryDirectory(prefix="cheby-wheel-test-") as tmp:
            temporary_root = Path(tmp)
            project = temporary_root / "project"
            project.mkdir()
            self.copy_distribution_sources(project)
            wheelhouse = temporary_root / "wheelhouse"
            wheelhouse.mkdir()

            self.run_command(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(wheelhouse),
                    str(project),
                ],
                cwd=temporary_root,
            )

            wheels = list(wheelhouse.glob("cheby_tools-*.whl"))
            self.assertEqual(len(wheels), 1, wheels)
            wheel = wheels[0]
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
                for member in (
                    "cheby_tools/__init__.py",
                    "cheby_tools/field.py",
                    "cheby_tools/spectral.py",
                    "cheby_tools/tecio.py",
                ):
                    self.assertIn(member, names)
                self.assertFalse(
                    any(name.startswith("spec_forge/") for name in names)
                )
                self.assertFalse(any(name.startswith("discr/") for name in names))
                self.assertFalse(any(name.startswith("stats/") for name in names))
                top_level_packages = {
                    name.split("/", 1)[0]
                    for name in names
                    if "/" in name and ".dist-info/" not in name
                }
                self.assertEqual(top_level_packages, {"cheby_tools"})
                metadata_name = next(
                    name for name in names if name.endswith(".dist-info/METADATA")
                )
                metadata = archive.read(metadata_name).decode("utf-8")
                self.assertIn("Requires-Python: >=3.11", metadata)
                self.assertIn("Requires-Dist: numpy>=1.23", metadata)
                self.assertNotIn("Provides-Extra: io", metadata)
                self.assertIn("Provides-Extra: dev", metadata)

            installed = temporary_root / "installed"
            self.run_command(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    str(installed),
                    str(wheel),
                ],
                cwd=temporary_root,
            )
            outside = temporary_root / "outside"
            outside.mkdir()
            self.smoke_installed_package(installed, outside)
            self.run_quickstart(
                project / "examples" / "field_quickstart.py",
                installed,
                outside,
            )

    def test_sdist_contains_complete_native_build_inputs_and_installs(self):
        with tempfile.TemporaryDirectory(prefix="cheby-sdist-test-") as tmp:
            temporary_root = Path(tmp)
            project = temporary_root / "project"
            project.mkdir()
            self.copy_distribution_sources(project)
            output_directory = temporary_root / "dist"
            output_directory.mkdir()

            self.run_command(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--sdist",
                    "--no-isolation",
                    "--outdir",
                    str(output_directory),
                    str(project),
                ],
                cwd=temporary_root,
            )

            archives = list(output_directory.glob("cheby_tools-*.tar.gz"))
            self.assertEqual(len(archives), 1, archives)
            with tarfile.open(archives[0], "r:gz") as archive:
                names = archive.getnames()

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
            for fragment in (
                "/tests/",
                "/docs/",
                "/stats/",
                "/discr/",
                "/spec_forge/",
                ".lay",
                ".eps",
                ".png",
            ):
                self.assertFalse(any(fragment in name for name in names), fragment)

            installed = temporary_root / "installed"
            self.run_command(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "--target",
                    str(installed),
                    str(archives[0]),
                ],
                cwd=temporary_root,
            )
            outside = temporary_root / "outside"
            outside.mkdir()
            self.smoke_installed_package(installed, outside)
            extracted = temporary_root / "extracted"
            with tarfile.open(archives[0], "r:gz") as archive:
                archive.extractall(extracted, filter="data")
            source_root = next(extracted.glob("cheby_tools-*"))
            self.run_quickstart(
                source_root / "examples" / "field_quickstart.py",
                installed,
                outside,
            )


class RepositoryCleanupTests(unittest.TestCase):
    def test_application_specific_sources_and_assets_are_absent(self):
        removed_paths = (
            "stats",
            "discr",
            "spec_forge",
            "compile_h5py.sh",
            "env_h5py.sh",
            "cfg_adastra.sh",
            "examples/build_ercoftac_db.py",
            "examples/discr_example.py",
            "examples/tecio_example.py",
            "examples/spectral_quickstart.py",
            "examples/figs",
        )
        for relative_path in removed_paths:
            with self.subTest(path=relative_path):
                self.assertFalse((REPOSITORY_ROOT / relative_path).exists())

        for pattern in ("**/*.lay", "**/*.eps", "**/*.png"):
            with self.subTest(pattern=pattern):
                self.assertEqual(list((REPOSITORY_ROOT / "examples").glob(pattern)), [])

    def test_legacy_implementation_names_are_absent(self):
        spectral_source = (REPOSITORY_ROOT / "cheby_tools/spectral.py").read_text(
            encoding="utf-8"
        )
        cmake_source = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        gitignore_lines = (REPOSITORY_ROOT / ".gitignore").read_text(
            encoding="utf-8"
        ).splitlines()

        self.assertNotIn("_FourierInterpBetween1DLegacy", spectral_source)
        self.assertNotIn("tecio_wrapper", cmake_source)
        self.assertNotIn("CHEBY_INSTALL_TECIO_WRAPPER", cmake_source)
        self.assertIn("*.plt", gitignore_lines)
        self.assertNotIn("*.plt ", gitignore_lines)
        self.assertFalse(any("tecio_wrapper" in line for line in gitignore_lines))
        self.assertFalse(any("stats_hii" in line for line in gitignore_lines))


if __name__ == "__main__":
    unittest.main()
