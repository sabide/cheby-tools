import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def copy_distribution_sources(self, destination):
        destination = Path(destination)
        for filename in ("pyproject.toml", "README.md", "MANIFEST.in"):
            source = REPOSITORY_ROOT / filename
            if source.exists():
                shutil.copy2(source, destination / filename)

        for directory in ("spec_forge", "discr", "stats", "examples", "tests"):
            shutil.copytree(
                REPOSITORY_ROOT / directory,
                destination / directory,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        shutil.copytree(
            REPOSITORY_ROOT / "tecio_wrapper",
            destination / "tecio_wrapper",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (destination / "external").mkdir()
        (destination / "external" / "vendor-marker.txt").write_text(
            "must not ship in the Python source distribution\n",
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

    def test_wheel_installs_and_runs_outside_source_tree(self):
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
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                self.assertIn("spec_forge/__init__.py", names)
                self.assertIn("discr/__init__.py", names)
                self.assertIn("stats/__init__.py", names)
                self.assertFalse(any(name.startswith("tecio_wrapper/") for name in names))
                top_level_packages = {
                    name.split("/", 1)[0]
                    for name in names
                    if "/" in name and ".dist-info/" not in name
                }
                self.assertEqual(
                    top_level_packages,
                    {"discr", "spec_forge", "stats"},
                )
                metadata_name = next(
                    name for name in names if name.endswith(".dist-info/METADATA")
                )
                metadata = archive.read(metadata_name).decode("utf-8")
                self.assertIn("Requires-Python: >=3.11", metadata)
                self.assertIn("Requires-Dist: numpy>=1.23", metadata)
                self.assertIn("Provides-Extra: io", metadata)
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
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(installed)
            example_source = project / "examples" / "spectral_quickstart.py"
            self.assertTrue(example_source.is_file())
            example = outside / "spectral_quickstart.py"
            shutil.copy2(example_source, example)
            example_output = self.run_command(
                [sys.executable, str(example)],
                cwd=outside,
                env=environment,
            )
            self.assertIn("derivative max error:", example_output)
            self.assertIn("interpolation max error:", example_output)

            smoke_code = """
import json
from importlib.metadata import version
import numpy as np
import spec_forge
from spec_forge import SpectralDiscretization

ops = SpectralDiscretization([0.0], [2.0 * np.pi], [24], ["fourier"])
x = ops.nodes[0]
error = float(np.max(np.abs(ops.ddx(np.sin(3.0 * x)) - 3.0 * np.cos(3.0 * x))))
print(json.dumps({"error": error, "module": spec_forge.__file__, "version": version("cheby-tools")}))
"""
            output = self.run_command(
                [sys.executable, "-c", smoke_code],
                cwd=outside,
                env=environment,
            )
            result = json.loads(output.strip().splitlines()[-1])
            self.assertLess(result["error"], 2.0e-12)
            self.assertEqual(result["version"], "0.1.0")
            self.assertTrue(Path(result["module"]).is_relative_to(installed))

    def test_sdist_contains_core_sources_without_vendor_trees(self):
        with tempfile.TemporaryDirectory(prefix="cheby-sdist-test-") as tmp:
            temporary_root = Path(tmp)
            project = temporary_root / "project"
            project.mkdir()
            self.copy_distribution_sources(project)
            (project / "dist").mkdir()

            self.run_command(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--sdist",
                    "--no-isolation",
                    "--outdir",
                    str(project / "dist"),
                    str(project),
                ],
                cwd=temporary_root,
            )

            archives = list((project / "dist").glob("cheby_tools-*.tar.gz"))
            self.assertEqual(len(archives), 1, archives)
            with tarfile.open(archives[0], "r:gz") as archive:
                names = archive.getnames()
                example_name = next(
                    name
                    for name in names
                    if name.endswith("/examples/spectral_quickstart.py")
                )
                example_contents = archive.extractfile(example_name).read()

            self.assertTrue(any(name.endswith("/pyproject.toml") for name in names))
            self.assertTrue(any(name.endswith("/spec_forge/__init__.py") for name in names))
            self.assertTrue(any(name.endswith("/discr/__init__.py") for name in names))
            self.assertTrue(any(name.endswith("/stats/__init__.py") for name in names))
            self.assertTrue(
                any(name.endswith("/examples/spectral_quickstart.py") for name in names)
            )
            self.assertFalse(any("/external/" in name for name in names))
            self.assertFalse(any("/tecio_wrapper/" in name for name in names))

            installed = temporary_root / "installed-sdist"
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

            outside = temporary_root / "outside-sdist"
            outside.mkdir()
            example = outside / "spectral_quickstart.py"
            example.write_bytes(example_contents)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(installed)
            output = self.run_command(
                [sys.executable, str(example)],
                cwd=outside,
                env=environment,
            )
            self.assertIn("derivative max error:", output)
            self.assertIn("interpolation max error:", output)


class OptionalDependencyTests(unittest.TestCase):
    def test_stats_imports_without_h5py_and_explains_the_io_extra(self):
        saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "stats" or name.startswith("stats.")
        }
        for name in saved_modules:
            del sys.modules[name]

        original_import = __import__

        def import_without_h5py(name, *args, **kwargs):
            if name == "h5py":
                raise ModuleNotFoundError("No module named 'h5py'", name="h5py")
            return original_import(name, *args, **kwargs)

        try:
            with mock.patch("builtins.__import__", side_effect=import_without_h5py):
                try:
                    import stats
                except ModuleNotFoundError as exc:
                    self.fail(f"import stats unexpectedly requires h5py: {exc}")

                self.assertTrue(callable(stats.wall_profile))
                with self.assertRaisesRegex(ImportError, r"cheby-tools\[io\]"):
                    _ = stats.H5DB
        finally:
            for name in list(sys.modules):
                if name == "stats" or name.startswith("stats."):
                    del sys.modules[name]
            sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
