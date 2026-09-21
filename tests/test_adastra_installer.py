import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPOSITORY_ROOT / "install_adastra.sh"


class AdastraInstallerTests(unittest.TestCase):
    def run_installer(self, *arguments):
        return subprocess.run(
            ["bash", str(INSTALLER), *map(str, arguments)],
            cwd=REPOSITORY_ROOT,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_target_venv_path_is_required(self):
        result = self.run_installer()

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("Usage:", result.stdout)

    def test_existing_non_venv_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="cheby-adastra-invalid-") as tmp:
            target = Path(tmp) / "not-a-venv"
            target.mkdir()
            sentinel = target / "keep.txt"
            sentinel.write_text("keep me\n", encoding="utf-8")

            result = self.run_installer(target)

            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("not a valid Python virtual environment", result.stdout)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep me\n")

    def test_counterfeit_python_executable_is_not_accepted_as_a_venv(self):
        with tempfile.TemporaryDirectory(prefix="cheby-adastra-counterfeit-") as tmp:
            target = Path(tmp) / "counterfeit"
            (target / "bin").mkdir(parents=True)
            (target / "pyvenv.cfg").write_text(
                "home = /counterfeit\n", encoding="utf-8"
            )
            (target / "bin" / "activate").touch()
            fake_python = target / "bin" / "python"
            fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_python.chmod(0o755)

            result = self.run_installer(target)

            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("not a valid Python virtual environment", result.stdout)
            self.assertNotIn("Installation ADASTRA terminée", result.stdout)

    def test_compiler_resolution_returns_an_absolute_executable(self):
        with tempfile.TemporaryDirectory(prefix="cheby-adastra-compiler-") as tmp:
            root = Path(tmp)
            relative_bin = root / "relative-bin"
            relative_bin.mkdir()
            compiler = relative_bin / "CC"
            compiler.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            compiler.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"relative-bin:{environment['PATH']}"

            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; resolve_executable CC',
                    "bash",
                    str(INSTALLER),
                ],
                cwd=root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stdout.strip(), str(compiler.resolve()))

    @unittest.skipUnless(
        os.environ.get("CHEBY_RUN_ADASTRA_INSTALL_TESTS") == "1",
        "set CHEBY_RUN_ADASTRA_INSTALL_TESTS=1 to run the native installer",
    )
    def test_creates_venv_and_installs_native_backend(self):
        with tempfile.TemporaryDirectory(prefix="cheby-adastra-install-") as tmp:
            target = Path(tmp) / "project with spaces" / ".venv"

            result = self.run_installer(target)

            self.assertEqual(result.returncode, 0, result.stdout)
            second_result = self.run_installer(target)
            self.assertEqual(second_result.returncode, 0, second_result.stdout)
            python = target / "bin" / "python"
            smoke = subprocess.run(
                [
                    str(python),
                    "-c",
                    (
                        "from pathlib import Path; "
                        "from cheby_tools import Field, SpectralDiscretization, _tecio; "
                        "root = Path(__import__('sys').prefix).resolve(); "
                        "backend = Path(_tecio.__file__).resolve(); "
                        "assert backend.is_relative_to(root), (backend, root); "
                        "print(backend)"
                    ),
                ],
                cwd=Path(tmp),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(smoke.returncode, 0, smoke.stdout)
            self.assertIn("Installation ADASTRA terminée", result.stdout)
            escaped_path = str(target / "bin" / "activate").replace(" ", "\\ ")
            activation = f"source {escaped_path}"
            self.assertIn(activation, result.stdout)


if __name__ == "__main__":
    unittest.main()
