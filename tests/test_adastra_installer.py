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

    @unittest.skipUnless(
        os.environ.get("CHEBY_RUN_ADASTRA_INSTALL_TESTS") == "1",
        "set CHEBY_RUN_ADASTRA_INSTALL_TESTS=1 to run the native installer",
    )
    def test_creates_venv_and_installs_native_backend(self):
        with tempfile.TemporaryDirectory(prefix="cheby-adastra-install-") as tmp:
            target = Path(tmp) / "project" / ".venv"

            result = self.run_installer(target)

            self.assertEqual(result.returncode, 0, result.stdout)
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
            self.assertIn(f"source {target}/bin/activate", result.stdout)


if __name__ == "__main__":
    unittest.main()
