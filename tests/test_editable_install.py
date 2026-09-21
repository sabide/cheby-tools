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
        self.run_command(
            [sys.executable, "-m", "venv", str(environment)], cwd=root
        )
        return environment / "bin" / "python"

    def run_probe(self, python, directory, code):
        output = self.run_command([str(python), "-c", code], cwd=directory)
        return json.loads(output.strip().splitlines()[-1])

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
import cheby_tools
from cheby_tools import Field, SpectralDiscretization, _tecio
from cheby_tools.tecio import write_plt
grid = SpectralDiscretization([0.0], [1.0], [8], ["fourier"])
output = Path("editable.plt")
write_plt(output, Field(np.ones(8), grid, "u"))
print(json.dumps({
    "backend": _tecio.__file__,
    "header": output.read_bytes()[:8].decode("ascii"),
    "package": cheby_tools.__file__,
}))
""",
            )
            self.assertIn("_tecio", probe["backend"])
            self.assertNotIn(str(REPOSITORY_ROOT), probe["backend"])
            self.assertEqual(probe["header"], "#!TDV112")
            self.assertTrue(
                Path(probe["package"]).is_relative_to(REPOSITORY_ROOT)
            )

    def test_explicit_opt_out_installs_working_python_core(self):
        with tempfile.TemporaryDirectory(
            prefix="cheby-editable-python-"
        ) as tmp:
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


if __name__ == "__main__":
    unittest.main()
