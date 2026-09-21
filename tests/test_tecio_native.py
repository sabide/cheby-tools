import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

from cheby_tools import Field, SpectralDiscretization
from cheby_tools import tecio


@unittest.skipIf(tecio._backend is None, "native TecIO backend is not installed")
class NativeTecIOTests(unittest.TestCase):
    def test_multifield_output_has_classic_header_and_spaced_name(self):
        grid = SpectralDiscretization(
            [0.0, -1.0],
            [2.0 * np.pi, 1.0],
            [8, 7],
            ["fourier", "chebyshev"],
        )
        x, y = grid.meshgrid()
        fields = [
            Field(np.sin(x), grid, "u"),
            Field(y**2, grid, "velocity magnitude"),
            Field(x - y, grid, "pressure, total"),
            Field(x + y, grid, "é" * 64),
        ]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fields.plt"
            tecio.write_plt(output, fields)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(output.read_bytes()[:8], b"#!TDV112")

    def test_private_backend_rejects_oversized_variable_name(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fields.plt"
            with self.assertRaisesRegex(ValueError, "128-byte limit"):
                tecio._backend.write_plt(
                    str(output),
                    ["x" * 129],
                    [np.ones(2, dtype=np.float64)],
                )

    def test_absolute_output_uses_parent_directory_for_scratch_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            read_only = root / "read-only"
            output_directory = root / "output"
            read_only.mkdir()
            output_directory.mkdir()
            read_only.chmod(0o555)
            try:
                probe = read_only / "probe"
                try:
                    probe.write_text("probe", encoding="utf-8")
                except PermissionError:
                    pass
                else:
                    probe.unlink()
                    self.skipTest("test user can write to a mode-0555 directory")

                output = (output_directory / "field.plt").resolve()
                package_root = Path(tecio.__file__).resolve().parents[1]
                code = f"""
import os
from pathlib import Path
import numpy as np
import cheby_tools
from cheby_tools import Field, SpectralDiscretization
from cheby_tools.tecio import write_plt
expected_root = Path(os.environ[\"CHEBY_TEST_PACKAGE_ROOT\"]).resolve()
actual_root = Path(cheby_tools.__file__).resolve().parents[1]
if actual_root != expected_root:
    raise SystemExit(f\"loaded {{actual_root}}, expected {{expected_root}}\")
grid = SpectralDiscretization([0.0], [2.0 * np.pi], [8], [\"fourier\"])
field = Field(np.sin(grid.nodes[0]), grid, \"u\")
write_plt({str(output)!r}, field)
"""
                environment = os.environ.copy()
                environment["CHEBY_TEST_PACKAGE_ROOT"] = str(package_root)
                environment["PYTHONPATH"] = os.pathsep.join(
                    filter(
                        None,
                        (str(package_root), environment.get("PYTHONPATH")),
                    )
                )
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=read_only,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout)
            finally:
                read_only.chmod(0o755)

            self.assertGreater(output.stat().st_size, 0)

    @unittest.skipIf(os.name == "nt", "backslash is a path separator on Windows")
    def test_posix_backslash_in_filename_is_not_a_path_separator(self):
        grid = SpectralDiscretization([0.0], [1.0], [4], ["fourier"])
        field = Field(np.ones(4), grid, "u")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "field\\name.plt"
            tecio.write_plt(output, field)

            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
