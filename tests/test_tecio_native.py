import os
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
        grid = SpectralDiscretization(
            [0.0], [2.0 * np.pi], [8], ["fourier"]
        )
        field = Field(np.sin(grid.nodes[0]), grid, "u")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            read_only = root / "read-only"
            output_directory = root / "output"
            read_only.mkdir()
            output_directory.mkdir()
            read_only.chmod(0o555)
            original_directory = Path.cwd()
            try:
                os.chdir(read_only)
                output = (output_directory / "field.plt").resolve()
                tecio.write_plt(output, field)
            finally:
                os.chdir(original_directory)
                read_only.chmod(0o755)

            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
