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
            [0.0, -1.0],
            [2.0 * np.pi, 1.0],
            [8, 7],
            ["fourier", "chebyshev"],
        )
        x, y = grid.meshgrid()
        fields = [Field(np.sin(x), grid, "u"), Field(y**2, grid, "v")]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fields.plt"
            tecio.write_plt(output, fields)

            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
