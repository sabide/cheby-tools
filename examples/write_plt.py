import numpy as np

from cheby_tools import Field, SpectralDiscretization
from cheby_tools.tecio import write_plt


grid = SpectralDiscretization(
    [0.0, -1.0],
    [2.0 * np.pi, 1.0],
    [64, 33],
    ["fourier", "chebyshev"],
)
x, y = grid.meshgrid()
u = Field(np.sin(x) * (1.0 - y**2), grid, "u")
temperature = Field(np.cos(2.0 * x) + y, grid, "temperature")
write_plt("fields.plt", [u, temperature])
print("wrote fields.plt")
