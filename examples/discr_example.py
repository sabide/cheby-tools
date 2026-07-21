import numpy as np

from spec_forge import SpectralDiscretization

xmin, xmax = [-1.0, -2.0], [2.0, 3.0]
n = [48, 64]
ops = SpectralDiscretization(
    xmin=xmin,
    xmax=xmax,
    n=n,
    bases=["chebyshev", "chebyshev"],
)

X, Y = ops.meshgrid()
field = np.sin(np.pi * X) * np.cos(3.0 * np.pi * Y)
expected_dx = np.pi * np.cos(np.pi * X) * np.cos(3.0 * np.pi * Y)
error = np.max(np.abs(ops.ddx(field) - expected_dx))
print(f"max derivative error: {error:.3e}")
