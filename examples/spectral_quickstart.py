"""Minimal analytical check for an installed cheby-tools package."""

import numpy as np

from spec_forge import SpectralDiscretization, SpectralInterpolate


def main():
    coarse = SpectralDiscretization(
        [0.0], [2.0 * np.pi], [24], ["fourier"]
    )
    fine = SpectralDiscretization(
        [0.0], [2.0 * np.pi], [48], ["fourier"]
    )

    x_coarse = coarse.nodes[0]
    x_fine = fine.nodes[0]
    field = np.sin(3.0 * x_coarse) + 0.25 * np.cos(5.0 * x_coarse)

    derivative = coarse.ddx(field)
    expected_derivative = (
        3.0 * np.cos(3.0 * x_coarse)
        - 1.25 * np.sin(5.0 * x_coarse)
    )
    derivative_error = np.max(np.abs(derivative - expected_derivative))

    interpolated = SpectralInterpolate(coarse, fine) @ field
    expected_fine = np.sin(3.0 * x_fine) + 0.25 * np.cos(5.0 * x_fine)
    interpolation_error = np.max(np.abs(interpolated - expected_fine))

    tolerance = 2.0e-12
    print(f"derivative max error:    {derivative_error:.3e}")
    print(f"interpolation max error: {interpolation_error:.3e}")
    if derivative_error > tolerance or interpolation_error > tolerance:
        raise SystemExit("spectral quickstart validation failed")


if __name__ == "__main__":
    main()
