import numpy as np

from cheby_tools import Field, SpectralDiscretization


def main():
    coarse = SpectralDiscretization(
        [0.0], [2.0 * np.pi], [24], ["fourier"]
    )
    fine = SpectralDiscretization(
        [0.0], [2.0 * np.pi], [48], ["fourier"]
    )
    x_coarse, x_fine = coarse.nodes[0], fine.nodes[0]
    field = Field(
        np.sin(3.0 * x_coarse) + 0.25 * np.cos(5.0 * x_coarse),
        coarse,
        "temperature",
    )
    derivative_error = np.max(
        np.abs(
            field.derivative(0).values
            - (
                3.0 * np.cos(3.0 * x_coarse)
                - 1.25 * np.sin(5.0 * x_coarse)
            )
        )
    )
    interpolation_error = np.max(
        np.abs(
            field.interpolate(fine).values
            - (
                np.sin(3.0 * x_fine)
                + 0.25 * np.cos(5.0 * x_fine)
            )
        )
    )
    print(f"derivative max error:    {derivative_error:.3e}")
    print(f"interpolation max error: {interpolation_error:.3e}")
    if derivative_error > 2.0e-12 or interpolation_error > 2.0e-12:
        raise SystemExit("field quick-start validation failed")


if __name__ == "__main__":
    main()
