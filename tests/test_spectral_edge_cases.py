import unittest

import numpy as np

from cheby_tools import SpectralDiscretization
from cheby_tools.spectral import SpectralInterpolate


class FourierResamplingTests(unittest.TestCase):
    @staticmethod
    def grid(size):
        return SpectralDiscretization(
            [0.0], [2.0 * np.pi], [size], ["fourier"]
        )

    def test_refining_source_nyquist_preserves_real_amplitude(self):
        source = self.grid(8)
        destination = self.grid(16)
        field = np.cos(4.0 * source.nodes[0])

        actual = SpectralInterpolate(source, destination) @ field
        expected = np.cos(4.0 * destination.nodes[0])

        np.testing.assert_allclose(actual, expected, atol=2e-12)

    def test_coarsening_to_target_nyquist_preserves_real_amplitude(self):
        source = self.grid(16)
        destination = self.grid(8)
        field = np.cos(4.0 * source.nodes[0])

        actual = SpectralInterpolate(source, destination) @ field
        expected = np.cos(4.0 * destination.nodes[0])

        np.testing.assert_allclose(actual, expected, atol=2e-12)

    def test_coarsening_complex_field_preserves_resolved_modes(self):
        source = self.grid(20)
        destination = self.grid(8)
        field = (
            np.exp(1j * source.nodes[0])
            + 0.25 * np.exp(-2j * source.nodes[0])
        )

        actual = SpectralInterpolate(source, destination) @ field
        expected = (
            np.exp(1j * destination.nodes[0])
            + 0.25 * np.exp(-2j * destination.nodes[0])
        )

        np.testing.assert_allclose(actual, expected, atol=2e-12)

    def test_low_complex_modes_survive_even_odd_size_combinations(self):
        for source_size, destination_size in ((7, 12), (8, 13), (13, 8), (12, 7)):
            with self.subTest(
                source_size=source_size, destination_size=destination_size
            ):
                source = self.grid(source_size)
                destination = self.grid(destination_size)
                field = (
                    0.75
                    + np.exp(2j * source.nodes[0])
                    - 0.3j * np.exp(-1j * source.nodes[0])
                )
                expected = (
                    0.75
                    + np.exp(2j * destination.nodes[0])
                    - 0.3j * np.exp(-1j * destination.nodes[0])
                )

                actual = SpectralInterpolate(source, destination) @ field

                np.testing.assert_allclose(actual, expected, atol=2e-12)

    def test_source_nyquist_uses_symmetric_interpolant_for_complex_data(self):
        source = self.grid(8)
        destination = self.grid(16)
        field = (1.0 + 2.0j) * np.cos(4.0 * source.nodes[0])

        actual = SpectralInterpolate(source, destination) @ field
        expected = (1.0 + 2.0j) * np.cos(4.0 * destination.nodes[0])

        np.testing.assert_allclose(actual, expected, atol=2e-12)

    def test_point_evaluation_uses_same_symmetric_nyquist_convention(self):
        discretization = self.grid(8)
        target = np.pi / 8.0

        for amplitude in (1.0, 1.0 + 2.0j):
            with self.subTest(amplitude=amplitude):
                field = amplitude * np.cos(4.0 * discretization.nodes[0])
                actual = discretization.interpolate_point(field, target)
                expected = amplitude * np.cos(4.0 * target)
                self.assertAlmostEqual(abs(actual - expected), 0.0, places=13)

    def test_first_derivative_uses_symmetric_nyquist_convention(self):
        discretization = self.grid(8)

        for amplitude in (1.0, 1.0 + 2.0j):
            with self.subTest(amplitude=amplitude):
                field = amplitude * np.cos(4.0 * discretization.nodes[0])
                actual = discretization.ddx(field)
                np.testing.assert_allclose(actual, 0.0, atol=2e-12)

    def test_unresolved_modes_are_removed_when_coarsening(self):
        source = self.grid(16)
        destination = self.grid(8)
        field = np.exp(6j * source.nodes[0])

        actual = SpectralInterpolate(source, destination) @ field

        np.testing.assert_allclose(actual, 0.0, atol=2e-12)


class InputValidationTests(unittest.TestCase):
    def test_grid_sizes_must_be_integer_values(self):
        invalid_sizes = (8.9, 8.0, True)

        for size in invalid_sizes:
            with self.subTest(size=size):
                with self.assertRaises(TypeError):
                    SpectralDiscretization(
                        [0.0], [1.0], [size], ["fourier"]
                    )

    def test_bounds_must_be_finite_and_strictly_increasing(self):
        invalid_bounds = (
            (1.0, 0.0),
            (1.0, 1.0),
            (np.nan, 1.0),
            (0.0, np.inf),
        )

        for lower, upper in invalid_bounds:
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaises(ValueError):
                    SpectralDiscretization(
                        [lower], [upper], [8], ["fourier"]
                    )

    def test_domain_length_must_be_finite(self):
        largest = np.finfo(float).max

        with self.assertRaises(ValueError):
            SpectralDiscretization(
                [-largest], [largest], [8], ["fourier"]
            )

    def test_chebyshev_interpolation_rejects_targets_outside_domain(self):
        discretization = SpectralDiscretization(
            [0.0], [1.0], [9], ["chebyshev"]
        )
        field = discretization.nodes[0]

        for target in (-0.01, 1.01):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    discretization.interpolate_point(field, target)

    def test_chebyshev_interpolation_accepts_endpoint_roundoff(self):
        discretization = SpectralDiscretization(
            [0.0], [1.0], [9], ["chebyshev"]
        )
        field = discretization.nodes[0]
        target = 1.0 + 4.0 * np.finfo(float).eps

        actual = discretization.interpolate_point(field, target)

        self.assertAlmostEqual(float(np.real(actual)), 1.0, places=14)

    def test_chebyshev_tolerance_does_not_dwarf_a_tiny_domain(self):
        discretization = SpectralDiscretization(
            [0.0], [1.0e-20], [9], ["chebyshev"]
        )
        field = discretization.nodes[0]

        with self.assertRaises(ValueError):
            discretization.interpolate_point(field, 1.0e-15)

    def test_chebyshev_accepts_one_ulp_beyond_large_offset_endpoint(self):
        lower = 1.0e12
        upper = lower + 1.0
        discretization = SpectralDiscretization(
            [lower], [upper], [9], ["chebyshev"]
        )
        field = discretization.nodes[0] - lower
        target = np.nextafter(upper, np.inf)

        actual = discretization.interpolate_point(field, target)

        self.assertAlmostEqual(float(np.real(actual)), 1.0, places=12)

    def test_fourier_interpolation_remains_periodic_outside_base_interval(self):
        discretization = SpectralDiscretization(
            [0.0], [2.0 * np.pi], [16], ["fourier"]
        )
        field = np.sin(discretization.nodes[0])

        actual = discretization.interpolate_point(field, 2.0 * np.pi + 0.25)

        self.assertAlmostEqual(float(np.real(actual)), np.sin(0.25), places=13)

    def test_interpolation_targets_must_be_finite(self):
        for basis, size in (("chebyshev", 9), ("fourier", 8)):
            discretization = SpectralDiscretization(
                [0.0], [1.0], [size], [basis]
            )
            field = np.ones(size)

            with self.subTest(basis=basis):
                with self.assertRaises(ValueError):
                    discretization.interpolate_point(field, np.nan)


if __name__ == "__main__":
    unittest.main()
