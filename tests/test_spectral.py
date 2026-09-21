import unittest

import numpy as np

import cheby_tools
from cheby_tools import SpectralDiscretization
from cheby_tools.spectral import SpectralInterpolate


class PublicApiTests(unittest.TestCase):
    def test_public_api_exports_only_supported_symbols(self):
        self.assertEqual(cheby_tools.__all__, ["SpectralDiscretization"])
        self.assertFalse(hasattr(cheby_tools, "SpectralInterpolate"))
        self.assertFalse(hasattr(cheby_tools, "np"))


class SpectralDiscretizationTests(unittest.TestCase):
    def test_chebyshev_derivative_and_roundtrip(self):
        ops = SpectralDiscretization([0.0], [1.0], [41], ["chebyshev"])
        (x,) = ops.nodes
        field = np.exp(x) + x**4
        expected = np.exp(x) + 4.0 * x**3
        np.testing.assert_allclose(ops.ddx(field), expected, atol=2e-11)
        np.testing.assert_allclose(ops.nodal_values(ops.spectral_coeffs(field)), field, atol=2e-13)

    def test_fourier_derivative(self):
        ops = SpectralDiscretization([0.0], [2.0 * np.pi], [48], ["fourier"])
        (x,) = ops.nodes
        field = np.sin(3.0 * x) + 0.25 * np.cos(5.0 * x)
        expected = 3.0 * np.cos(3.0 * x) - 1.25 * np.sin(5.0 * x)
        np.testing.assert_allclose(ops.ddx(field), expected, atol=2e-12)

    def test_mixed_2d_derivatives_and_integral(self):
        ops = SpectralDiscretization(
            [0.0, -1.0], [2.0 * np.pi, 1.0], [32, 35], ["fourier", "chebyshev"]
        )
        X, Y = ops.meshgrid()
        field = np.sin(2.0 * X) * (1.0 + Y**2)
        np.testing.assert_allclose(ops.ddx(field), 2.0 * np.cos(2.0 * X) * (1.0 + Y**2), atol=2e-11)
        np.testing.assert_allclose(ops.ddy(field), 2.0 * Y * np.sin(2.0 * X), atol=2e-11)
        self.assertAlmostEqual(float(ops.integrate(field)), 0.0, places=12)

    def test_interpolate_point_from_nodal_values(self):
        ops = SpectralDiscretization(
            [0.0, -1.0], [2.0 * np.pi, 1.0], [48, 41], ["fourier", "chebyshev"]
        )
        X, Y = ops.meshgrid()
        field = np.sin(2.0 * X) * np.cos(np.pi * Y)
        x0, y0 = 0.731, -0.234
        expected = np.sin(2.0 * x0) * np.cos(np.pi * y0)
        self.assertAlmostEqual(float(np.real(ops.interpolate_point(field, x0, y0))), expected, places=11)
        self.assertAlmostEqual(float(np.real(ops.interpolate_point(field, [x0, y0]))), expected, places=11)

    def test_interpolation_between_grids(self):
        coarse = SpectralDiscretization(
            [0.0, -1.0], [2.0 * np.pi, 1.0], [20, 21], ["fourier", "chebyshev"]
        )
        fine = SpectralDiscretization(
            [0.0, -1.0], [2.0 * np.pi, 1.0], [40, 41], ["fourier", "chebyshev"]
        )
        Xc, Yc = coarse.meshgrid()
        field = np.cos(3.0 * Xc) * (1.0 + Yc + Yc**3)
        Xf, Yf = fine.meshgrid()
        expected = np.cos(3.0 * Xf) * (1.0 + Yf + Yf**3)
        np.testing.assert_allclose(SpectralInterpolate(coarse, fine) @ field, expected, atol=2e-12)

if __name__ == "__main__":
    unittest.main()
