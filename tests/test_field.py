import unittest

import numpy as np

from cheby_tools import Field, SpectralDiscretization


class FieldTests(unittest.TestCase):
    def setUp(self):
        self.grid = SpectralDiscretization(
            [0.0, -1.0],
            [2.0 * np.pi, 1.0],
            [24, 17],
            ["fourier", "chebyshev"],
        )
        self.x, self.y = self.grid.meshgrid()
        self.values = np.sin(3.0 * self.x) * (1.0 + self.y**2)

    def test_constructor_preserves_values_grid_and_name(self):
        field = Field(self.values, self.grid, "temperature")

        self.assertIs(field.values, self.values)
        self.assertIs(field.discretization, self.grid)
        self.assertEqual(field.name, "temperature")

    def test_constructor_accepts_one_two_and_three_dimensional_shapes(self):
        cases = (
            ([0.0], [1.0], [5], ["chebyshev"]),
            ([0.0, -1.0], [1.0, 1.0], [4, 5], ["fourier", "chebyshev"]),
            (
                [0.0, -1.0, 2.0],
                [1.0, 1.0, 3.0],
                [4, 5, 6],
                ["fourier", "chebyshev", "fourier"],
            ),
        )

        for xmin, xmax, sizes, bases in cases:
            with self.subTest(sizes=sizes):
                grid = SpectralDiscretization(xmin, xmax, sizes, bases)
                values = np.zeros(tuple(sizes))
                field = Field(values, grid, "u")
                self.assertEqual(field.values.shape, tuple(sizes))

    def test_constructor_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, r"expected shape.*24, 17"):
            Field(self.values[:, :-1], self.grid, "temperature")

    def test_constructor_rejects_invalid_grid_and_name(self):
        with self.assertRaises(TypeError):
            Field(self.values, object(), "u")
        with self.assertRaises(TypeError):
            Field(self.values, self.grid, 4)
        for name in ("", "   "):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Field(self.values, self.grid, name)

    def test_derivative_returns_new_field_without_mutating_source(self):
        source = Field(self.values.copy(), self.grid, "u")
        before = source.values.copy()

        result = source.derivative(axis=0)

        expected = 3.0 * np.cos(3.0 * self.x) * (1.0 + self.y**2)
        self.assertIsNot(result, source)
        self.assertIs(result.discretization, self.grid)
        self.assertEqual(result.name, "u")
        np.testing.assert_allclose(result.values, expected, atol=2e-11)
        np.testing.assert_array_equal(source.values, before)

    def test_second_fourier_derivative_handles_nyquist(self):
        grid = SpectralDiscretization(
            [0.0], [2.0 * np.pi], [8], ["fourier"]
        )
        values = np.cos(4.0 * grid.nodes[0])

        result = Field(values, grid, "mode").derivative(0, order=2)

        np.testing.assert_allclose(result.values, -16.0 * values, atol=2e-12)

    def test_invalid_orders_are_rejected(self):
        field = Field(self.values, self.grid, "u")
        for order in (True, 1.5):
            with self.subTest(order=order), self.assertRaises(TypeError):
                field.derivative(0, order)
        for order in (0, -1):
            with self.subTest(order=order), self.assertRaises(ValueError):
                field.derivative(0, order)

    def test_invalid_axes_are_rejected(self):
        field = Field(self.values, self.grid, "u")
        for axis in (-1, 2):
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                field.derivative(axis)
        for axis in (True, 0.5):
            with self.subTest(axis=axis), self.assertRaises(TypeError):
                field.derivative(axis)

    def test_interpolation_returns_field_on_target(self):
        target = SpectralDiscretization(
            [0.0, -1.0],
            [2.0 * np.pi, 1.0],
            [48, 25],
            ["fourier", "chebyshev"],
        )

        result = Field(self.values, self.grid, "u").interpolate(target)

        xt, yt = target.meshgrid()
        np.testing.assert_allclose(
            result.values,
            np.sin(3.0 * xt) * (1.0 + yt**2),
            atol=2e-11,
        )
        self.assertIs(result.discretization, target)
        self.assertEqual(result.name, "u")

    def test_interpolation_rejects_non_discretization_target(self):
        with self.assertRaises(TypeError):
            Field(self.values, self.grid, "u").interpolate(object())

    def test_non_contiguous_complex_values_are_preserved(self):
        values = np.asfortranarray(self.values) * (1.0 + 0.5j)

        field = Field(values, self.grid, "mode")

        self.assertIs(field.values, values)
        self.assertTrue(np.iscomplexobj(field.values))
        np.testing.assert_array_equal(field.values, values)


if __name__ == "__main__":
    unittest.main()
