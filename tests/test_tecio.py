import unittest
from unittest import mock

import numpy as np

from cheby_tools import Field, SpectralDiscretization
from cheby_tools import tecio


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def write_plt(self, filename, names, arrays):
        self.calls.append((filename, names, arrays))


class TecIOAdapterTests(unittest.TestCase):
    def make_grid(self, xmax=2.0):
        return SpectralDiscretization(
            [0.0, -1.0],
            [xmax, 1.0],
            [4, 3],
            ["fourier", "chebyshev"],
        )

    def setUp(self):
        self.grid = self.make_grid()
        self.x, self.y = self.grid.meshgrid()

    def test_multiple_fields_use_i_fastest_float64_arrays(self):
        u = Field(self.x + self.y, self.grid, "u")
        v = Field(self.x - self.y, self.grid, "v")
        backend = RecordingBackend()

        with mock.patch.object(tecio, "_backend", backend):
            tecio.write_plt("velocity.plt", [u, v])

        filename, names, arrays = backend.calls[0]
        self.assertEqual(filename, "velocity.plt")
        self.assertEqual(names, ["x", "y", "u", "v"])
        self.assertEqual([array.shape for array in arrays], [(3, 4)] * 4)
        for source, converted in zip((self.x, self.y, u.values, v.values), arrays):
            self.assertTrue(converted.flags.c_contiguous)
            self.assertEqual(converted.dtype, np.float64)
            np.testing.assert_array_equal(converted, source.transpose(1, 0))

    def test_single_field_is_accepted(self):
        backend = RecordingBackend()
        field = Field(self.x, self.grid, "u")

        with mock.patch.object(tecio, "_backend", backend):
            tecio.write_plt("field.plt", field)

        self.assertEqual(len(backend.calls), 1)

    def test_field_names_may_contain_spaces_but_not_tecio_separators(self):
        backend = RecordingBackend()
        fields = [
            Field(self.x, self.grid, "velocity magnitude"),
            Field(self.y, self.grid, "pressure, total"),
        ]

        with mock.patch.object(tecio, "_backend", backend):
            tecio.write_plt("field.plt", fields)

        self.assertEqual(
            backend.calls[0][1],
            ["x", "y", "velocity magnitude", "pressure, total"],
        )

        for name in ("a" * 128, "é" * 64):
            with self.subTest(accepted_name=name), mock.patch.object(
                tecio, "_backend", backend
            ):
                tecio.write_plt(
                    "field.plt", Field(self.x, self.grid, name)
                )

        for name in ("line\nbreak", "null\0byte"):
            with self.subTest(name=name), mock.patch.object(
                tecio, "_backend", backend
            ):
                with self.assertRaisesRegex(ValueError, "TecIO") as caught:
                    tecio.write_plt(
                        "field.plt", Field(self.x, self.grid, name)
                    )
                self.assertIn(repr(name), str(caught.exception))

        for name in ("a" * 129, "é" * 65):
            with self.subTest(rejected_name=name), mock.patch.object(
                tecio, "_backend", backend
            ):
                with self.assertRaisesRegex(
                    ValueError, "128 UTF-8 bytes"
                ) as caught:
                    tecio.write_plt(
                        "field.plt", Field(self.x, self.grid, name)
                    )
                self.assertIn(repr(name), str(caught.exception))

    def test_equivalent_distinct_grids_are_accepted(self):
        other = self.make_grid()
        fields = [
            Field(self.x, self.grid, "u"),
            Field(other.meshgrid()[1], other, "v"),
        ]
        backend = RecordingBackend()

        with mock.patch.object(tecio, "_backend", backend):
            tecio.write_plt("fields.plt", fields)

        self.assertEqual(len(backend.calls), 1)

    def test_invalid_field_collections_are_rejected(self):
        values = np.ones(tuple(self.grid.n))
        other = self.make_grid(xmax=3.0)
        cases = (
            [],
            [object()],
            [Field(values, self.grid, "u"), Field(values, self.grid, "u")],
            [Field(values, self.grid, "x")],
            [
                Field(values, self.grid, "u"),
                Field(np.ones(tuple(other.n)), other, "v"),
            ],
            [Field(values.astype(complex), self.grid, "u")],
        )

        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(
                (TypeError, ValueError)
            ):
                tecio.write_plt("fields.plt", fields)

    def test_suffix_is_exactly_lowercase_plt(self):
        field = Field(np.ones(tuple(self.grid.n)), self.grid, "u")
        for path in ("fields.szplt", "fields.PLT", "fields"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                tecio.write_plt(path, field)

    def test_missing_backend_is_actionable(self):
        field = Field(np.ones(tuple(self.grid.n)), self.grid, "u")
        load_error = ImportError("missing TecIO shared library")

        with mock.patch.object(tecio, "_backend", None), mock.patch.object(
            tecio, "_backend_import_error", load_error
        ):
            with self.assertRaisesRegex(ImportError, r"CMake.*TecIO") as caught:
                tecio.write_plt("field.plt", field)

        self.assertIs(caught.exception.__cause__, load_error)

    def test_one_and_three_dimensional_backend_shapes(self):
        cases = (
            (
                SpectralDiscretization([0.0], [1.0], [5], ["fourier"]),
                (5,),
            ),
            (
                SpectralDiscretization(
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    [2, 3, 4],
                    ["fourier", "fourier", "fourier"],
                ),
                (4, 3, 2),
            ),
        )

        for grid, expected_shape in cases:
            with self.subTest(dim=grid.dim):
                backend = RecordingBackend()
                field = Field(np.ones(tuple(grid.n)), grid, "u")
                with mock.patch.object(tecio, "_backend", backend):
                    tecio.write_plt("field.plt", field)
                arrays = backend.calls[0][2]
                self.assertTrue(all(a.shape == expected_shape for a in arrays))

    def test_noncontiguous_float32_source_is_not_modified(self):
        values = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
        self.assertFalse(values.flags.c_contiguous)
        field = Field(values, self.grid, "u")
        before = values.copy()
        dtype = values.dtype
        strides = values.strides
        backend = RecordingBackend()

        with mock.patch.object(tecio, "_backend", backend):
            tecio.write_plt("field.plt", field)

        np.testing.assert_array_equal(values, before)
        self.assertEqual(values.dtype, dtype)
        self.assertEqual(values.strides, strides)
        converted = backend.calls[0][2][-1]
        self.assertEqual(converted.dtype, np.float64)
        self.assertTrue(converted.flags.c_contiguous)
        np.testing.assert_array_equal(converted, values.transpose(1, 0))


if __name__ == "__main__":
    unittest.main()
