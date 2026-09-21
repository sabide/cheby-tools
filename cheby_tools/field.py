"""High-level fields defined on spectral discretizations."""

from numbers import Integral

import numpy as np

from .spectral import SpectralDiscretization, SpectralInterpolate


class Field:
    """Nodal values associated with one spectral discretization."""

    def __init__(self, values, discretization, name):
        if not isinstance(discretization, SpectralDiscretization):
            raise TypeError("discretization must be a SpectralDiscretization.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        name = name.strip()
        if not name:
            raise ValueError("name must not be empty.")

        values = np.asarray(values)
        expected = tuple(discretization.n)
        if values.shape != expected:
            raise ValueError(
                f"Field {name!r} expected shape {expected}, got {values.shape}."
            )

        self.values = values
        self.discretization = discretization
        self.name = name

    def derivative(self, axis, order=1):
        """Return a new field differentiated along one grid axis."""
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, Integral):
            raise TypeError("axis must be an integer.")
        if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral):
            raise TypeError("order must be an integer.")

        axis = int(axis)
        order = int(order)
        if not 0 <= axis < self.discretization.dim:
            raise ValueError(
                f"Invalid axis {axis} for dim={self.discretization.dim}."
            )
        if order < 1:
            raise ValueError("order must be at least 1.")

        values = self.discretization.diff(self.values, axis, order=order)
        return type(self)(values, self.discretization, self.name)

    def interpolate(self, target):
        """Return this field interpolated onto ``target``."""
        if not isinstance(target, SpectralDiscretization):
            raise TypeError("target must be a SpectralDiscretization.")
        values = SpectralInterpolate(self.discretization, target) @ self.values
        return type(self)(values, target, self.name)
