"""Compatibility facade for the historical :mod:`discr` API.

New code should use :class:`spec_forge.SpectralDiscretization` directly.
"""

from __future__ import annotations

import warnings

import numpy as np

from spec_forge import SpectralDiscretization


class discr_2d:
    """Deprecated 2D Chebyshev facade backed by ``spec_forge``."""

    def __init__(self, xmin, xmax, n):
        warnings.warn(
            "discr.discr_2d is deprecated; use "
            "spec_forge.SpectralDiscretization instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._ops = SpectralDiscretization(
            xmin=xmin,
            xmax=xmax,
            n=n,
            bases=["chebyshev", "chebyshev"],
        )

        self._x_min_, self._y_min_ = self._ops.xmin
        self._x_max_, self._y_max_ = self._ops.xmax
        self._nx_, self._ny_ = self._ops.n
        self._x_phys_, self._y_phys_ = self._ops.nodes
        self.X, self.Y = self._ops.meshgrid()
        self.Dx, self.Dy = self._ops.D

    cheb_nodes = staticmethod(SpectralDiscretization.cheb_nodes)
    cheb_D = staticmethod(SpectralDiscretization.cheb_D)
    T_eval_matrix = staticmethod(SpectralDiscretization.cheb_eval_matrix)
    build_PSGL_1D = staticmethod(SpectralDiscretization.build_cheb_transform)

    @staticmethod
    def affine_map(x_hat, a, b):
        return SpectralDiscretization.affine_map_cheb(x_hat, a, b)

    def dx(self, f):
        return self._ops.ddx(f)

    def dy(self, f):
        return self._ops.ddy(f)

    def cheb_coeffs_2d(self, fxy):
        return self._ops.spectral_coeffs(fxy)

    def interpolate(self, fxy, x_target, y_target):
        coeffs = self._ops.spectral_coeffs(fxy)
        return self._ops.interpolate(coeffs, x_target, y_target)

    def interpolate_line_x(self, fxy, x0, y_target=None):
        if y_target is None:
            y_target = self._y_phys_
        if not self._x_min_ <= x0 <= self._x_max_:
            raise ValueError(f"x0={x0} is outside [{self._x_min_}, {self._x_max_}]")
        return self.interpolate(fxy, [x0], y_target).reshape(-1)

    def interpolate_line_y(self, fxy, y0, x_target=None):
        if x_target is None:
            x_target = self._x_phys_
        if not self._y_min_ <= y0 <= self._y_max_:
            raise ValueError(f"y0={y0} is outside [{self._y_min_}, {self._y_max_}]")
        return self.interpolate(fxy, x_target, [y0]).reshape(-1)

    def grid(self):
        return self.X, self.Y
