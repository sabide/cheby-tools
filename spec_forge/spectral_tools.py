
import numpy as np
from numpy import pi
import string


# ============================================================
# Base classes for 1D operators acting along one axis
# ============================================================

class AxisOp1D:
    def __init__(self, axis=0, basis="generic", name="A"):
        self.axis = int(axis)
        self.basis = str(basis)
        self.name = str(name)

    def __matmul__(self, arr):
        return self.apply(arr)

    def apply(self, arr):
        raise NotImplementedError

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"axis={self.axis}, basis={self.basis!r})"
        )


class MatrixOp1D(AxisOp1D):
    """
    Dense linear operator A acting along one axis via np.einsum.
    If A has shape (m,n), the target axis of arr must have length n,
    and the output axis will have length m.
    """
    def __init__(self, A, axis=0, basis="generic", name="A"):
        super().__init__(axis=axis, basis=basis, name=name)
        self.A = np.asarray(A)

        if self.A.ndim != 2:
            raise ValueError("A must be a 2D matrix.")

    def apply(self, arr):
        arr = np.asarray(arr)
        ndim = arr.ndim

        if ndim < 1 or ndim > 3:
            raise ValueError("arr must be 1D, 2D, or 3D.")

        if not (0 <= self.axis < ndim):
            raise ValueError(f"axis={self.axis} incompatible with arr.ndim={ndim}")

        if arr.shape[self.axis] != self.A.shape[1]:
            raise ValueError(
                f"Size mismatch on axis {self.axis}: "
                f"arr.shape[{self.axis}]={arr.shape[self.axis]} != A.shape[1]={self.A.shape[1]}"
            )

        letters = string.ascii_lowercase
        idx_in = list(letters[:ndim])
        j = idx_in[self.axis]
        idx_out = idx_in.copy()
        idx_out[self.axis] = "i"

        subs_A = f"i{j}"
        subs_in = "".join(idx_in)
        subs_out = "".join(idx_out)
        subs = f"{subs_A},{subs_in}->{subs_out}"

        return np.einsum(subs, self.A, arr)


class DiffOp1D(MatrixOp1D):
    pass


class ExpandOp1D(MatrixOp1D):
    pass


class InterpOp1D(MatrixOp1D):
    pass


# ============================================================
# Fourier operators through FFT
# ============================================================

class FourierExpandOp1D(AxisOp1D):
    """
    Nodal -> Fourier coefficients using FFT.
    Convention:
        a_k = fft(f) / N
    """
    def __init__(self, axis=0, name="E"):
        super().__init__(axis=axis, basis="fourier", name=name)

    def apply(self, arr):
        arr = np.asarray(arr)
        N = arr.shape[self.axis]
        return np.fft.fft(arr, axis=self.axis) / N


class FourierInterpOp1D(AxisOp1D):
    """
    Fourier coefficients -> nodal values.
    Inverse convention of FourierExpandOp1D.
    """
    def __init__(self, axis=0, name="I", real_output_if_close=True):
        super().__init__(axis=axis, basis="fourier", name=name)
        self.real_output_if_close = bool(real_output_if_close)

    def apply(self, coeffs):
        coeffs = np.asarray(coeffs)
        N = coeffs.shape[self.axis]
        out = np.fft.ifft(N * coeffs, axis=self.axis)
        return np.real_if_close(out) if self.real_output_if_close else out


class FourierDiffOp1D(AxisOp1D):
    """
    Fourier derivative through FFT.

    The physical domain is [a,b), length L=b-a.
    Wave numbers are built with np.fft.fftfreq.
    """
    def __init__(self, N, a, b, axis=0, order=1, name="d", real_output_if_close=True):
        super().__init__(axis=axis, basis="fourier", name=name)
        self.N = int(N)
        self.a = float(a)
        self.b = float(b)
        self.order = int(order)
        self.real_output_if_close = bool(real_output_if_close)

        L = self.b - self.a
        self.k = 2.0 * np.pi * np.fft.fftfreq(self.N, d=L / self.N)

    def apply(self, arr):
        arr = np.asarray(arr)

        if arr.shape[self.axis] != self.N:
            raise ValueError(
                f"Size mismatch on axis {self.axis}: "
                f"arr.shape[{self.axis}]={arr.shape[self.axis]} != N={self.N}"
            )

        ahat = np.fft.fft(arr, axis=self.axis)

        factor = (1j * self.k) ** self.order
        shape = [1] * arr.ndim
        shape[self.axis] = self.N
        factor = factor.reshape(shape)

        out = np.fft.ifft(factor * ahat, axis=self.axis)
        return np.real_if_close(out) if self.real_output_if_close else out


# ============================================================
# Main discretization class
# ============================================================

class SpectralDiscretization:
    """
    Tensor-product spectral discretization in 1D / 2D / 3D.

    bases[d] must be:
      - 'chebyshev'
      - 'fourier'

    Chebyshev:
      - dense derivative matrix
      - dense nodal <-> coeff transforms
      - Clenshaw-Curtis quadrature weights

    Fourier:
      - derivative through FFT
      - nodal <-> coeff through FFT / IFFT
      - trapezoidal quadrature weights
    """

    def __init__(self, xmin, xmax, n, bases):
        self.dim = len(n)

        if not (len(xmin) == len(xmax) == len(n) == len(bases)):
            raise ValueError("xmin, xmax, n, and bases must have the same length.")

        if self.dim < 1 or self.dim > 3:
            raise ValueError("Only 1D, 2D, and 3D are supported.")

        self.xmin = [float(v) for v in xmin]
        self.xmax = [float(v) for v in xmax]
        self.n = [int(v) for v in n]
        self.bases = [str(b).lower() for b in bases]

        self.L = [float(xmax[d] - xmin[d]) for d in range(self.dim)]

        self.nodes_hat = []
        self.nodes = []

        self.D = []
        self.E = []
        self.I = []
        self.W = []

        for axis in range(self.dim):
            N = self.n[axis]
            a = self.xmin[axis]
            b = self.xmax[axis]
            basis = self.bases[axis]

            if basis == "chebyshev":
                if N < 2:
                    raise ValueError("Chebyshev requires N >= 2.")

                xhat = self.cheb_nodes(N)
                x = self.affine_map_cheb(xhat, a, b)

                Dhat = self.cheb_D(N)
                D = (2.0 / (b - a)) * Dhat
                E = self.build_cheb_transform(N)
                I = self.build_cheb_inverse_transform(N)
                W = 0.5 * (b - a) * self.cheb_quadrature_weights(N)

            elif basis == "fourier":
                if N < 1:
                    raise ValueError("Fourier requires N >= 1.")

                xhat = self.fourier_nodes(N)
                x = self.affine_map_fourier(xhat, a, b)

                D = None
                E = None
                I = None
                W = self.fourier_quadrature_weights(N, a, b)

            else:
                raise ValueError(
                    f"Unknown basis '{basis}' on axis {axis}. "
                    "Use 'chebyshev' or 'fourier'."
                )

            self.nodes_hat.append(xhat)
            self.nodes.append(x)
            self.D.append(D)
            self.E.append(E)
            self.I.append(I)
            self.W.append(W)

        self._build_axis_operators()
        self._build_aliases()

    # --------------------------------------------------------
    # Build operator objects
    # --------------------------------------------------------
    def _build_axis_operators(self):
        self.diff_ops = []
        self.expand_ops = []
        self.interp_ops = []

        for axis in range(self.dim):
            N = self.n[axis]
            a = self.xmin[axis]
            b = self.xmax[axis]
            basis = self.bases[axis]

            if basis == "chebyshev":
                d_op = DiffOp1D(self.D[axis], axis=axis, basis=basis, name=f"d{axis}")
                e_op = ExpandOp1D(self.E[axis], axis=axis, basis=basis, name=f"E{axis}")
                i_op = InterpOp1D(self.I[axis], axis=axis, basis=basis, name=f"I{axis}")

            elif basis == "fourier":
                d_op = FourierDiffOp1D(N, a, b, axis=axis, order=1, name=f"d{axis}")
                e_op = FourierExpandOp1D(axis=axis, name=f"E{axis}")
                i_op = FourierInterpOp1D(axis=axis, name=f"I{axis}")

            self.diff_ops.append(d_op)
            self.expand_ops.append(e_op)
            self.interp_ops.append(i_op)

    def _build_aliases(self):
        self.dx = self.diff_ops[0]
        self.Ex = self.expand_ops[0]
        self.Ix = self.interp_ops[0]
        self.Wx = self.W[0]

        if self.dim >= 2:
            self.dy = self.diff_ops[1]
            self.Ey = self.expand_ops[1]
            self.Iy = self.interp_ops[1]
            self.Wy = self.W[1]

        if self.dim >= 3:
            self.dz = self.diff_ops[2]
            self.Ez = self.expand_ops[2]
            self.Iz = self.interp_ops[2]
            self.Wz = self.W[2]

        self._x_phys_ = self.nodes[0]
        self._x_min_ = self.xmin[0]
        self._x_max_ = self.xmax[0]
        self._nx_ = self.n[0]

        if self.dim >= 2:
            self._y_phys_ = self.nodes[1]
            self._y_min_ = self.xmin[1]
            self._y_max_ = self.xmax[1]
            self._ny_ = self.n[1]

        if self.dim >= 3:
            self._z_phys_ = self.nodes[2]
            self._z_min_ = self.xmin[2]
            self._z_max_ = self.xmax[2]
            self._nz_ = self.n[2]

    # --------------------------------------------------------
    # Nodes and affine maps
    # --------------------------------------------------------
    @staticmethod
    def cheb_nodes(N):
        """
        Chebyshev-Lobatto nodes in [-1,1], increasing order.
        """
        return -np.cos(np.arange(N) * pi / (N - 1))

    @staticmethod
    def fourier_nodes(N):
        """
        Fourier nodes in [0,2π).
        """
        return 2.0 * pi * np.arange(N) / N

    @staticmethod
    def affine_map_cheb(x_hat, a, b):
        return 0.5 * (x_hat + 1.0) * (b - a) + a

    @staticmethod
    def affine_map_fourier(x_hat, a, b):
        return a + (b - a) * x_hat / (2.0 * pi)

    # --------------------------------------------------------
    # Chebyshev differentiation and transforms
    # --------------------------------------------------------
    @staticmethod
    def cheb_D(N):
        """
        Chebyshev-Lobatto differentiation matrix on [-1,1],
        consistent with increasing nodes.
        """
        x = -np.cos(pi * np.arange(N) / (N - 1))
        if (N - 1) % 2 == 0:
            x[(N - 1) // 2] = 0.0

        c = np.ones(N)
        c[0] = 2.0
        c[-1] = 2.0
        c = c * (-1.0) ** np.arange(N)
        c = c.reshape(N, 1)

        X = np.tile(x.reshape(N, 1), (1, N))
        dX = X - X.T

        D = (c @ (1.0 / c).T) / (dX + np.eye(N))
        D = D - np.diag(D.sum(axis=1))
        return D

    @staticmethod
    def cheb_eval_matrix(x_hat, N):
        """
        Evaluation matrix T_k(x_m), shape (M,N).
        """
        x_hat = np.asarray(x_hat)
        theta = np.arccos(np.clip(x_hat, -1.0, 1.0))
        k = np.arange(N)[:, None]
        return np.cos(k * theta).T

    @staticmethod
    def build_cheb_transform(N):
        """
        Nodal -> Chebyshev coefficients on increasing Lobatto nodes.
        """
        j = np.arange(N)
        k = j[:, None]

        C = np.cos(k * j * pi / (N - 1))

        w = np.ones(N)
        w[0] = 0.5
        w[-1] = 0.5

        E = (2.0 / (N - 1)) * C * w
        E[0, :] *= 0.5
        E[-1, :] *= 0.5

        # reverse columns because stored nodes are increasing
        E = E[:, ::-1]
        return E

    @staticmethod
    def build_cheb_inverse_transform(N):
        """
        Chebyshev coefficients -> nodal values on increasing Lobatto nodes.
        """
        x = SpectralDiscretization.cheb_nodes(N)
        return SpectralDiscretization.cheb_eval_matrix(x, N)

    # --------------------------------------------------------
    # Quadrature
    # --------------------------------------------------------
    @staticmethod
    def cheb_quadrature_weights(N):
        """
        Clenshaw-Curtis quadrature weights on [-1,1]
        for Chebyshev-Lobatto nodes in increasing order.
        """
        if N == 1:
            return np.array([2.0], dtype=float)

        n = N - 1
        w = np.zeros(N, dtype=float)

        ii = np.arange(1, N - 1)
        v = np.ones(N - 2, dtype=float)

        if n % 2 == 0:
            w[0] = 1.0 / (n**2 - 1.0)
            w[-1] = w[0]
            for k in range(1, n // 2):
                v -= 2.0 * np.cos(2.0 * k * np.pi * ii / n) / (4.0 * k**2 - 1.0)
            v -= np.cos(n * np.pi * ii / n) / (n**2 - 1.0)
        else:
            w[0] = 1.0 / n**2
            w[-1] = w[0]
            for k in range(1, (n + 1) // 2):
                v -= 2.0 * np.cos(2.0 * k * np.pi * ii / n) / (4.0 * k**2 - 1.0)

        w[ii] = 2.0 * v / n
        return w

    @staticmethod
    def fourier_quadrature_weights(N, a, b):
        return np.full(N, (b - a) / N, dtype=float)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def get_operator(self, axis, kind="diff"):
        if kind == "diff":
            return self.diff_ops[axis]
        if kind == "expand":
            return self.expand_ops[axis]
        if kind == "interp":
            return self.interp_ops[axis]
        raise ValueError(f"Unknown kind={kind}")

    def get_matrix(self, axis, kind="diff"):
        if self.bases[axis] == "fourier":
            raise ValueError("Fourier operators are FFT-based, not dense matrices.")
        if kind == "diff":
            return self.D[axis]
        if kind == "expand":
            return self.E[axis]
        if kind == "interp":
            return self.I[axis]
        raise ValueError(f"Unknown kind={kind}")

    def get_nodes(self, axis):
        return self.nodes[axis]

    def meshgrid(self):
        return np.meshgrid(*self.nodes, indexing="ij")

    # --------------------------------------------------------
    # Differential / transform interface
    # --------------------------------------------------------
    def diff(self, phi, axis):
        return self.diff_ops[axis] @ phi

    def expand(self, phi, axis):
        return self.expand_ops[axis] @ phi

    def interp_axis(self, a, axis):
        return self.interp_ops[axis] @ a

    def spectral_coeffs(self, phi):
        out = phi
        for axis in range(self.dim):
            out = self.expand_ops[axis] @ out
        return out

    def nodal_values(self, coeffs):
        out = coeffs
        for axis in range(self.dim):
            out = self.interp_ops[axis] @ out
        return out

    # --------------------------------------------------------
    # Quadrature / norms
    # --------------------------------------------------------
    def integrate(self, f, axis=None):
        """
        Tensor-product quadrature.
        """
        out = np.asarray(f)

        if out.ndim < 1 or out.ndim > 3:
            raise ValueError("f must be 1D, 2D, or 3D.")

        if out.ndim != self.dim:
            raise ValueError(
                f"Input has ndim={out.ndim}, but discretization dim={self.dim}."
            )

        if axis is None:
            axes = tuple(range(self.dim))
        elif np.isscalar(axis):
            axes = (int(axis),)
        else:
            axes = tuple(int(a) for a in axis)

        for ax in axes:
            if not (0 <= ax < self.dim):
                raise ValueError(f"Invalid axis {ax} for dim={self.dim}")

        seen = set()
        axes = tuple(ax for ax in axes if not (ax in seen or seen.add(ax)))

        for ax in sorted(axes, reverse=True):
            w = self.W[ax]

            if out.shape[ax] != len(w):
                raise ValueError(
                    f"Size mismatch on axis {ax}: "
                    f"out.shape[{ax}]={out.shape[ax]} != len(w)={len(w)}"
                )

            shape = [1] * out.ndim
            shape[ax] = len(w)
            wv = w.reshape(shape)

            out = np.sum(wv * out, axis=ax)

        return out

    def integrate_x(self, f):
        return self.integrate(f, axis=0)

    def integrate_y(self, f):
        if self.dim < 2:
            raise ValueError("No y-axis in 1D.")
        return self.integrate(f, axis=1)

    def integrate_z(self, f):
        if self.dim < 3:
            raise ValueError("No z-axis in dimension < 3.")
        return self.integrate(f, axis=2)

    def inner(self, f, g, axis=None):
        return self.integrate(np.conjugate(f) * g, axis=axis)

    def norm2(self, f):
        return np.sqrt(np.real(self.inner(f, f)))

    # --------------------------------------------------------
    # Interpolation from full spectral coefficients
    # --------------------------------------------------------
    @staticmethod
    def fourier_eval_matrix(x_hat, N):
        """
        Evaluation matrix for Fourier coefficients in FFT ordering:
            M[m,k] = exp(i * k_fft[k] * x_hat[m])
        """
        x_hat = np.asarray(x_hat)
        kk = 2.0 * np.pi * np.fft.fftfreq(N, d=(2.0 * np.pi) / N)
        return np.exp(1j * np.outer(x_hat, kk))

    def interpolate(self, coeffs, *x_targets):
        """
        Evaluate full spectral coefficients on arbitrary tensor-product target grids.
        """
        if len(x_targets) != self.dim:
            raise ValueError("Provide one target array per direction.")

        out = coeffs

        for axis in range(self.dim):
            basis = self.bases[axis]
            xt = np.asarray(x_targets[axis])
            a = self.xmin[axis]
            b = self.xmax[axis]
            N = self.n[axis]

            if basis == "chebyshev":
                xhat = 2.0 * (xt - a) / (b - a) - 1.0
                M = self.cheb_eval_matrix(xhat, N)

            elif basis == "fourier":
                xhat = 2.0 * np.pi * (xt - a) / (b - a)
                M = self.fourier_eval_matrix(xhat, N)

            else:
                raise ValueError(f"Unknown basis {basis!r} on axis {axis}")

            op = InterpOp1D(M, axis=axis, basis=basis, name=f"Ieval_{axis}")
            out = op @ out

        return out

    # --------------------------------------------------------
    # Convenient aliases
    # --------------------------------------------------------
    def ddx(self, phi):
        return self.dx @ phi

    def ddy(self, phi):
        if self.dim < 2:
            raise ValueError("No y-axis in 1D.")
        return self.dy @ phi

    def ddz(self, phi):
        if self.dim < 3:
            raise ValueError("No z-axis in dimension < 3.")
        return self.dz @ phi

    def make_interpolator(self, ops_dst):
        return SpectralInterpolate(self, ops_dst)


# ============================================================
# Interpolation operators between two discretizations
# ============================================================

class InterpBetween1D:
    def __init__(self, axis=0, basis="generic", name="T"):
        self.axis = int(axis)
        self.basis = str(basis)
        self.name = str(name)

    def __matmul__(self, arr):
        return self.apply(arr)

    def apply(self, arr):
        raise NotImplementedError

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"axis={self.axis}, basis={self.basis!r})"
        )


class MatrixInterpBetween1D(InterpBetween1D):
    """
    Dense interpolation operator acting along one axis with np.einsum.

    If T has shape (m,n), and arr.shape[axis] = n,
    then output has shape with axis-length m.
    """
    def __init__(self, T, axis=0, basis="generic", name="T"):
        super().__init__(axis=axis, basis=basis, name=name)
        self.T = np.asarray(T)

        if self.T.ndim != 2:
            raise ValueError("T must be a 2D matrix.")

    def apply(self, arr):
        arr = np.asarray(arr)
        ndim = arr.ndim

        if ndim < 1 or ndim > 3:
            raise ValueError("arr must be 1D, 2D, or 3D.")

        if not (0 <= self.axis < ndim):
            raise ValueError(f"axis={self.axis} incompatible with arr.ndim={ndim}")

        if arr.shape[self.axis] != self.T.shape[1]:
            raise ValueError(
                f"Size mismatch on axis {self.axis}: "
                f"arr.shape[{self.axis}]={arr.shape[self.axis]} != T.shape[1]={self.T.shape[1]}"
            )

        letters = string.ascii_lowercase
        idx_in = list(letters[:ndim])
        j = idx_in[self.axis]
        idx_out = idx_in.copy()
        idx_out[self.axis] = "i"

        subs_T = f"i{j}"
        subs_in = "".join(idx_in)
        subs_out = "".join(idx_out)
        subs = f"{subs_T},{subs_in}->{subs_out}"

        return np.einsum(subs, self.T, arr)


class FourierInterpBetween1D(InterpBetween1D):
    """
    Uniform periodic grid -> uniform periodic grid interpolation on [a,b)
    using FFT + zero-padding / truncation.

    Input  : nodal values on Nsrc Fourier grid
    Output : nodal values on Ndst Fourier grid
    """
    def __init__(self, Nsrc, Ndst, a, b, axis=0, name="T", real_output_if_close=True):
        super().__init__(axis=axis, basis="fourier", name=name)
        self.Nsrc = int(Nsrc)
        self.Ndst = int(Ndst)
        self.a = float(a)
        self.b = float(b)
        self.real_output_if_close = bool(real_output_if_close)

        if self.Nsrc < 1 or self.Ndst < 1:
            raise ValueError("Fourier interpolation requires Nsrc >= 1 and Ndst >= 1.")

    def apply(self, arr):
        arr = np.asarray(arr)

        if arr.ndim < 1 or arr.ndim > 3:
            raise ValueError("arr must be 1D, 2D, or 3D.")

        if not (0 <= self.axis < arr.ndim):
            raise ValueError(f"axis={self.axis} incompatible with arr.ndim={arr.ndim}")

        if arr.shape[self.axis] != self.Nsrc:
            raise ValueError(
                f"Size mismatch on axis {self.axis}: "
                f"arr.shape[{self.axis}]={arr.shape[self.axis]} != Nsrc={self.Nsrc}"
            )

        ahat = np.fft.fft(arr, axis=self.axis) / self.Nsrc
        ahat = np.moveaxis(ahat, self.axis, 0)

        out_hat = np.zeros((self.Ndst,) + ahat.shape[1:], dtype=complex)

        ncopy = min(self.Nsrc, self.Ndst)

        if ncopy % 2 == 0:
            nh = ncopy // 2 + 1
        else:
            nh = (ncopy + 1) // 2
        nt = ncopy - nh

        out_hat[:nh, ...] = ahat[:nh, ...]
        if nt > 0:
            out_hat[-nt:, ...] = ahat[-nt:, ...]

        out = np.fft.ifft(self.Ndst * out_hat, axis=0)
        out = np.moveaxis(out, 0, self.axis)

        return np.real_if_close(out) if self.real_output_if_close else out


class SpectralInterpolate:
    """
    Reusable spectral interpolation operator from ops_src grid to ops_dst grid.

    Requirements:
      - same dimension
      - same basis type on each axis
      - same physical interval on each axis

    Strategy:
      - Chebyshev: dense nodal->nodal interpolation matrix:
            T = M_eval(dst_nodes) @ E_src
        applied with np.einsum
      - Fourier: FFT + zero-padding / truncation
    """
    def __init__(self, ops_src, ops_dst):
        self.src = ops_src
        self.dst = ops_dst
        self.dim = ops_src.dim

        if ops_src.dim != ops_dst.dim:
            raise ValueError("ops_src and ops_dst must have same dimension.")

        self.ops = []

        for axis in range(self.dim):
            basis_src = ops_src.bases[axis]
            basis_dst = ops_dst.bases[axis]

            if basis_src != basis_dst:
                raise ValueError(
                    f"Basis mismatch on axis {axis}: "
                    f"{basis_src!r} != {basis_dst!r}"
                )

            a_src = ops_src.xmin[axis]
            b_src = ops_src.xmax[axis]
            a_dst = ops_dst.xmin[axis]
            b_dst = ops_dst.xmax[axis]

            if abs(a_src - a_dst) > 1e-14 or abs(b_src - b_dst) > 1e-14:
                raise ValueError(
                    f"Domain mismatch on axis {axis}: "
                    f"[{a_src},{b_src}] != [{a_dst},{b_dst}]"
                )

            Nsrc = ops_src.n[axis]
            Ndst = ops_dst.n[axis]

            if basis_src == "chebyshev":
                xdst = ops_dst.nodes[axis]
                xhat_dst = 2.0 * (xdst - a_src) / (b_src - a_src) - 1.0

                M_eval = ops_src.cheb_eval_matrix(xhat_dst, Nsrc)
                E_src = ops_src.E[axis]
                T = M_eval @ E_src

                op = MatrixInterpBetween1D(
                    T,
                    axis=axis,
                    basis="chebyshev",
                    name=f"T{axis}"
                )

            elif basis_src == "fourier":
                op = FourierInterpBetween1D(
                    Nsrc=Nsrc,
                    Ndst=Ndst,
                    a=a_src,
                    b=b_src,
                    axis=axis,
                    name=f"T{axis}"
                )

            else:
                raise ValueError(f"Unknown basis {basis_src!r} on axis {axis}")

            self.ops.append(op)

    def apply(self, arr):
        out = np.asarray(arr)

        if out.ndim != self.dim:
            raise ValueError(
                f"Input has ndim={out.ndim}, but interpolator dim={self.dim}."
            )

        for axis in range(self.dim):
            if out.shape[axis] != self.src.n[axis]:
                raise ValueError(
                    f"Input shape mismatch on axis {axis}: "
                    f"got {out.shape[axis]}, expected {self.src.n[axis]}"
                )

        for op in self.ops:
            out = op @ out

        return out

    def __matmul__(self, arr):
        return self.apply(arr)


# ============================================================
# Tests
# ============================================================

def _test_1d_cheb_diff():
    print("=== test 1D Chebyshev derivative ===")
    ops = SpectralDiscretization(
        xmin=[-1.0],
        xmax=[1.0],
        n=[48],
        bases=["chebyshev"]
    )

    x = ops.nodes[0]
    phi = np.sin(pi * x)
    dphi_true = pi * np.cos(pi * x)
    dphi_num = ops.dx @ phi

    err = np.max(np.abs(dphi_num - dphi_true))
    print(f"[1D cheb dx] max error = {err:.3e}")
    assert err < 1e-8


def _test_1d_fourier_diff():
    print("=== test 1D Fourier derivative ===")
    ops = SpectralDiscretization(
        xmin=[0.0],
        xmax=[2*pi],
        n=[64],
        bases=["fourier"]
    )

    x = ops.nodes[0]
    phi = np.sin(3.0 * x) + 0.4 * np.cos(5.0 * x)
    dphi_true = 3.0 * np.cos(3.0 * x) - 2.0 * np.sin(5.0 * x)
    dphi_num = ops.dx @ phi

    err = np.max(np.abs(dphi_num - dphi_true))
    print(f"[1D fourier dx] max error = {err:.3e}")
    assert err < 1e-12


def _test_1d_cheb_expand_interp():
    print("=== test 1D Chebyshev Ex/Ix ===")
    ops = SpectralDiscretization(
        xmin=[-1.0],
        xmax=[1.0],
        n=[32],
        bases=["chebyshev"]
    )

    x = ops.nodes[0]
    phi = x**4 - 0.2*x + 1.0

    a = ops.Ex @ phi
    phi_back = ops.Ix @ a

    err = np.max(np.abs(phi_back - phi))
    print(f"[1D cheb Ex/Ix] max error = {err:.3e}")
    assert err < 1e-12


def _test_1d_fourier_expand_interp():
    print("=== test 1D Fourier Ex/Ix ===")
    ops = SpectralDiscretization(
        xmin=[0.0],
        xmax=[2*pi],
        n=[32],
        bases=["fourier"]
    )

    x = ops.nodes[0]
    phi = np.sin(2*x) + 0.3*np.cos(3*x)

    a = ops.Ex @ phi
    phi_back = ops.Ix @ a

    err = np.max(np.abs(phi_back - phi))
    print(f"[1D fourier Ex/Ix] max error = {err:.3e}")
    assert err < 1e-12


def _test_1d_quadrature():
    print("=== test quadrature ===")

    ops_c = SpectralDiscretization(
        xmin=[-1.0],
        xmax=[1.0],
        n=[65],
        bases=["chebyshev"]
    )
    x = ops_c.nodes[0]
    val = ops_c.integrate(x**4)
    true = 2.0 / 5.0
    err = abs(val - true)
    print(f"[1D cheb quad] error = {err:.3e}")
    assert err < 1e-12

    ops_f = SpectralDiscretization(
        xmin=[0.0],
        xmax=[2*pi],
        n=[64],
        bases=["fourier"]
    )
    x = ops_f.nodes[0]
    val = ops_f.integrate(np.sin(3*x) + 2.0)
    true = 4.0 * pi
    err = abs(val - true)
    print(f"[1D fourier quad] error = {err:.3e}")
    assert err < 1e-12


def _test_2d_fourier_cheb():
    print("=== test 2D Fourier-Chebyshev ===")
    ops = SpectralDiscretization(
        xmin=[0.0, -1.0],
        xmax=[2*pi, 1.0],
        n=[64, 41],
        bases=["fourier", "chebyshev"]
    )

    x, y = ops.nodes
    X, Y = np.meshgrid(x, y, indexing="ij")

    phi = np.sin(2.0 * X) * np.cos(pi * Y)

    dx_true = 2.0 * np.cos(2.0 * X) * np.cos(pi * Y)
    dy_true = -pi * np.sin(2.0 * X) * np.sin(pi * Y)

    dx_num = ops.dx @ phi
    dy_num = ops.dy @ phi

    err_dx = np.max(np.abs(dx_num - dx_true))
    err_dy = np.max(np.abs(dy_num - dy_true))

    print(f"[2D dx] max error = {err_dx:.3e}")
    print(f"[2D dy] max error = {err_dy:.3e}")

    assert err_dx < 1e-12
    assert err_dy < 1e-8

    a = ops.Ey @ (ops.Ex @ phi)
    phi_back = ops.Ix @ (ops.Iy @ a)

    err_rec = np.max(np.abs(phi_back - phi))
    print(f"[2D Ex/Ey + Ix/Iy] max error = {err_rec:.3e}")
    assert err_rec < 1e-11

    integ = ops.integrate(np.ones_like(phi))
    true = (2*pi - 0.0) * (1.0 - (-1.0))
    err_int = abs(integ - true)
    print(f"[2D integral] error = {err_int:.3e}")
    assert err_int < 1e-11


def _test_3d_fourier_cheb_fourier():
    print("=== test 3D Fourier-Chebyshev-Fourier ===")
    ops = SpectralDiscretization(
        xmin=[0.0, -1.0, 0.0],
        xmax=[2*pi, 1.0, 2*pi],
        n=[24, 25, 28],
        bases=["fourier", "chebyshev", "fourier"]
    )

    x, y, z = ops.nodes
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    phi = np.sin(X) * np.cos(pi * Y) * np.sin(2.0 * Z)

    dx_true = np.cos(X) * np.cos(pi * Y) * np.sin(2.0 * Z)
    dy_true = -pi * np.sin(X) * np.sin(pi * Y) * np.sin(2.0 * Z)
    dz_true = 2.0 * np.sin(X) * np.cos(pi * Y) * np.cos(2.0 * Z)

    dx_num = ops.dx @ phi
    dy_num = ops.dy @ phi
    dz_num = ops.dz @ phi

    err_dx = np.max(np.abs(dx_num - dx_true))
    err_dy = np.max(np.abs(dy_num - dy_true))
    err_dz = np.max(np.abs(dz_num - dz_true))

    print(f"[3D dx] max error = {err_dx:.3e}")
    print(f"[3D dy] max error = {err_dy:.3e}")
    print(f"[3D dz] max error = {err_dz:.3e}")

    assert err_dx < 1e-12
    assert err_dy < 1e-8
    assert err_dz < 1e-12


def _test_3d_partial_integrals():
    print("=== test 3D partial integrals ===")

    ops = SpectralDiscretization(
        xmin=[0.0, -1.0, 0.0],
        xmax=[2*pi, 1.0, 2*pi],
        n=[48, 49, 50],
        bases=["fourier", "chebyshev", "fourier"]
    )

    x, y, z = ops.nodes
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    f = (np.sin(2.0 * X) + 2.0) * (Y**2 + 1.0) * (np.cos(3.0 * Z) + 3.0)

    Ix_num = ops.integrate(f, axis=0)
    Iy_num = ops.integrate(f, axis=1)
    Iz_num = ops.integrate(f, axis=2)

    Ix_true = (4.0 * pi) * (Y[0, :, :]**2 + 1.0) * (np.cos(3.0 * Z[0, :, :]) + 3.0)
    Iy_true = (8.0 / 3.0) * (np.sin(2.0 * X[:, 0, :]) + 2.0) * (np.cos(3.0 * Z[:, 0, :]) + 3.0)
    Iz_true = (6.0 * pi) * (np.sin(2.0 * X[:, :, 0]) + 2.0) * (Y[:, :, 0]**2 + 1.0)

    err_Ix = np.max(np.abs(Ix_num - Ix_true))
    err_Iy = np.max(np.abs(Iy_num - Iy_true))
    err_Iz = np.max(np.abs(Iz_num - Iz_true))

    print(f"[3D -> 2D, int over x] max error = {err_Ix:.3e}")
    print(f"[3D -> 2D, int over y] max error = {err_Iy:.3e}")
    print(f"[3D -> 2D, int over z] max error = {err_Iz:.3e}")

    assert err_Ix < 1e-11
    assert err_Iy < 1e-11
    assert err_Iz < 1e-11

    Ixy_num = ops.integrate(f, axis=(0, 1))
    Ixz_num = ops.integrate(f, axis=(0, 2))
    Iyz_num = ops.integrate(f, axis=(1, 2))

    Ixy_true = (32.0 * pi / 3.0) * (np.cos(3.0 * z) + 3.0)
    Ixz_true = (24.0 * pi**2) * (y**2 + 1.0)
    Iyz_true = (16.0 * pi) * (np.sin(2.0 * x) + 2.0)

    err_Ixy = np.max(np.abs(Ixy_num - Ixy_true))
    err_Ixz = np.max(np.abs(Ixz_num - Ixz_true))
    err_Iyz = np.max(np.abs(Iyz_num - Iyz_true))

    print(f"[3D -> 1D, int over x,y] max error = {err_Ixy:.3e}")
    print(f"[3D -> 1D, int over x,z] max error = {err_Ixz:.3e}")
    print(f"[3D -> 1D, int over y,z] max error = {err_Iyz:.3e}")

    assert err_Ixy < 1e-11
    assert err_Ixz < 1e-11
    assert err_Iyz < 1e-11

    Ixyz_num = ops.integrate(f)
    Ixyz_true = 64.0 * pi**2

    err_Ixyz = abs(Ixyz_num - Ixyz_true)

    print(f"[3D -> scalar] error = {err_Ixyz:.3e}")
    assert err_Ixyz < 1e-11


def _demo_usage():
    print("=== demo usage ===")
    ops = SpectralDiscretization(
        xmin=[0.0, -1.0, 0.0],
        xmax=[2*pi, 1.0, 2*pi],
        n=[16, 17, 20],
        bases=["fourier", "chebyshev", "fourier"]
    )

    x, y, z = ops.nodes
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    phi = np.sin(X) * np.cos(pi * Y) * np.sin(Z)

    dx_phi = ops.dx @ phi
    dy_phi = ops.dy @ phi
    dz_phi = ops.dz @ phi

    a = ops.Ez @ (ops.Ey @ (ops.Ex @ phi))
    phi_back = ops.Ix @ (ops.Iy @ (ops.Iz @ a))

    print("phi.shape      =", phi.shape)
    print("dx_phi.shape   =", dx_phi.shape)
    print("dy_phi.shape   =", dy_phi.shape)
    print("dz_phi.shape   =", dz_phi.shape)
    print("a.shape        =", a.shape)
    print("phi_back.shape =", phi_back.shape)
    print("reconstruction error =", np.max(np.abs(phi_back - phi)))
    print("integral of |phi|^2  =", ops.inner(phi, phi))
    print("L2 norm             =", ops.norm2(phi))


# ============================================================
# Interpolation tests
# ============================================================

def _test_interp_1d_cheb():
    print("=== test interpolation 1D Chebyshev ===")

    ops_src = SpectralDiscretization(
        xmin=[-1.0],
        xmax=[1.0],
        n=[33],
        bases=["chebyshev"]
    )

    ops_dst = SpectralDiscretization(
        xmin=[-1.0],
        xmax=[1.0],
        n=[57],
        bases=["chebyshev"]
    )

    x_src = ops_src.nodes[0]
    x_dst = ops_dst.nodes[0]

    phi_src = np.cos(3.0 * x_src) + x_src**4 - 0.2 * x_src
    phi_true = np.cos(3.0 * x_dst) + x_dst**4 - 0.2 * x_dst

    interp = SpectralInterpolate(ops_src, ops_dst)
    phi_dst = interp @ phi_src

    err = np.max(np.abs(phi_dst - phi_true))
    print(f"[1D cheb src->dst] max error = {err:.3e}")
    assert err < 1e-10


def _test_interp_1d_fourier():
    print("=== test interpolation 1D Fourier ===")

    ops_src = SpectralDiscretization(
        xmin=[0.0],
        xmax=[2*pi],
        n=[48],
        bases=["fourier"]
    )

    ops_dst = SpectralDiscretization(
        xmin=[0.0],
        xmax=[2*pi],
        n=[123],
        bases=["fourier"]
    )

    x_src = ops_src.nodes[0]
    x_dst = ops_dst.nodes[0]

    phi_src = np.sin(3.0 * x_src) + 0.4 * np.cos(5.0 * x_src) - 0.2 * np.sin(7.0 * x_src)
    phi_true = np.sin(3.0 * x_dst) + 0.4 * np.cos(5.0 * x_dst) - 0.2 * np.sin(7.0 * x_dst)

    interp = SpectralInterpolate(ops_src, ops_dst)
    phi_dst = interp @ phi_src

    err = np.max(np.abs(phi_dst - phi_true))
    print(f"[1D fourier src->dst] max error = {err:.3e}")
    assert err < 1e-12


def _test_interp_2d_fourier_cheb():
    print("=== test interpolation 2D Fourier-Chebyshev ===")

    ops_src = SpectralDiscretization(
        xmin=[0.0, -1.0],
        xmax=[2*pi, 1.0],
        n=[48, 41],
        bases=["fourier", "chebyshev"]
    )

    ops_dst = SpectralDiscretization(
        xmin=[0.0, -1.0],
        xmax=[2*pi, 1.0],
        n=[123, 32],
        bases=["fourier", "chebyshev"]
    )

    x_src, y_src = ops_src.nodes
    x_dst, y_dst = ops_dst.nodes

    Xs, Ys = np.meshgrid(x_src, y_src, indexing="ij")
    Xd, Yd = np.meshgrid(x_dst, y_dst, indexing="ij")

    phi_src = (
        np.sin(2.0 * Xs) * np.cos(pi * Ys)
        + 0.3 * np.cos(5.0 * Xs) * (Ys**2 + 1.0)
    )

    phi_true = (
        np.sin(2.0 * Xd) * np.cos(pi * Yd)
        + 0.3 * np.cos(5.0 * Xd) * (Yd**2 + 1.0)
    )

    interp = SpectralInterpolate(ops_src, ops_dst)
    phi_dst = interp @ phi_src

    err = np.max(np.abs(phi_dst - phi_true))
    print(f"[2D fourier-cheb src->dst] max error = {err:.3e}")
    assert err < 1e-10


def _test_interp_3d_fourier_cheb_fourier():
    print("=== test interpolation 3D Fourier-Chebyshev-Fourier ===")

    ops_src = SpectralDiscretization(
        xmin=[0.0, -1.0, 0.0],
        xmax=[2*pi, 1.0, 2*pi],
        n=[48, 49, 50],
        bases=["fourier", "chebyshev", "fourier"]
    )

    ops_dst = SpectralDiscretization(
        xmin=[0.0, -1.0, 0.0],
        xmax=[2*pi, 1.0, 2*pi],
        n=[123, 32, 256],
        bases=["fourier", "chebyshev", "fourier"]
    )

    x_src, y_src, z_src = ops_src.nodes
    x_dst, y_dst, z_dst = ops_dst.nodes

    Xs, Ys, Zs = np.meshgrid(x_src, y_src, z_src, indexing="ij")
    Xd, Yd, Zd = np.meshgrid(x_dst, y_dst, z_dst, indexing="ij")

    phi_src = (
        np.sin(2.0 * Xs) * np.cos(pi * Ys) * np.sin(3.0 * Zs)
        + 0.2 * np.cos(5.0 * Xs) * (Ys**2 + 1.0) * np.cos(4.0 * Zs)
        - 0.1 * np.sin(7.0 * Xs) * Ys * np.sin(2.0 * Zs)
    )

    phi_true = (
        np.sin(2.0 * Xd) * np.cos(pi * Yd) * np.sin(3.0 * Zd)
        + 0.2 * np.cos(5.0 * Xd) * (Yd**2 + 1.0) * np.cos(4.0 * Zd)
        - 0.1 * np.sin(7.0 * Xd) * Yd * np.sin(2.0 * Zd)
    )

    interp = SpectralInterpolate(ops_src, ops_dst)
    phi_dst = interp @ phi_src

    err = np.max(np.abs(phi_dst - phi_true))
    print(f"[3D fourier-cheb-fourier src->dst] max error = {err:.3e}")
    assert err < 1e-10


def _test_interp_3d_roundtrip():
    print("=== test interpolation 3D roundtrip ===")

    ops_coarse = SpectralDiscretization(
        xmin=[0.0, -1.0, 0.0],
        xmax=[2*pi, 1.0, 2*pi],
        n=[32, 33, 36],
        bases=["fourier", "chebyshev", "fourier"]
    )

    ops_fine = SpectralDiscretization(
        xmin=[0.0, -1.0, 0.0],
        xmax=[2*pi, 1.0, 2*pi],
        n=[96, 49, 120],
        bases=["fourier", "chebyshev", "fourier"]
    )

    x, y, z = ops_coarse.nodes
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    phi = (
        np.sin(2.0 * X) * np.cos(pi * Y) * np.sin(3.0 * Z)
        + 0.3 * np.cos(4.0 * X) * (Y**2 + 1.0) * np.cos(2.0 * Z)
    )

    interp_up = SpectralInterpolate(ops_coarse, ops_fine)
    interp_dn = SpectralInterpolate(ops_fine, ops_coarse)

    phi_fine = interp_up @ phi
    phi_back = interp_dn @ phi_fine

    err = np.max(np.abs(phi_back - phi))
    print(f"[3D roundtrip coarse->fine->coarse] max error = {err:.3e}")
    assert err < 1e-10


if __name__ == "__main__":
    _test_1d_cheb_diff()
    _test_1d_fourier_diff()
    _test_1d_cheb_expand_interp()
    _test_1d_fourier_expand_interp()
    _test_1d_quadrature()
    _test_2d_fourier_cheb()
    _test_3d_fourier_cheb_fourier()
    _test_3d_partial_integrals()

    _test_interp_1d_cheb()
    _test_interp_1d_fourier()
    _test_interp_2d_fourier_cheb()
    _test_interp_3d_fourier_cheb_fourier()
    _test_interp_3d_roundtrip()

    _demo_usage()
    print("✓ All tests passed.")
