
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

    def nearest_index(self, coords):
        """
        Return the nearest grid indices (i,j,k,...) for a given physical point.
        
        Parameters
        ----------
        coords : sequence of floats
        Physical coordinates [x0], [x0,y0], or [x0,y0,z0]
    
        Returns
        -------
        indices : tuple of ints
        Indices of the closest grid point in each direction
        """
        coords = tuple(coords)
        
        if len(coords) != self.dim:
            raise ValueError(
                f"Expected {self.dim} coordinates, got {len(coords)}."
            )
        
        idx = []
        
        for d in range(self.dim):
            x = self.nodes[d]
            x0 = coords[d]
            
            i = int(np.argmin(np.abs(x - x0)))
            idx.append(i)
            
        return tuple(idx)
        

    
    def interpolate_point(self, phi, *coords):
        """
        Interpolate nodal values at a single point.
        
        Accepts:
        interpolate_point(phi, x0, y0, z0)
        interpolate_point(phi, [x0, y0, z0])
        """

        if len(coords) == 1 and np.iterable(coords[0]):
            coords = tuple(coords[0])
    
        if len(coords) != self.dim:
            raise ValueError(
                f"Expected {self.dim} coordinates, got {len(coords)}."
            )

        phi = np.asarray(phi)
        expected_shape = tuple(self.n)
        if phi.shape != expected_shape:
            raise ValueError(
                f"Expected nodal field shape {expected_shape}, got {phi.shape}."
            )

        coeffs = self.spectral_coeffs(phi)
        targets = [np.array([c], dtype=float) for c in coords]
        out = self.interpolate(coeffs, *targets)
        return np.asarray(out).reshape(-1)[0]


    
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
    using FFT zero-padding / truncation along one axis.

    Input  : nodal values on Nsrc Fourier grid
    Output : nodal values on Ndst Fourier grid

    If input is real, uses rfft/irfft to preserve Hermitian symmetry exactly.
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

        # ----------------------------------------------------
        # Real input: use rfft/irfft to preserve reality exactly
        # ----------------------------------------------------
        if np.isrealobj(arr):
            ahat = np.fft.rfft(arr, axis=self.axis) / self.Nsrc

            ahat = np.moveaxis(ahat, self.axis, 0)   # spectral axis first
            Ksrc = ahat.shape[0]
            Kdst = self.Ndst // 2 + 1

            out_hat = np.zeros((Kdst,) + ahat.shape[1:], dtype=ahat.dtype)

            kcopy = min(Ksrc, Kdst)
            out_hat[:kcopy, ...] = ahat[:kcopy, ...]

            # Nyquist handling when both sizes are even
            # rfft stores Nyquist as the last coefficient, which must remain real.
            if self.Nsrc % 2 == 0 and self.Ndst % 2 == 0 and kcopy == Ksrc == Kdst:
                out_hat[-1, ...] = np.real(out_hat[-1, ...])

            out = np.fft.irfft(self.Ndst * out_hat, n=self.Ndst, axis=0)
            out = np.moveaxis(out, 0, self.axis)
            return out

        # ----------------------------------------------------
        # Complex input: fallback to full fft/ifft
        # ----------------------------------------------------
        ahat = np.fft.fft(arr, axis=self.axis) / self.Nsrc
        ahat = np.moveaxis(ahat, self.axis, 0)

        out_hat = np.zeros((self.Ndst,) + ahat.shape[1:], dtype=ahat.dtype)

        # copy low positive frequencies + low negative frequencies
        if self.Nsrc % 2 == 0:
            kpos_src = self.Nsrc // 2
            out_hat[:kpos_src, ...] = ahat[:kpos_src, ...]
            out_hat[-(self.Nsrc - kpos_src):, ...] = ahat[kpos_src:, ...]
        else:
            kpos_src = (self.Nsrc + 1) // 2
            out_hat[:kpos_src, ...] = ahat[:kpos_src, ...]
            out_hat[-(self.Nsrc - kpos_src):, ...] = ahat[kpos_src:, ...]

        out = np.fft.ifft(self.Ndst * out_hat, axis=0)
        out = np.moveaxis(out, 0, self.axis)

        return np.real_if_close(out) if self.real_output_if_close else out

    
#------------
class _FourierInterpBetween1DLegacy(InterpBetween1D):
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
