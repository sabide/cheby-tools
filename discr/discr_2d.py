# discr_2d.py
import numpy as np
from numpy import pi


class discr_2d:
    """
    2D Chebyshev-Lobatto discretization:
    - interpolation on arbitrary target grids
    - differentiation in x and y
    Grid shape: (nx, ny) with indexing='ij' (axis 0 -> x, axis 1 -> y)
    """

    def __init__(self, xmin, xmax, n):
        """
        Parameters
        ----------
        xmin : list or tuple of float
            [x_min, y_min] domain minimum coordinates.
        xmax : list or tuple of float
            [x_max, y_max] domain maximum coordinates.
        n : list or tuple of int
            [nx, ny] number of Chebyshev-Lobatto points in each direction.
        """
        self._x_min_, self._y_min_ = float(xmin[0]), float(xmin[1])
        self._x_max_, self._y_max_ = float(xmax[0]), float(xmax[1])
        self._nx_, self._ny_ = int(n[0]), int(n[1])

        # Lobatto nodes in [-1,1]
        self._x_hat_ = self.cheb_nodes(self._nx_)
        self._y_hat_ = self.cheb_nodes(self._ny_)

        # Physical nodes and grid (indexing ij: (nx, ny))
        self._x_phys_ = self.affine_map(self._x_hat_, self._x_min_, self._x_max_)
        self._y_phys_ = self.affine_map(self._y_hat_, self._y_min_, self._y_max_)
        self.X, self.Y = np.meshgrid(self._x_phys_, self._y_phys_, indexing='ij')  # (nx, ny)

        # Differentiation matrices in [-1,1]
        Dx_hat = self.cheb_D(self._nx_)
        Dy_hat = self.cheb_D(self._ny_)

        # Scale to physical domain
        self.Dx = (2.0 / (self._x_max_ - self._x_min_)) * Dx_hat  # (nx, nx)
        self.Dy = (2.0 / (self._y_max_ - self._y_min_)) * Dy_hat  # (ny, ny)

        # PSGL (nodal -> coeffs) 1D with 1/(c[i]*c[j]) matrix factor
        self._PSGL_X_ = self.build_PSGL_1D(self._nx_)
        self._PSGL_Y_ = self.build_PSGL_1D(self._ny_)

    # ---------- utilities ----------
    @staticmethod
    def cheb_nodes(N):
        """Chebyshev-Lobatto nodes in [-1,1]."""
        return np.cos(np.arange(N) * pi / (N - 1))

    @staticmethod
    def affine_map(x_hat, a, b):
        """Affine map from [-1,1] to [a,b]."""
        return 0.5 * (x_hat + 1.0) * (b - a) + a

    @staticmethod
    def cheb_D(N):
        """Trefethen's Chebyshev differentiation matrix on Lobatto nodes in [-1,1]."""
        x = np.cos(pi * np.arange(N) / (N - 1))
        if (N - 1) % 2 == 0:
            x[(N - 1)//2] = 0.0
        c = np.ones(N); c[0] = 2.0; c[-1] = 2.0
        c = c * (-1.0) ** np.arange(N)
        c = c.reshape(N, 1)
        X = np.tile(x.reshape(N, 1), (1, N))
        dX = X - X.T
        D = (c @ (1.0 / c).T) / (dX + np.eye(N))
        D = D - np.diag(D.sum(axis=1))
        return D  # d/dx_hat

    @staticmethod
    def T_eval_matrix(x_hat, N):
        """Matrix T_k(x) = cos(k arccos x) for evaluating Chebyshev series."""
        theta = np.arccos(np.clip(x_hat, -1.0, 1.0))
        k = np.arange(N)[:, None]              # (N,1)
        return np.cos(k * theta).T             # (M,N)

    @staticmethod
    def build_PSGL_1D(N):
        """
        Nodal -> Chebyshev coefficients matrix.
        PSGL[i,j] = (2/(N-1)) * cos(i*j*pi/(N-1)) * 1/(c[i]*c[j]),
        with c[0]=c[N-1]=2, otherwise 1.
        """
        j = np.arange(N)
        i = j[:, None]
        C = np.cos(i * j * np.pi / (N - 1))     # (N,N)
        c = np.ones(N); c[0] = c[-1] = 2.0
        W = 1.0 / np.outer(c, c)                # (N,N)
        P = (2.0 / (N - 1)) * (C * W)
        return P

    # ---------- derivatives ----------
    def dx(self, f):
        """df/dx on the grid (nx, ny)."""
        f = np.asarray(f)
        assert f.shape == (self._nx_, self._ny_), "f must be (nx, ny)"
        return self.Dx @ f

    def dy(self, f):
        """df/dy on the grid (nx, ny)."""
        f = np.asarray(f)
        assert f.shape == (self._nx_, self._ny_), "f must be (nx, ny)"
        return f @ self.Dy.T

    # ---------- nodal<->coeffs & interpolation ----------
    def cheb_coeffs_2d(self, fxy):
        """Return Chebyshev coefficients a_{p,q} from nodal values fxy."""
        fxy = np.asarray(fxy)
        assert fxy.shape == (self._nx_, self._ny_), "fxy must be (nx, ny)"
        A = self._PSGL_X_ @ fxy @ self._PSGL_Y_.T
        return A

    def interpolate(self, fxy, x_target, y_target):
        """
        Interpolate fxy from Chebyshev-Lobatto nodes to target physical coordinates.
        Returns F: (len(x_target), len(y_target))
        """
        A = self.cheb_coeffs_2d(fxy)
        x_target = np.asarray(x_target)
        y_target = np.asarray(y_target)
        xi  = 2*(x_target - self._x_min_)/(self._x_max_ - self._x_min_) - 1
        eta = 2*(y_target - self._y_min_)/(self._y_max_ - self._y_min_) - 1
        Cx = self.T_eval_matrix(xi,  self._nx_)
        Cy = self.T_eval_matrix(eta, self._ny_)
        F = Cx @ A @ Cy.T
        return F
    
    def interpolate_line_x(self, fxy, x0, y_target=None):
        """
            Interpolate f(x,y) along a vertical line x = x0.
        Returns a 1D array with shape (len(y_target),).
        If y_target is None, uses self._y_phys_ (nodal y-line).
        """
        # 1) coefficients from nodal values
        A = self.cheb_coeffs_2d(fxy)
        
        # 2) map x0 and y_target to [-1,1]
        if not (self._x_min_ <= x0 <= self._x_max_):
            raise ValueError(f"x0={x0} is outside [{self._x_min_}, {self._x_max_}]")
        if y_target is None:
            y_target = self._y_phys_

        x0 = float(x0)
        y_target = np.asarray(y_target)
        xi0  = 2*(x0 - self._x_min_)/(self._x_max_ - self._x_min_) - 1.0
        eta  = 2*(y_target - self._y_min_)/(self._y_max_ - self._y_min_) - 1.0

        # 3) evaluation matrices
        Cx0 = self.T_eval_matrix(np.array([xi0]), self._nx_)   # (1, nx)
        Cy  = self.T_eval_matrix(eta, self._ny_)               # (my, ny)

        # 4) line interpolation: shape (1, my) -> ravel
        prof = (Cx0 @ A @ Cy.T).ravel()
        return prof

    def interpolate_line_y(self, fxy, y0, x_target=None):
        """
        Interpolate f(x,y) along a horizontal line y = y0.
        Returns a 1D array with shape (len(x_target),).
        If x_target is None, uses self._x_phys_ (nodal x-line).
        """
        # 1) coefficients from nodal values
        A = self.cheb_coeffs_2d(fxy)

        # 2) map y0 and x_target to [-1,1]
        if not (self._y_min_ <= y0 <= self._y_max_):
            raise ValueError(f"y0={y0} is outside [{self._y_min_}, {self._y_max_}]")
        if x_target is None:
            x_target = self._x_phys_

        y0 = float(y0)
        x_target = np.asarray(x_target)
        eta0 = 2*(y0 - self._y_min_)/(self._y_max_ - self._y_min_) - 1.0
        xi   = 2*(x_target - self._x_min_)/(self._x_max_ - self._x_min_) - 1.0

        # 3) evaluation matrices
        Cx  = self.T_eval_matrix(xi,  self._nx_)               # (mx, nx)
        Cy0 = self.T_eval_matrix(np.array([eta0]), self._ny_)  # (1,  ny)

        # 4) line interpolation: shape (mx, 1) -> ravel
        prof = (Cx @ A @ Cy0.T).ravel()
        return prof

    

    def grid(self):
        """Return physical grids X, Y with shape (nx, ny)."""
        return self.X, self.Y


# =========================
# Self-test
# =========================
def _test_module():
    xmin, xmax = [-1.0, -2.0], [2.0, 3.0]
    n = [48, 64]
    ops = discr_2d(xmin=xmin, xmax=xmax, n=n)

    # analytic function on grid
    X, Y = ops.grid()
    F = np.sin(np.pi * X) * np.cos(3*np.pi * Y)

    # interpolation back on nodes -> identity
    F_on_nodes = ops.interpolate(F, ops._x_phys_, ops._y_phys_)
    err_nodes = np.max(np.abs(F_on_nodes - F))
    print(f"[interp @nodes] max error = {err_nodes:.3e}")

    # interpolation on target grid
    n_dst = [24, 18]
    x_dst = np.linspace(xmin[0], xmax[0], n_dst[0])
    y_dst = np.linspace(xmin[1], xmax[1], n_dst[1])
    F_dst = ops.interpolate(F, x_dst, y_dst)
    Xd, Yd = np.meshgrid(x_dst, y_dst, indexing='ij')
    F_true = np.sin(np.pi * Xd) * np.cos(3*np.pi * Yd)
    err_dst = np.max(np.abs(F_dst - F_true))
    print(f"[interp @target] max error = {err_dst:.3e}")

    # derivatives
    dFx_num = ops.dx(F)
    dFy_num = ops.dy(F)
    dFx_true =    np.pi * np.cos(np.pi * X) * np.cos(3*np.pi * Y)
    dFy_true = -3*np.pi * np.sin(np.pi * X) * np.sin(3*np.pi * Y)
    err_dx = np.max(np.abs(dFx_num - dFx_true))
    err_dy = np.max(np.abs(dFy_num - dFy_true))
    print(f"[dx] max error = {err_dx:.3e}")
    print(f"[dy] max error = {err_dy:.3e}")

    # assertions
    assert err_nodes < 1e-10, "Interpolation on nodes failed"
    assert err_dst   < 1e-8,  "Interpolation on target grid failed"
    assert err_dx    < 1e-8,  "dx derivative failed"
    assert err_dy    < 1e-8,  "dy derivative failed"


    # --- line profiles on nodal lines (identity) ---
    ix = n[0] // 3
    iy = n[1] // 4
    x0_nodal = ops._x_phys_[ix]
    y0_nodal = ops._y_phys_[iy]

    prof_y_num = ops.interpolate_line_x(F, x0_nodal)  # f(x0, y_phys)
    prof_y_ref = np.sin(np.pi * x0_nodal) * np.cos(3*np.pi * ops._y_phys_)
    err_line_x_nodal = np.max(np.abs(prof_y_num - prof_y_ref))

    prof_x_num = ops.interpolate_line_y(F, y0_nodal)               # f(x_phys, y0)
    prof_x_ref = np.sin(np.pi * ops._x_phys_) * np.cos(3*np.pi * y0_nodal)
    err_line_y_nodal = np.max(np.abs(prof_x_num - prof_x_ref))

    print(f"[line x=x0 @nodes] max error = {err_line_x_nodal:.3e}")
    print(f"[line y=y0 @nodes] max error = {err_line_y_nodal:.3e}")
    assert err_line_x_nodal < 1e-10, "Line profile at x=x0 (nodal) failed"
    assert err_line_y_nodal < 1e-10, "Line profile at y=y0 (nodal) failed"

    # --- line profiles on arbitrary lines (non-nodal) ---
    x0 = 0.37*(xmax[0]-xmin[0]) + xmin[0]
    y0 = -0.23*(ymax := xmax[1]) + 1.23*(ymin := xmin[1])  # any linear combo in [ymin,ymax]
    # ensure inside bounds
    y0 = max(min(y0, ymax), ymin)

    y_dst = np.linspace(xmin[1], xmax[1], 81)
    x_dst = np.linspace(xmin[0], xmax[0], 77)

    prof_y_num = ops.interpolate_line_x(F, x0, y_dst)
    prof_y_ref = np.sin(np.pi * x0) * np.cos(3*np.pi * y_dst)
    prof_x_num = ops.interpolate_line_y(F, y0, x_dst)
    prof_x_ref = np.sin(np.pi * x_dst) * np.cos(3*np.pi * y0)

    err_line_x = np.max(np.abs(prof_y_num - prof_y_ref))
    err_line_y = np.max(np.abs(prof_x_num - prof_x_ref))
    print(f"[line x=x0] max error = {err_line_x:.3e}")
    print(f"[line y=y0] max error = {err_line_y:.3e}")

    assert err_line_x < 1e-8, "Line profile at x=x0 failed"
    assert err_line_y < 1e-8, "Line profile at y=y0 failed"

    print("✓ All tests passed.")


if __name__ == "__main__":
    _test_module()
