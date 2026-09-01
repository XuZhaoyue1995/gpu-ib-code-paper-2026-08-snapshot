"""ib_immersed.py — immersed-boundary spread/interp (Uhlmann direct forcing),
single-device (CPU/GPU) vectorized port of pure_v0.1's Eul_2_Lag_new_s /
Lag_2_Eul_new_s_var_exp + Index_cell + delta_h3.

On a single device the whole Eulerian field and all Lagrangian markers live
together, so there is NO rank-0 master/gather/scatter (that bottleneck is a
multi-rank MPI artifact).  IB = a kernel over markers, each touching a 5x5x5
box of Eulerian cells via the structured Index_cell(i,j,k) O(1) lookup.
"""
import numpy as np
from backend import xp, asnumpy, asreal


def delta_h3(r, h):                       # Roma-Peskin 3-pt regularized delta (vectorized)
    a = xp.abs(r / h)
    out = xp.zeros_like(a)
    m1 = a <= 0.5
    out = xp.where(m1, (1.0 + xp.sqrt(xp.clip(1.0 - 3.0 * a * a, 0.0, None))) / (3.0 * h), out)
    m2 = (a > 0.5) & (a <= 1.5)
    out = xp.where(m2, (5.0 - 3.0 * a - xp.sqrt(xp.clip(1.0 - 3.0 * (1.0 - a) ** 2, 0.0, None))) / (6.0 * h), out)
    return out


class ImmersedBoundary:
    """Structured Index_cell lookup + delta spread/interp for a uniform Cartesian
    block.  cpos must be the cell centers of that block."""
    def __init__(self, mesh, hh=None):
        # Use the mesh's FINEST-layer structured lookup (make_euler provides index_cell/
        # pos_ref/dx_uniform for layer 0).  Works for uniform AND nested meshes: the IB
        # body must live in the finest uniform region (its 5^3 stencils stay in fine cells).
        idx = np.asarray(mesh.index_cell)                            # (nx,ny,nz) -> global fine cell id
        self.n = np.asarray(idx.shape)
        pos_ref = np.asarray(mesh.pos_ref)                           # center of fine cell (0,0,0)
        self.dx = float(np.asarray(mesh.dx_uniform)[0])
        self.hh = float(self.dx if hh is None else hh)
        self.ncell = mesh.cells.shape[0]
        self.pos_ref = asreal(pos_ref)              # float32/float64 working dtype
        self.index_cell = xp.asarray(idx)
        self._imp = None                            # implicit coef-matrix cache (prepare_implicit)

    def _cells_weights(self, markers):
        """For each marker return its 5^3 support cell ids (M,125) and delta weights (M,125)."""
        m = xp.asarray(markers)
        l = xp.rint((m - self.pos_ref) / self.dx).astype(xp.int64)   # (M,3) nearest cell 0-based
        offs = xp.arange(-2, 3)
        # build (M,5,5,5) index grids
        di, dj, dk = xp.meshgrid(offs, offs, offs, indexing="ij")
        di = di.ravel(); dj = dj.ravel(); dk = dk.ravel()            # (125,)
        ii = l[:, 0:1] + di[None, :]                                 # (M,125)
        jj = l[:, 1:2] + dj[None, :]
        kk = l[:, 2:3] + dk[None, :]
        nx, ny, nz = int(self.n[0]), int(self.n[1]), int(self.n[2])
        inb = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
        iic = xp.clip(ii, 0, nx - 1); jjc = xp.clip(jj, 0, ny - 1); kkc = xp.clip(kk, 0, nz - 1)
        cell = self.index_cell[iic, jjc, kkc]                        # (M,125)
        valid = inb & (cell >= 0)
        gx = self.pos_ref[0] + ii * self.dx - m[:, 0:1]              # grid point - marker
        gy = self.pos_ref[1] + jj * self.dx - m[:, 1:2]
        gz = self.pos_ref[2] + kk * self.dx - m[:, 2:3]
        w = delta_h3(gx, self.hh) * delta_h3(gy, self.hh) * delta_h3(gz, self.hh)
        w = xp.where(valid, w, 0.0)
        cell = xp.where(valid, cell, 0)
        return cell, w

    def interp(self, vc, markers):
        """Eulerian -> Lagrangian: vel[i] = sum_cells vc[cell]*delta*hh^3."""
        cell, w = self._cells_weights(markers)                       # (M,125)
        w = w * self.hh ** 3
        out = xp.zeros((cell.shape[0], vc.shape[1]), dtype=vc.dtype)
        for c in range(vc.shape[1]):
            out[:, c] = (vc[cell, c] * w).sum(axis=1)
        return out

    def spread(self, flag, markers, ds):
        """Lagrangian -> Eulerian: F[cell] += sum_markers flag[i]*delta*ds^3."""
        cell, w = self._cells_weights(markers)                       # (M,125)
        ds = asreal(ds)
        w = w * (ds ** 3)[:, None]
        out = xp.zeros((self.ncell, flag.shape[1]), dtype=flag.dtype)
        for c in range(flag.shape[1]):
            contrib = (flag[:, c:c + 1] * w).ravel()
            out[:, c] = xp.bincount(cell.ravel(), contrib, minlength=self.ncell).astype(flag.dtype, copy=False)
        return out

    def direct_force(self, vc, markers, vel_desired, dt, ds):
        """Uhlmann direct forcing: f_L = (u_desired - interp(vc))/dt ; F_E = spread(f_L)."""
        vel_ib = self.interp(vc, markers)
        f_lag = (vel_desired - vel_ib) / dt
        return self.spread(f_lag, markers, ds), f_lag

    def prepare_implicit(self, markers, ds, reg=1e-8):
        """Build the (small) marker-coupling coef matrix M = D @ S (Fortran's `coef`)
        ONCE and cache its factorization.  M[i,j] = sum_cell delta_i*delta_j*ds_j^3*hh^3
        = interp(spread(.)) restricted to markers -> nmk x nmk (markers only couple
        within the delta support, so M is small/banded; we store it dense + inverse).
        Markers are static for a fixed body, so this is one-time."""
        import scipy.sparse as sp
        cell, w = self._cells_weights(markers)               # (M,125)
        ds = asreal(ds)
        w_int = w * self.hh ** 3                             # Eul->Lag (interp)
        w_spr = w * (ds ** 3)[:, None]                       # Lag->Eul (spread)
        nmk = int(cell.shape[0])
        cell_np = asnumpy(cell); wint_np = asnumpy(w_int); wspr_np = asnumpy(w_spr)
        rows = np.repeat(np.arange(nmk), cell_np.shape[1])
        cols = cell_np.ravel()
        D = sp.csr_matrix((wint_np.ravel(), (rows, cols)), shape=(nmk, self.ncell))
        S = sp.csr_matrix((wspr_np.ravel(), (cols, rows)), shape=(self.ncell, nmk))
        Mmat = np.asarray((D @ S).todense())                 # (nmk,nmk) coef matrix
        Mmat += reg * np.trace(Mmat) / nmk * np.eye(nmk)     # tiny jitter for SPD safety
        Minv = np.linalg.inv(Mmat)
        # cache on the backend (Minv built in float64 for accuracy, cast to working dtype)
        self._imp = dict(cell=cell, w_int=w_int, w_spr=w_spr,
                         Minv=asreal(Minv), nmk=nmk)
        return self

    def invalidate_implicit(self):
        """Markers moved -> the cached coef matrix M (position-dependent) is stale.
        Called by the driver every step when prescribed kinematics are attached."""
        self._imp = None

    def implicit_force(self, vc, markers, vel_desired, dt, ds):
        """Implicit direct forcing (port of pure_v0.1 Force_calculation_p):
        solve  M f = (u_desired - interp(vc))/dt  via the prebuilt M^-1, then
        F_E = spread(f).  Corrects the M != I delta overlap the explicit single-shot
        direct_force ignores -> drives no-slip to ~0.  prepare_implicit() must have
        been called (auto-called here on first use / if marker count changed)."""
        if getattr(self, "_imp", None) is None or self._imp["nmk"] != markers.shape[0]:
            self.prepare_implicit(markers, ds)
        c = self._imp; cell = c["cell"]; w_int = c["w_int"]; w_spr = c["w_spr"]; Minv = c["Minv"]
        nd = vc.shape[1]
        vel_ib = xp.empty((cell.shape[0], nd), dtype=vc.dtype)
        for k in range(nd):
            vel_ib[:, k] = (vc[cell, k] * w_int).sum(axis=1)
        rhs = (asreal(vel_desired) - vel_ib) / dt            # (nmk,nd)
        f = Minv @ rhs                                       # solve M f = rhs (small dense)
        cflat = cell.ravel()
        fE = xp.zeros((self.ncell, nd), dtype=vc.dtype)
        for k in range(nd):
            fE[:, k] = xp.bincount(cflat, (f[:, k:k + 1] * w_spr).ravel(), minlength=self.ncell).astype(vc.dtype, copy=False)
        return fE, f


if __name__ == "__main__":
    import sys
    from pathlib import Path
    solver_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(solver_dir.parent))
    sys.path.insert(0, str(solver_dir))
    sys.argv.append("--nometis")
    import make_euler_mesh as M
    from ib_core_np import IBCore
    mesh = M.make_nested_mesh(0.125, [M._build_box(("center+size", (0, 0, 0), (2, 2, 2)))], refinement_ratio=2)
    core = IBCore(mesh)
    cpos = np.asarray(core.cpos)
    ib = ImmersedBoundary(mesh)

    rng = np.random.default_rng(0)
    mk = rng.uniform(-0.5, 0.5, size=(200, 3))                       # interior markers (margin from bdry)

    # (1) partition of unity: interp of constant 1 == 1 at every marker
    one = xp.ones((core.ncell, 1))
    pu = ib.interp(one, mk)[:, 0]
    pu = np.asarray(pu) if xp.__name__ == "numpy" else xp.asnumpy(pu)
    print(f"[partition-of-unity] interp(1): max|.-1| = {np.max(np.abs(pu - 1.0)):.3e}")

    # (2) linear reproduction: interp of f=x_cell == x_marker
    fx = xp.asarray(cpos[:, 0:1])
    ix = np.asarray(ib.interp(fx, mk)[:, 0])
    print(f"[linear-reproduction] interp(x): max|.-x_marker| = {np.max(np.abs(ix - mk[:, 0])):.3e}")

    # (3) force conservation: sum_E spread(f)*hh^3 == sum_L f*ds^3
    ds = np.full(mk.shape[0], 0.125)
    f_lag = xp.asarray(rng.standard_normal((mk.shape[0], 3)))
    FE = ib.spread(f_lag, mk, ds)
    lhs = np.asarray((FE.sum(axis=0)) * ib.hh ** 3)
    rhs = np.asarray((f_lag * xp.asarray(ds)[:, None] ** 3).sum(axis=0))
    print(f"[force-conservation] |sum_E F*hh^3 - sum_L f*ds^3| = {np.max(np.abs(lhs - rhs)):.3e}")
