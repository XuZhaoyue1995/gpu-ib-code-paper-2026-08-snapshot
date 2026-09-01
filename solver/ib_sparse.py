"""ib_sparse.py — assemble the (constant) viscous elliptic operator flow_mat as a
scipy sparse matrix, so it can be LU-factored ONCE and solved directly each step
(no CG iterations -> O(N) per step instead of O(N^{4/3})).

A = R @ Imat @ Vblock @ P @ C   (= rot . integr_c2f . [viscous] . interp_f2c . curl)
Self-validated: A @ s must equal the matrix-free core.flow_mat(s) (the oracle).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

try:
    from backend import asnumpy, GPU, FP32
except Exception:                       # standalone (no backend on path) -> CPU/numpy
    asnumpy = lambda x: np.asarray(x); GPU = False; FP32 = False

_a = lambda x: np.asarray(asnumpy(x))   # core array (numpy OR cupy) -> numpy, for scipy assembly


def _curl_matrix(core):
    # CURL[face] += fe_sign*edge over the flat link_fe (handles 2:1 hanging nodes)
    rows = _a(core.lf_face); cols = _a(core.lf_edge); data = _a(core.fe_sign_link)
    return sp.coo_matrix((data, (rows, cols)), shape=(core.nface, core.nedge)).tocsr()


def _interp_f2c_matrix(core):           # face -> cellvec, component-major idx = d*ncell + c
    nc = core.ncell; m1 = _a(core.m1)
    F2C = _a(core.F2C); vol_ci = _a(core.vol_ci); rf1 = _a(core.rf1); rf2 = _a(core.rf2)
    f = np.arange(core.nface)
    rows = []; cols = []; data = []
    for d in range(3):
        rows.append(d * nc + F2C[:, 0]); cols.append(f); data.append(vol_ci[F2C[:, 0]] * rf1[:, d])
        rows.append(d * nc + F2C[m1, 1]); cols.append(f[m1]); data.append(-vol_ci[F2C[m1, 1]] * rf2[m1, d])
    return sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(3 * nc, core.nface)).tocsr()


def _integr_c2f_matrix(core):           # cellvec -> face
    nc = core.ncell; m1 = _a(core.m1)
    F2C = _a(core.F2C); rf1 = _a(core.rf1); rf2 = _a(core.rf2)
    f = np.arange(core.nface)
    rows = []; cols = []; data = []
    for d in range(3):
        rows.append(f); cols.append(d * nc + F2C[:, 0]); data.append(rf1[:, d])
        rows.append(f[m1]); cols.append(d * nc + F2C[m1, 1]); data.append(-rf2[m1, d])
    return sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(core.nface, 3 * nc)).tocsr()


def _cdiv_matrix(core):                 # face -> cell
    m1 = _a(core.m1); F2C = _a(core.F2C)
    rows = np.concatenate([F2C[:, 0], F2C[m1, 1]])
    cols = np.concatenate([np.arange(core.nface), np.arange(core.nface)[m1]])
    data = np.concatenate([np.ones(core.nface), -np.ones(int(m1.sum()))])
    return sp.coo_matrix((data, (rows, cols)), shape=(core.ncell, core.nface)).tocsr()


def _cgrad_ghost0_matrix(core):         # cell -> face (boundary ghost == 0)
    m1 = _a(core.m1); F2C = _a(core.F2C)
    rows = np.concatenate([np.arange(core.nface), np.arange(core.nface)[m1]])
    cols = np.concatenate([F2C[:, 0], F2C[m1, 1]])
    data = np.concatenate([-np.ones(core.nface), np.ones(int(m1.sum()))])
    return sp.coo_matrix((data, (rows, cols)), shape=(core.nface, core.ncell)).tocsr()


def build_A(core, alpd=0.0, visc=0.0, rou=1.0, dt=0.0, to_backend=True):
    """Assemble flow_mat as a sparse nedge x nedge matrix (no DirBC applied).
    Assembled on CPU (scipy); if running on GPU and to_backend, returned as a
    cupyx.scipy.sparse matrix so Asp @ p works on device-resident edge vectors."""
    nc = core.ncell
    C = _curl_matrix(core)
    P = _interp_f2c_matrix(core)
    I = _integr_c2f_matrix(core)
    R = C.T.tocsr()                                  # ROT = adjoint of CURL
    if alpd != 0.0:
        CDIVm = _cdiv_matrix(core)
        Gm = _cgrad_ghost0_matrix(core)
        Dvol = sp.diags(_a(core.vol_ci)); DAv = sp.diags(_a(core.AoL) * visc)
        V = rou * sp.eye(nc) - 0.5 * dt * (Dvol @ CDIVm @ DAv @ Gm)
        Vblock = sp.block_diag([V, V, V]).tocsr()
        A = (R @ I @ Vblock @ P @ C).tocsr()
    else:
        A = (R @ I @ P @ C).tocsr()
    if FP32:
        A = A.astype(np.float32)
    if to_backend and GPU:
        import cupyx.scipy.sparse as csp
        return csp.csr_matrix(A)
    return A


class DirectHelm:
    """Replaces the CG Helm_s with a one-time LU factorization + back-substitution.
    Solves on the interior (non-Dirichlet) edge subspace; A is constant across steps."""
    def __init__(self, core, dir_mask, alpd, visc, rou, dt):
        A = build_A(core, alpd, visc, rou, dt)
        self.free = np.where(~dir_mask)[0]
        Aii = A[self.free][:, self.free].tocsc()
        self.lu = sla.splu(Aii)
        self.dir_mask = dir_mask

    def solve(self, res_e, s):
        # A delta = res_e on interior, delta=0 on Dirichlet; s_new = s + delta
        x = s.copy()
        delta = self.lu.solve(res_e[self.free])
        x[self.free] += delta
        return x, 0


if __name__ == "__main__":
    import sys
    from pathlib import Path
    solver_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(solver_dir.parent))
    sys.path.insert(0, str(solver_dir))
    sys.argv.append("--nometis")
    import make_euler_mesh as M
    from ib_core_np import IBCore
    mesh = M.make_nested_mesh(0.25, [M._build_box(("center+size", (0, 0, 0), (2, 2, 2)))], refinement_ratio=2)
    core = IBCore(mesh)
    rng = np.random.default_rng(1)
    s = rng.standard_normal(core.nedge)
    for alpd, visc, dt in [(0.0, 0.0, 0.0), (0.5, 0.01, 0.01)]:
        A = build_A(core, alpd=alpd, visc=visc, rou=1.0, dt=dt)
        w_sparse = A @ s
        w_free = core.flow_mat(s, alpd=alpd, visc=visc, rou=1.0, dt=dt)
        d = np.max(np.abs(w_sparse - w_free)); sc = np.max(np.abs(w_free))
        print(f"alpd={alpd} visc={visc}: ‖A@s - flow_mat(s)‖inf = {d:.3e}  rel={d/sc:.2e}  "
              f"{'OK' if d < 1e-9 * sc else '*** DIFFER ***'}")
