"""ib_driver_np.py — new IB solver driver (CG elliptic solve, BC, time loop).

Built on the oracle-validated IBCore (ib_core_np.py).  Mirrors pure_v0.1's
Helmholtz.f90 (Helm_s/CG_m/flow_mat/DirBC_s) and NS_solver_exp.f90 (FlowSolve_exp).

This file is being grown incrementally, each piece validated:
  [1] CG elliptic solver on flow_mat  <- manufactured-solution test (this commit)
  [2] stream-function BC (BC_s/BC_vc) + ghost cells
  [3] Crank-Nicolson time loop + convection/diffusion RHS
  [4] full-run compare vs Fortran oracle trajectory
"""
import numpy as np
from backend import xp, GPU, FP32, asreal, real, asnumpy   # numpy/cupy; FP32; real = working dtype
from ib_core_np import IBCore

# cupy's default memory pool retains freed blocks AND fragments during the cold
# first-step CG (1000s of iterations) -> peak balloons to ~8GB for a working set of
# ~3GB.  Periodically releasing free blocks keeps the peak ~= the live set, which is
# what lets a finer mesh fit on a small GPU.  No-op on CPU.
if GPU:
    import cupy as _cp
    _mempool = _cp.get_default_memory_pool()
    def free_gpu_pool():
        _mempool.free_all_blocks()
else:
    def free_gpu_pool():
        pass


# ---------------------------------------------------------------------------
# [1] CG elliptic solver  (Helm_s/CG_m with flow_mat as the matrix-free matvec)
# ---------------------------------------------------------------------------
def apply_A(core, pp, dir_mask, **kw):
    """Elliptic operator A = flow_mat + DirBC_s (zero residual at Dirichlet edges)."""
    w = core.flow_mat(pp, **kw)
    w[dir_mask] = 0.0
    return w


def cg_solve(core, b, dir_mask, tol=1e-10, maxiter=3000, prec=None, **kw):
    """Preconditioned CG for A x = b on the non-Dirichlet edge subspace.
    Returns (x, iters, rel_residual).  prec = diagonal preconditioner (None -> identity)."""
    if prec is None:
        prec = np.ones(core.nedge)
    x = np.zeros(core.nedge)
    b = b.copy(); b[dir_mask] = 0.0
    r = b - apply_A(core, x, dir_mask, **kw)        # x0 = 0
    r[dir_mask] = 0.0
    z = r * prec
    p = z.copy()
    eta = z.dot(r)
    eta0 = max(eta, 1e-300)
    last = 1.0
    m = 0
    for m in range(1, maxiter + 1):
        w = apply_A(core, p, dir_mask, **kw)
        denom = p.dot(w)
        if denom == 0.0:
            break
        alf = eta / denom
        x += alf * p
        r -= alf * w
        z = r * prec
        eta_new = z.dot(r)
        last = abs(r.dot(r)) / (b.dot(b) + 1e-300)
        if last < tol * tol:
            break
        p = z + (eta_new / eta) * p
        eta = eta_new
    return x, m, np.sqrt(last)


def boundary_edge_mask(core, lo=-1.0, hi=1.0, eps=1e-9):
    """Edges on the domain box faces (|coord| at lo/hi) -> Dirichlet for the test."""
    e = core.epos
    on = np.zeros(core.nedge, dtype=bool)
    for k in range(3):
        on |= (np.abs(e[:, k] - lo) < eps) | (np.abs(e[:, k] - hi) < eps)
    return on


def diag_preconditioner(core, dir_mask):
    """Diagonal preconditioner (Helm_s): qf=(rf.nf)/Af^2 ; prec=sumup_F2E(qf); invert.
    sumup_F2E[e] = sum over faces touching e of qf[f]  (|fe_sign|, unsigned face->edge sum)."""
    qf = (core.rf * core.nf).sum(axis=1) / (core.Af * core.Af)
    # SUMUP_F2E over the flat link_fe (unsigned face->edge sum)
    prec = xp.bincount(core.lf_edge, xp.abs(core.fe_sign_link) * qf[core.lf_face], minlength=core.nedge)
    prec[dir_mask] = 0.0
    inv = xp.zeros(core.nedge, dtype=real)
    nz = prec != 0.0
    inv[nz] = 1.0 / prec[nz]
    return inv


# ---------------------------------------------------------------------------
# [2][3] full FlowSolve_exp_3 driver: ghost cells + stream-function BC + CN time loop
# ---------------------------------------------------------------------------
class Driver:
    """Crank-Nicolson stream-function/vorticity stepper (pure_v0.1 FlowSolve_exp_3,
    IB_key=0).  Ghost cell per boundary face; BC families 1..6 = INLET/OUTLET/TOP/BOT/
    FRONT/BACK (matches make_euler _outer_box_boundary_faces + BC_IC.txt)."""

    def __init__(self, core, mesh, visc, rou=1.0, ac=0.5, ad=0.5):
        self.core = core
        self.visc = visc; self.rou = rou; self.ac = ac; self.ad = ad
        self.bc_edge = xp.asarray(mesh.bc_edge)
        self.bc_face = xp.asarray(mesh.bc_face)
        bf = xp.where(~core.m1)[0]                       # boundary faces
        self.bf = bf
        self.nghost = int(bf.shape[0])
        self.face_ghost = xp.full(core.nface, -1, dtype=xp.int64)
        self.face_ghost[bf] = core.ncell + xp.arange(self.nghost)
        self.F2C_ext = core.F2C.copy()
        self.F2C_ext[bf, 1] = self.face_ghost[bf]
        self.bf_fam = self.bc_face[bf]                  # 1..6 per boundary face
        self.inlet_edges = self.bc_edge == 1            # LINEAR stream BC
        self.dir_mask = xp.isin(self.bc_edge, xp.asarray([3, 4, 5, 6]))   # FIXED -> DirBC_s residual zero
        self.prec = diag_preconditioner(core, self.dir_mask)
        self.Asp = None                                 # optional sparse matrix for fast CG matvec
        self.ib = None                                  # optional immersed body (set_ib)
        self.ib_alpha = 1.0                             # IB force amplification (compensate Helm^-1 attenuation)
        self.ib_implicit = False                        # True -> solve M f = rhs (coef matrix) instead of explicit f=rhs
        self._step_count = 0                            # for cold-start handling
        self.cold_cg = 0                                # >0: cap CG iters for the first 3 steps (limits cold-CG pool
        #                                                 fragmentation so fine meshes fit a small GPU; transient only)
        self.free_each_step = False                     # >0 GPU mesh: release cupy pool each step (cap cross-step growth)
        self.cavity = False                             # lid-driven cavity BC (no-slip walls + moving lid) instead of inlet/outlet
        self.motion = None                              # optional prescribed kinematics (ib_kinematics), updates markers each step
        self.time = 0.0                                 # physical time, advanced by each step()

    def set_cavity(self, U_lid=1.0, spanwise_zerograd=True):
        """Lid-driven cavity: psi=0 (Dirichlet) on ALL boundary edges (walls are
        streamlines); the y=max wall is the moving lid (u=U_lid), the z-faces (thin
        spanwise slab) are ZEROGRAD by default for quasi-2D; pass
        spanwise_zerograd=False for six no-slip walls. Boundary
        faces are classified BY POSITION (robust to the family numbering).  Ghia 1982."""
        self.cavity = True
        self.U_lid = float(U_lid)
        self._lid_vec = xp.asarray([U_lid, 0.0, 0.0], dtype=real)
        self.inlet_edges = xp.zeros(self.core.nedge, dtype=bool)   # no inlet
        self.dir_mask = self.bc_edge > 0                           # psi=0 on every wall edge
        self.prec = diag_preconditioner(self.core, self.dir_mask)
        # classify boundary faces by position: lid=y-max, z-faces=spanwise slab, rest=no-slip
        import numpy as _np
        fp = _np.asarray(asnumpy(self.core.fpos)); bf = asnumpy(self.bf)
        fy = fp[bf, 1]; fz = fp[bf, 2]
        tol = 1e-6 + 1e-3 * (fy.max() - fy.min())
        self._lid_m = xp.asarray(fy > fy.max() - tol)
        z_faces = (fz < fz.min() + tol) | (fz > fz.max() - tol)
        if not spanwise_zerograd:
            z_faces = _np.zeros_like(z_faces, dtype=bool)
        self._zf_m = xp.asarray(z_faces)
        return self

    def set_ib(self, ib, markers, ds, vel_desired, force_mode='replacement'):
        """Attach an immersed body (ib_immersed.ImmersedBoundary).  markers (M,3),
        ds (M,) marker spacing, vel_desired (M,3) prescribed marker velocity.
        For a moving body, update markers/vel_desired before each step()."""
        if force_mode not in {'replacement', 'fortran_accumulated'}:
            raise ValueError(f'unsupported IB force_mode: {force_mode!r}')
        self.ib = ib
        self.ib_markers = asreal(markers)
        self.ib_ds = asreal(ds)
        self.ib_vel = asreal(vel_desired)
        self.ib_force_mode = force_mode
        self.ib_force_E = xp.zeros((self.core.ncell, 3), dtype=self.core.cpos.dtype)
        self.ib_force_L = xp.zeros_like(self.ib_vel)
        self.ib_force_increment_E = xp.zeros_like(self.ib_force_E)
        self.ib_force_increment_L = xp.zeros_like(self.ib_force_L)
        self.ib_force_update_count = 0
        self.ib_force_updates_last_step = 0
        return self

    def ib_state_dict(self):
        '''Return persistent IB state as detached CPU data for np.savez.'''
        if self.ib is None:
            return None
        return {
            'force_mode': self.ib_force_mode,
            'force_E': asnumpy(self.ib_force_E).copy(),
            'force_L': asnumpy(self.ib_force_L).copy(),
            'force_update_count': int(self.ib_force_update_count),
        }

    def load_ib_state_dict(self, state):
        '''Restore persistent IB state after set_ib has been called.'''
        if self.ib is None:
            raise RuntimeError('set_ib must be called before loading IB state')
        mode = str(state['force_mode'])
        if mode != self.ib_force_mode:
            raise ValueError(
                f'IB force-mode mismatch: checkpoint={mode!r}, run={self.ib_force_mode!r}'
            )
        force_E = xp.asarray(state['force_E'], dtype=self.ib_force_E.dtype)
        force_L = xp.asarray(state['force_L'], dtype=self.ib_force_L.dtype)
        if force_E.shape != self.ib_force_E.shape:
            raise ValueError(
                f'Eulerian IB-force shape mismatch: {force_E.shape} != {self.ib_force_E.shape}'
            )
        if force_L.shape != self.ib_force_L.shape:
            raise ValueError(
                f'Lagrangian IB-force shape mismatch: {force_L.shape} != {self.ib_force_L.shape}'
            )
        self.ib_force_E = force_E.copy()
        self.ib_force_L = force_L.copy()
        self.ib_force_increment_E = xp.zeros_like(self.ib_force_E)
        self.ib_force_increment_L = xp.zeros_like(self.ib_force_L)
        self.ib_force_update_count = int(state.get('force_update_count', 0))
        self.ib_force_updates_last_step = 0
        return self

    def set_motion(self, motion, t0=0.0):
        """Attach prescribed kinematics (ib_kinematics object: motion(t) ->
        (markers, vel)).  Each step() advances the clock to t+dt and sets
        ib_markers/ib_vel at the NEW time level (Fortran updates the mesh before
        the force calc).  The implicit coef matrix M = interp.spread depends on
        the marker positions, so it is invalidated (rebuilt) every step."""
        self.motion = motion
        self.time = float(t0)
        return self

    def enable_sparse(self, dt):
        """Assemble flow_mat as a constant sparse matrix; CG then uses A@p (fast matvec)
        instead of the python operator chain.  CG (not direct LU) because the stream-
        function operator has a gauge null space that LU can't factor but CG handles."""
        from ib_sparse import build_A
        self.Asp = build_A(self.core, alpd=self.ad, visc=self.visc, rou=self.rou, dt=dt)
        return self

    # cell-vec operators on ghost-extended fields ------------------------------
    def cgrad_ext(self, cell_ext_col):   # scalar component (ncell+nghost,)
        return cell_ext_col[self.F2C_ext[:, 1]] - cell_ext_col[self.F2C_ext[:, 0]]

    def wavg_ext(self, cell_ext_col, wf1, wf2):
        return cell_ext_col[self.F2C_ext[:, 0]] * wf1 + cell_ext_col[self.F2C_ext[:, 1]] * wf2

    # BC + velocity ------------------------------------------------------------
    def calc_vel(self, s):
        """BC_s (INLET=LINEAR) -> U=curl(s) -> vc=interp_f2c(U) -> BC_vc ghosts."""
        core = self.core
        s = s.copy()
        s[self.inlet_edges] = core.epos[self.inlet_edges, 1] * core.te[self.inlet_edges, 2]
        U = core.curl(s)
        vc = xp.zeros((core.ncell + self.nghost, 3), dtype=real)
        vc[:core.ncell] = core.interp_f2c(U)
        bf = self.bf; c1 = core.F2C[bf, 0]; g = self.face_ghost[bf]
        if self.cavity:
            # lid-driven cavity ghosts: face velocity = 0 (no-slip wall), = (U_lid,0,0)
            # (moving lid), ZEROGRAD on the spanwise z-faces.  ghost = 2*face_target - interior.
            v1 = vc[c1]
            gv = -v1                                     # default no-slip walls: face vel 0
            gv[self._lid_m] = 2.0 * self._lid_vec - v1[self._lid_m]   # moving lid: face vel = U_lid
            gv[self._zf_m] = v1[self._zf_m]              # z-faces: ZEROGRAD (quasi-2D slab)
            vc[g] = gv
            return s, U, vc
        nf = core.nf[bf]; Af2 = (core.Af[bf] ** 2)[:, None]; Uf = U[bf][:, None]
        zero = self.bf_fam == 1                          # INLET: ZERO tangential velocity
        vc[g[zero]] = Uf[zero] * nf[zero] / Af2[zero]
        zg = ~zero                                       # others: ZEROGRAD
        v1 = vc[c1[zg]]
        vc[g[zg]] = v1 + (Uf[zg] - (v1 * nf[zg]).sum(axis=1, keepdims=True)) * nf[zg] / Af2[zg]
        return s, U, vc

    def conv_diff(self, U, vc):
        """convection_cell + diffusion_cell (conv_id=1: CENTRAL, no cross-diffusion)."""
        core = self.core; ncell = core.ncell
        conv = xp.zeros((ncell, 3), dtype=real); diff = xp.zeros((ncell, 3), dtype=real)
        wf = 0.5 * U                                     # set_conv CENTRAL: wf1=wf2=0.5*U
        for j in range(3):
            rd = self.cgrad_ext(vc[:, j])                # CGRAD(vc_j) with ghost
            diff[:, j] = core.vol_ci * core.cdiv(core.AoL * self.visc * rd)
            vf = self.wavg_ext(vc[:, j], wf, wf)         # WAVG_C2F(vc_j, wf)
            conv[:, j] = core.vol_ci * core.cdiv(vf)
        return conv, diff

    def helm_solve(self, res_e, s, dt, tol1=1e-10, tol2=1e-9, cg_max=1000):
        """Helm_s: CG on A x = b, warm-start x=s, r=res_e is the current residual b-A x."""
        if FP32:                                          # float32 CG floor ~1e-6; 1e-9 unreachable
            tol1 = max(tol1, 3e-5); tol2 = max(tol2, 3e-5)
        core = self.core
        kw = dict(alpd=self.ad, visc=self.visc, rou=self.rou, dt=dt)
        r = res_e.copy(); r[self.dir_mask] = 0.0
        x = s.copy()
        pp = r * self.prec
        eta = pp.dot(r); rho0 = eta
        xnorm = x.dot(x) + 1e-16
        err_tot = xnorm * (tol1 ** 2) + (tol2 ** 2) * rho0
        if float(eta) < err_tot:                         # already converged (Helm_s: goto 100)
            return x, 0
        check = 20 if GPU else 1                          # GPU: check convergence every K iters (avoid per-iter device sync)
        for m in range(1, cg_max + 1):
            w = (self.Asp @ pp) if self.Asp is not None else core.flow_mat(pp, **kw)
            w[self.dir_mask] = 0.0
            alf = eta / pp.dot(w)
            x += alf * pp
            r -= alf * w
            z = r * self.prec
            eta_new = z.dot(r)
            if m % check == 0 and float(eta_new) < err_tot:
                break
            pp = z + (eta_new / eta) * pp
            eta = eta_new
        self.last_helm_m = m
        return x, m

    def step(self, s, s_old, dt, dt_old, inner=3):
        """One FlowSolve_exp_3 (IB_key=0) step.  Returns (s_new, vc_new, s_old_new)."""
        core = self.core; rou = self.rou; ac = self.ac; ad = self.ad
        accumulated = (
            self.ib is not None
            and self.ib_force_mode == 'fortran_accumulated'
        )
        if accumulated and inner < 2:
            raise ValueError('fortran_accumulated requires at least 2 fluid inner iterations')
        self._step_count += 1
        self.time += dt
        if self.ib is not None:
            self.ib_force_updates_last_step = 0
        if self.ib is not None and self.motion is not None:
            mk, vel = self.motion(self.time)             # body state at the new time level
            self.ib_markers = asreal(mk)
            self.ib_vel = asreal(vel)
            if self.ib_implicit:
                self.ib.invalidate_implicit()            # M is position-dependent -> rebuild
            if accumulated:
                # The held Lagrangian force follows a moving body.
                self.ib_force_E = self.ib.spread(
                    self.ib_force_L, self.ib_markers, self.ib_ds
                )
        cg = min(self.cold_cg, 1000) if (self.cold_cg and self._step_count <= 3) else 1000
        # state from current s
        _, U, vc = self.calc_vel(s)
        conv0, diff0 = self.conv_diff(U, vc)
        res_c0 = rou * vc[:core.ncell] - (1 - ac) * dt * rou * conv0 + (1 - ad) * dt * diff0
        # Guess_s
        qe = s.copy()
        s = s + (s - s_old) * dt / dt_old
        s, U, vc = self.calc_vel(s)
        s_old = qe
        for i in range(1, inner + 1):
            conv, diff = self.conv_diff(U, vc)
            res_c = res_c0 - rou * vc[:core.ncell] - ac * dt * rou * conv + ad * dt * diff
            if self.ib is not None:
                if accumulated:
                    # All solves use the currently held total force.
                    res_c = res_c + dt * self.ib_force_E
                else:
                    # Legacy Python path: replace force at every inner iteration.
                    if self.ib_implicit:
                        fE, fL = self.ib.implicit_force(
                            vc[:core.ncell], self.ib_markers, self.ib_vel, dt, self.ib_ds
                        )
                    else:
                        fE, fL = self.ib.direct_force(
                            vc[:core.ncell], self.ib_markers, self.ib_vel, dt, self.ib_ds
                        )
                    self.ib_force_E = self.ib_alpha * fE
                    self.ib_force_L = self.ib_alpha * fL
                    self.ib_force_increment_E = self.ib_force_E
                    self.ib_force_increment_L = self.ib_force_L
                    self.ib_force_update_count += 1
                    self.ib_force_updates_last_step += 1
                    res_c = res_c + dt * self.ib_force_E
            res_f = core.integr_c2f(res_c)
            res_e = core.rot(res_f)
            s, _ = self.helm_solve(res_e, s, dt, cg_max=cg)
            s, U, vc = self.calc_vel(s)
            if accumulated and i == 1:
                # Active Fortran get_IB_force_exp_inc: evaluate once from the
                # post-predictor velocity, then add to persistent force_*_OLD.
                if self.ib_implicit:
                    delta_E, delta_L = self.ib.implicit_force(
                        vc[:core.ncell], self.ib_markers, self.ib_vel, dt, self.ib_ds
                    )
                else:
                    delta_E, delta_L = self.ib.direct_force(
                        vc[:core.ncell], self.ib_markers, self.ib_vel, dt, self.ib_ds
                    )
                self.ib_force_increment_E = self.ib_alpha * delta_E
                self.ib_force_increment_L = self.ib_alpha * delta_L
                self.ib_force_E = self.ib_force_E + self.ib_force_increment_E
                self.ib_force_L = self.ib_force_L + self.ib_force_increment_L
                self.ib_force_update_count += 1
                self.ib_force_updates_last_step += 1
        if self.free_each_step:
            free_gpu_pool()                             # release pool between steps (fine-mesh memory cap)
        return s, vc, s_old


if __name__ == "__main__":
    import sys
    from pathlib import Path
    solver_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(solver_dir.parent))
    sys.path.insert(0, str(solver_dir))
    sys.argv.append("--nometis")
    import make_euler_mesh as M
    mesh = M.make_nested_mesh(0.25, [M._build_box(("center+size", (0, 0, 0), (2, 2, 2)))],
                              refinement_ratio=2)
    core = IBCore(mesh)
    dmask = boundary_edge_mask(core)
    print(f"edges={core.nedge}  Dirichlet(boundary)={int(dmask.sum())}  interior={int((~dmask).sum())}")

    # ---- manufactured-solution test: pick x_true (0 on boundary), b = A x_true, solve, compare ----
    rng = np.random.default_rng(0)
    e = core.epos
    smooth = np.cos(e[:, 0]) * np.cos(e[:, 1]) * np.cos(e[:, 2])      # smooth interior field
    x_true = smooth.copy(); x_true[dmask] = 0.0
    b = apply_A(core, x_true, dmask)

    prec = diag_preconditioner(core, dmask)
    for label, P in [("no-prec", None), ("diag-prec", prec)]:
        x, iters, res = cg_solve(core, b, dmask, tol=1e-12, maxiter=5000, prec=P)
        rA = apply_A(core, x, dmask) - b
        res_rel = np.linalg.norm(rA) / (np.linalg.norm(b) + 1e-300)
        # solution error on interior (operator may have a null space -> report both)
        err = np.linalg.norm(x - x_true) / (np.linalg.norm(x_true) + 1e-300)
        print(f"  [{label:9s}] iters={iters:4d}  ‖Ax-b‖/‖b‖={res_rel:.2e}  ‖x-x_true‖/‖x_true‖={err:.2e}")
