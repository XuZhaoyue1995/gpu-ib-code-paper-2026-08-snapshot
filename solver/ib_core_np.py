"""ib_core_np.py — new IB solver core (serial NumPy reference).

Every operator here is a faithful NumPy port of pure_v0.1's Fortran, each
VALIDATED bit-exact against the Fortran oracle (parallel run, reassembled by
position) in work/_geom_validate.py.  This module just lifts that validated code
into a clean, reusable class so the new solver can be built on it and then
accelerated (CuPy/GPU, multigrid, block-structured, distributed IB).

Conventions (match the validated port):
  - global (single-domain) mesh from make_euler_mesh; no MPI partition here.
  - cell fields have length ncell (volume cells only); boundary handled via the
    `m1` internal-face mask + boundary face geometry (ghost center == face center).
  - face order / F2C column order is preserved from make_euler (writer col0->col0),
    so the c1/c2 sign convention matches the Fortran exactly.

Stream-function / vorticity formulation:  s (edge stream fn) -> U=curl(s) (face
flux) -> vc=interp_f2c(U) (cell velocity).  Elliptic operator = flow_mat.
"""
import numpy as np
from backend import xp, asreal, real   # numpy/cupy backend; asreal casts floats to working dtype; real = working float dtype


class IBCore:
    def __init__(self, mesh, ND=3):
        self.ND = ND
        n = mesh.nodes
        self.E2N = E2N = mesh.edges_to_nodes        # (ne,2)
        self.F2N = mesh.faces_to_nodes
        self.F2C = F2C = mesh.faces_to_cells        # (nf,2), -1 == boundary
        self.F2E = F2E = mesh.faces_to_edges        # (nf,4)
        self.ncell = mesh.cells.shape[0]
        self.nface = F2C.shape[0]
        self.nedge = E2N.shape[0]
        self.m1 = m1 = F2C[:, 1] >= 0               # internal-face mask

        # ---- full face-edge incidence (handles 2:1 hanging nodes) ----
        # = F2E (4 base edges/face, collapsed coarse edges already -> first fine sub-edge)
        #   + link_fe_insert (the remaining fine sub-edges).  Flat list = pure_v0.1's link_fe.
        lfi = np.asarray(getattr(mesh, "link_fe_insert", np.zeros((0, 2), dtype=np.int64)))
        self.lf_face = lf_face = np.concatenate([np.repeat(np.arange(self.nface), 4), lfi[:, 0]])
        self.lf_edge = lf_edge = np.concatenate([F2E.ravel(), lfi[:, 1]])

        # ---- geometry over the flat link_fe (uniform: lfi empty -> exactly 4 edges/face) ----
        self.epos = epos = n[E2N].mean(axis=1)                       # edge center
        self.te = te = n[E2N[:, 1]] - n[E2N[:, 0]]                   # edge tangent
        # fpos = avg of the face's edges (variable count per face for hanging-node faces)
        fsum = np.zeros((self.nface, 3)); fcnt = np.zeros(self.nface)
        np.add.at(fsum, lf_face, epos[lf_edge]); np.add.at(fcnt, lf_face, 1.0)
        self.fpos = fpos = fsum / fcnt[:, None]                      # face center
        csum = np.zeros((self.ncell, 3)); ccnt = np.zeros(self.ncell)
        np.add.at(csum, F2C[:, 0], fpos);          np.add.at(ccnt, F2C[:, 0], 1.0)
        np.add.at(csum, F2C[m1, 1], fpos[m1]);     np.add.at(ccnt, F2C[m1, 1], 1.0)
        self.cpos = cpos = csum / ccnt[:, None]                      # cell center (avg of faces)

        # rf = rf1 - rf2 (= cpos(c2)-cpos(c1) internal, fpos-cpos(c1) boundary)
        rf = np.zeros((self.nface, 3))
        rf[m1]  = cpos[F2C[m1, 1]] - cpos[F2C[m1, 0]]
        rf[~m1] = fpos[~m1] - cpos[F2C[~m1, 0]]
        self.rf = rf
        self.rf1 = rf1 = fpos - cpos[F2C[:, 0]]                      # fpos - cpos(c1)
        self.rf2 = rf2 = np.zeros((self.nface, 3))
        rf2[m1] = fpos[m1] - cpos[F2C[m1, 1]]                        # 0 on boundary faces

        # nf = 1/(ND-1) * sum_edges sign(avec.rf)*avec ; avec=(epos-fpos)x te  (per link)
        avec = np.cross(epos[lf_edge] - fpos[lf_face], te[lf_edge])  # (nlink,3)
        s = np.sign((avec * rf[lf_face]).sum(axis=1)); s[s == 0] = 1.0
        self.fe_sign_link = fe_sign_link = s                        # per-link sign (for rot/curl)
        nf = np.zeros((self.nface, 3))
        np.add.at(nf, lf_face, avec * s[:, None])
        nf *= 1.0 / (ND - 1)
        self.nf = nf
        self.Af = Af = np.linalg.norm(nf, axis=1)

        # ---- Modify_geom: 2:1 hanging-node geometry correction (Geometry.f90 Modify_geom) ----
        # Without it, interp_f2c reconstructs spurious cross-flow at refinement interfaces
        # (the moment sum_faces nf(x)rf is not diagonal there).  No-op on uniform meshes
        # (regular faces have |rfe|^2/Af=0.25, |rf|^2/Af=0.25, neither matches 0.36 or 4/9).
        def _build_cpos(fp):
            cs = np.zeros((self.ncell, 3)); cc = np.zeros(self.ncell)
            np.add.at(cs, F2C[:, 0], fp);        np.add.at(cc, F2C[:, 0], 1.0)
            np.add.at(cs, F2C[m1, 1], fp[m1]);   np.add.at(cc, F2C[m1, 1], 1.0)
            return cs / cc[:, None]
        tol = 1e-6 * Af
        # (a) interface face centers: links with |epos-fpos|^2 == 0.36*Af  ->  fpos += rfe/6
        rfe = epos[lf_edge] - fpos[lf_face]
        hitf = np.abs((rfe * rfe).sum(1) - 0.36 * Af[lf_face]) < tol[lf_face]
        np.add.at(fpos, lf_face[hitf], rfe[hitf] / 6.0)
        # (b) recompute cpos, rf from modified fpos
        cpos = _build_cpos(fpos)
        rf1 = fpos - cpos[F2C[:, 0]]
        rf2 = np.zeros((self.nface, 3)); rf2[m1] = fpos[m1] - cpos[F2C[m1, 1]]
        # (c) interface cell centers: |rf1|^2 == 4/9*Af -> cpos[c1]+=rf1/4 ; elif |rf2|^2 == 4/9*Af -> cpos[c2]+=rf2/4
        A49 = (4.0 / 9.0) * Af
        h1 = np.abs((rf1 * rf1).sum(1) - A49) < tol
        np.add.at(cpos, F2C[h1, 0], rf1[h1] / 4.0)
        h2 = (~h1) & m1 & (np.abs((rf2 * rf2).sum(1) - A49) < tol)
        np.add.at(cpos, F2C[h2, 1], rf2[h2] / 4.0)
        # (d) final rf, rf1, rf2 from modified fpos + cpos
        self.fpos = fpos; self.cpos = cpos
        self.rf1 = rf1 = fpos - cpos[F2C[:, 0]]
        self.rf2 = rf2 = np.zeros((self.nface, 3)); rf2[m1] = fpos[m1] - cpos[F2C[m1, 1]]
        self.rf = rf = rf1 - rf2

        # vol_ci = ND / sum(dot(nf,rf1) over c1 - dot(nf,rf2) over c2) ; boundary cells 0
        Vraw = np.zeros(self.ncell)
        np.add.at(Vraw, F2C[:, 0], (nf * rf1).sum(axis=1))
        np.add.at(Vraw, F2C[m1, 1], -(nf[m1] * rf2[m1]).sum(axis=1))
        self.vol_ci = np.where(np.abs(Vraw) > 1e-14, ND / Vraw, 0.0)

        # AoL = (rf.nf)/(rf.rf)   (face geometry for the diffusion term)
        self.AoL = (rf * nf).sum(axis=1) / (rf * rf).sum(axis=1)

        # matr (3x3 per cell) for gradient reconstruction:  matr[c] = 0.5*sum_faces outer(nf,rf)
        # (Matrix(): matr(:,k,:)=Visc_F2C(rf(k),nf,dx); Visc_F2C wf hardcoded 0.5; identity on BC cells)
        contrib = 0.5 * nf[:, :, None] * rf[:, None, :]             # (nf,3,3) = 0.5*outer(nf,rf)
        matr = np.zeros((self.ncell, 3, 3))
        np.add.at(matr, F2C[:, 0], contrib)
        np.add.at(matr, F2C[m1, 1], contrib[m1])
        bc = self.vol_ci <= 0
        matr[bc] = np.eye(3)
        self.matr = matr

        # ---- move operator arrays to the active backend (np on CPU, cupy on GPU) ----
        for name in ("F2C", "lf_face", "lf_edge", "fe_sign_link", "rf", "rf1", "rf2", "m1",
                     "vol_ci", "AoL", "Af", "nf", "matr", "cpos", "epos", "te"):
            setattr(self, name, asreal(getattr(self, name)))   # floats -> working dtype (FP32/FP64), ints/bool unchanged

    # ---- operators (all bit-exact validated; np.bincount instead of slow np.add.at) ----
    def rot(self, face):                 # face -> edge ; ROT[e] += sum_links fe_sign*face[lf_face]
        # bincount always returns float64; cast back to the working dtype to keep FP32 pure
        return xp.bincount(self.lf_edge, self.fe_sign_link * face[self.lf_face], minlength=self.nedge).astype(real, copy=False)

    def curl(self, edge):                # edge -> face ; CURL[f] += sum_links fe_sign*edge[lf_edge] (adjoint)
        return xp.bincount(self.lf_face, self.fe_sign_link * edge[self.lf_edge], minlength=self.nface).astype(real, copy=False)

    def cdiv(self, face):                # face -> cell ; CDIV[c1]+=face, CDIV[c2]-=face
        m1 = self.m1
        return (xp.bincount(self.F2C[:, 0], face, minlength=self.ncell)
                - xp.bincount(self.F2C[m1, 1], face[m1], minlength=self.ncell)).astype(real, copy=False)

    def cgrad(self, cell_ext):           # cell -> face ; face = cell[c2]-cell[c1]  (cell_ext incl ghost values)
        return cell_ext[self.F2C[:, 1]] - cell_ext[self.F2C[:, 0]]

    def integr_c2f(self, cellvec):       # cell vec -> face ; dot(cell[c1],rf1)-dot(cell[c2],rf2)
        out = (cellvec[self.F2C[:, 0]] * self.rf1).sum(axis=1)
        m1 = self.m1
        out[m1] -= (cellvec[self.F2C[m1, 1]] * self.rf2[m1]).sum(axis=1)
        return out

    def interp_f2c(self, face):          # face -> cell vec ; vol_ci*(sum_c1 face*rf1 - sum_c2 face*rf2)
        m1 = self.m1; c0 = self.F2C[:, 0]; c1 = self.F2C[m1, 1]
        fm = face[m1]
        out = xp.empty((self.ncell, 3), dtype=real)
        for d in range(3):
            out[:, d] = (xp.bincount(c0, face * self.rf1[:, d], minlength=self.ncell)
                         - xp.bincount(c1, fm * self.rf2[m1, d], minlength=self.ncell))
        return out * self.vol_ci[:, None]

    def grad_recon(self, face_jump):     # face jumps (=cgrad(phi)) -> cell gradient via matr LU
        # RHS[c] = 0.5*sum_faces face_jump*nf (added to BOTH c1,c2) ; grad = matr^{-1} RHS
        m1 = self.m1; w = 0.5 * face_jump
        rhs = xp.empty((self.ncell, 3), dtype=real)
        for d in range(3):
            rhs[:, d] = (xp.bincount(self.F2C[:, 0], w * self.nf[:, d], minlength=self.ncell)
                         + xp.bincount(self.F2C[m1, 1], w[m1] * self.nf[m1, d], minlength=self.ncell))
        return xp.linalg.solve(self.matr, rhs[..., None])[..., 0]

    def flow_mat(self, pp, alpd=0.0, visc=0.0, rou=1.0, dt=0.0):
        # elliptic operator: w = rot(integr_c2f([viscous] interp_f2c(curl(pp))))   (Helmholtz.f90 flow_mat)
        vcq = self.interp_f2c(self.curl(pp))
        if alpd != 0.0:
            for k in range(self.ND):
                qf = self.AoL * visc * self.cgrad_cellscalar(vcq[:, k])
                vcq[:, k] = rou * vcq[:, k] - 0.5 * dt * self.vol_ci * self.cdiv(qf)
        return self.rot(self.integr_c2f(vcq))   # + DirBC_s applied by the driver

    def cgrad_cellscalar(self, cell):
        # cgrad on a volume-cell scalar; ghost value == 0 (vcq=interp_f2c(...) has vcq[ghost]=0
        # because vol_ci[ghost]=0), so boundary face jump = 0 - cell[c1] = -cell[c1]
        out = xp.zeros(self.nface); m1 = self.m1
        out[m1] = cell[self.F2C[m1, 1]] - cell[self.F2C[m1, 0]]
        out[~m1] = -cell[self.F2C[~m1, 0]]
        return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    sys.argv.append("--nometis")
    import make_euler_mesh as M
    mesh = M.make_nested_mesh(0.25, [M._build_box(("center+size", (0, 0, 0), (2, 2, 2)))],
                              refinement_ratio=2)
    c = IBCore(mesh)
    print(f"ncell={c.ncell} nface={c.nface} nedge={c.nedge}")
    # sanity: gradient reconstruction of a linear field must be the constant unit vector
    worst = 0.0
    for k in range(3):
        phi_ext = c.cpos[:, k]                         # linear field = k-th coordinate (volume cells)
        # build extended cgrad incl boundary ghost = face center coord (matches Fortran cpos(ghost))
        jump = np.empty(c.nface); m1 = c.m1
        jump[m1] = phi_ext[c.F2C[m1, 1]] - phi_ext[c.F2C[m1, 0]]
        jump[~m1] = c.fpos[~m1, k] - phi_ext[c.F2C[~m1, 0]]
        g = c.grad_recon(jump)
        e_k = np.zeros(3); e_k[k] = 1.0
        err = np.max(np.abs(g[c.vol_ci > 0] - e_k))
        worst = max(worst, err)
    print(f"grad-recon(linear)=e_k  max err = {worst:.3e}  {'OK' if worst < 1e-9 else 'FAIL'}")
    # sanity: vol_ci>0 count == interior cells; Af all positive
    print(f"interior cells={int((c.vol_ci>0).sum())}  Af>0: {bool((c.Af>0).all())}")
