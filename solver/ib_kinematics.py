"""ib_kinematics.py — prescribed kinematics for the immersed boundary (port of
pure_v0.1 kinematics.f90's prescribed-motion core, vectorized).

A motion object is attached with Driver.set_motion(motion); each step() the
driver advances its clock and calls motion(t_new) -> (markers(t), vel(t)),
feeding the multi-direct-forcing IB.  The interface is plain (M,3) arrays, NOT
rigid-body specific: morphing bodies (per-marker velocities, e.g. the flapping
wing built from Fourier control points + weights in kinematics.f90) plug in by
returning their own arrays.

Velocities are ANALYTIC (not finite-differenced): the IB force is
f = (u_body - interp(vc))/dt, so noise in u_body feeds straight into the force.
"""
import numpy as np


class RigidMotion:
    """markers(t) = ref + xc(t) (one rigid translation for all markers).
    xc_fn(t)->(3,) displacement, uc_fn(t)->(3,) velocity (analytic pair)."""
    def __init__(self, ref_markers, xc_fn, uc_fn):
        self.ref = np.asarray(ref_markers, dtype=float)
        self.xc_fn = xc_fn
        self.uc_fn = uc_fn

    def __call__(self, t):
        mk = self.ref + np.asarray(self.xc_fn(t), dtype=float)
        vel = np.tile(np.asarray(self.uc_fn(t), dtype=float), (self.ref.shape[0], 1))
        return mk, vel


class Oscillation(RigidMotion):
    """Sinusoidal rigid translation  x_c(t) = A*sin(2*pi*freq*t + phase)*axis."""
    def __init__(self, ref_markers, axis=(0.0, 1.0, 0.0), A=0.1, freq=1.0, phase=0.0):
        ax = np.asarray(axis, dtype=float)
        ax = ax / np.linalg.norm(ax)
        w = 2.0 * np.pi * freq
        super().__init__(ref_markers,
                         lambda t: A * np.sin(w * t + phase) * ax,
                         lambda t: A * w * np.cos(w * t + phase) * ax)
        self.A = A; self.freq = freq; self.axis = ax


class FourierMotion(RigidMotion):
    """kinematics.f90 get_pos:  x_c(t) = a[:,0] + sum_{k=1..K} a[:,k]*cos(k*2*pi*t/T)
    + b[:,k]*sin(k*2*pi*t/T)  per coordinate (Fortran: K=6 harmonics, period 1).
    a, b: (3, K+1) coefficient arrays (b[:,0] unused, as in the Fortran files).
    This is the format of the butterfly/bat control-point files (wingtip/wrist/
    digit5), so the future morphing-wing kinematics builds on this class."""
    def __init__(self, ref_markers, a, b, T=1.0):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        K = a.shape[1] - 1
        w0 = 2.0 * np.pi / T
        k = np.arange(1, K + 1)

        def xc(t):
            return a[:, 0] + a[:, 1:] @ np.cos(k * w0 * t) + b[:, 1:] @ np.sin(k * w0 * t)

        def uc(t):
            return (a[:, 1:] * (-k * w0)) @ np.sin(k * w0 * t) + (b[:, 1:] * (k * w0)) @ np.cos(k * w0 * t)

        super().__init__(ref_markers, xc, uc)


class PitchPlunge:
    """Planar rigid motion (translation + rotation about z through the body pivot):
        positions(t) = c(t) + Rz(alpha(t)) @ ref
        velocities(t) = c'(t) + alpha'(t) * z_hat x (Rz @ ref)
    All four callables analytic (xc_fn/uc_fn (3,), alpha_fn/alphadot_fn scalar).
    This is the flapping-plate primitive (stroke + pitch flip)."""
    def __init__(self, ref_markers, xc_fn, uc_fn, alpha_fn, alphadot_fn):
        self.ref = np.asarray(ref_markers, dtype=float)
        self.xc_fn = xc_fn; self.uc_fn = uc_fn
        self.alpha_fn = alpha_fn; self.alphadot_fn = alphadot_fn

    def __call__(self, t):
        a = float(self.alpha_fn(t)); ad = float(self.alphadot_fn(t))
        ca, sa = np.cos(a), np.sin(a)
        x, y, z = self.ref[:, 0], self.ref[:, 1], self.ref[:, 2]
        rx = ca * x - sa * y                      # Rz(a) @ ref
        ry = sa * x + ca * y
        mk = np.stack([rx, ry, z], 1) + np.asarray(self.xc_fn(t), dtype=float)
        vel = np.stack([-ad * ry, ad * rx, np.zeros_like(z)], 1) \
            + np.asarray(self.uc_fn(t), dtype=float)
        return mk, vel


class WangHover(PitchPlunge):
    """Wang/Birch/Dickinson (JFM 2004) normal hovering kinematics:
        x_c(t) = (A0/2) cos(2 pi f t) * stroke_axis(x)
        alpha(t) = alpha0 + beta sin(2 pi f t + phi)   (pitch about z, chord ref along x)
    Standard case: A0/c=2.8, alpha0=pi/2, beta=pi/4, phi=0, Re=u_max*c/nu=75,
    u_max = pi f A0."""
    def __init__(self, ref_markers, A0, freq, alpha0=np.pi / 2, beta=np.pi / 4, phi=0.0):
        w = 2 * np.pi * freq
        super().__init__(ref_markers,
                         lambda t: np.array([0.5 * A0 * np.cos(w * t), 0.0, 0.0]),
                         lambda t: np.array([-0.5 * A0 * w * np.sin(w * t), 0.0, 0.0]),
                         lambda t: alpha0 + beta * np.sin(w * t + phi),
                         lambda t: beta * w * np.cos(w * t + phi))
        self.A0 = A0; self.freq = freq; self.u_max = np.pi * freq * A0


if __name__ == "__main__":
    # self-test: analytic velocity == centered finite difference of position
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((50, 3))
    eps = 1e-6
    cases = [
        ("Oscillation", Oscillation(ref, axis=(0, 1, 0), A=0.0625, freq=0.8)),
        ("FourierMotion", FourierMotion(ref, a=rng.standard_normal((3, 7)) * 0.1,
                                        b=rng.standard_normal((3, 7)) * 0.1, T=1.0)),
        ("WangHover", WangHover(ref, A0=2.8, freq=0.114)),
    ]
    for name, mo in cases:
        worst = 0.0
        for t in (0.0, 0.13, 0.41, 0.77, 1.9):
            mk_p, _ = mo(t + eps)
            mk_m, _ = mo(t - eps)
            _, vel = mo(t)
            fd = (mk_p - mk_m) / (2 * eps)
            worst = max(worst, float(np.max(np.abs(fd - vel))))
        print(f"[{name:13s}] max|d(pos)/dt - vel| = {worst:.3e}  {'OK' if worst < 1e-5 else 'FAIL'}")
    # rigid consistency: every marker displaces by the same vector
    mo = cases[0][1]
    mk0, _ = mo(0.0); mk1, _ = mo(0.3)
    d = mk1 - mk0
    print(f"[rigid       ] per-marker displacement spread = {np.max(np.abs(d - d[0])):.3e}")
