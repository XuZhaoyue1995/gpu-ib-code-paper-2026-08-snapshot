"""ib_run.py — entry point for the modernized IB solver (deliverable program).

    python ib_run.py --dx 0.125 --steps 100 --re 100 [--gpu] [--sparse] [--out result]

Generates a Cartesian IB mesh, runs the Crank-Nicolson stream-function/vorticity
solver, and exposes a per-step callback hook for loosely-coupled ML control
(read velocity/forces -> controller -> adjust kinematics/BC).  CPU by default
(NumPy, oracle-validated); --gpu runs the identical code on an A100 via CuPy.
"""
import os
import sys
import time
import argparse


def main():
    ap = argparse.ArgumentParser(description="Modernized IB DNS solver")
    ap.add_argument("--dx", type=float, default=0.125, help="uniform cell size in [-1,1]^3")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--gpu", action="store_true", help="run on GPU via CuPy (IB_GPU=1)")
    ap.add_argument("--sparse", action="store_true", help="CPU: sparse-matrix CG matvec (~2-3x)")
    ap.add_argument("--out", type=str, default="", help="save final field to <out>.npz")
    args = ap.parse_args()

    if args.gpu:
        os.environ["IB_GPU"] = "1"                       # backend reads this at import
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.dirname(here))
    sys.argv = [sys.argv[0], "--nometis"]                # make_euler_mesh flag

    import numpy as np
    import make_euler_mesh as M
    from backend import xp, asnumpy, GPU
    from ib_core_np import IBCore
    from ib_driver_np import Driver

    mesh = M.make_nested_mesh(args.dx, [M._build_box(("center+size", (0, 0, 0), (2, 2, 2)))],
                              refinement_ratio=2)
    core = IBCore(mesh)
    drv = Driver(core, mesh, visc=1.0 / args.re)
    if args.sparse and not GPU:
        drv.enable_sparse(args.dt)
    print(f"backend={'GPU/cupy' if GPU else 'CPU/numpy'}  ncell={core.ncell}  nedge={core.nedge}  "
          f"Re={args.re}  dt={args.dt}")

    # initial stream function (Init_Stream LINEAR + small 3D perturbation)
    e = core.epos
    s_raw = e[:, 1] * core.te[:, 2] + 0.1 * xp.sin(e[:, 0]) * xp.sin(e[:, 1]) * xp.sin(e[:, 2])
    s, U, vc = drv.calc_vel(s_raw)
    s_old = s_raw.copy()

    def control_hook(step, s, vc):
        """Loosely-coupled ML hook: read state (vc / IB forces) -> controller ->
        adjust kinematics or BC for the next step.  No-op for the static case;
        wire an RL/control policy here (gradient need NOT pass through the solver)."""
        return None

    every = max(1, args.steps // 10)
    t0 = time.perf_counter()
    for n in range(1, args.steps + 1):
        s, vc, s_old = drv.step(s, s_old, args.dt, dt_old=args.dt, inner=3)
        control_hook(n, s, vc)
        if n % every == 0:
            mv = float(asnumpy(xp.max(xp.abs(vc[:core.ncell]))))
            print(f"  step {n:5d}  t={n * args.dt:7.3f}  max|vc|={mv:.5f}")
    ms = (time.perf_counter() - t0) / args.steps * 1e3
    print(f"done: {args.steps} steps, {ms:.1f} ms/step")

    if args.out:
        np.savez(args.out, s=asnumpy(s), vc=asnumpy(vc[:core.ncell]),
                 cpos=asnumpy(core.cpos), epos=asnumpy(core.epos))
        print(f"saved {args.out}.npz")


if __name__ == "__main__":
    main()
