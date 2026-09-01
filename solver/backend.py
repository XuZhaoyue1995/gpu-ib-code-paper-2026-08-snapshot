"""Array backend: NumPy (CPU, default, validated) or CuPy (GPU, set IB_GPU=1).

The whole solver writes `xp.*` instead of `np.*`; CPU and GPU share one code path.
When CUDA is unavailable the code defaults to NumPy. Set the environment variable
IB_GPU=1 to run the same array-level implementation through CuPy. Mesh generation
(make_euler_mesh) stays on the host; IBCore converts the operator arrays to `xp`.
"""
import os

GPU = os.environ.get("IB_GPU", "0") == "1"
# FP32 can improve throughput and halve storage on GPUs with limited FP64 rate.
# It also permits a finer mesh (higher D/dx) within a fixed memory budget.
FP32 = os.environ.get("IB_FP32", "0") == "1"

if GPU:                                   # CUDA/CuPy path
    import cupy as xp
    import cupyx.scipy.sparse as xsp
    import cupyx.scipy.sparse.linalg as xspla

    def asnumpy(a):
        return xp.asnumpy(a)
else:                                     # local / default path (validated)
    import numpy as xp
    import scipy.sparse as xsp
    import scipy.sparse.linalg as xspla

    def asnumpy(a):
        import numpy as _np
        return _np.asarray(a)


real = xp.float32 if FP32 else xp.float64   # working float dtype


def asarray(a):
    return xp.asarray(a)


def asreal(a):
    """To the active backend, casting FLOAT arrays to the working dtype (real);
    integer/bool index arrays pass through unchanged."""
    a = xp.asarray(a)
    return a.astype(real, copy=False) if a.dtype.kind == "f" else a
