"""
IBM nested-Cartesian mesh generator — single-file pipeline.

Usage:
    1. Edit the CONFIG block right below this docstring.
    2. Run:   python make_euler_mesh.py
    3. Outputs go to <OUT_DIR>:
         - <CASE_NAME>.h5         HDF5 main format (future JAX solver)
         - <CASE_NAME>.vtu        ParaView visualization
         - <CASE_NAME>_<rank>.{str,siz,msh,BC,cm}  per-rank solver input
         - <CASE_NAME>.mts        METIS input file (for the legacy
                                  partnmesh/mpmetis + Partition_3D toolchain;
                                  the modern pipeline does NOT need this)

Deps:
    numpy, h5py, meshio  -- always required.
    pymetis              -- REQUIRED. Install via:
                              conda install -c conda-forge pymetis
                            (or `pip install pymetis`). The script will refuse
                            to run without it, because the only fallback is
                            a pure-Python recursive bisection that produces
                            partitions roughly 2x worse in worst-rank comm
                            volume — silent 2x MPI slowdown is worse than a
                            clear failure. To opt in to the slow path despite
                            this, run as:  python make_euler_mesh.py --nometis

"""
from __future__ import annotations   # required by Python syntax to be first

# =============================================================================
#                                  CONFIG
#                       (edit only this block in normal use)
# =============================================================================

CASE_NAME = "test"

# --- Spacing & refinement: how the per-layer dx is computed ---
# INNER_SPACING is dx for the innermost (layer 0) uniform region. Each outer
# layer is COARSER by a fixed factor of 2:
#
#         dx[k] = INNER_SPACING * 2^k                (k = 0, 1, 2, ...)
#
# With INNER_SPACING=0.025 and 3 layers, dx is 0.025, 0.05, 0.1 from inside out.
#
# NOTE: the refinement ratio is HARDCODED to 2 (no CONFIG knob). The mesh
# generator itself supports r>=3 (see _finalize_hanging_node_connectivity), but
# the Fortran solver's Modify_geom (Geometry.f90, around line 166) is hardcoded
# with r=2 magic constants (0.36*Af, 4/9*Af) and won't correct hanging-node
# geometry for r>=3. Re-expose this as a CONFIG knob once the solver supports it.
INNER_SPACING    = 0.025

# --- Nested layers, INNERMOST FIRST ---
# How to write a valid BOXES list (given the dx schedule above):
#
# Add or remove entries to control how many layers you want; the mesh generator
# accepts any number of layers >= 1. Each entry describes one box in one of two
# forms (pick whichever reads naturally per box — they are interchangeable):
#
#   Mode 1:  ("center+size", center_tuple, size_tuple)
#       e.g.  ("center+size", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
#       →     cube of side 2 centered at origin, i.e. [-1, 1] x [-1, 1] x [-1, 1].
#       Best when the box is symmetric about a known center.
#
#   Mode 2:  ("bounds", (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi))
#       e.g.  ("bounds", (-5.0, 5.0, -5.0, 5.0, -4.0, 4.0))
#       →     asymmetric box: x in [-5, 5], y in [-5, 5], z in [-4, 4].
#       Best when you naturally think in terms of corners / extents.
#
# The generator will refuse the input with a clear error message if any of
# these geometric constraints is violated:
#   1. Strict nesting: box[k] must lie STRICTLY INSIDE box[k+1].
#   2. Grid-aligned size: each box[k]'s side length must be exactly divisible
#      by dx[k] (so layer k tiles cleanly: side / dx[k] is an integer).
#   3. Grid-aligned offset: the offset between box[k]'s corner and box[k+1]'s
#      corner must be exactly divisible by dx[k+1] (so layer k's nodes land on
#      layer k+1's grid where they overlap — this is what makes the hanging-
#      node 2:1 interface work).
#
# Tip: if the generator rejects your input, the error message tells you which
# axis failed which constraint and which dx made it fail.

BOXES = [
    # Layer 0 (innermost) — IBM workspace; uniform fine cells, Index_cell lives here
    ("center+size", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
    # Layer 1 — middle buffer
    ("bounds",      (-2.0, 2.0,  -2.0, 2.0,  -2.0, 2.0)),
    # Layer 2 (outermost) — far field; carries the domain boundary FAMs
    ("bounds",      (-5.0, 5.0,  -5.0, 5.0,  -4.0, 4.0)),
]

# Number of FLUID MPI ranks (solver runs with NUM_PARTS+1 total ranks --
# the +1 is the IB rank, rank 0).
NUM_PARTS = 255

# Output directory. Relative paths are resolved against the current working
# directory (where you ran `python make_euler_mesh.py` from), like any standard CLI tool.
# Absolute paths are used as-is. The resolved absolute path is printed below.
OUT_DIR = "./"

# Which output formats to write
WRITE_HDF5         = False   # main format, future JAX solver reads this
WRITE_VTU          = False   # ParaView visualization
WRITE_SOLVER_INPUT = True   # per-rank files for current Fortran solver
WRITE_MTS          = False   # emergency: feed external partnmesh/mpmetis manually

# Set True to also print internal progress (memory estimate, partition phases,
# detailed cell-volume statistics, etc.). Default is False — keeps the output
# focused on what you actually need to know. Turn on if a step seems stuck.
VERBOSE = False


# =============================================================================
#                              IMPLEMENTATION
#                            (do not edit below)
# =============================================================================

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import h5py
import meshio


# ----------------------------------------------------------------------------
# Specs
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class BoxSpec:
    """Axis-aligned box. Two ways to construct:

    Mode 1: BoxSpec(center=(0, 0, 0), size=(2.0, 2.0, 2.0))
    Mode 2: BoxSpec.from_bounds(-1, 1, -1, 1, -1, 1)
    """
    center: tuple
    size: tuple

    @classmethod
    def from_bounds(cls, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) -> "BoxSpec":
        if not (x_lo < x_hi and y_lo < y_hi and z_lo < z_hi):
            raise ValueError(
                f"from_bounds requires lo < hi in each axis; got "
                f"x=[{x_lo}, {x_hi}], y=[{y_lo}, {y_hi}], z=[{z_lo}, {z_hi}]"
            )
        center = ((x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0, (z_lo + z_hi) / 2.0)
        size = (x_hi - x_lo, y_hi - y_lo, z_hi - z_lo)
        return cls(center=center, size=size)

    @property
    def min(self) -> np.ndarray:
        return np.array(self.center, dtype=np.float64) - np.array(self.size, dtype=np.float64) / 2.0

    @property
    def max(self) -> np.ndarray:
        return np.array(self.center, dtype=np.float64) + np.array(self.size, dtype=np.float64) / 2.0


@dataclass(frozen=True)
class CombinedMesh:
    """Multi-layer nested Cartesian hex mesh with full topology + comm metadata.

    Connectivity is POST-aliasing: coarse interface faces have been removed,
    and coarse interface edges have been collapsed onto fine sub-edges per
    Patch_mesh's Modify_edge / Collapse_face convention. link_fe_insert holds
    the extra (face, fine_edge) entries that non-interface coarse faces gain
    when their referenced coarse edges decompose into multiple fine edges.
    """
    nodes: np.ndarray
    cells: np.ndarray
    n_layers: int
    layer_cell_counts: tuple
    layer_node_counts: tuple
    refinement_ratio: int
    boundary_faces: np.ndarray
    boundary_fam: np.ndarray
    pos_ref: np.ndarray
    dx_uniform: np.ndarray
    index_cell: np.ndarray
    faces_to_nodes: np.ndarray
    faces_to_cells: np.ndarray
    edges_to_nodes: np.ndarray
    faces_to_edges: np.ndarray
    bc_node: np.ndarray
    bc_edge: np.ndarray
    bc_face: np.ndarray
    bc_cell: np.ndarray
    interface_pairs: np.ndarray
    link_fe_insert: np.ndarray = field(  # (K, 2) -- extra link_fe entries beyond 4*num_face
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64))

    @property
    def n_inner_cells(self) -> int:
        return self.layer_cell_counts[0]

    @property
    def n_outer_cells(self) -> int:
        return int(self.cells.shape[0]) - self.layer_cell_counts[0]

    @property
    def n_outer_nodes(self) -> int:
        return int(self.nodes.shape[0]) - self.layer_node_counts[0]


# Hex-corner ordering (preserved from legacy MeshGenerator.py)
CELL_CORNER_ORDER = np.array([
    (1, 0, 1), (1, 0, 0), (0, 0, 1), (0, 0, 0),
    (1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 1, 0),
], dtype=np.int64)

LOCAL_FACES_OF_HEX = np.array([
    [3, 1, 0, 2],  # y=0 face
    [7, 5, 4, 6],  # y=+y face
    [3, 1, 5, 7],  # z=0 face
    [2, 0, 4, 6],  # z=+z face
    [3, 7, 6, 2],  # x=0 face
    [1, 5, 4, 0],  # x=+x face
], dtype=np.int64)

LOCAL_EDGES_OF_FACE = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
], dtype=np.int64)


# ----------------------------------------------------------------------------
# Validation + memory estimate
# ----------------------------------------------------------------------------

def _validate_nested_spec(inner_spacing, boxes, refinement_ratio):
    if not (isinstance(refinement_ratio, (int, np.integer)) and refinement_ratio >= 1):
        raise ValueError(f"refinement_ratio must be a positive integer, got {refinement_ratio}")
    N = len(boxes)
    if N < 1:
        raise ValueError("boxes must contain at least one BoxSpec")
    h = np.atleast_1d(np.asarray(inner_spacing, dtype=np.float64))
    if h.size == 1:
        v = float(h.flat[0])
        h = np.array([v, v, v], dtype=np.float64)
    if h.shape != (3,):
        raise ValueError(f"inner_spacing must be a scalar or 3-tuple, got shape {h.shape}")
    if (h <= 0).any():
        raise ValueError(f"inner_spacing must be positive, got {h}")
    r = int(refinement_ratio)
    dx = [h * (r ** k) for k in range(N)]
    n_cells = []
    i_lo = []
    for k, box in enumerate(boxes):
        sz = np.asarray(box.size, dtype=np.float64)
        ratio = sz / dx[k]
        n = np.round(ratio).astype(np.int64)
        if not np.allclose(sz, n * dx[k], atol=1e-8):
            bad = ", ".join(
                f"axis {ax} size {sz[ax]:.6g} / dx {dx[k][ax]:.6g} = {ratio[ax]:.6g} (not integer)"
                for ax in range(3) if not np.isclose(sz[ax], n[ax] * dx[k][ax], atol=1e-8)
            )
            raise ValueError(
                f"layer {k} box size not divisible by spacing dx={tuple(dx[k])}:\n  {bad}\n"
                f"  Adjust box {k} size or check inner_spacing."
            )
        n_cells.append(tuple(int(v) for v in n))
    for k in range(1, N):
        b_in = boxes[k-1]
        b_out = boxes[k]
        if (b_in.min < b_out.min - 1e-9).any() or (b_in.max > b_out.max + 1e-9).any():
            raise ValueError(
                f"layer {k-1} box must lie inside layer {k} box.\n"
                f"  layer {k-1}: min={tuple(b_in.min.tolist())}, max={tuple(b_in.max.tolist())}\n"
                f"  layer {k}:   min={tuple(b_out.min.tolist())}, max={tuple(b_out.max.tolist())}"
            )
        offset = b_in.min - b_out.min
        ratio = offset / dx[k]
        off_int = np.round(ratio).astype(np.int64)
        if not np.allclose(offset, off_int * dx[k], atol=1e-8):
            bad = ", ".join(
                f"axis {ax}: (inner_min {b_in.min[ax]:.6g} - outer_min {b_out.min[ax]:.6g}) / "
                f"dx {dx[k][ax]:.6g} = {ratio[ax]:.6g} (not integer)"
                for ax in range(3) if not np.isclose(offset[ax], off_int[ax] * dx[k][ax], atol=1e-8)
            )
            raise ValueError(
                f"layer {k-1} corners not aligned with layer {k}'s grid:\n  {bad}\n"
                f"  Adjust box {k-1} center/size or box {k} center."
            )
        i_lo.append(off_int)
    i_lo = [np.zeros(3, dtype=np.int64)] + i_lo
    return dx, n_cells, i_lo


def _report_memory_estimate(n_cells_per_layer, r):
    # For each layer k > 0, subtract the cells of the cavity (= box[k-1]
    # expressed in dx[k]'s grid). Since dx[k] = dx[k-1] * r and box[k-1].size
    # = n_cells[k-1] * dx[k-1], box[k-1] occupies (n_cells[k-1] / r) cells per
    # axis in layer k's grid.
    total_cells = 0
    for k, nc in enumerate(n_cells_per_layer):
        layer_cells = nc[0] * nc[1] * nc[2]
        if k > 0 and r > 1:
            prev = n_cells_per_layer[k-1]
            cavity = (prev[0] // r) * (prev[1] // r) * (prev[2] // r)
            layer_cells -= cavity
        total_cells += layer_cells
    n_nodes_est = total_cells
    n_faces_est = 3 * total_cells
    n_edges_est = 3 * total_cells
    bytes_total = (
        n_nodes_est * 3 * 8 +
        total_cells * 8 * 8 +
        n_faces_est * 4 * 8 +
        n_edges_est * 2 * 8 +
        n_faces_est * 4 * 8 +
        (8 * total_cells + 4 * n_faces_est // 2) * 2 * 8
    )
    # Auto-format memory: MB for small meshes, GB otherwise.
    if bytes_total < 1e9:
        mem_str = f"{bytes_total / 1e6:.1f} MB"
    else:
        mem_str = f"{bytes_total / 1e9:.2f} GB"
    print(f"[mesh] expected mesh size: {total_cells:,} cells, "
          f"peak memory ~{mem_str} (rough estimate)")


# ----------------------------------------------------------------------------
# Core builder
# ----------------------------------------------------------------------------

def make_nested_mesh(inner_spacing, boxes, refinement_ratio=2, *, verbose=False) -> CombinedMesh:
    dx, n_cells, i_lo = _validate_nested_spec(inner_spacing, boxes, refinement_ratio)
    r = int(refinement_ratio)
    N = len(boxes)

    if verbose:
        _report_memory_estimate(n_cells, r)

    node_shape = [(nc[0] + 1, nc[1] + 1, nc[2] + 1) for nc in n_cells]
    layer_min = [boxes[k].min for k in range(N)]

    layer_ids = [None] * N
    layer_node_counts = [0] * N
    next_id = 0

    k = N - 1
    layer_ids[k] = np.full(node_shape[k], -1, dtype=np.int64)
    if N > 1:
        n_in_k = np.round(np.asarray(boxes[k-1].size) / dx[k]).astype(np.int64)
        a_lo, b_lo, c_lo = i_lo[k]
        a_hi, b_hi, c_hi = a_lo + n_in_k[0], b_lo + n_in_k[1], c_lo + n_in_k[2]
        is_valid = np.ones(node_shape[k], dtype=bool)
        is_valid[a_lo+1:a_hi, b_lo+1:b_hi, c_lo+1:c_hi] = False
    else:
        is_valid = np.ones(node_shape[k], dtype=bool)
    n_new = int(is_valid.sum())
    layer_ids[k][is_valid] = np.arange(next_id, next_id + n_new, dtype=np.int64)
    layer_node_counts[k] = n_new
    next_id += n_new

    for k in range(N - 2, -1, -1):
        layer_ids[k] = np.full(node_shape[k], -1, dtype=np.int64)
        a_idx, b_idx, c_idx = np.indices(node_shape[k])
        coincident = (a_idx % r == 0) & (b_idx % r == 0) & (c_idx % r == 0)
        offset = i_lo[k + 1]
        ca = offset[0] + a_idx // r
        cb = offset[1] + b_idx // r
        cc = offset[2] + c_idx // r
        ca_c = ca[coincident]
        cb_c = cb[coincident]
        cc_c = cc[coincident]
        coarse_ids = layer_ids[k + 1][ca_c, cb_c, cc_c]
        layer_ids[k][coincident] = coarse_ids

        if k > 0:
            n_in_k = np.round(np.asarray(boxes[k-1].size) / dx[k]).astype(np.int64)
            a_lo2, b_lo2, c_lo2 = i_lo[k]
            a_hi2, b_hi2, c_hi2 = a_lo2 + n_in_k[0], b_lo2 + n_in_k[1], c_lo2 + n_in_k[2]
            in_next_inner = np.zeros(node_shape[k], dtype=bool)
            in_next_inner[a_lo2+1:a_hi2, b_lo2+1:b_hi2, c_lo2+1:c_hi2] = True
        else:
            in_next_inner = np.zeros(node_shape[k], dtype=bool)

        needs_new = (layer_ids[k] < 0) & ~in_next_inner
        n_new = int(needs_new.sum())
        layer_ids[k][needs_new] = np.arange(next_id, next_id + n_new, dtype=np.int64)
        layer_node_counts[k] = n_new
        next_id += n_new

    total_nodes = next_id

    nodes = np.zeros((total_nodes, 3), dtype=np.float64)
    for k in range(N):
        idx = layer_ids[k] >= 0
        a, b, c = np.where(idx)
        ids_here = layer_ids[k][a, b, c]
        coords = np.column_stack([
            layer_min[k][0] + a * dx[k][0],
            layer_min[k][1] + b * dx[k][1],
            layer_min[k][2] + c * dx[k][2],
        ])
        nodes[ids_here] = coords

    layer_cells = []
    layer_cell_counts = [0] * N
    for k in range(N):
        cell_mask = np.ones(n_cells[k], dtype=bool)
        if k > 0:
            n_in_k = np.round(np.asarray(boxes[k-1].size) / dx[k]).astype(np.int64)
            a_lo3, b_lo3, c_lo3 = i_lo[k]
            a_hi3, b_hi3, c_hi3 = a_lo3 + n_in_k[0], b_lo3 + n_in_k[1], c_lo3 + n_in_k[2]
            cell_mask[a_lo3:a_hi3, b_lo3:b_hi3, c_lo3:c_hi3] = False
        c_arr = _hex_cells_from_id_grid(layer_ids[k], cell_mask=cell_mask)
        layer_cells.append(c_arr)
        layer_cell_counts[k] = int(c_arr.shape[0])

    cells = np.concatenate(layer_cells, axis=0)

    boundary_faces, boundary_fam = _outer_box_boundary_faces(layer_ids[N - 1])

    inner_min = layer_min[0]
    pos_ref = inner_min + dx[0] / 2.0
    Nix, Niy, Niz = n_cells[0]
    index_cell = np.arange(Nix * Niy * Niz, dtype=np.int64).reshape(Nix, Niy, Niz)

    f2n, f2c, e2n, f2e, bc_node, bc_edge, bc_face, bc_cell = _build_connectivity(
        cells, boundary_faces, boundary_fam, total_nodes
    )

    interface_pairs = _detect_interface_pairs(
        cells, layer_cell_counts, layer_ids, n_cells, i_lo, r, f2n, f2c
    ) if r > 1 and N > 1 else np.zeros((0, 1 + r * r), dtype=np.int64)

    # Apply Patch_mesh hanging-node finalization eagerly so all downstream
    # writers see post-aliasing connectivity (coarse interface faces removed,
    # coarse interface edges collapsed onto fine sub-edges, link_fe_insert
    # generated for non-interface coarse faces that referenced collapsed edges).
    if r > 1 and N > 1 and interface_pairs.shape[0] > 0:
        f2n, f2c, f2e, bc_face, e2n, bc_edge, link_fe_insert = (
            _finalize_hanging_node_connectivity(
                f2n, f2c, f2e, e2n, bc_face, bc_edge, interface_pairs, nodes, r
            )
        )
        if verbose:
            print(f"[mesh] interface stitching: {f2n.shape[0]:,} faces, "
                  f"{e2n.shape[0]:,} edges, "
                  f"{link_fe_insert.shape[0]:,} extra face-edge links")
    else:
        link_fe_insert = np.zeros((0, 2), dtype=np.int64)

    return CombinedMesh(
        nodes=nodes, cells=cells,
        n_layers=N,
        layer_cell_counts=tuple(layer_cell_counts),
        layer_node_counts=tuple(layer_node_counts),
        refinement_ratio=r,
        boundary_faces=boundary_faces, boundary_fam=boundary_fam,
        pos_ref=pos_ref, dx_uniform=dx[0].copy(), index_cell=index_cell,
        faces_to_nodes=f2n, faces_to_cells=f2c,
        edges_to_nodes=e2n, faces_to_edges=f2e,
        bc_node=bc_node, bc_edge=bc_edge, bc_face=bc_face, bc_cell=bc_cell,
        interface_pairs=interface_pairs,
        link_fe_insert=link_fe_insert,
    )


def _hex_cells_from_id_grid(id_grid, cell_mask=None):
    Nx = id_grid.shape[0] - 1
    Ny = id_grid.shape[1] - 1
    Nz = id_grid.shape[2] - 1
    if cell_mask is None:
        cell_mask = np.ones((Nx, Ny, Nz), dtype=bool)
    corners = np.stack(
        [id_grid[ox:ox + Nx, oy:oy + Ny, oz:oz + Nz]
         for (ox, oy, oz) in CELL_CORNER_ORDER],
        axis=-1,
    )
    valid = cell_mask & (corners >= 0).all(axis=-1)
    return corners[valid].astype(np.int64, copy=False)


def _outer_box_boundary_faces(outer_id):
    Nox = outer_id.shape[0] - 1
    Noy = outer_id.shape[1] - 1
    Noz = outer_id.shape[2] - 1

    def stack4(c0, c1, c2, c3):
        return np.stack([c0, c1, c2, c3], axis=-1).reshape(-1, 4)

    fam1 = stack4(
        outer_id[0, 0:Noy,     1:Noz+1],
        outer_id[0, 1:Noy+1,   1:Noz+1],
        outer_id[0, 1:Noy+1,   0:Noz],
        outer_id[0, 0:Noy,     0:Noz],
    )
    fam2 = stack4(
        outer_id[Nox, 1:Noy+1, 1:Noz+1],
        outer_id[Nox, 0:Noy,   1:Noz+1],
        outer_id[Nox, 0:Noy,   0:Noz],
        outer_id[Nox, 1:Noy+1, 0:Noz],
    )
    fam3 = stack4(
        outer_id[0:Nox,   0:Noy,   Noz],
        outer_id[1:Nox+1, 0:Noy,   Noz],
        outer_id[1:Nox+1, 1:Noy+1, Noz],
        outer_id[0:Nox,   1:Noy+1, Noz],
    )
    fam4 = stack4(
        outer_id[0:Nox,   0:Noy,   0],
        outer_id[0:Nox,   1:Noy+1, 0],
        outer_id[1:Nox+1, 1:Noy+1, 0],
        outer_id[1:Nox+1, 0:Noy,   0],
    )
    fam5 = stack4(
        outer_id[0:Nox,   Noy, 1:Noz+1],
        outer_id[1:Nox+1, Noy, 1:Noz+1],
        outer_id[1:Nox+1, Noy, 0:Noz],
        outer_id[0:Nox,   Noy, 0:Noz],
    )
    fam6 = stack4(
        outer_id[1:Nox+1, 0, 1:Noz+1],
        outer_id[0:Nox,   0, 1:Noz+1],
        outer_id[0:Nox,   0, 0:Noz],
        outer_id[1:Nox+1, 0, 0:Noz],
    )

    groups = [fam1, fam2, fam3, fam4, fam5, fam6]
    faces = np.concatenate(groups, axis=0).astype(np.int64, copy=False)
    fam = np.concatenate(
        [np.full(g.shape[0], tag, dtype=np.int64) for tag, g in enumerate(groups, start=1)]
    )
    return faces, fam


def _build_connectivity(cells, boundary_faces, boundary_fam, n_nodes):
    M = cells.shape[0]
    all_faces = cells[:, LOCAL_FACES_OF_HEX].reshape(-1, 4)
    cell_of_face = np.repeat(np.arange(M, dtype=np.int64), 6)
    keys = np.sort(all_faces, axis=1)
    order = np.lexsort(keys.T[::-1])
    sorted_keys = keys[order]
    sorted_cells = cell_of_face[order]
    sorted_orig = all_faces[order]
    is_new = np.empty(sorted_keys.shape[0], dtype=bool)
    is_new[0] = True
    is_new[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    group_id = np.cumsum(is_new) - 1
    n_faces = int(group_id[-1] + 1)
    counts = np.bincount(group_id, minlength=n_faces)
    if (counts > 2).any():
        raise ValueError("non-manifold: a face is adjacent to >2 cells")
    starts = np.where(is_new)[0]
    F2N = sorted_orig[starts].copy()
    F2C = np.full((n_faces, 2), -1, dtype=np.int64)
    F2C[:, 0] = sorted_cells[starts]
    is_internal = counts == 2
    F2C[is_internal, 1] = sorted_cells[starts[is_internal] + 1]

    bc_face = np.zeros(n_faces, dtype=np.int64)
    if boundary_faces.shape[0] > 0:
        unique_keys = sorted_keys[starts]
        bf_keys = np.sort(boundary_faces, axis=1)
        row_bytes = unique_keys.shape[1] * 8
        keys_be = np.ascontiguousarray(unique_keys.astype('>u8'))
        bf_be = np.ascontiguousarray(bf_keys.astype('>u8'))
        keys_packed = keys_be.view(f"V{row_bytes}").ravel()
        bf_packed = bf_be.view(f"V{row_bytes}").ravel()
        fids = np.searchsorted(keys_packed, bf_packed)
        fids_clipped = np.minimum(fids, n_faces - 1)
        mismatch = (fids >= n_faces) | (keys_packed[fids_clipped] != bf_packed)
        if mismatch.any():
            i_bad = int(np.where(mismatch)[0][0])
            raise ValueError(f"boundary face {boundary_faces[i_bad].tolist()} not in enumerated faces")
        if (counts[fids] != 1).any():
            i_bad = int(np.where(counts[fids] != 1)[0][0])
            raise ValueError(f"boundary face matched an internal face (counts={counts[fids[i_bad]]})")
        bc_face[fids] = boundary_fam.astype(np.int64)

    face_edges = F2N[:, LOCAL_EDGES_OF_FACE].reshape(-1, 2)
    edge_keys = np.sort(face_edges, axis=1)
    e_order = np.lexsort(edge_keys.T[::-1])
    e_sorted_keys = edge_keys[e_order]
    e_is_new = np.empty(e_sorted_keys.shape[0], dtype=bool)
    e_is_new[0] = True
    e_is_new[1:] = np.any(e_sorted_keys[1:] != e_sorted_keys[:-1], axis=1)
    e_group_id = np.cumsum(e_is_new) - 1
    n_edges = int(e_group_id[-1] + 1)
    E2N = e_sorted_keys[np.where(e_is_new)[0]].copy()
    edge_id_flat = np.empty(face_edges.shape[0], dtype=np.int64)
    edge_id_flat[e_order] = e_group_id
    F2E = edge_id_flat.reshape(n_faces, 4)

    bc_node = np.zeros(n_nodes, dtype=np.int64)
    bc_edge = np.zeros(n_edges, dtype=np.int64)
    bc_cell = np.zeros(M, dtype=np.int64)
    bm = bc_face != 0
    if bm.any():
        b_F2N = F2N[bm]
        b_F2E = F2E[bm]
        b_fam = bc_face[bm]
        np.maximum.at(bc_node, b_F2N.ravel(), np.repeat(b_fam, 4))
        np.maximum.at(bc_edge, b_F2E.ravel(), np.repeat(b_fam, 4))

    return F2N, F2C, E2N, F2E, bc_node, bc_edge, bc_face, bc_cell


def _detect_interface_pairs(cells, layer_cell_counts, layer_ids, n_cells, i_lo, r, f2n, f2c):
    if r <= 1:
        return np.zeros((0, 1 + r * r), dtype=np.int64)
    f2n_sorted = np.sort(f2n, axis=1)
    key_to_face = {tuple(int(v) for v in k): i for i, k in enumerate(f2n_sorted)}
    pairs = []
    N = len(layer_cell_counts)
    for k in range(N - 1):
        a_lo = i_lo[k + 1]
        nk = np.asarray([n_cells[k][i] for i in range(3)], dtype=np.int64)
        n_in_kp1 = nk // r
        a_hi = a_lo + n_in_kp1
        cell_offset = sum(layer_cell_counts[:k])
        coarse_offset = sum(layer_cell_counts[:k + 1])
        ncp = n_cells[k + 1]
        cell_mask_kp1 = np.ones(ncp, dtype=bool)
        cell_mask_kp1[a_lo[0]:a_hi[0], a_lo[1]:a_hi[1], a_lo[2]:a_hi[2]] = False
        n_alive_kp1 = int(cell_mask_kp1.sum())
        cell_id_kp1 = np.full(ncp, -1, dtype=np.int64)
        cell_id_kp1[cell_mask_kp1] = np.arange(coarse_offset, coarse_offset + n_alive_kp1, dtype=np.int64)

        nck = n_cells[k]
        cell_mask_k = np.ones(nck, dtype=bool)
        if k > 0:
            i_loK = i_lo[k]
            nkm1_in_k = np.asarray(n_cells[k-1], dtype=np.int64) // r
            cell_mask_k[
                i_loK[0]:i_loK[0]+nkm1_in_k[0],
                i_loK[1]:i_loK[1]+nkm1_in_k[1],
                i_loK[2]:i_loK[2]+nkm1_in_k[2],
            ] = False
        n_alive_k = int(cell_mask_k.sum())
        cell_id_k = np.full(nck, -1, dtype=np.int64)
        cell_id_k[cell_mask_k] = np.arange(cell_offset, cell_offset + n_alive_k, dtype=np.int64)

        for side in ("x-", "x+", "y-", "y+", "z-", "z+"):
            _collect_interface_side(
                pairs, side, a_lo, a_hi, n_in_kp1, nk,
                cell_id_kp1, cell_id_k, layer_ids[k], layer_ids[k + 1],
                cells, f2n_sorted, key_to_face, r,
            )

    if not pairs:
        return np.zeros((0, 1 + r * r), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


def _collect_interface_side(pairs, side, a_lo, a_hi, n_in_kp1, nk,
                            cell_id_kp1, cell_id_k, layer_id_k, layer_id_kp1,
                            cells, f2n_sorted, key_to_face, r):
    side_info = {
        "x-": (0, False, 4, 5),
        "x+": (0, True,  5, 4),
        "y-": (1, False, 0, 1),
        "y+": (1, True,  1, 0),
        "z-": (2, False, 2, 3),
        "z+": (2, True,  3, 2),
    }
    axis, is_max, fine_local_face, coarse_local_face = side_info[side]
    other = [a for a in (0, 1, 2) if a != axis]
    if is_max:
        coarse_axis_idx = a_hi[axis]
    else:
        coarse_axis_idx = a_lo[axis] - 1
    coarse_range_o0 = range(a_lo[other[0]], a_hi[other[0]])
    coarse_range_o1 = range(a_lo[other[1]], a_hi[other[1]])
    for co0 in coarse_range_o0:
        for co1 in coarse_range_o1:
            coarse_idx = [0, 0, 0]
            coarse_idx[axis] = coarse_axis_idx
            coarse_idx[other[0]] = co0
            coarse_idx[other[1]] = co1
            coarse_id = cell_id_kp1[coarse_idx[0], coarse_idx[1], coarse_idx[2]]
            if coarse_id < 0:
                continue
            coarse_face_nodes = cells[coarse_id][LOCAL_FACES_OF_HEX[coarse_local_face]]
            cf_key = tuple(int(v) for v in np.sort(coarse_face_nodes))
            coarse_face_id = key_to_face.get(cf_key, -1)
            if coarse_face_id < 0:
                continue
            fine_axis_idx = (nk[axis] - 1) if is_max else 0
            fine_ids = []
            for d0 in range(r):
                for d1 in range(r):
                    fine_idx = [0, 0, 0]
                    fine_idx[axis] = fine_axis_idx
                    fine_idx[other[0]] = (co0 - a_lo[other[0]]) * r + d0
                    fine_idx[other[1]] = (co1 - a_lo[other[1]]) * r + d1
                    fid_cell = cell_id_k[fine_idx[0], fine_idx[1], fine_idx[2]]
                    if fid_cell < 0:
                        fine_ids = None
                        break
                    fine_face_nodes = cells[fid_cell][LOCAL_FACES_OF_HEX[fine_local_face]]
                    ff_key = tuple(int(v) for v in np.sort(fine_face_nodes))
                    fine_face_id = key_to_face.get(ff_key, -1)
                    if fine_face_id < 0:
                        fine_ids = None
                        break
                    fine_ids.append(fine_face_id)
                if fine_ids is None:
                    break
            if fine_ids is None or len(fine_ids) != r * r:
                continue
            pairs.append([coarse_face_id] + fine_ids)


# ----------------------------------------------------------------------------
# HDF5 + VTU + legacy writers
# ----------------------------------------------------------------------------

SCHEMA_VERSION = "0.3"
SCHEMA_KIND = "ibzhaoyue.combined_mesh"
_HDF5_CHUNK_THRESHOLD = 1_000_000
_HDF5_GZIP_LEVEL = 4

VTK_HEX_NODE_ORDER = np.array([3, 1, 0, 2, 7, 5, 4, 6], dtype=np.int64)


def _save_array(group, name, data):
    data = np.ascontiguousarray(data)
    if data.size >= _HDF5_CHUNK_THRESHOLD and data.ndim >= 1:
        first_chunk = max(1, _HDF5_CHUNK_THRESHOLD // max(1, int(np.prod(data.shape[1:]))))
        chunks = (min(first_chunk, data.shape[0]),) + data.shape[1:]
        group.create_dataset(name, data=data, chunks=chunks,
                              compression="gzip", compression_opts=_HDF5_GZIP_LEVEL)
    else:
        group.create_dataset(name, data=data)


def write_hdf5(mesh, path):
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["kind"] = SCHEMA_KIND
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        f.attrs["n_layers"] = mesh.n_layers
        f.attrs["refinement_ratio"] = mesh.refinement_ratio
        f.attrs["layer_cell_counts"] = np.asarray(mesh.layer_cell_counts, dtype=np.int64)
        f.attrs["layer_node_counts"] = np.asarray(mesh.layer_node_counts, dtype=np.int64)
        g = f.create_group("geometry")
        _save_array(g, "nodes", mesh.nodes)
        c = f.create_group("connectivity")
        _save_array(c, "cells", mesh.cells)
        _save_array(c, "boundary_faces", mesh.boundary_faces)
        _save_array(c, "boundary_fam", mesh.boundary_fam)
        _save_array(c, "faces_to_nodes", mesh.faces_to_nodes)
        _save_array(c, "faces_to_cells", mesh.faces_to_cells)
        _save_array(c, "edges_to_nodes", mesh.edges_to_nodes)
        _save_array(c, "faces_to_edges", mesh.faces_to_edges)
        b = f.create_group("bc")
        _save_array(b, "bc_node", mesh.bc_node)
        _save_array(b, "bc_edge", mesh.bc_edge)
        _save_array(b, "bc_face", mesh.bc_face)
        _save_array(b, "bc_cell", mesh.bc_cell)
        i = f.create_group("ibm")
        _save_array(i, "index_cell", mesh.index_cell)
        _save_array(i, "pos_ref", mesh.pos_ref)
        _save_array(i, "dx_uniform", mesh.dx_uniform)
        if mesh.interface_pairs.shape[0] > 0:
            ip = f.create_group("interface")
            _save_array(ip, "coarse_to_fine", mesh.interface_pairs)


def write_vtu(mesh, path):
    vtk_hex = mesh.cells[:, VTK_HEX_NODE_ORDER]
    cells_block = [("hexahedron", vtk_hex), ("quad", mesh.boundary_faces)]
    region_hex = np.zeros(mesh.cells.shape[0], dtype=np.int32)
    region_hex[:mesh.n_inner_cells] = 1
    region_quad = np.zeros(mesh.boundary_faces.shape[0], dtype=np.int32)
    fam_hex = np.zeros(mesh.cells.shape[0], dtype=np.int32)
    fam_quad = mesh.boundary_fam.astype(np.int32)
    cell_data = {
        "region": [region_hex, region_quad],
        "fam":    [fam_hex, fam_quad],
    }
    point_data = {"bc_node": mesh.bc_node.astype(np.int32)}
    m = meshio.Mesh(points=mesh.nodes, cells=cells_block,
                     point_data=point_data, cell_data=cell_data)
    m.write(str(path))


def _finalize_hanging_node_connectivity(F2N, F2C, F2E, E2N, bc_face, bc_edge,
                                         interface_pairs, nodes, r):
    """Apply Patch_mesh-style hanging-node finalization to the raw connectivity.

    Two operations applied jointly:

    A) Face aliasing: each coarse interface face is removed from F2N/F2C/F2E/bc_face;
       each of its r*r fine sub-faces gets F2C(2) = coarse cell.

    B) Edge collapse + link_fe_insert: each coarse interface edge has (r-1)
       hanging nodes lying on it at parametric positions i/r (i=1..r-1) — for
       r=2 that's a single midpoint, for r=3 the 1/3 and 2/3 points, etc. —
       and physically equals r fine sub-edges. Non-interface coarse faces
       (those still in F2E after aliasing) that reference this coarse edge get
       the reference replaced by the FIRST fine sub-edge; the remaining (r-1)
       fine sub-edges are emitted as link_fe_insert rows (face, sub_edge_id).
       Finally the coarse interface edges are removed from E2N and remaining
       edges are renumbered.

    Matches Patch_mesh/source/Patch.f90 (Modify_edge + Collapse_face) for r=2;
    naturally generalizes to r>=3 by walking (r-1) hanging nodes per coarse
    edge instead of one midpoint.

    Returns: F2N, F2C, F2E, bc_face, E2N, bc_edge, link_fe_insert
    """
    # ---- A. Face aliasing -------------------------------------------------
    if interface_pairs.shape[0] > 0:
        coarse_ids = interface_pairs[:, 0]
        fine_groups = interface_pairs[:, 1:]
        coarse_cells = F2C[coarse_ids, 0]
        F2C_new = F2C.copy()
        for j in range(fine_groups.shape[1]):
            fine_ids = fine_groups[:, j]
            if (F2C_new[fine_ids, 1] != -1).any():
                raise ValueError("interface fine face already has a second neighbor")
            F2C_new[fine_ids, 1] = coarse_cells
        keep_face = np.ones(F2N.shape[0], dtype=bool)
        keep_face[coarse_ids] = False
    else:
        F2C_new = F2C
        keep_face = np.ones(F2N.shape[0], dtype=bool)

    # ---- B. Edge collapse + link_fe_insert -------------------------------
    n_edge_old = E2N.shape[0]
    F2E_new = F2E.copy()
    link_fe_insert = np.zeros((0, 2), dtype=np.int64)

    if r >= 2 and interface_pairs.shape[0] > 0:
        # Detect coarse interface edges and build their r sub-edges.
        # An edge from n1 to n2 is a coarse interface edge iff there is a node
        # at parametric position t = 1/r along it (the first hanging node).
        # For r=2 that's the midpoint; for r=3 it's the 1/3 point; etc.
        # Quantize positions by min-edge-length * 1e-3 for robust matching.
        edge_vec = nodes[E2N[:, 1]] - nodes[E2N[:, 0]]
        edge_len = np.linalg.norm(edge_vec, axis=1)
        eps = float(edge_len.min()) * 1e-3
        node_key = np.round(nodes / eps).astype(np.int64)
        node_dict = {tuple(k): i for i, k in enumerate(map(tuple, node_key))}

        t_probe = 1.0 / r
        probe_pos = nodes[E2N[:, 0]] + t_probe * (nodes[E2N[:, 1]] - nodes[E2N[:, 0]])
        probe_key = np.round(probe_pos / eps).astype(np.int64)

        coarse_mask = np.zeros(n_edge_old, dtype=bool)
        for ei in range(n_edge_old):
            k = tuple(probe_key[ei])
            if k in node_dict:
                hn = node_dict[k]
                if hn != int(E2N[ei, 0]) and hn != int(E2N[ei, 1]):
                    coarse_mask[ei] = True

        coarse_iface = np.where(coarse_mask)[0]

        if coarse_iface.size > 0:
            # Build (sorted node-pair) -> edge_id dictionary
            edge_dict = {}
            for i in range(n_edge_old):
                a, b = int(E2N[i, 0]), int(E2N[i, 1])
                edge_dict[(min(a, b), max(a, b))] = i

            # For each coarse interface edge, locate the r-1 hanging nodes at
            # parametric positions i/r (i=1..r-1) and assemble r sub-edges that
            # connect consecutive points (n1, h_1, h_2, ..., h_{r-1}, n2).
            coarse_to_fine = {}
            for ce in coarse_iface:
                n1 = int(E2N[ce, 0]); n2 = int(E2N[ce, 1])
                p1 = nodes[n1]; p2 = nodes[n2]
                points = [n1]
                ok = True
                for i in range(1, r):
                    t = i / r
                    pos = p1 + t * (p2 - p1)
                    pk = tuple(np.round(pos / eps).astype(np.int64))
                    if pk not in node_dict:
                        ok = False
                        break
                    points.append(node_dict[pk])
                if not ok:
                    continue
                points.append(n2)

                sub_edges = []
                for j in range(r):
                    a, b = points[j], points[j + 1]
                    key = (min(a, b), max(a, b))
                    if key not in edge_dict:
                        sub_edges = None
                        break
                    sub_edges.append(edge_dict[key])
                if sub_edges is not None:
                    coarse_to_fine[int(ce)] = sub_edges

            # Update F2E entries that reference coarse edges → first fine edge;
            # collect link_fe_insert rows for the remaining r-1 fine edges.
            insert_rows = []
            for face_id in range(F2E.shape[0]):
                if not keep_face[face_id]:
                    continue
                for slot in range(F2E.shape[1]):
                    ce = int(F2E_new[face_id, slot])
                    if ce in coarse_to_fine:
                        fine_list = coarse_to_fine[ce]
                        F2E_new[face_id, slot] = fine_list[0]
                        for fe in fine_list[1:]:
                            insert_rows.append((face_id, fe))

            # Remove the coarse interface edges; renumber remaining edges.
            keep_edge = np.ones(n_edge_old, dtype=bool)
            keep_edge[list(coarse_to_fine.keys())] = False
            edge_old_to_new = np.full(n_edge_old, -1, dtype=np.int64)
            edge_old_to_new[keep_edge] = np.arange(int(keep_edge.sum()), dtype=np.int64)

            F2E_new = edge_old_to_new[F2E_new]
            if (F2E_new[keep_face] < 0).any():
                raise ValueError("a kept face still references a removed coarse edge")

            E2N_new = E2N[keep_edge]
            bc_edge_new = bc_edge[keep_edge]

            link_fe_insert = (np.array(insert_rows, dtype=np.int64)
                              if insert_rows else np.zeros((0, 2), dtype=np.int64))
            if link_fe_insert.size:
                link_fe_insert[:, 1] = edge_old_to_new[link_fe_insert[:, 1]]
        else:
            E2N_new = E2N
            bc_edge_new = bc_edge
    else:
        E2N_new = E2N
        bc_edge_new = bc_edge

    # Apply face mask and renumber link_fe_insert face IDs to post-aliasing space.
    face_old_to_new = np.full(F2N.shape[0], -1, dtype=np.int64)
    face_old_to_new[keep_face] = np.arange(int(keep_face.sum()), dtype=np.int64)
    if link_fe_insert.size:
        link_fe_insert[:, 0] = face_old_to_new[link_fe_insert[:, 0]]
        if (link_fe_insert < 0).any():
            raise ValueError("link_fe_insert references a removed face")

    F2N_new = F2N[keep_face]
    F2C_new = F2C_new[keep_face]
    F2E_new = F2E_new[keep_face]
    bc_face_new = bc_face[keep_face]

    return F2N_new, F2C_new, F2E_new, bc_face_new, E2N_new, bc_edge_new, link_fe_insert


def _write_10I10(f, arr):
    """Write integer array in Fortran 10I10 format.
    For EMPTY arrays, still emit a blank line — Fortran list-directed Read(q,*)
    expects a record separator even for zero-length arrays. Without this, the
    next Read consumes the WRONG record and the file appears truncated (EOF).
    """
    flat = np.asarray(arr).ravel()
    if flat.size == 0:
        f.write("\n")
        return
    for i in range(0, flat.size, 10):
        f.write("".join(f"{int(v):10d}" for v in flat[i:i + 10]) + "\n")


def write_legacy_mesh_files(mesh, prefix):
    """Write the GLOBAL legacy-format files. mesh is already post-aliasing
    (faces / edges / link_fe_insert) per Patch_mesh convention."""
    prefix = Path(prefix)
    n_vol = mesh.cells.shape[0]
    n_bd = mesh.boundary_faces.shape[0]
    n_node = mesh.nodes.shape[0]
    n_edge = mesh.edges_to_nodes.shape[0]
    n_face = mesh.faces_to_nodes.shape[0]
    n_cell = n_vol + n_bd
    max_npvc = 8
    link_fe_insert = (mesh.link_fe_insert
                      if mesh.link_fe_insert is not None
                      else np.zeros((0, 2), dtype=np.int64))
    num_link_fe = 4 * n_face + int(link_fe_insert.shape[0])
    num_link_cn = 8 * n_vol + 4 * n_bd

    with open(f"{prefix}.siz", "w", newline="\n") as f:
        for val in (max_npvc, 3, n_node, n_edge, n_face, n_cell, num_link_fe, num_link_cn):
            f.write(f"{val:12d}\n")

    face_ids = np.repeat(np.arange(1, n_face + 1, dtype=np.int64), 4)
    edge_ids = (mesh.faces_to_edges.ravel() + 1).astype(np.int64)
    link_fe_regular = np.column_stack([face_ids, edge_ids])
    if link_fe_insert.shape[0] > 0:
        insert_1idx = link_fe_insert.copy()
        insert_1idx += 1
        link_fe = np.concatenate([link_fe_regular, insert_1idx], axis=0)
    else:
        link_fe = link_fe_regular
    vol_cell = np.repeat(np.arange(1, n_vol + 1, dtype=np.int64), 8)
    vol_node = (mesh.cells.ravel() + 1).astype(np.int64)
    bd_cell = np.repeat(np.arange(n_vol + 1, n_vol + n_bd + 1, dtype=np.int64), 4)
    bd_node = (mesh.boundary_faces.ravel() + 1).astype(np.int64)
    link_cn = np.column_stack([
        np.concatenate([vol_cell, bd_cell]),
        np.concatenate([vol_node, bd_node]),
    ])
    E2N = mesh.edges_to_nodes + 1
    # F2C 1-indexed. Boundary faces (faces_to_cells[:,1]==-1) must get the
    # boundary-face-element ID (n_vol+bf+1) in column 2, NOT 0 -- this matches
    # Patch_mesh after Prelabel_cell+Collapse_cell, and Partition_3D's get_part
    # relies on F2C never being 0. (Mirrors the per-rank F2C_ext convention.)
    F2C = mesh.faces_to_cells + 1
    if n_bd > 0:
        rb = 4 * 8
        f2n_keys = np.ascontiguousarray(np.sort(mesh.faces_to_nodes, axis=1).astype('>u8')).view(f"V{rb}").ravel()
        bf_keys = np.ascontiguousarray(np.sort(mesh.boundary_faces, axis=1).astype('>u8')).view(f"V{rb}").ravel()
        order = np.argsort(f2n_keys)
        bf_to_face_id = order[np.searchsorted(f2n_keys[order], bf_keys)]
        F2C[bf_to_face_id, 1] = n_vol + np.arange(1, n_bd + 1, dtype=np.int64)
    F2C[F2C < 0] = 0

    with open(f"{prefix}.msh", "w", newline="\n") as f:
        for row in mesh.nodes:
            f.write(f"{row[0]:25.16E} {row[1]:25.16E} {row[2]:25.16E}\n")
        _write_10I10(f, E2N)
        _write_10I10(f, F2C)
        _write_10I10(f, link_fe)
        _write_10I10(f, link_cn)

    bc_cell_out = np.concatenate([
        np.zeros(n_vol, dtype=np.int64),
        mesh.boundary_fam.astype(np.int64),
    ])
    with open(f"{prefix}.BC", "w", newline="\n") as f:
        _write_10I10(f, mesh.bc_node)
        _write_10I10(f, mesh.bc_edge)
        _write_10I10(f, mesh.bc_face)
        _write_10I10(f, bc_cell_out)


def write_mts(mesh, path, *, metis5_compatible=False):
    """METIS input file. Default produces Patch_mesh-compatible "<n> 3" header;
    set metis5_compatible=True to omit the type code (for mpmetis)."""
    path = Path(path)
    n_vol = mesh.cells.shape[0]
    with open(path, "w", newline="\n") as f:
        if metis5_compatible:
            f.write(f"{n_vol:10d}\n")
        else:
            f.write(f"{n_vol:10d}{3:10d}\n")
        for cell in mesh.cells:
            f.write("".join(f"{int(n)+1:12d}" for n in cell) + "\n")


# ----------------------------------------------------------------------------
# Partition (METIS / pure-Python fallback)
# ----------------------------------------------------------------------------

def _partition_cells(mesh, num_parts, *, gtype="nodal", seed=42,
                     allow_fallback=False):
    """Return (cell_partition, node_partition, backend_name).

    REQUIRES pymetis. If not installed, raises ImportError with an actionable
    message. We intentionally do NOT silently fall back to pure-Python recursive
    bisection: measurements at NUM_PARTS=255 (see _comm_load_compare.py) show
    the fallback's worst-rank comm volume roughly DOUBLES, which translates to
    ~2x slower MPI runs. Silent slowdown is worse than a clear failure.

    If you absolutely must run without pymetis (e.g. air-gapped HPC), opt in
    explicitly via one of:
      - CLI:     `python make_euler_mesh.py --nometis`  (main() picks it up from sys.argv)
      - library: pass allow_fallback=True to this function.
    The slow path will then be used with a loud warning.
    """
    n_cells = mesh.cells.shape[0]
    if num_parts < 1:
        raise ValueError("num_parts must be >= 1")
    if num_parts == 1:
        print("[partition] backend: trivial (num_parts=1, all cells to rank 0)")
        return (np.zeros(n_cells, dtype=np.int64),
                np.zeros(mesh.nodes.shape[0], dtype=np.int64),
                "trivial")

    # CLI users can pass --nometis on the command line; library users pass
    # allow_fallback=True. Either route reaches the same fallback path.
    fallback_ok = allow_fallback or ("--nometis" in sys.argv)

    try:
        import pymetis  # type: ignore
    except ImportError as exc:
        if fallback_ok:
            print("=" * 72)
            print("[partition] WARNING: pymetis missing; using pure-Python bisection.")
            print("            This typically makes MPI runs ~2x slower (worst-rank")
            print("            comm volume doubles). Install pymetis for production:")
            print("                conda install -c conda-forge pymetis")
            print("=" * 72)
            result = _partition_python_bisection(mesh, num_parts, seed=seed)
            print(f"[partition] backend: pure-python (recursive spatial bisection)")
            return result, None, "pure-python"
        raise ImportError(
            "pymetis is required for mesh partitioning but is not installed.\n"
            "\n"
            "Install it with one of:\n"
            "    conda install -c conda-forge pymetis    (recommended)\n"
            "    pip install pymetis\n"
            "\n"
            "Why pymetis is required: the pure-Python bisection fallback\n"
            "produces partitions whose worst-rank communication volume is ~2x\n"
            "larger than METIS's, which directly slows MPI runs by ~2x. The\n"
            "fallback is kept for emergency use only. To opt in despite the\n"
            "performance hit, rerun as:  python make_euler_mesh.py --nometis"
        ) from exc

    cell_part, node_part = _partition_pymetis(mesh, num_parts, gtype=gtype, seed=seed)
    print(f"[partition] using metis (pymetis)")
    return cell_part, node_part, "pymetis"


def _partition_pymetis(mesh, num_parts, *, gtype, seed):
    """Use pymetis.part_mesh which mirrors METIS's mpmetis CLI: takes mesh
    connectivity (cells -> 8 nodes each) and returns BOTH element_part AND
    vertex_part. This matches what mpmetis writes to .mts.epart.N and
    .mts.npart.N -- the inputs Partition_3D consumes in the legacy workflow.
    """
    import pymetis  # type: ignore
    options = None
    try:
        options = pymetis.Options()
        if hasattr(options, "seed"):
            options.seed = seed
    except Exception:
        options = None
    # part_mesh expects connectivity as list-of-lists; mesh.cells is (n_vol, 8)
    connectivity = mesh.cells.tolist()
    gtype_enum = pymetis.GType.NODAL if gtype == "nodal" else pymetis.GType.DUAL
    kwargs = {"n_parts": num_parts, "connectivity": connectivity,
              "options": options, "gtype": gtype_enum}
    if gtype == "dual":
        kwargs["ncommon"] = 4  # hex face has 4 nodes
    result = pymetis.part_mesh(**kwargs)
    cell_part = np.asarray(result.element_part, dtype=np.int64)
    node_part = np.asarray(result.vertex_part, dtype=np.int64)
    return cell_part, node_part


def _partition_python_bisection(mesh, num_parts, *, seed):
    # Deterministic recursive spatial bisection (sorted split on the longest
    # axis). `seed` is accepted only for API parity with the pymetis path; the
    # split is deterministic, so it has no effect.
    del seed
    n_cells = mesh.cells.shape[0]
    centers = mesh.nodes[mesh.cells].mean(axis=1)
    ranks = np.zeros(n_cells, dtype=np.int64)
    _bisect_recursive(centers, ranks, np.arange(n_cells, dtype=np.int64), 0, num_parts)
    return ranks


def _bisect_recursive(centers, ranks, indices, rank_base, k):
    if k == 1:
        ranks[indices] = rank_base
        return
    sub = centers[indices]
    spans = sub.max(axis=0) - sub.min(axis=0)
    axis = int(np.argmax(spans))
    half = k // 2
    order = np.argsort(sub[:, axis], kind="stable")
    n_left = (len(indices) * (k - half)) // k
    left = indices[order[:n_left]]
    right = indices[order[n_left:]]
    _bisect_recursive(centers, ranks, left, rank_base, k - half)
    _bisect_recursive(centers, ranks, right, rank_base + (k - half), half)


# ----------------------------------------------------------------------------
# Per-rank mesh + communication tables
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class _LocalMesh:
    """Per-rank local mesh -- Partition_3D OWNED-only convention.

    All entities are OWNED by this rank. No ghosts. Cross-rank references in
    link tables (E2N, F2C, link_fe, link_cn) use 0 in the second column
    (matches localize() in Partition_3D/localization.F90).
    Local IDs are 1-indexed.
    """
    rank: int
    nodes_local: np.ndarray        # (num_node, 3) -- OWNED nodes only
    e2n_local: np.ndarray          # (num_edge, 2) -- OWNED edges, second col = 0 if cross-rank
    f2c_local: np.ndarray          # (num_face, 2) -- OWNED faces, second col = 0 if cross-rank
    link_fe_local: np.ndarray      # (num_link_fe, 2) -- OWNED, second col = 0 if cross-rank
    link_cn_local: np.ndarray      # (num_link_cn, 2) -- OWNED, second col = 0 if cross-rank
    bc_node: np.ndarray            # (num_node+1,) leading -1 dummy + OWNED BC values
    bc_edge: np.ndarray            # (num_edge+1,)
    bc_face: np.ndarray            # (num_face+1,)
    bc_cell: np.ndarray            # (num_cell+1,) -- bd face elements carry FAM tag
    nd: int
    ncn_max: int
    num_node: int
    num_edge: int
    num_face: int
    num_cell: int
    num_link_fe: int
    num_link_cn: int


def _compute_global_partitions(mesh, cell_to_rank_mpi, node_to_rank_mpi=None):
    """Compute partition assignment for ALL entity types following Partition_3D's rules.

    Rules (from partitions.F90:get_part):
        Part_edge[i]   = Part_node[E2N(1, i)]                   # first endpoint
        Part_face[i]   = Part_cell[F2C(1, i)]                   # first cell
        Part_link_fe[i] = Part_face[link_fe(1, i)]              # the face of this link
        Part_link_cn[i] = Part_cell[link_cn(1, i)]              # the cell of this link
        # Boundary face elements (legacy "cells" with BC_face!=0) inherit:
        Part_cell_ext[bf_element_id] = Part_cell[adjacent_volume]

    Part_node SOURCE:
      - If node_to_rank_mpi is given (METIS's vertex_part + 1, the normal path),
        use it directly. This matches the legacy MeshGenerator -> 3Dconvert ->
        Patch_mesh -> mpmetis -> Partition_3D workflow.
      - Else fall back to min(Part_cell touching). This branch is only reachable
        when the user explicitly opts in to the bisection fallback (via
        `python make_euler_mesh.py --nometis`), because the bisection backend doesn't
        return a node partition. The fallback biases low ranks to own most
        shared nodes and inflates cross-rank comm-table sizes at scale --
        safe only for small debug runs.

    Returns dict with:
        Part_cell_ext : (n_volumes + n_bd,) — 1..num_parts. Legacy "cells" partition.
        Part_node     : (n_nodes,)
        Part_edge     : (n_edges,)
        Part_face     : (n_faces,)
        Part_link_fe  : (4 * n_faces,) — global link_fe table partition
        Part_link_cn  : (8 * n_volumes + 4 * n_bd,) — global link_cn table partition
        bf_to_face_id : (n_bd,) — for each boundary_faces row, its face_id in faces_to_nodes
        F2C_ext       : (n_faces, 2) — F2C with second column filled with bd_face_element ID
                         for boundary faces (instead of -1)
    """
    n_volumes = mesh.cells.shape[0]
    n_bd      = mesh.boundary_faces.shape[0]
    n_nodes   = mesh.nodes.shape[0]
    n_edges   = mesh.edges_to_nodes.shape[0]
    n_faces   = mesh.faces_to_nodes.shape[0]

    # bf_to_face_id: lookup via sorted-node-key
    f2n_sorted = np.sort(mesh.faces_to_nodes, axis=1).astype('>u8')
    bf_sorted  = np.sort(mesh.boundary_faces, axis=1).astype('>u8')
    row_bytes  = 4 * 8
    f2n_keys = np.ascontiguousarray(f2n_sorted).view(f"V{row_bytes}").ravel()
    bf_keys  = np.ascontiguousarray(bf_sorted ).view(f"V{row_bytes}").ravel()
    order = np.argsort(f2n_keys)
    sorted_keys = f2n_keys[order]
    pos = np.searchsorted(sorted_keys, bf_keys)
    bf_to_face_id = order[pos]                              # global face id per boundary_faces row

    # Extended Part_cell: volumes (from METIS) + boundary face elements (inherit from adjacent volume)
    Part_cell_ext = np.zeros(n_volumes + n_bd, dtype=np.int64)
    Part_cell_ext[:n_volumes] = cell_to_rank_mpi
    if n_bd > 0:
        bf_adj_vol = mesh.faces_to_cells[bf_to_face_id, 0]  # adjacent volume cell (always >= 0)
        Part_cell_ext[n_volumes:] = cell_to_rank_mpi[bf_adj_vol]

    # F2C_ext: replace -1 in second col with the bd face element's id
    F2C_ext = mesh.faces_to_cells.copy()
    # For each boundary face's global face id, set F2C_ext[face_id, 1] = n_volumes + bf_idx
    F2C_ext[bf_to_face_id, 1] = n_volumes + np.arange(n_bd, dtype=np.int64)

    # Part_node: use METIS output if provided, else fall back to min(touching cells)
    if node_to_rank_mpi is not None:
        Part_node = np.asarray(node_to_rank_mpi, dtype=np.int64)
        if Part_node.shape[0] != n_nodes:
            raise ValueError(
                f"node_to_rank_mpi length {Part_node.shape[0]} != n_nodes {n_nodes}"
            )
    else:
        INF = np.iinfo(np.int64).max
        Part_node = np.full(n_nodes, INF, dtype=np.int64)
        cell_ranks_rep = np.repeat(cell_to_rank_mpi, 8)
        np.minimum.at(Part_node, mesh.cells.ravel(), cell_ranks_rep)

    # Part_edge: from first endpoint's partition
    Part_edge = Part_node[mesh.edges_to_nodes[:, 0]]

    # Part_face: from first cell's partition (F2C[:, 0] always points to a volume)
    Part_face = cell_to_rank_mpi[mesh.faces_to_cells[:, 0]]

    # Part_link_fe: 4 per face from the regular F2E, plus 1 per link_fe_insert
    # entry (those carry the face's partition since the insert row's first
    # column is a face ID).
    link_fe_insert = (mesh.link_fe_insert
                      if mesh.link_fe_insert is not None
                      else np.zeros((0, 2), dtype=np.int64))
    Part_link_fe_regular = np.repeat(Part_face, 4)
    if link_fe_insert.shape[0] > 0:
        Part_link_fe_insert = Part_face[link_fe_insert[:, 0]]
        Part_link_fe = np.concatenate([Part_link_fe_regular, Part_link_fe_insert])
    else:
        Part_link_fe = Part_link_fe_regular

    # Part_link_cn: 8 per volume + 4 per bd face element, each sharing the cell's partition
    Part_link_cn = np.concatenate([
        np.repeat(cell_to_rank_mpi, 8),
        np.repeat(Part_cell_ext[n_volumes:], 4) if n_bd > 0 else np.empty(0, dtype=np.int64),
    ])

    return {
        "Part_cell_ext": Part_cell_ext,
        "Part_node":     Part_node,
        "Part_edge":     Part_edge,
        "Part_face":     Part_face,
        "Part_link_fe":  Part_link_fe,
        "Part_link_cn":  Part_link_cn,
        "bf_to_face_id": bf_to_face_id,
        "F2C_ext":       F2C_ext,
    }


def _build_per_rank_data(mesh, num_parts, cell_to_rank_mpi, *,
                          node_to_rank_mpi=None, verbose=False):
    """Build per-rank OWNED-only local meshes + comm tables.

    Matches Partition_3D's localize()/write_parts() convention exactly:
      - .siz declares OWNED counts (no ghosts)
      - Link tables (E2N/F2C/link_fe/link_cn) store OWNED entries only;
        cross-rank references in column 2 are encoded as 0
      - BC arrays: dummy -1 followed by OWNED values

    Returns (locals_list, comm_tables_list), index 0 = fluid rank 1.
    """
    parts = _compute_global_partitions(mesh, cell_to_rank_mpi,
                                        node_to_rank_mpi=node_to_rank_mpi)
    Part_cell_ext = parts["Part_cell_ext"]
    Part_node     = parts["Part_node"]
    Part_edge     = parts["Part_edge"]
    Part_face     = parts["Part_face"]
    Part_link_fe  = parts["Part_link_fe"]
    Part_link_cn  = parts["Part_link_cn"]
    F2C_ext       = parts["F2C_ext"]

    n_volumes = mesh.cells.shape[0]
    n_bd      = mesh.boundary_faces.shape[0]

    # Global link tables (0-indexed). The link_fe table is the regular
    # 4*num_face entries plus link_fe_insert (extra (face, fine_edge) rows
    # generated by hanging-node edge collapse for non-interface coarse faces).
    global_link_fe_regular = np.column_stack([
        np.repeat(np.arange(mesh.faces_to_nodes.shape[0]), 4),
        mesh.faces_to_edges.ravel(),
    ])
    link_fe_insert = (mesh.link_fe_insert
                      if mesh.link_fe_insert is not None
                      else np.zeros((0, 2), dtype=np.int64))
    if link_fe_insert.shape[0] > 0:
        global_link_fe = np.concatenate([global_link_fe_regular, link_fe_insert], axis=0)
    else:
        global_link_fe = global_link_fe_regular
    if n_bd > 0:
        cn_cells = np.concatenate([
            np.repeat(np.arange(n_volumes), 8),
            np.repeat(np.arange(n_volumes, n_volumes + n_bd), 4),
        ])
        cn_nodes = np.concatenate([mesh.cells.ravel(), mesh.boundary_faces.ravel()])
    else:
        cn_cells = np.repeat(np.arange(n_volumes), 8)
        cn_nodes = mesh.cells.ravel()
    global_link_cn = np.column_stack([cn_cells, cn_nodes])

    # Per-rank OWNED indices + G2L maps (0 = not owned; 1..N = local 1-indexed)
    n_nodes = len(Part_node)
    n_edges = len(Part_edge)
    n_faces = len(Part_face)
    n_cells_ext = len(Part_cell_ext)
    n_lfe = len(Part_link_fe)
    n_lcn = len(Part_link_cn)

    def _owned_and_g2l(part_array, target_rank, length):
        owned = np.where(part_array == target_rank)[0]
        g2l = np.zeros(length, dtype=np.int64)   # 0 = not owned
        g2l[owned] = np.arange(1, len(owned) + 1, dtype=np.int64)
        return owned, g2l

    per_rank = []
    for r in range(1, num_parts + 1):
        owned_nodes,     node_g2l = _owned_and_g2l(Part_node,     r, n_nodes)
        owned_edges,     edge_g2l = _owned_and_g2l(Part_edge,     r, n_edges)
        owned_faces,     face_g2l = _owned_and_g2l(Part_face,     r, n_faces)
        owned_cells_ext, cell_g2l = _owned_and_g2l(Part_cell_ext, r, n_cells_ext)
        owned_lfe,       lfe_g2l  = _owned_and_g2l(Part_link_fe,  r, n_lfe)
        owned_lcn,       lcn_g2l  = _owned_and_g2l(Part_link_cn,  r, n_lcn)
        per_rank.append({
            "owned_nodes": owned_nodes, "owned_edges": owned_edges,
            "owned_faces": owned_faces, "owned_cells_ext": owned_cells_ext,
            "owned_lfe": owned_lfe,     "owned_lcn": owned_lcn,
            "node_g2l": node_g2l, "edge_g2l": edge_g2l,
            "face_g2l": face_g2l, "cell_g2l": cell_g2l,
            "lfe_g2l": lfe_g2l,   "lcn_g2l": lcn_g2l,
        })

    # Comm tables (iterate global links; OWNED-only G2L)
    fw_per_rank = [{t: {} for t in ("E2N", "fe", "F2C", "cn")} for _ in range(num_parts + 1)]
    rv_per_rank = [{t: {} for t in ("N2E", "ef", "C2F", "nc")} for _ in range(num_parts + 1)]

    if verbose:
        print(f"[partition] computing inter-rank communication tables...")

    edge_g2ls = [None] + [per_rank[r-1]["edge_g2l"] for r in range(1, num_parts + 1)]
    node_g2ls = [None] + [per_rank[r-1]["node_g2l"] for r in range(1, num_parts + 1)]
    face_g2ls = [None] + [per_rank[r-1]["face_g2l"] for r in range(1, num_parts + 1)]
    cell_g2ls = [None] + [per_rank[r-1]["cell_g2l"] for r in range(1, num_parts + 1)]
    lfe_g2ls  = [None] + [per_rank[r-1]["lfe_g2l"]  for r in range(1, num_parts + 1)]
    lcn_g2ls  = [None] + [per_rank[r-1]["lcn_g2l"]  for r in range(1, num_parts + 1)]

    def _build_one_type(fw_key, rv_key,
                        link_table, part_link, part_geom2, geom2_g2ls, link_g2ls):
        g2_arr = link_table[:, 1]
        p1_arr = part_link
        p2_arr = part_geom2[g2_arr]
        cross  = p1_arr != p2_arr
        for i in np.where(cross)[0]:
            p1, p2 = int(p1_arr[i]), int(p2_arr[i])
            fw_per_rank[p1][fw_key].setdefault(p2, []).append(int(link_g2ls[p1][i]))
            rv_per_rank[p2][rv_key].setdefault(p1, []).append(int(geom2_g2ls[p2][g2_arr[i]]))

    _build_one_type("E2N", "N2E", mesh.edges_to_nodes, Part_edge, Part_node, node_g2ls, edge_g2ls)
    _build_one_type("fe",  "ef",  global_link_fe,      Part_link_fe, Part_edge, edge_g2ls, lfe_g2ls)
    _build_one_type("F2C", "C2F", F2C_ext,             Part_face, Part_cell_ext, cell_g2ls, face_g2ls)
    _build_one_type("cn",  "nc",  global_link_cn,      Part_link_cn, Part_node, node_g2ls, lcn_g2ls)

    if verbose:
        print(f"[partition] building per-rank mesh data...")
    locals_list = []
    comm_tables_list = []
    for r in range(1, num_parts + 1):
        d = per_rank[r-1]
        owned_nodes     = d["owned_nodes"]
        owned_edges     = d["owned_edges"]
        owned_faces     = d["owned_faces"]
        owned_cells_ext = d["owned_cells_ext"]
        owned_lfe       = d["owned_lfe"]
        owned_lcn       = d["owned_lcn"]
        node_g2l        = d["node_g2l"]
        edge_g2l        = d["edge_g2l"]
        face_g2l        = d["face_g2l"]
        cell_g2l        = d["cell_g2l"]

        n_node = len(owned_nodes)
        n_edge = len(owned_edges)
        n_face = len(owned_faces)
        n_cell = len(owned_cells_ext)
        n_lfe_owned = len(owned_lfe)
        n_lcn_owned = len(owned_lcn)

        # OWNED node coords
        nodes_local = mesh.nodes[owned_nodes] if n_node > 0 else np.empty((0, 3), dtype=np.float64)

        # E2N: (n1_local, n2_local_or_0) -- node_g2l returns 0 for non-owned nodes
        if n_edge > 0:
            e2n_g = mesh.edges_to_nodes[owned_edges]
            e2n_local = np.column_stack([node_g2l[e2n_g[:, 0]], node_g2l[e2n_g[:, 1]]])
        else:
            e2n_local = np.empty((0, 2), dtype=np.int64)

        # F2C: (c1_local, c2_local_or_0)
        if n_face > 0:
            f2c_g = F2C_ext[owned_faces]
            f2c_local = np.column_stack([cell_g2l[f2c_g[:, 0]], cell_g2l[f2c_g[:, 1]]])
        else:
            f2c_local = np.empty((0, 2), dtype=np.int64)

        # link_fe: (face_local, edge_local_or_0)
        if n_lfe_owned > 0:
            lfe_g = global_link_fe[owned_lfe]
            link_fe_local = np.column_stack([face_g2l[lfe_g[:, 0]], edge_g2l[lfe_g[:, 1]]])
        else:
            link_fe_local = np.empty((0, 2), dtype=np.int64)

        # link_cn: (cell_local, node_local_or_0)
        if n_lcn_owned > 0:
            lcn_g = global_link_cn[owned_lcn]
            link_cn_local = np.column_stack([cell_g2l[lcn_g[:, 0]], node_g2l[lcn_g[:, 1]]])
        else:
            link_cn_local = np.empty((0, 2), dtype=np.int64)

        # BC arrays: leading -1 dummy + OWNED values
        bc_node = np.full(n_node + 1, -1, dtype=np.int64)
        if n_node > 0:
            bc_node[1:] = mesh.bc_node[owned_nodes]
        bc_edge = np.full(n_edge + 1, -1, dtype=np.int64)
        if n_edge > 0:
            bc_edge[1:] = mesh.bc_edge[owned_edges]
        bc_face = np.full(n_face + 1, -1, dtype=np.int64)
        if n_face > 0:
            bc_face[1:] = mesh.bc_face[owned_faces]
        # bc_cell: volumes carry 0, boundary face elements carry FAM tag
        bc_cell = np.full(n_cell + 1, -1, dtype=np.int64)
        if n_cell > 0:
            cell_bc = np.zeros(n_cell, dtype=np.int64)
            bd_mask = owned_cells_ext >= n_volumes
            if np.any(bd_mask):
                bf_rows = owned_cells_ext[bd_mask] - n_volumes
                cell_bc[bd_mask] = mesh.boundary_fam[bf_rows]
            bc_cell[1:] = cell_bc

        local_mesh = _LocalMesh(
            rank=r,
            nodes_local=nodes_local,
            e2n_local=e2n_local,
            f2c_local=f2c_local,
            link_fe_local=link_fe_local,
            link_cn_local=link_cn_local,
            bc_node=bc_node, bc_edge=bc_edge, bc_face=bc_face, bc_cell=bc_cell,
            nd=3, ncn_max=8,
            num_node=n_node, num_edge=n_edge, num_face=n_face, num_cell=n_cell,
            num_link_fe=n_lfe_owned, num_link_cn=n_lcn_owned,
        )
        locals_list.append(local_mesh)

        comm_table = {}
        for t in ("E2N", "fe", "F2C", "cn"):
            comm_table[t] = _assemble_comm_arrays(fw_per_rank[r][t], num_parts)
        for t in ("N2E", "ef", "C2F", "nc"):
            comm_table[t] = _assemble_comm_arrays(rv_per_rank[r][t], num_parts)
        comm_tables_list.append(comm_table)

    return locals_list, comm_tables_list


def _assemble_comm_arrays(by_neighbor, num_parts):
    """Pack per-neighbor lists into Partition_3D-style (pt, lp, part) arrays.

    Order within each (p1, p2) bucket is the global-link iteration order, which
    forward (link index) and reverse (geom2 id) both share by construction.
    Duplicates are preserved -- forward[i] pairs with reverse[i] for symmetric
    send/recv; deduping would break that 1:1 correspondence.
    """
    neighbors_sorted = sorted(by_neighbor.keys())
    pt = []
    lp = [0]
    for p in range(1, num_parts + 1):
        if p in by_neighbor:
            pt.extend(by_neighbor[p])
        lp.append(len(pt))
    part = np.array(neighbors_sorted, dtype=np.int64)
    return {"pt": np.array(pt, dtype=np.int64),
            "lp": np.array(lp, dtype=np.int64),
            "part": part}


def _write_rank_files(prefix, local, comm, num_parts):
    label = f"{local.rank:05d}"
    base = prefix.parent / f"{prefix.name}_{label}"
    with open(f"{base}.siz", "w", newline="\n") as f:
        for v in (local.nd, local.ncn_max, local.num_node, local.num_edge,
                   local.num_face, local.num_cell, local.num_link_fe, local.num_link_cn):
            f.write(f"{v:12d}\n")
    with open(f"{base}.msh", "w", newline="\n") as f:
        f.write(f"{0.0:25.16E} {0.0:25.16E} {0.0:25.16E}\n")
        for row in local.nodes_local:
            f.write(f"{row[0]:25.16E} {row[1]:25.16E} {row[2]:25.16E}\n")
        for row in local.e2n_local:
            f.write(f"{int(row[0]):12d} {int(row[1]):12d}\n")
        for row in local.f2c_local:
            f.write(f"{int(row[0]):12d} {int(row[1]):12d}\n")
        for row in local.link_fe_local:
            f.write(f"{int(row[0]):12d} {int(row[1]):12d}\n")
        for row in local.link_cn_local:
            f.write(f"{int(row[0]):12d} {int(row[1]):12d}\n")
    with open(f"{base}.BC", "w", newline="\n") as f:
        _write_10I10(f, local.bc_node)
        _write_10I10(f, local.bc_edge)
        _write_10I10(f, local.bc_face)
        _write_10I10(f, local.bc_cell)
    c = comm
    with open(f"{base}.cm", "w", newline="\n") as f:
        f.write(f"{num_parts:12d}\n")
        f.write(" ".join(f"{c[t]['pt'].size:12d}" for t in ("E2N", "fe", "F2C", "cn")) + "\n")
        _write_10I10(f, c["E2N"]["lp"]); _write_10I10(f, c["fe"]["lp"])
        _write_10I10(f, c["F2C"]["lp"]); _write_10I10(f, c["cn"]["lp"])
        _write_10I10(f, c["E2N"]["pt"]); _write_10I10(f, c["fe"]["pt"])
        _write_10I10(f, c["F2C"]["pt"]); _write_10I10(f, c["cn"]["pt"])
        f.write(" ".join(f"{c[t]['pt'].size:12d}" for t in ("N2E", "ef", "C2F", "nc")) + "\n")
        _write_10I10(f, c["N2E"]["lp"]); _write_10I10(f, c["ef"]["lp"])
        _write_10I10(f, c["C2F"]["lp"]); _write_10I10(f, c["nc"]["lp"])
        _write_10I10(f, c["N2E"]["pt"]); _write_10I10(f, c["ef"]["pt"])
        _write_10I10(f, c["C2F"]["pt"]); _write_10I10(f, c["nc"]["pt"])
        f.write(" ".join(f"{c[t]['part'].size:12d}" for t in ("E2N", "fe", "F2C", "cn")) + "\n")
        _write_10I10(f, c["E2N"]["part"]); _write_10I10(f, c["fe"]["part"])
        _write_10I10(f, c["F2C"]["part"]); _write_10I10(f, c["cn"]["part"])
        f.write(" ".join(f"{c[t]['part'].size:12d}" for t in ("N2E", "ef", "C2F", "nc")) + "\n")
        _write_10I10(f, c["N2E"]["part"]); _write_10I10(f, c["ef"]["part"])
        _write_10I10(f, c["C2F"]["part"]); _write_10I10(f, c["nc"]["part"])


def _write_str_file(prefix, mesh, num_parts, Part_cell_ext):
    """Write the IB rank's .str file.

    Per Partition_3D convention (output.F90:319-326), num_cell counts the
    EXTENDED cell set (volumes + boundary face elements) and num_cell_p_lp
    is the per-rank prefix sum over Part_cell_ext. Index_cell entries get
    permuted to the GLOBAL partition-contiguous 1-indexed cell IDs so the
    IB rank's `Vc(:, Index_cell(i,j,k))` lookup hits the correct rank chunk
    after recv_flow scatters per-rank send_flow buffers into Vc.
    """
    base = prefix.parent / f"{prefix.name}_00000.str"
    n_volumes = mesh.cells.shape[0]
    n_global_cells = len(Part_cell_ext)               # volumes + bd_face_elements

    # Per-rank owned cell_ext count + cumulative
    num_cell_p = np.bincount(Part_cell_ext, minlength=num_parts + 1)[1:num_parts + 1]
    num_cell_p_lp = np.zeros(num_parts + 1, dtype=np.int64)
    num_cell_p_lp[1:] = np.cumsum(num_cell_p)

    # Build cell_ext_perm: old_id (0-based) -> new_id (1-based, partition-contiguous).
    # Iterating old IDs in ascending order matches Partition_3D's sort_on_partition,
    # which preserves intra-rank original-order ordering (so this matches the per-rank
    # local 1-indexed ordering built in _build_per_rank_data via np.where).
    cell_ext_perm = np.empty(n_global_cells, dtype=np.int64)
    counter = num_cell_p_lp[:num_parts].copy()        # one cursor per rank
    for old_id in range(n_global_cells):
        r = int(Part_cell_ext[old_id])
        counter[r - 1] += 1
        cell_ext_perm[old_id] = counter[r - 1]        # 1-based new global ID

    # Index_cell stores 0-based volume cell IDs; remap to new global IDs
    new_index_cell = cell_ext_perm[mesh.index_cell]

    num_x, num_y, num_z = mesh.index_cell.shape
    with open(base, "w", newline="\n") as f:
        f.write(f"{3:12d}\n")
        f.write(f"{n_global_cells:12d}\n")
        _write_10I10(f, num_cell_p_lp)
        f.write(f"{num_x:12d} {num_y:12d} {num_z:12d}\n")
        f.write(" ".join(f"{v:25.16E}" for v in mesh.pos_ref) + "\n")
        f.write(" ".join(f"{v:25.16E}" for v in mesh.dx_uniform) + "\n")
        # Fortran reads Index_cell(num_x,num_y,num_z) in column-major (i fastest).
        # Python C-order ravel is k fastest, so write with order='F' to match.
        _write_10I10(f, new_index_cell.ravel(order='F'))


def diagnose_vol_ci(mesh, *, verbose=True):
    """Replicate the solver's cell_vol_new accumulation in Python and report
    any volume cell whose accumulated vol_ci would be zero (which would make
    the solver trap on `dim_scale/vol_ci(i)` at Geometry.f90:566).

    This is a GLOBAL diagnostic — it sums face contributions across all faces
    (no partitioning), so it tells us whether the underlying mesh has a
    structural problem independent of MPI.

    Boundary face elements naturally accumulate to 0 (rf2=0): their cell
    "center" is by construction the boundary face center, so face_center -
    cell_center = 0. The Fortran solver now guards against this in
    Geometry.f90 cell_vol_new (`if (abs(vol_ci(i)) > 1.0d-14) ...`); legacy
    builds without this guard relied on `-fpe0` being off and the subsequent
    `where(BC_cell/=0) vol_ci=0` cleaning up the resulting Inf. We list bd
    elements separately here so they don't drown out the real signal —
    namely, ANY volume cell (BC_cell==0) with vol_ci=0, which would be a
    true mesh-connectivity bug and crash the solver regardless of guards.

    Returns dict with keys:
        n_zero_volume_cells, zero_volume_cell_ids,
        n_zero_bd_elements,  zero_bd_element_ids,
        vol_ci_volumes (sample stats)
    """
    n_vol = mesh.cells.shape[0]
    n_bd = mesh.boundary_faces.shape[0]

    # We need cpos_cg for each "extended" cell (volumes + bd elements).
    # cpos_cg(cell) = mean of fpos_cg over faces referencing this cell as c1 or c2.
    # fpos_cg(face) = mean of epos_cg over the face's edges (link_fe).
    # epos_cg(edge) = mean of npos over edge's 2 nodes.

    n_face = mesh.faces_to_nodes.shape[0]
    n_edge = mesh.edges_to_nodes.shape[0]

    epos = mesh.nodes[mesh.edges_to_nodes].mean(axis=1)        # (n_edge, 3)

    # Build link_fe table (4 per face + link_fe_insert)
    link_fe_insert = (mesh.link_fe_insert
                      if mesh.link_fe_insert is not None
                      else np.zeros((0, 2), dtype=np.int64))
    lfe_face = np.concatenate([
        np.repeat(np.arange(n_face), 4),
        link_fe_insert[:, 0] if link_fe_insert.size else np.empty(0, dtype=np.int64),
    ])
    lfe_edge = np.concatenate([
        mesh.faces_to_edges.ravel(),
        link_fe_insert[:, 1] if link_fe_insert.size else np.empty(0, dtype=np.int64),
    ])

    fpos = np.zeros((n_face, 3), dtype=np.float64)
    fcnt = np.zeros(n_face, dtype=np.int64)
    np.add.at(fpos, lfe_face, epos[lfe_edge])
    np.add.at(fcnt, lfe_face, 1)
    fpos /= fcnt[:, None]

    # cpos_cg for extended cells (volumes + bd elements).
    # F2C_ext: for boundary faces, second cell = n_vol + bf_idx.
    f2n_sorted = np.sort(mesh.faces_to_nodes, axis=1).astype('>u8')
    bf_sorted = np.sort(mesh.boundary_faces, axis=1).astype('>u8')
    row_bytes = 4 * 8
    f2n_keys = np.ascontiguousarray(f2n_sorted).view(f"V{row_bytes}").ravel()
    bf_keys = np.ascontiguousarray(bf_sorted).view(f"V{row_bytes}").ravel()
    order = np.argsort(f2n_keys)
    pos = np.searchsorted(f2n_keys[order], bf_keys)
    bf_to_face_id = order[pos]

    F2C_ext = mesh.faces_to_cells.copy()
    if n_bd > 0:
        F2C_ext[bf_to_face_id, 1] = n_vol + np.arange(n_bd, dtype=np.int64)

    n_cells_ext = n_vol + n_bd
    cpos = np.zeros((n_cells_ext, 3), dtype=np.float64)
    ccnt = np.zeros(n_cells_ext, dtype=np.int64)
    for col in (0, 1):
        cells_col = F2C_ext[:, col]
        valid = cells_col >= 0
        np.add.at(cpos, cells_col[valid], fpos[valid])
        np.add.at(ccnt, cells_col[valid], 1)
    # Avoid div-by-zero in cpos averaging (shouldn't happen)
    safe = ccnt > 0
    cpos[safe] /= ccnt[safe, None]

    # Now compute nf, rf1, rf2 per face and accumulate vol_ci.
    # For nf computation, we use the same formula as Operators.face_geom_new
    # but simplified: nf(F) = (1/2) * Σ_edges fe_sign(F,e) * cross(ref(F,e), te(e))
    # where ref(F,e) = epos(e) - fpos(F), te(e) = node(E2N(2)) - node(E2N(1)).
    # For axis-aligned hex meshes, |nf| = face area in the normal direction.

    te = mesh.nodes[mesh.edges_to_nodes[:, 1]] - mesh.nodes[mesh.edges_to_nodes[:, 0]]
    ref = epos[lfe_edge] - fpos[lfe_face]                # (n_lfe, 3)
    # rf direction for sign: rf = rf1 - rf2 = cpos(c2) - cpos(c1) (with cpos[-1]=0)
    rf = cpos[F2C_ext[:, 1]] - cpos[F2C_ext[:, 0]]       # (n_face, 3)
    avec = np.cross(ref, te[lfe_edge])                    # (n_lfe, 3)
    tmp = np.einsum('ij,ij->i', avec, rf[lfe_face])       # (n_lfe,)
    fe_sign = np.sign(tmp)                                # ±1 per entry (or 0)
    nf = np.zeros((n_face, 3), dtype=np.float64)
    np.add.at(nf, lfe_face, avec * fe_sign[:, None])
    nf *= 0.5

    rf1 = fpos - cpos[F2C_ext[:, 0]]                      # (n_face, 3)
    rf2 = fpos - cpos[F2C_ext[:, 1]]

    vol_ci = np.zeros(n_cells_ext, dtype=np.float64)
    contrib_c1 = np.einsum('ij,ij->i', nf, rf1)
    contrib_c2 = np.einsum('ij,ij->i', nf, rf2)
    np.add.at(vol_ci, F2C_ext[:, 0], contrib_c1)
    np.add.at(vol_ci, F2C_ext[:, 1], -contrib_c2)

    vol_ci_volumes = vol_ci[:n_vol]
    vol_ci_bd      = vol_ci[n_vol:]
    zero_vol_mask = np.abs(vol_ci_volumes) < 1e-15
    zero_bd_mask  = np.abs(vol_ci_bd) < 1e-15
    zero_vol_ids  = np.where(zero_vol_mask)[0]
    zero_bd_ids   = np.where(zero_bd_mask)[0]

    if verbose:
        print(f"[check] volume cells: {n_vol:,}   "
              f"with zero volume: {int(zero_vol_mask.sum())}")
        print(f"[check] boundary-face elements: {n_bd:,}   "
              f"with zero volume: {int(zero_bd_mask.sum())}  "
              f"(zero by construction; solver clears them)")
        print(f"[check] cell volume stats: min={vol_ci_volumes.min():.3e}, "
              f"max={vol_ci_volumes.max():.3e}, "
              f"median={np.median(np.abs(vol_ci_volumes)):.3e}")
        if zero_vol_ids.size > 0:
            print(f"[check] *** PROBLEM: {zero_vol_ids.size} volume cells have zero volume ***")
            print(f"[check] first 10 bad cell IDs: {zero_vol_ids[:10].tolist()}")
            for cid in zero_vol_ids[:5]:
                print(f"    cell {cid}: center={cpos[cid].tolist()}, "
                      f"face count={ccnt[cid]}, "
                      f"layer cell counts={mesh.layer_cell_counts}")

    return {
        "n_zero_volume_cells": int(zero_vol_mask.sum()),
        "zero_volume_cell_ids": zero_vol_ids,
        "n_zero_bd_elements": int(zero_bd_mask.sum()),
        "zero_bd_element_ids": zero_bd_ids,
        "vol_ci_volumes": vol_ci_volumes,
    }


def write_solver_input(mesh, prefix, num_parts, *, gtype="nodal", seed=42,
                       verbose=False):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if num_parts < 1:
        raise ValueError("num_parts must be >= 1 (number of fluid ranks)")
    cell_partition_0idx, node_partition_0idx, backend_used = _partition_cells(
        mesh, num_parts, gtype=gtype, seed=seed
    )
    cell_to_rank_mpi = cell_partition_0idx + 1
    node_to_rank_mpi = (node_partition_0idx + 1) if node_partition_0idx is not None else None
    parts = _compute_global_partitions(mesh, cell_to_rank_mpi,
                                       node_to_rank_mpi=node_to_rank_mpi)
    locals_list, comms = _build_per_rank_data(
        mesh, num_parts, cell_to_rank_mpi,
        node_to_rank_mpi=node_to_rank_mpi, verbose=verbose
    )
    if verbose:
        print(f"[write] writing per-rank files...")
    _write_str_file(prefix, mesh, num_parts, parts["Part_cell_ext"])
    for local, comm in zip(locals_list, comms):
        _write_rank_files(prefix, local, comm, num_parts)
    # File count: 1 .str (IB rank 0) + 4 files per fluid rank (.siz, .msh, .BC, .cm)
    n_files_written = 1 + 4 * num_parts
    summary = {
        "num_parts": num_parts,
        "num_cells_global": int(mesh.cells.shape[0] + mesh.boundary_faces.shape[0]),
        "cells_per_rank": [int(loc.num_cell) for loc in locals_list],
        "nodes_per_rank": [int(loc.num_node) for loc in locals_list],
        "partition_backend": backend_used,
        "n_files_written": n_files_written,
    }
    return summary


# =============================================================================
#                                  MAIN
# =============================================================================

def _build_box(spec):
    mode = spec[0]
    if mode == "center+size":
        return BoxSpec(center=spec[1], size=spec[2])
    if mode == "bounds":
        x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = spec[1]
        return BoxSpec.from_bounds(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    raise ValueError(f"Unknown box mode {mode!r} (use 'center+size' or 'bounds')")


def main() -> int:
    import time
    t_start = time.perf_counter()

    if len(BOXES) < 1:
        raise ValueError("BOXES must contain at least one entry (innermost layer).")
    boxes = [_build_box(spec) for spec in BOXES]
    n_layers = len(boxes)

    # Per-layer dx is derived from INNER_SPACING * 2^k. Showing it here makes
    # config mistakes (e.g. INNER_SPACING off by 10x) immediately obvious.
    dx_per_layer = [INNER_SPACING * (2 ** k) for k in range(n_layers)]
    dx_str = ", ".join(f"{d:g}" for d in dx_per_layer)

    print(f"=== Building '{CASE_NAME}': {n_layers}-layer nested mesh ===")
    for k, b in enumerate(boxes):
        tag = " (innermost)" if k == 0 else ""
        print(f"  layer {k}{tag}: center={tuple(b.center)}  size={tuple(b.size)}")
    print(f"  inner_spacing={INNER_SPACING}    per-layer dx: {dx_str}")
    # Only emit an extra blank in VERBOSE mode (so make_nested_mesh's verbose
    # progress lines have breathing room between them and the config header).
    if VERBOSE:
        print()

    mesh = make_nested_mesh(
        inner_spacing=INNER_SPACING,
        boxes=boxes,
        refinement_ratio=2,
        verbose=VERBOSE,
    )

    # Compact 3-line summary of what got built.
    print()
    n_vol = mesh.cells.shape[0]
    print(f"[mesh] {n_vol:,} cells / {mesh.nodes.shape[0]:,} nodes / "
          f"{mesh.faces_to_nodes.shape[0]:,} faces / {mesh.edges_to_nodes.shape[0]:,} edges")
    print(f"       layer cell counts: {mesh.layer_cell_counts}")
    print(f"       interface pairs: {mesh.interface_pairs.shape[0]:,}    "
          f"IBM index_cell: {tuple(mesh.index_cell.shape)} [Nx, Ny, Nz]")

    # Sanity check: every volume cell must have a non-zero geometric volume,
    # otherwise the solver would later trap on a divide-by-zero. We catch it
    # here before writing any files.
    print()
    diag = diagnose_vol_ci(mesh, verbose=VERBOSE)
    if diag["n_zero_volume_cells"] > 0:
        print(f"!!! [check] FAILED: {diag['n_zero_volume_cells']} volume cells have zero volume.")
        print(f"!!! The solver would crash on these. Aborting before write.")
        return 1
    print(f"[check] all {n_vol:,} volume cells have non-zero volume")

    # Resolve OUT_DIR against the current working directory (standard CLI
    # behavior). Absolute paths pass through unchanged. .resolve() returns
    # the full absolute path, which we print so the user sees exactly where
    # files will go.
    out_dir = Path(OUT_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f"[write] output directory: {out_dir}")

    if WRITE_HDF5:
        h5 = out_dir / f"{CASE_NAME}.h5"
        write_hdf5(mesh, h5)
        print(f"[write]   HDF5: {h5.name}  ({h5.stat().st_size/1024:.1f} KB)")

    if WRITE_VTU:
        vtu = out_dir / f"{CASE_NAME}.vtu"
        write_vtu(mesh, vtu)
        print(f"[write]   VTU:  {vtu.name}  ({vtu.stat().st_size/1024:.1f} KB)")

    if WRITE_MTS:
        mts = out_dir / f"{CASE_NAME}.mts"
        write_mts(mesh, mts)
        print(f"[write]   MTS:  {mts.name}  ({mts.stat().st_size/1024:.1f} KB)")

    if WRITE_SOLVER_INPUT:
        prefix = out_dir / CASE_NAME
        summary = write_solver_input(mesh, prefix, num_parts=NUM_PARTS, verbose=VERBOSE)
        cpr = np.asarray(summary['cells_per_rank'])
        print(f"[write]   {summary['n_files_written']} files written "
              f"({CASE_NAME}_00000.str + {CASE_NAME}_00001..{NUM_PARTS:05d}.{{siz,msh,BC,cm}})")
        print(f"[write]   cells per rank: min={cpr.min()}, max={cpr.max()}, "
              f"mean={cpr.mean():.0f}, median={int(np.median(cpr))}")

    elapsed = time.perf_counter() - t_start
    print()
    print(f"DONE in {elapsed:.1f} s.")
    if WRITE_VTU:
        print(f"To view in ParaView, open: {out_dir / (CASE_NAME + '.vtu')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
