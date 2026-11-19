"""Mesh repair / decimation helpers shared by both modes."""
from __future__ import annotations

import math

import fast_simplification
import numpy as np
import rhino3dm as r
import trimesh


def basic_repair(tm: trimesh.Trimesh, merge_tol: float) -> tuple[trimesh.Trimesh, dict]:
    """Weld vertices, drop degenerate/duplicate faces, make winding consistent."""
    before_v, before_f = len(tm.vertices), len(tm.faces)
    tm = tm.copy()
    digits = max(0, int(round(-math.log10(max(merge_tol, 1e-9)))))
    tm.merge_vertices(digits_vertex=digits, merge_tex=True, merge_norm=True)
    tm.update_faces(tm.nondegenerate_faces(height=merge_tol * 1e-3))
    tm.update_faces(tm.unique_faces())
    tm.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(tm)
    if tm.is_watertight and tm.volume < 0:
        tm.invert()
    trimesh.repair.fill_holes(tm)
    return tm, {
        "welded_vertices": before_v - len(tm.vertices),
        "removed_faces": before_f - len(tm.faces),
    }


def decimate(tm: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(tm.faces) <= target_faces:
        return tm
    reduction = 1.0 - target_faces / len(tm.faces)
    v, f = fast_simplification.simplify(
        tm.vertices.astype(np.float32), tm.faces.astype(np.int32), target_reduction=reduction
    )
    return trimesh.Trimesh(v.astype(float), f, process=True)


def sensible_face_target(tm: trimesh.Trimesh, max_faces: int, edge_ratio: float = 1500) -> int:
    """How many faces this mesh needs: edges ~ size/edge_ratio, capped by max_faces."""
    diag = float(np.linalg.norm(tm.extents)) or 1.0
    edge = diag / edge_ratio
    needed = int(tm.area / (0.433 * edge * edge))  # area of an equilateral triangle
    return int(max(5000, min(max_faces, needed)))


def to_rhino_mesh(vertices: np.ndarray, faces: np.ndarray) -> r.Mesh:
    m = r.Mesh()
    for x, y, z in vertices:
        # AddPoint3d stores double-precision vertices (Add would truncate to float32).
        m.Vertices.AddPoint3d(float(x), float(y), float(z))
    for a, b, c in faces:
        m.Faces.AddFace(int(a), int(b), int(c))
    try:
        m.Normals.ComputeNormals()
    except Exception:
        pass
    try:
        m.Compact()
    except Exception:
        pass
    return m
