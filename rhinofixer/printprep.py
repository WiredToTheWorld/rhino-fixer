"""Mode 2: turn a Rhino model into a single watertight, printable mesh.

Strategy
1. Mesh every printable object, convert to millimetres and scale to the
   requested print size.
2. If every body is already closed, union them exactly with Manifold.
3. Otherwise voxel-remesh: rasterise the surface, close small gaps, fill the
   inside, thicken open sheets to a minimum wall, then marching-cubes back to a
   mesh. That result is watertight by construction.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import manifold3d as mf
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from . import meshtools
from .analyze import Analysis

RESOLUTION_MM = {"draft": 0.8, "standard": 0.4, "fine": 0.25}
VOXEL_BUDGET = {"draft": 25_000_000, "standard": 60_000_000, "fine": 110_000_000}

PRINTERS = {
    "a1mini": {"name": "Bambu Lab A1 mini", "bed": [180, 180, 180]},
    "a1": {"name": "Bambu Lab A1", "bed": [256, 256, 256]},
    "p1s": {"name": "Bambu Lab P1S / P1P", "bed": [256, 256, 256]},
    "x1c": {"name": "Bambu Lab X1 Carbon / X1E", "bed": [256, 256, 256]},
    "p2s": {"name": "Bambu Lab P2S", "bed": [256, 256, 256]},
    "h2d": {"name": "Bambu Lab H2D (single nozzle)", "bed": [325, 320, 325]},
    "h2s": {"name": "Bambu Lab H2S", "bed": [340, 320, 340]},
}


@dataclass
class PrintOptions:
    scale: float = 1.0              # uniform scale applied after converting to mm
    printer: str = "x1c"
    resolution: str = "standard"    # draft | standard | fine
    min_wall_mm: float = 0.8        # thicken open sheets to at least this
    close_gaps_mm: float = 1.0      # gaps/holes narrower than this are sealed
    min_part_mm3: float = 1.0       # drop floating crumbs smaller than this
    force_remesh: bool = False
    smooth: bool = True
    max_faces: int = 400_000


def _log(steps, msg):
    steps.append(msg)


def gather_mesh(a: Analysis) -> trimesh.Trimesh:
    skip = a.outliers | a.invalid | a.degenerate | a.duplicates
    parts = [m for i, m in a.meshes.items() if i not in skip]
    if not parts:
        raise ValueError("No printable geometry (surfaces, solids, SubDs or meshes) in this file.")
    tm = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0].copy()
    tm.apply_scale(a.unit_mm)  # -> millimetres
    return tm


# ---------------------------------------------------------------- exact path
def _to_manifold(tm: trimesh.Trimesh) -> mf.Manifold | None:
    m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(tm.vertices, np.float32),
                            tri_verts=np.asarray(tm.faces, np.uint32)))
    return m if m.status() == mf.Error.NoError and not m.is_empty() else None


def _from_manifold(m: mf.Manifold) -> trimesh.Trimesh:
    mesh = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)


def try_exact(tm: trimesh.Trimesh, steps) -> trimesh.Trimesh | None:
    rep, stats = meshtools.basic_repair(tm, 1e-3)
    bodies = rep.split(only_watertight=False)
    closed = [b for b in bodies if b.is_watertight and b.is_winding_consistent]
    _log(steps, f"Exact repair: {len(closed)} of {len(bodies)} bodies are closed after welding "
                f"{stats['welded_vertices']:,} vertices.")
    if len(closed) != len(bodies):
        return None
    mans = []
    for b in bodies:
        if b.volume < 0:
            b.invert()
        m = _to_manifold(b)
        if m is None:
            _log(steps, "A body is closed but not a valid manifold - switching to voxel remesh.")
            return None
        mans.append(m)
    result = mf.Manifold.batch_boolean(mans, mf.OpType.Add) if len(mans) > 1 else mans[0]
    _log(steps, f"Unioned {len(mans)} closed bodies into one solid (exact geometry kept).")
    return _from_manifold(result)


# ---------------------------------------------------------------- voxel path
def _rasterize_surface(v, f, origin, pitch, shape) -> np.ndarray:
    grid = np.zeros(shape, dtype=bool)
    tri = v[f]
    edges = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(axis=1)
    n = np.maximum(1, np.ceil(edges / (pitch * 0.5))).astype(np.int64)
    for k in np.unique(n):
        sel = tri[n == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = (i + j) <= k
        w = np.stack([i[keep], j[keep], k - i[keep] - j[keep]], axis=1).astype(float) / k  # (m,3)
        chunk = max(1, 3_000_000 // len(w))
        for s in range(0, len(sel), chunk):
            pts = np.einsum("mk,tkd->tmd", w, sel[s:s + chunk]).reshape(-1, 3)
            idx = np.floor((pts - origin) / pitch).astype(np.int64)
            idx = np.clip(idx, 0, np.array(shape) - 1)
            grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def voxel_remesh(tm: trimesh.Trimesh, o: PrintOptions, steps) -> tuple[trimesh.Trimesh, dict]:
    ext = tm.extents
    pitch = RESOLUTION_MM.get(o.resolution, 0.4)
    budget = VOXEL_BUDGET.get(o.resolution, 60_000_000)
    pitch = max(pitch, (float(np.prod(ext + 4)) / budget) ** (1 / 3))
    close_r = max(0, math.ceil(o.close_gaps_mm / (2 * pitch)))
    wall_t = max(0, math.ceil((o.min_wall_mm / pitch - 1) / 2))
    pad = close_r + wall_t + 3
    origin = tm.bounds[0] - pad * pitch
    shape = tuple((np.ceil(ext / pitch).astype(int) + 2 * pad + 1).tolist())
    _log(steps, f"Voxel remesh at {pitch:.2f} mm resolution ({shape[0]}x{shape[1]}x{shape[2]} grid).")

    t0 = time.time()
    surf = _rasterize_surface(np.asarray(tm.vertices), np.asarray(tm.faces), origin, pitch, shape)
    struct = ndimage.generate_binary_structure(3, 1)

    # Close gaps, then fill enclosed interior.
    if close_r:
        closed = ndimage.binary_dilation(surf, struct, iterations=close_r)
        closed = ndimage.binary_fill_holes(closed)
        closed = ndimage.binary_erosion(closed, struct, iterations=close_r)
        solid = closed | surf
    else:
        solid = ndimage.binary_fill_holes(surf)
    interior = solid & ~ndimage.binary_dilation(surf, struct)
    filled_vox = int(interior.sum())

    # Thicken sheets (surface voxels not bounding a filled interior).
    sheet_vox = 0
    if wall_t:
        near_interior = ndimage.binary_dilation(interior, struct, iterations=2)
        sheet = surf & ~near_interior
        sheet_vox = int(sheet.sum())
        if sheet_vox:
            solid |= ndimage.binary_dilation(sheet, struct, iterations=wall_t)
    del surf, interior

    # Drop crumbs.
    labels, nlab = ndimage.label(solid, structure=struct)
    removed_parts = 0
    if nlab > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        min_vox = max(1, int(o.min_part_mm3 / pitch ** 3))
        small = np.where((sizes < min_vox) & (sizes > 0))[0]
        if len(small) and len(small) < nlab:
            solid[np.isin(labels, small)] = False
            removed_parts = len(small)
    parts = nlab - removed_parts
    del labels

    # Crop, soften and contour.
    nz = np.argwhere(solid)
    lo = np.maximum(nz.min(axis=0) - 2, 0)
    hi = np.minimum(nz.max(axis=0) + 3, np.array(shape))
    field = solid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    del solid
    field = np.pad(field, 1)
    if o.smooth:
        field = ndimage.gaussian_filter(field, sigma=0.65)
    verts, faces, _, _ = measure.marching_cubes(field, level=0.5, allow_degenerate=False)
    verts = (verts - 1 + lo) * pitch + origin + pitch / 2
    out = trimesh.Trimesh(verts, faces[:, ::-1], process=True)
    if out.volume < 0:
        out.invert()
    _log(steps, f"Solidified in {time.time() - t0:.1f}s: "
                f"{'sealed gaps up to ' + format(o.close_gaps_mm, 'g') + ' mm, ' if close_r else ''}"
                f"{'thickened open surfaces to >= ' + format(o.min_wall_mm, 'g') + ' mm walls, ' if sheet_vox else ''}"
                f"{'filled enclosed volumes, ' if filled_vox else ''}"
                f"{f'removed {removed_parts} tiny fragment(s), ' if removed_parts else ''}"
                f"{parts} separate part(s).")
    return out, {"pitch": pitch, "parts": parts, "removed_fragments": removed_parts,
                 "thickened": bool(sheet_vox), "filled": bool(filled_vox)}


def _finalize(tm: trimesh.Trimesh, o: PrintOptions, steps, tolerance: float) -> trimesh.Trimesh:
    m = _to_manifold(tm)
    if m is not None:
        try:
            simp = m.simplify(tolerance)
            if not simp.is_empty() and simp.num_tri() < len(tm.faces):
                _log(steps, f"Simplified flat areas within {tolerance:g} mm: "
                            f"{len(tm.faces):,} -> {simp.num_tri():,} triangles.")
                m = simp
        except Exception:
            pass
        tm = _from_manifold(m)
    if len(tm.faces) > o.max_faces:
        dec = meshtools.decimate(tm, o.max_faces)
        if dec.is_watertight and dec.is_winding_consistent:
            _log(steps, f"Reduced to {len(dec.faces):,} triangles.")
            tm = dec
    return tm


def prepare(a: Analysis, o: PrintOptions) -> tuple[trimesh.Trimesh, dict]:
    steps: list[str] = []
    tm = gather_mesh(a)
    src_dims = tm.extents.copy()
    tm.apply_scale(o.scale)
    _log(steps, f"Scaled to {o.scale * 100:g}% ({src_dims[0]:.1f} x {src_dims[1]:.1f} x {src_dims[2]:.1f} mm -> "
                f"{tm.extents[0]:.1f} x {tm.extents[1]:.1f} x {tm.extents[2]:.1f} mm).")

    target = tm.extents.copy()
    method, vinfo = "exact", {}
    result = None if o.force_remesh else try_exact(tm, steps)
    if result is None:
        method = "voxel"
        result, vinfo = voxel_remesh(tm, o, steps)
        result = _finalize(result, o, steps, tolerance=0.05)
        # Thickening grows open sheets outward a little; scale back to the requested size.
        k = float(target.max() / result.extents.max())
        if abs(k - 1) > 1e-4:
            result.apply_scale(k)
            _log(steps, f"Rescaled by {k * 100:.2f}% to restore the requested size.")
    else:
        result = _finalize(result, o, steps, tolerance=0.005)

    # Place on the build plate: centred, sitting on Z=0.
    bed = PRINTERS.get(o.printer, PRINTERS["x1c"])["bed"]
    b = result.bounds
    result.apply_translation([bed[0] / 2 - (b[0][0] + b[1][0]) / 2,
                              bed[1] / 2 - (b[0][1] + b[1][1]) / 2, -b[0][2]])
    dims = result.extents.tolist()
    fits = all(d <= bb + 1e-6 for d, bb in zip(dims, bed))
    watertight = bool(result.is_watertight and result.is_winding_consistent)
    return result, {
        "method": method,
        "steps": steps,
        "dims_mm": dims,
        "bed_mm": bed,
        "printer": PRINTERS.get(o.printer, PRINTERS["x1c"])["name"],
        "fits_bed": fits,
        "watertight": watertight,
        "volume_cm3": float(result.volume) / 1000 if watertight else None,
        "triangles": int(len(result.faces)),
        "bodies": int(len(result.split(only_watertight=False))) if len(result.faces) < 2_000_000 else None,
        **vinfo,
    }
