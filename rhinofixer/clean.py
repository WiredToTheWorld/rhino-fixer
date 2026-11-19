"""Mode 1: clean up a Rhino model and write a repaired .3dm.

NURBS / SubD / Brep geometry is kept exactly (only moved). Meshes are repaired
and over-dense meshes are decimated. Layers, materials and other tables are
preserved because the original file is edited in place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import rhino3dm as r

from . import meshtools
from . import model_io as mio
from .analyze import Analysis


@dataclass
class CleanOptions:
    origin: str = "base_center"      # base_center | center | min_corner | none
    remove_outliers: bool = True
    remove_invalid: bool = True
    remove_degenerate: bool = True
    remove_duplicates: bool = True
    remove_curves: bool = False
    repair_meshes: bool = True
    decimate_meshes: bool = True
    max_mesh_faces: int = 200_000


def _target_point(lo, hi, mode) -> np.ndarray | None:
    c = (lo + hi) / 2
    if mode == "base_center":
        return np.array([c[0], c[1], lo[2]])
    if mode == "center":
        return c
    if mode == "min_corner":
        return lo
    return None


def clean(a: Analysis, opts: CleanOptions, out_path: str) -> dict:
    model = r.File3dm.Read(a.path)
    actions: list[str] = []
    infos = {i.index: i for i in a.infos}

    remove: set[int] = set()
    if opts.remove_outliers and a.outliers:
        remove |= a.outliers
        actions.append(f"Deleted {len(a.outliers)} stray object(s) lying far away from the model.")
    if opts.remove_invalid and a.invalid:
        remove |= a.invalid
        actions.append(f"Deleted {len(a.invalid)} invalid object(s).")
    if opts.remove_degenerate and a.degenerate:
        remove |= a.degenerate
        actions.append(f"Deleted {len(a.degenerate)} zero-size object(s).")
    if opts.remove_duplicates and a.duplicates:
        dup = a.duplicates - remove
        remove |= dup
        if dup:
            actions.append(f"Deleted {len(dup)} duplicate object(s).")
    if opts.remove_curves:
        cur = {i.index for i in a.infos if i.type in ("Curve", "Point")} - remove
        remove |= cur
        if cur:
            actions.append(f"Deleted {len(cur)} curve/point object(s).")

    keep = [i for i in a.infos if i.index not in remove]
    kept_idx = {i.index for i in keep}
    bb = mio.overall_bbox(a.infos, kept_idx)

    # Translation from current position to the requested origin placement.
    shift = np.zeros(3)
    if bb is not None:
        tgt = _target_point(bb[0], bb[1], opts.origin)
        if tgt is not None:
            shift = -tgt
            dist = float(np.linalg.norm(tgt)) * a.unit_mm
            if dist > 1e-6:
                where = {"base_center": "centred on the origin, sitting on the ground plane",
                         "center": "centred on the origin", "min_corner": "with its corner at the origin"}[opts.origin]
                actions.append(f"Moved the model {mio.fmt_len(dist)} so it is {where}.")
    move = r.Vector3d(*shift.tolist())

    # Collect replacement geometry, then delete originals and re-add.
    new_objects = []
    mesh_fixes, faces_before, faces_after = 0, 0, 0
    for info in keep:
        obj = model.Objects[info.index]
        attrs = obj.Attributes
        geom = obj.Geometry
        if info.type == "Mesh" and info.index in a.meshes and (opts.repair_meshes or opts.decimate_meshes):
            tm = a.meshes[info.index]  # relative to a.offset -> precise
            faces_before += len(tm.faces)
            if opts.repair_meshes:
                tm, _ = meshtools.basic_repair(tm, a.tolerance)
            if opts.decimate_meshes and info.index in a.dense:
                tm = meshtools.decimate(tm, meshtools.sensible_face_target(tm, opts.max_mesh_faces))
            elif opts.decimate_meshes and len(tm.faces) > opts.max_mesh_faces:
                tm = meshtools.decimate(tm, opts.max_mesh_faces)
            faces_after += len(tm.faces)
            mesh_fixes += 1
            verts = tm.vertices + (a.offset + shift)
            geom = meshtools.to_rhino_mesh(verts, tm.faces)
        else:
            geom = geom.Duplicate()
            if np.any(shift):
                geom.Translate(move)
        new_objects.append((geom, attrs))

    if mesh_fixes:
        msg = f"Repaired {mesh_fixes} mesh(es) (welded vertices, removed degenerate/duplicate faces, fixed normals)"
        if faces_after < faces_before:
            msg += f"; triangles {faces_before:,} -> {faces_after:,}"
        actions.append(msg + ".")

    # Remove every top-level object, then add the cleaned versions.
    ids = [str(model.Objects[i.index].Attributes.Id) for i in a.infos]
    for oid in ids:
        model.Objects.Delete(oid)
    for geom, attrs in new_objects:
        model.Objects.Add(geom, attrs)

    if not actions:
        actions.append("No problems needed fixing - the model was rewritten unchanged.")
    # Save in the same Rhino version as the upload so older Rhino can still open it
    # (ArchiveVersion is 50/60/70/80 for Rhino 5-8).
    av = int(model.ArchiveVersion or 80)
    version = min(8, max(5, av // 10 if av >= 50 else av))
    ok = model.Write(out_path, version) or (version != 8 and model.Write(out_path, 8))
    if not ok:
        raise RuntimeError("Failed to write the cleaned .3dm file.")

    dims = ((bb[1] - bb[0]) * a.unit_mm).tolist() if bb else [0, 0, 0]
    return {
        "actions": actions,
        "objects_before": len(a.infos),
        "objects_after": len(new_objects),
        "file_size_before": a.file_size,
        "file_size_after": os.path.getsize(out_path),
        "dims_mm": dims,
        "shift": shift.tolist(),
        "removed": sorted(remove),
    }
