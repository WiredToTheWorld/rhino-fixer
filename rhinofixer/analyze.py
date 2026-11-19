"""Inspect a .3dm file and list the problems that make it slow, huge or crash-prone."""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import rhino3dm as r
import trimesh

from . import model_io as mio

# A mesh is "unreasonably dense" if it has more faces than this...
DENSE_FACE_LIMIT = 250_000
# ...or its typical edge is tinier than its own size / this ratio (with >50k faces).
DENSE_EDGE_RATIO = 4000


@dataclass
class Issue:
    severity: str  # "error" | "warn" | "info"
    code: str
    title: str
    detail: str
    objects: list[int] = field(default_factory=list)

    def to_dict(self):
        return {"severity": self.severity, "code": self.code, "title": self.title,
                "detail": self.detail, "count": len(self.objects)}


@dataclass
class Analysis:
    path: str
    file_size: int
    units: str
    unit_mm: float
    tolerance: float
    infos: list[mio.ObjInfo]
    offset: np.ndarray                 # local origin used for meshing (model units)
    meshes: dict[int, trimesh.Trimesh]  # object index -> mesh relative to offset
    mesh_notes: list[str]
    issues: list[Issue]
    outliers: set[int]
    invalid: set[int]
    degenerate: set[int]
    duplicates: set[int]
    dense: set[int]
    lines: dict[int, list[np.ndarray]] = field(default_factory=dict)  # display edges, relative to offset

    @property
    def main_indices(self) -> set[int]:
        bad = self.outliers | self.invalid | self.degenerate
        return {i.index for i in self.infos} - bad

    def main_bbox(self):
        return mio.overall_bbox(self.infos, self.main_indices)


def _find_outliers(infos: list[mio.ObjInfo], unit_mm: float) -> set[int]:
    """Objects lying absurdly far away from everything else (stray points, lost copies)."""
    boxed = [i for i in infos if i.bbox is not None]
    if len(boxed) < 3:
        return set()
    centers = np.array([(b.bbox[0] + b.bbox[1]) / 2 for b in boxed])
    med = np.median(centers, axis=0)
    dist = np.linalg.norm(centers - med, axis=1)
    core = [b for b, d in zip(boxed, dist) if d <= np.percentile(dist, 80)]
    lo = np.min([b.bbox[0] for b in core], axis=0)
    hi = np.max([b.bbox[1] for b in core], axis=0)
    diag = max(float(np.linalg.norm(hi - lo)), 1.0 / unit_mm)
    out = set()
    for b in boxed:
        c = (b.bbox[0] + b.bbox[1]) / 2
        gap = np.linalg.norm(np.maximum(0, np.maximum(lo - c, c - hi)))
        if gap > 20 * diag and gap * unit_mm > 1000:
            out.add(b.index)
    return out


def analyze(path: str, subd_levels: int = 3) -> Analysis:
    model = r.File3dm.Read(path)
    if model is None:
        raise ValueError("Could not read this file - is it a valid Rhino .3dm file?")
    unit_mm = mio.unit_to_mm(model)
    tol = model.Settings.ModelAbsoluteTolerance or 0.01
    infos = mio.describe_objects(model)
    issues: list[Issue] = []

    invalid = {i.index for i in infos if not i.valid}
    degenerate = {i.index for i in infos if i.bbox is None or
                  (i.type != "Point" and np.linalg.norm(i.bbox[1] - i.bbox[0]) < tol)}
    degenerate -= invalid
    outliers = _find_outliers(infos, unit_mm) - invalid - degenerate

    # Duplicate objects: same type, same face count, same bounding box (to tolerance).
    seen, duplicates = {}, set()
    for i in infos:
        if i.bbox is None:
            continue
        key = (i.type, i.face_count, tuple(np.round(np.r_[i.bbox[0], i.bbox[1]] / (tol * 10)).astype(np.int64)))
        if key in seen:
            duplicates.add(i.index)
        else:
            seen[key] = i.index

    main = {i.index for i in infos} - outliers - invalid - degenerate
    bb = mio.overall_bbox(infos, main) or mio.overall_bbox(infos)
    offset = (bb[0] + bb[1]) / 2 if bb else np.zeros(3)

    # Mesh everything meshable (relative to the local offset for precision).
    meshes, notes, lines = {}, [], {}
    for i in infos:
        # Invalid meshes are usually just degenerate faces - repairable, so mesh them anyway.
        if not i.meshable or (i.index in invalid and i.type != "Mesh"):
            continue
        obj_notes = []
        try:
            tm = mio.object_to_trimesh(model, model.Objects[i.index].Geometry, offset, obj_notes, subd_levels)
        except Exception as e:  # never let one bad object kill the analysis
            obj_notes.append(f"could not be meshed ({e})")
            tm = None
        if tm is not None and len(tm.faces):
            meshes[i.index] = tm
            try:
                lines[i.index] = mio.object_lines(model, model.Objects[i.index].Geometry, offset, subd_levels)
            except Exception:
                pass  # lines are cosmetic only
        for n in obj_notes:
            i.notes.append(n)
            notes.append(f"{i.type} on layer '{i.layer}': {n}")

    bad_meshes = {i for i in invalid if i in meshes}
    invalid -= bad_meshes

    by_index = infos_by_index(infos)
    dense = set()
    for idx, tm in meshes.items():
        info = by_index[idx]
        if info.type != "Mesh":
            continue
        nf = len(tm.faces)
        if nf > DENSE_FACE_LIMIT:
            dense.add(idx)
        elif nf > 50_000:
            diag = float(np.linalg.norm(tm.extents))
            med_edge = float(np.median(tm.edges_unique_length))
            if med_edge > 0 and diag / med_edge > DENSE_EDGE_RATIO:
                dense.add(idx)

    # ---- issues -----------------------------------------------------------
    if bb is not None:
        far = float(np.max(np.abs(np.r_[bb[0], bb[1]]))) * unit_mm
        size = float(np.max(bb[1] - bb[0])) * unit_mm
        if far > 1_000_000 or (far > 10 * max(size, 1) and far > 100_000):
            issues.append(Issue("error", "far_from_origin", "Model is a long way from the origin",
                                f"Geometry sits {mio.fmt_len(far)} from 0,0,0 but is only {mio.fmt_len(size)} across. "
                                "Huge coordinates exhaust floating-point precision, which causes jittery display, "
                                "failed booleans/joins and crashes."))
        elif far > 10 * max(size, 1) and far > 10_000:
            issues.append(Issue("warn", "far_from_origin", "Model is offset from the origin",
                                f"Geometry sits {mio.fmt_len(far)} from 0,0,0. Moving it to the origin improves precision."))
    if outliers:
        issues.append(Issue("error", "outliers", "Stray objects far away from the model",
                            "These objects are thousands of model-sizes away. They inflate the model extents, "
                            "break Zoom Extents and hurt precision.", sorted(outliers)))
    if invalid:
        issues.append(Issue("error", "invalid", "Invalid geometry",
                            "Objects that fail Rhino's validity check are a common cause of crashes.", sorted(invalid)))
    if bad_meshes:
        issues.append(Issue("error", "bad_meshes", "Damaged meshes",
                            "Meshes with degenerate faces (often caused by sitting far from the origin). "
                            "Clean-up repairs them.", sorted(bad_meshes)))
    if degenerate:
        issues.append(Issue("warn", "degenerate", "Zero-size / degenerate objects",
                            "Objects smaller than the model tolerance.", sorted(degenerate)))
    if duplicates:
        issues.append(Issue("warn", "duplicates", "Duplicate objects",
                            "Objects stacked exactly on top of an identical copy.", sorted(duplicates)))
    if dense:
        tot = sum(len(meshes[d].faces) for d in dense)
        issues.append(Issue("warn", "dense_meshes", "Unreasonably dense meshes",
                            f"{tot:,} triangles in meshes far denser than their size needs.", sorted(dense)))
    open_solids = [i.index for i in infos if i.meshable and i.solid is False and i.type in ("Brep", "Mesh", "SubD")]
    if open_solids:
        issues.append(Issue("info", "open_geometry", "Open (non-solid) surfaces or meshes",
                            "Fine for modelling, but these can't be 3D printed directly. "
                            "The 3D-print option will solidify them.", open_solids))
    loose = [i.index for i in infos if i.type in ("Curve", "Point")]
    if loose:
        issues.append(Issue("info", "loose_curves", "Curves / points",
                            "Not printable; ignored by the 3D-print option.", loose))
    if notes:
        issues.append(Issue("warn", "mesh_notes", "Some objects could not be fully meshed", "; ".join(notes[:5])))
    if not meshes:
        issues.append(Issue("error", "nothing_meshable", "No printable geometry",
                            "No surfaces, solids, SubDs or meshes with display meshes were found."))
    if os.path.getsize(path) > 100 * 1024 * 1024:
        issues.append(Issue("warn", "large_file", "Very large file",
                            f"{os.path.getsize(path) / 1e6:,.0f} MB."))

    return Analysis(path=path, file_size=os.path.getsize(path), units=mio.unit_name(model), unit_mm=unit_mm,
                    tolerance=tol, infos=infos, offset=offset, meshes=meshes, mesh_notes=notes, issues=issues,
                    outliers=outliers, invalid=invalid, degenerate=degenerate, duplicates=duplicates, dense=dense,
                    lines=lines)


def infos_by_index(infos):
    return {i.index: i for i in infos}


def summary(a: Analysis) -> dict:
    bb = a.main_bbox()
    dims_mm = ((bb[1] - bb[0]) * a.unit_mm).tolist() if bb else [0, 0, 0]
    far = float(np.max(np.abs(np.r_[bb[0], bb[1]]))) * a.unit_mm if bb else 0
    counts = Counter(i.type for i in a.infos)
    return {
        "file_size": a.file_size,
        "units": a.units,
        "tolerance": a.tolerance,
        "object_count": len(a.infos),
        "counts": dict(counts),
        "triangles": int(sum(len(m.faces) for m in a.meshes.values())),
        "dims_mm": dims_mm,
        "distance_from_origin_mm": far,
        "issues": [i.to_dict() for i in a.issues],
    }
