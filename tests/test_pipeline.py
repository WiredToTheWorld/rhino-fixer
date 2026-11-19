"""End-to-end checks for both modes.

Run:  .venv\\Scripts\\python -m pytest tests -q      (or just run this file)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import rhino3dm as r
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhinofixer import analyze, clean, meshtools, printprep  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "sample files" / "the broken model.3dm"
FAR = 5_000_000.0  # 5 km away, in mm


def make_messy_model(path: str, offset: float = FAR) -> None:
    """Over-dense sphere + overlapping box (both closed), a duplicate, and a stray point 50 km out."""
    m = r.File3dm()
    m.Settings.ModelUnitSystem = r.UnitSystem.Millimeters
    sphere = trimesh.creation.icosphere(subdivisions=7, radius=20)        # ~327k faces: absurdly dense
    box = trimesh.creation.box(extents=[30, 30, 30])
    box.apply_translation([20, 0, 0])
    # Sphere stored single-precision (like older files): far out, its tiny triangles collapse.
    s = r.Mesh()
    for x, y, z in sphere.vertices + [offset, offset, 0]:
        s.Vertices.Add(float(x), float(y), float(z))
    for a, b, c in sphere.faces:
        s.Faces.AddFace(int(a), int(b), int(c))
    m.Objects.AddMesh(s)
    for tm in (box, box):                                                  # box added twice -> duplicate
        v = tm.vertices + [offset, offset, 0]
        m.Objects.AddMesh(meshtools.to_rhino_mesh(v, tm.faces))
    m.Objects.AddPoint(r.Point3d(offset + 50_000_000, offset, 0))
    assert m.Write(path, 8)


def test_messy_model():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "messy.3dm")
        make_messy_model(src)
        a = analyze.analyze(src)
        codes = {i.code for i in a.issues}
        # 5 km out, Rhino's float vertices collapse the tiny sphere triangles -> damaged mesh.
        assert {"far_from_origin", "outliers", "duplicates", "bad_meshes"} <= codes, codes

        out = os.path.join(d, "fixed.3dm")
        rep = clean.clean(a, clean.CleanOptions(), out)
        assert rep["objects_after"] == 2, rep
        fixed = analyze.analyze(out)
        left = {i.code for i in fixed.issues}
        assert not {"far_from_origin", "outliers", "duplicates", "dense_meshes", "bad_meshes", "invalid"} & left, left
        lo, hi = fixed.main_bbox()
        assert abs(lo[2]) < 0.01 and abs((lo[0] + hi[0]) / 2) < 0.01

        # Damaged geometry still comes out watertight (via the voxel path).
        mesh, prep = printprep.prepare(a, printprep.PrintOptions(scale=2.0, resolution="draft"))
        assert prep["watertight"], prep["steps"]
        assert abs(max(prep["dims_mm"]) - 2 * 55) < 1.0  # sphere -20..20 + box to 35 -> 55 mm wide


def test_closed_solids_union_exactly():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "near.3dm")
        make_messy_model(src, offset=0.0)
        a = analyze.analyze(src)
        assert "dense_meshes" in {i.code for i in a.issues}
        mesh, prep = printprep.prepare(a, printprep.PrintOptions(scale=2.0))
        assert prep["method"] == "exact", prep["steps"]
        assert prep["watertight"] and prep["bodies"] == 1
        assert abs(max(prep["dims_mm"]) - 2 * 55) < 0.05


def test_duplicates_need_matching_shape():
    """Different objects that merely share a bounding box must not count as duplicates."""
    with tempfile.TemporaryDirectory() as d:
        m = r.File3dm()
        m.Settings.ModelUnitSystem = r.UnitSystem.Millimeters
        m.Objects.AddLine(r.Point3d(0, 0, 0), r.Point3d(10, 10, 0))
        m.Objects.AddLine(r.Point3d(0, 10, 0), r.Point3d(10, 0, 0))   # same bbox, different line
        m.Objects.AddLine(r.Point3d(0, 0, 0), r.Point3d(10, 10, 0))   # true copy of the first
        p = os.path.join(d, "lines.3dm")
        assert m.Write(p, 8)
        assert analyze.analyze(p).duplicates == {2}


def test_clean_keeps_rhino_version():
    if not SAMPLE.exists():
        return
    with tempfile.TemporaryDirectory() as d:
        v7 = os.path.join(d, "v7.3dm")
        assert r.File3dm.Read(str(SAMPLE)).Write(v7, 7)
        out = os.path.join(d, "v7-fixed.3dm")
        clean.clean(analyze.analyze(v7), clean.CleanOptions(), out)
        assert r.File3dm.ReadArchiveVersion(out) == 70


def test_sample_model():
    if not SAMPLE.exists():
        return
    a = analyze.analyze(str(SAMPLE))
    assert "far_from_origin" in {i.code for i in a.issues}
    mesh, prep = printprep.prepare(a, printprep.PrintOptions(scale=0.25, resolution="draft"))
    assert prep["watertight"], prep["steps"]
    target = max(printprep.gather_mesh(a).extents) * 0.25
    assert abs(max(prep["dims_mm"]) - target) < 0.2, (prep["dims_mm"], target)


if __name__ == "__main__":
    test_messy_model()
    print("messy model: ok")
    test_closed_solids_union_exactly()
    print("closed solids: ok")
    test_duplicates_need_matching_shape()
    print("duplicate detection: ok")
    test_clean_keeps_rhino_version()
    print("rhino version kept: ok")
    test_sample_model()
    print("sample model: ok")
