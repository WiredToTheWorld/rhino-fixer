"""Reading .3dm files and turning Rhino geometry into triangle meshes.

rhino3dm has no meshing kernel, so Breps/Extrusions use the render meshes
cached in the file, SubDs are meshed by subdividing their control net, and
block instances are expanded from their definitions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import rhino3dm as r
import trimesh

# Scale factor from a Rhino unit system to millimetres.
_UNIT_TO_MM = {
    "Angstroms": 1e-7,
    "Nanometers": 1e-6,
    "Microns": 0.001,
    "Millimeters": 1.0,
    "Centimeters": 10.0,
    "Decimeters": 100.0,
    "Meters": 1000.0,
    "Dekameters": 10_000.0,
    "Hectometers": 100_000.0,
    "Kilometers": 1_000_000.0,
    "Microinches": 2.54e-5,
    "Mils": 0.0254,
    "Inches": 25.4,
    "Feet": 304.8,
    "Yards": 914.4,
    "Miles": 1_609_344.0,
    "NauticalMiles": 1_852_000.0,
    "PrinterPoints": 25.4 / 72,
    "PrinterPicas": 25.4 / 6,
    "None": 1.0,  # unitless: assume millimetres
}


def unit_name(model: r.File3dm) -> str:
    return str(model.Settings.ModelUnitSystem).split(".")[-1]


def unit_to_mm(model: r.File3dm) -> float:
    return _UNIT_TO_MM.get(unit_name(model), 1.0)


def bbox_tuple(bb) -> tuple[np.ndarray, np.ndarray] | None:
    if bb is None or not bb.IsValid:
        return None
    lo = np.array([bb.Min.X, bb.Min.Y, bb.Min.Z], dtype=float)
    hi = np.array([bb.Max.X, bb.Max.Y, bb.Max.Z], dtype=float)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return None
    return lo, hi


def rhino_mesh_to_arrays(mesh: r.Mesh, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (shifted by -offset, in float64) and triangulated faces.

    Must read the double-precision vertices: the single-precision copy that
    plain indexing returns is snapped to ~16 mm steps when a model sits
    hundreds of km from the origin.
    """
    pts = mesh.Vertices.ToPoint3dArray()
    v = np.array([(p.X, p.Y, p.Z) for p in pts], dtype=float).reshape(-1, 3)
    v -= offset
    nf = len(mesh.Faces)
    quads = np.empty((nf, 4), dtype=np.int64)
    faces = mesh.Faces
    for i in range(nf):
        quads[i] = faces[i]
    tri_a = quads[:, [0, 1, 2]]
    is_quad = quads[:, 2] != quads[:, 3]
    tri_b = quads[is_quad][:, [0, 2, 3]]
    return v, np.vstack([tri_a, tri_b])


@dataclass
class ObjInfo:
    index: int
    id: str
    type: str
    layer: str
    name: str
    valid: bool
    bbox: tuple[np.ndarray, np.ndarray] | None
    meshable: bool
    solid: bool | None = None
    face_count: int = 0          # mesh faces (for meshes) or brep faces
    notes: list[str] = field(default_factory=list)


MESHABLE = ("Mesh", "Brep", "Extrusion", "SubD", "InstanceReference")


def geom_type(g) -> str:
    if isinstance(g, r.Mesh):
        return "Mesh"
    if isinstance(g, r.Brep):
        return "Brep"
    if isinstance(g, r.Extrusion):
        return "Extrusion"
    if isinstance(g, r.SubD):
        return "SubD"
    if isinstance(g, r.InstanceReference):
        return "InstanceReference"
    if isinstance(g, r.Curve):
        return "Curve"
    if isinstance(g, (r.Point, r.PointCloud)):
        return "Point"
    return type(g).__name__


def _layer_names(model: r.File3dm) -> dict[int, str]:
    out = {}
    for i in range(len(model.Layers)):
        lay = model.Layers[i]
        out[lay.Index] = lay.FullPath or lay.Name
    return out


def describe_objects(model: r.File3dm) -> list[ObjInfo]:
    layers = _layer_names(model)
    infos = []
    for i in range(len(model.Objects)):
        o = model.Objects[i]
        if o.Attributes.IsInstanceDefinitionObject:
            continue  # lives inside a block definition; handled via its instances
        g = o.Geometry
        t = geom_type(g)
        try:
            valid = bool(g.IsValid)
        except Exception:
            valid = True
        info = ObjInfo(
            index=i,
            id=str(o.Attributes.Id),
            type=t,
            layer=layers.get(o.Attributes.LayerIndex, "?"),
            name=o.Attributes.Name or "",
            valid=valid,
            bbox=bbox_tuple(g.GetBoundingBox()),
            meshable=t in MESHABLE,
        )
        if t == "Mesh":
            info.face_count = len(g.Faces)
            info.solid = bool(g.IsClosed)
        elif t == "Brep":
            info.face_count = len(g.Faces)
            info.solid = bool(g.IsSolid)
        elif t == "SubD":
            info.face_count = g.FaceCount
            info.solid = bool(g.IsSolid)
        infos.append(info)
    return infos


def _mesh_brep(g: r.Brep, offset, notes) -> list[tuple[np.ndarray, np.ndarray]]:
    parts, missing = [], 0
    for i in range(len(g.Faces)):
        m = g.Faces[i].GetMesh(r.MeshType.Any)
        if m is None or len(m.Faces) == 0:
            missing += 1
            continue
        parts.append(rhino_mesh_to_arrays(m, offset))
    if missing:
        notes.append(
            f"{missing} of {len(g.Faces)} surface faces have no render mesh saved in the file "
            "(re-save from Rhino with shaded display to include them)."
        )
    return parts


def _mesh_subd(g: r.SubD, offset, levels: int) -> list[tuple[np.ndarray, np.ndarray]]:
    s = g.Duplicate()
    # Each Catmull-Clark step multiplies faces by ~4; cap the result.
    while levels > 0 and s.FaceCount * (4 ** levels) > 400_000:
        levels -= 1
    if levels > 0:
        s.Subdivide(levels)
    m = r.Mesh.CreateFromSubDControlNet(s, False)
    return [rhino_mesh_to_arrays(m, offset)] if m is not None else []


def mesh_geometry(model, g, offset, notes, subd_levels=3, depth=0) -> list[tuple[np.ndarray, np.ndarray]]:
    t = geom_type(g)
    if t == "Mesh":
        return [rhino_mesh_to_arrays(g, offset)]
    if t == "Brep":
        return _mesh_brep(g, offset, notes)
    if t == "Extrusion":
        m = g.GetMesh(r.MeshType.Any)
        if m is None:
            notes.append("Extrusion has no render mesh saved in the file.")
            return []
        return [rhino_mesh_to_arrays(m, offset)]
    if t == "SubD":
        return _mesh_subd(g, offset, subd_levels)
    if t == "InstanceReference" and depth < 8:
        idef = model.InstanceDefinitions.FindId(g.ParentIdefId)
        if idef is None:
            notes.append("Block instance references a missing block definition.")
            return []
        x = g.Xform
        xf = np.array([[x.M00, x.M01, x.M02, x.M03], [x.M10, x.M11, x.M12, x.M13],
                       [x.M20, x.M21, x.M22, x.M23], [x.M30, x.M31, x.M32, x.M33]])
        out = []
        for oid in idef.GetObjectIds():
            child = model.Objects.FindId(oid)
            if child is None:
                continue
            for v, f in mesh_geometry(model, child.Geometry, np.zeros(3), notes, subd_levels, depth + 1):
                vh = np.c_[v, np.ones(len(v))] @ xf.T
                out.append((vh[:, :3] / vh[:, 3:4] - offset, f))
        return out
    return []


# ------------------------------------------------------------------ display lines
# Recreates the black lines Rhino draws in Shaded view: Brep/Extrusion edges and
# SubD face edges. Each line is an (N, 3) polyline relative to `offset`.

def _p(pt) -> tuple[float, float, float]:
    return (pt.X, pt.Y, pt.Z)


MAX_LINE_POINTS = 600_000  # cap on preview line samples per object (keeps huge files fast)
DIJKSTRA_CHUNK_ENTRIES = 20_000_000  # max (starts x vertices) table per shortest-path batch


def _curve_polyline(c, offset, max_n: int = 64) -> np.ndarray:
    dom = c.Domain
    try:
        linear = c.IsLinear()
    except Exception:
        linear = False
    n = 2 if linear else int(min(max_n, max(4, c.SpanCount * max(c.Degree, 1) * 3)))
    ts = np.linspace(dom.T0, dom.T1, n)
    return np.array([_p(c.PointAt(float(t))) for t in ts]) - offset


def _brep_lines(g: r.Brep, offset) -> list[np.ndarray]:
    ne = len(g.Edges)
    # Fewer samples per edge on Breps with very many edges.
    max_n = int(min(64, max(4, MAX_LINE_POINTS // max(ne, 1))))
    return [_curve_polyline(g.Edges[i], offset, max_n) for i in range(ne)]


def _subd_lines(g: r.SubD, offset, levels: int, mesh=None) -> list[np.ndarray]:
    """SubD edges on the limit surface: shortest paths over a fine subdivision
    mesh between each edge's exact limit end points.

    `mesh` may be the (vertices, faces) already built by _mesh_subd for this
    SubD with the same offset, to avoid subdividing twice.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    if mesh is None:
        parts = _mesh_subd(g, offset, levels)
        if not parts:
            return []
        mesh = parts[0]
    v, f = np.asarray(mesh[0]), np.asarray(mesh[1])
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    w = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    graph = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                       shape=(len(v), len(v))).tocsr()
    tree = cKDTree(v)

    edges = []
    it = g.Edges
    edge = it.First()
    for _ in range(g.EdgeCount):
        if edge is None:
            break
        if edge.VertexCount == 2:
            a = np.array(_p(edge.Vertex(0).SurfacePoint)) - offset
            b = np.array(_p(edge.Vertex(1).SurfacePoint)) - offset
            edges.append((a, b))
        edge = it.Next()
    if not edges:
        return []
    ends = np.array([[a, b] for a, b in edges])
    _, ia = tree.query(ends[:, 0])
    _, ib = tree.query(ends[:, 1])

    # Dijkstra returns a dense (starts x vertices) predecessor table, so run it
    # in chunks of start vertices to keep memory bounded (~20M entries a chunk).
    # `limit` stops each search once it is well past the longest edge.
    edge_len = np.linalg.norm(ends[:, 0] - ends[:, 1], axis=1)
    limit = float(edge_len.max() * 3 + w.max() * 4) if len(w) else np.inf
    starts = np.unique(ia)
    chunk = max(1, DIJKSTRA_CHUNK_ENTRIES // max(len(v), 1))
    by_start: dict[int, list[int]] = {}
    for k, s in enumerate(ia):
        by_start.setdefault(int(s), []).append(k)
    lines: list[np.ndarray | None] = [None] * len(edges)
    for c0 in range(0, len(starts), chunk):
        block = starts[c0:c0 + chunk]
        _, pred = dijkstra(graph, indices=block, return_predecessors=True, limit=limit)
        for row, s in enumerate(block):
            for k in by_start[int(s)]:
                a, b = edges[k]
                t = int(ib[k])
                path, cur = [t], t
                while cur != s and cur >= 0 and len(path) < 10_000:
                    cur = int(pred[row, cur])
                    path.append(cur)
                if cur != s:
                    lines[k] = np.array([a, b])  # unreachable - straight segment
                    continue
                pts = v[path[::-1]].copy()
                pts[0], pts[-1] = a, b
                lines[k] = pts
        del pred
    return [pl for pl in lines if pl is not None]


def object_lines(model, g, offset, subd_levels=3, depth=0, mesh=None) -> list[np.ndarray]:
    t = geom_type(g)
    if t == "Brep":
        return _brep_lines(g, offset)
    if t == "Extrusion":
        b = g.ToBrep(False)
        return _brep_lines(b, offset) if b is not None else []
    if t == "SubD":
        return _subd_lines(g, offset, subd_levels, mesh)
    if t == "InstanceReference" and depth < 8:
        idef = model.InstanceDefinitions.FindId(g.ParentIdefId)
        if idef is None:
            return []
        x = g.Xform
        xf = np.array([[x.M00, x.M01, x.M02, x.M03], [x.M10, x.M11, x.M12, x.M13],
                       [x.M20, x.M21, x.M22, x.M23], [x.M30, x.M31, x.M32, x.M33]])
        out = []
        for oid in idef.GetObjectIds():
            child = model.Objects.FindId(oid)
            if child is None:
                continue
            for pl in object_lines(model, child.Geometry, np.zeros(3), subd_levels, depth + 1):
                ph = np.c_[pl, np.ones(len(pl))] @ xf.T
                out.append(ph[:, :3] / ph[:, 3:4] - offset)
        return out
    return []  # meshes: Rhino shows no wires in Shaded view either


def polylines_to_segments(lines: list[np.ndarray]) -> np.ndarray:
    """(M, 2, 3) segment pairs for a LineSegments buffer."""
    segs = [np.stack([pl[:-1], pl[1:]], axis=1) for pl in lines if len(pl) >= 2]
    return np.concatenate(segs) if segs else np.zeros((0, 2, 3))


def object_to_trimesh(model, obj_geom, offset, notes, subd_levels=3) -> trimesh.Trimesh | None:
    parts = mesh_geometry(model, obj_geom, offset, notes, subd_levels)
    parts = [p for p in parts if len(p[1])]
    if not parts:
        return None
    meshes = [trimesh.Trimesh(v, f, process=False) for v, f in parts]
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def overall_bbox(infos: list[ObjInfo], only=None):
    boxes = [i.bbox for i in infos if i.bbox is not None and (only is None or i.index in only)]
    if not boxes:
        return None
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    return lo, hi


def fmt_len(mm: float) -> str:
    a = abs(mm)
    if a >= 1_000_000:
        return f"{mm / 1_000_000:,.2f} km"
    if a >= 1000:
        return f"{mm / 1000:,.2f} m"
    return f"{mm:,.1f} mm"


def safe_float(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None
