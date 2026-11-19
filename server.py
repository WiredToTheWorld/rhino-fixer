"""Rhino Fixer web server.

Run:  .venv\\Scripts\\python server.py   then open http://127.0.0.1:8765

Environment variables (all optional): HOST, PORT, JOBS_DIR, MAX_UPLOAD_MB,
JOB_TTL_HOURS, MAX_CONCURRENT_JOBS, HOSTED=1 (changes the privacy note in the UI).
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import trimesh
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rhinofixer import analyze as an
from rhinofixer import clean as cl
from rhinofixer import meshtools
from rhinofixer import printprep as pp

ROOT = Path(__file__).parent
# Settings come from environment variables so the same code runs locally and hosted.
JOBS_DIR = Path(os.environ.get("JOBS_DIR", ROOT / "jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_MB", "1024")) * 1024 * 1024
JOB_TTL = float(os.environ.get("JOB_TTL_HOURS", "6")) * 3600
PREVIEW_FACES = 300_000
# Voxel remeshing can use ~1 GB, so only let a few heavy jobs run at once.
HEAVY_JOBS = threading.BoundedSemaphore(int(os.environ.get("MAX_CONCURRENT_JOBS", "4")))

app = FastAPI(title="Rhino Fixer")
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


@app.get("/api/config")
def config():
    return {"max_upload_mb": MAX_UPLOAD // (1024 * 1024), "job_ttl_hours": JOB_TTL / 3600,
            "hosted": os.environ.get("HOSTED", "") == "1"}

TYPE_COLORS = {"Brep": [170, 190, 215, 255], "Extrusion": [170, 190, 215, 255], "SubD": [215, 190, 160, 255],
               "Mesh": [180, 205, 175, 255], "InstanceReference": [200, 180, 210, 255]}


def _sweep_old_jobs():
    now = time.time()
    for d in JOBS_DIR.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > JOB_TTL:
            shutil.rmtree(d, ignore_errors=True)
            JOBS.pop(d.name, None)


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "", Path(name).stem).strip() or "model"
    return stem[:80]


def _preview_glb(meshes: list[tuple[trimesh.Trimesh, list[int]]], path: Path) -> None:
    total = sum(len(m.faces) for m, _ in meshes) or 1
    scene = trimesh.Scene()
    for i, (m, color) in enumerate(meshes):
        if total > PREVIEW_FACES:
            m = meshtools.decimate(m, max(500, int(len(m.faces) * PREVIEW_FACES / total)))
        m = m.copy()
        m.visual = trimesh.visual.ColorVisuals(m, face_colors=color)
        scene.add_geometry(m, geom_name=f"obj{i}")
    scene.export(path, file_type="glb")


def _write_lines(a: an.Analysis, skip: set[int], translate: np.ndarray, path: Path) -> bool:
    """Display edges as raw little-endian float32 xyz segment pairs, in mm."""
    from rhinofixer import model_io as mio

    lines = [pl for i, ls in a.lines.items() if i not in skip for pl in ls]
    segs = mio.polylines_to_segments(lines)
    if not len(segs):
        return False
    ((segs + translate) * a.unit_mm).astype("<f4").tofile(path)
    return True


def _job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "This session has expired - please upload the file again.")
    return job


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".3dm"):
        raise HTTPException(400, "Please upload a Rhino .3dm file.")
    with LOCK:
        _sweep_old_jobs()
    job_id = uuid.uuid4().hex[:12]
    d = JOBS_DIR / job_id
    d.mkdir()
    src = d / "input.3dm"
    size = 0
    with open(src, "wb") as fh:
        while chunk := file.file.read(4 * 1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                fh.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD // (1024 * 1024)} MB.")
            fh.write(chunk)
    try:
        with HEAVY_JOBS:
            a = an.analyze(str(src))
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(422, f"Could not read the model: {e}")

    infos = {i.index: i for i in a.infos}
    skip = a.outliers | a.invalid | a.degenerate
    prev = [(m.copy().apply_scale(a.unit_mm), TYPE_COLORS.get(infos[i].type, [190, 190, 190, 255]))
            for i, m in a.meshes.items() if i not in skip]
    if prev:
        _preview_glb(prev, d / "original.glb")
    has_lines = _write_lines(a, skip, np.zeros(3), d / "original-lines.bin")

    print_dims = None
    try:
        print_dims = pp.gather_mesh(a).extents.tolist()
    except ValueError:
        pass

    JOBS[job_id] = {"analysis": a, "dir": d, "stem": _safe_stem(file.filename)}
    return {
        "job": job_id,
        "name": file.filename,
        "summary": an.summary(a),
        "print_dims_mm": print_dims,
        "preview": f"/api/{job_id}/file/original.glb" if prev else None,
        "lines": f"/api/{job_id}/file/original-lines.bin" if has_lines else None,
        "printers": pp.PRINTERS,
    }


class CleanBody(BaseModel):
    origin: str = "base_center"
    remove_outliers: bool = True
    remove_invalid: bool = True
    remove_degenerate: bool = True
    remove_duplicates: bool = True
    remove_curves: bool = False
    repair_meshes: bool = True
    decimate_meshes: bool = True
    max_mesh_faces: int = 200_000


@app.post("/api/{job_id}/clean")
def clean(job_id: str, body: CleanBody):
    job = _job(job_id)
    a: an.Analysis = job["analysis"]
    name = f"{job['stem']}-fixed.3dm"
    out = job["dir"] / name
    try:
        with HEAVY_JOBS:
            rep = cl.clean(a, cl.CleanOptions(**body.model_dump()), str(out))
    except Exception as e:
        raise HTTPException(500, f"Clean-up failed: {e}")
    # Preview of the cleaned model (meshes moved by the same shift).
    infos = {i.index: i for i in a.infos}
    shift = np.array(rep["shift"]) + a.offset
    prev = []
    for i, m in a.meshes.items():
        if i in rep["removed"]:
            continue
        mm = m.copy()
        mm.apply_translation(shift)
        mm.apply_scale(a.unit_mm)
        prev.append((mm, TYPE_COLORS.get(infos[i].type, [190, 190, 190, 255])))
    if prev:
        _preview_glb(prev, job["dir"] / "cleaned.glb")
    has_lines = _write_lines(a, set(rep["removed"]), shift, job["dir"] / "cleaned-lines.bin")
    rep["download"] = f"/api/{job_id}/file/{name}"
    rep["preview"] = f"/api/{job_id}/file/cleaned.glb" if prev else None
    rep["lines"] = f"/api/{job_id}/file/cleaned-lines.bin" if has_lines else None
    rep["filename"] = name
    return rep


class PrintBody(BaseModel):
    scale: float = 1.0
    printer: str = "x1c"
    resolution: str = "standard"
    min_wall_mm: float = 0.8
    close_gaps_mm: float = 1.0
    min_part_mm3: float = 1.0
    force_remesh: bool = False
    smooth: bool = True


@app.post("/api/{job_id}/print")
def make_printable(job_id: str, body: PrintBody):
    job = _job(job_id)
    if not (0.0001 <= body.scale <= 10000):
        raise HTTPException(400, "Scale is out of range.")
    try:
        with HEAVY_JOBS:
            mesh, rep = pp.prepare(job["analysis"], pp.PrintOptions(**body.model_dump()))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except MemoryError:
        raise HTTPException(500, "Ran out of memory - try the Draft resolution or a smaller print size.")
    stem = f"{job['stem']}-print"
    d = job["dir"]
    mesh.export(d / f"{stem}.3mf", file_type="3mf")
    mesh.export(d / f"{stem}.stl", file_type="stl")
    _preview_glb([(mesh, [235, 140, 60, 255])], d / "print.glb")
    rep["downloads"] = {"3mf": f"/api/{job_id}/file/{stem}.3mf", "stl": f"/api/{job_id}/file/{stem}.stl"}
    rep["preview"] = f"/api/{job_id}/file/print.glb"
    return rep


@app.get("/api/{job_id}/file/{name}")
def get_file(job_id: str, name: str):
    d = (JOBS_DIR / job_id).resolve()
    p = (d / name).resolve()
    if d.parent != JOBS_DIR.resolve() or p.parent != d or not p.is_file() or name == "input.3dm":
        raise HTTPException(404, "File not found.")
    if p.suffix == ".glb":
        return FileResponse(p, media_type="model/gltf-binary")
    if p.suffix == ".bin":
        return FileResponse(p, media_type="application/octet-stream")
    return FileResponse(p, filename=name, media_type="application/octet-stream")


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8765"))
    host = os.environ.get("HOST", "127.0.0.1")  # the container sets 0.0.0.0
    uvicorn.run(app, host=host, port=port)
