"""Rhino Fixer web server.

Run:  .venv\\Scripts\\python server.py   then open http://127.0.0.1:8765

Environment variables (all optional): HOST, PORT, JOBS_DIR, MAX_UPLOAD_MB,
JOB_TTL_HOURS, MAX_CONCURRENT_JOBS, MAX_JOBS_IN_MEMORY, HOSTED=1 (changes the
privacy note in the UI).
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import secrets
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rhinofixer import analyze as an
from rhinofixer import clean as cl
from rhinofixer import meshtools
from rhinofixer import model_io as mio
from rhinofixer import printprep as pp

log = logging.getLogger("rhino-fixer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).parent
# Settings come from environment variables so the same code runs locally and hosted.
JOBS_DIR = Path(os.environ.get("JOBS_DIR", ROOT / "jobs")).resolve()
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = int(os.environ.get("MAX_UPLOAD_MB", "1024")) * 1024 * 1024
JOB_TTL = float(os.environ.get("JOB_TTL_HOURS", "6")) * 3600
# Each session keeps its analysed meshes in memory; cap how many are kept.
MAX_JOBS_IN_MEMORY = int(os.environ.get("MAX_JOBS_IN_MEMORY", "8"))
PREVIEW_FACES = 300_000
# Voxel remeshing can use ~1 GB, so only let a few heavy jobs run at once.
HEAVY_JOBS = threading.BoundedSemaphore(int(os.environ.get("MAX_CONCURRENT_JOBS", "4")))
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
FILE_NAME_RE = re.compile(r"^[A-Za-z0-9._ -]{1,120}$")

TYPE_COLORS = {"Brep": [170, 190, 215, 255], "Extrusion": [170, 190, 215, 255], "SubD": [215, 190, 160, 255],
               "Mesh": [180, 205, 175, 255], "InstanceReference": [200, 180, 210, 255]}

app = FastAPI(title="Rhino Fixer", docs_url=None, redoc_url=None, openapi_url=None)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


# ------------------------------------------------------------------ security headers
def _importmap_hash() -> str:
    """CSP hash of the inline <script type="importmap"> in index.html."""
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
    if not m:
        return ""
    digest = hashlib.sha256(m.group(1).encode("utf-8")).digest()
    return f"'sha256-{base64.b64encode(digest).decode()}'"


CSP = "; ".join([
    "default-src 'self'",
    f"script-src 'self' https://cdn.jsdelivr.net {_importmap_hash()}",
    "connect-src 'self' https://cdn.jsdelivr.net",
    "img-src 'self' data: blob:",
    "style-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "frame-ancestors 'none'",
    "form-action 'self'",
])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    # Reject oversized uploads before the body is read (FastAPI would otherwise
    # receive the whole multipart body before the endpoint can check its size).
    if request.method == "POST" and request.url.path == "/api/upload":
        declared = request.headers.get("content-length", "")
        if not declared.isdigit():
            return JSONResponse({"detail": "Upload size is required."}, status_code=411)
        if int(declared) > MAX_UPLOAD + 1024 * 1024:  # allow for multipart overhead
            return JSONResponse({"detail": f"File is larger than {MAX_UPLOAD // (1024 * 1024)} MB."},
                                status_code=413)
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


# ------------------------------------------------------------------ job store
def _sweep_jobs() -> None:
    """Forget sessions that are expired or beyond the memory cap, and delete their files.
    Never touches a session that is currently being processed."""
    now = time.time()
    with LOCK:
        idle = [(j["touched"], jid) for jid, j in JOBS.items() if j["busy"] == 0]
        expired = {jid for t, jid in idle if now - t > JOB_TTL}
        keep = sorted((x for x in idle if x[1] not in expired), reverse=True)
        busy_count = len(JOBS) - len(idle)
        overflow = {jid for _, jid in keep[max(0, MAX_JOBS_IN_MEMORY - busy_count):]}
        drop = expired | overflow
        for jid in drop:
            JOBS.pop(jid, None)
        known = set(JOBS)
    for jid in drop:
        shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)
    # Orphaned folders (e.g. from before a restart).
    for d in JOBS_DIR.iterdir():
        if d.is_dir() and d.name not in known and now - d.stat().st_mtime > JOB_TTL:
            shutil.rmtree(d, ignore_errors=True)


@contextmanager
def _use_job(job_id: str):
    """Look up a session and mark it busy so the sweeper leaves it alone."""
    with LOCK:
        job = JOBS.get(job_id) if JOB_ID_RE.match(job_id) else None
        if job is None:
            raise HTTPException(404, "This session has expired - please upload the file again.")
        job["busy"] += 1
        job["touched"] = time.time()
    try:
        yield job
    finally:
        with LOCK:
            job["busy"] -= 1
            job["touched"] = time.time()


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "", Path(name).stem).strip(" .") or "model"
    return stem[:80]


# ------------------------------------------------------------------ preview helpers
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
    lines = [pl for i, ls in a.lines.items() if i not in skip for pl in ls]
    segs = mio.polylines_to_segments(lines)
    if not len(segs):
        return False
    ((segs + translate) * a.unit_mm).astype("<f4").tofile(path)
    return True


def _print_dims(a: an.Analysis, skip: set[int]) -> list[float] | None:
    """Size in mm of everything the print option would use (no mesh copying)."""
    bounds = [m.bounds for i, m in a.meshes.items() if i not in skip]
    if not bounds:
        return None
    lo = np.min([b[0] for b in bounds], axis=0)
    hi = np.max([b[1] for b in bounds], axis=0)
    return ((hi - lo) * a.unit_mm).tolist()


# ------------------------------------------------------------------ API
@app.get("/api/config")
def config():
    return {"max_upload_mb": MAX_UPLOAD // (1024 * 1024), "job_ttl_hours": JOB_TTL / 3600,
            "hosted": os.environ.get("HOSTED", "") == "1"}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    name = file.filename or ""
    if not name.lower().endswith(".3dm"):
        raise HTTPException(400, "Please upload a Rhino .3dm file.")
    _sweep_jobs()

    job_id = secrets.token_hex(16)
    d = JOBS_DIR / job_id
    d.mkdir()
    try:
        src = d / "input.3dm"
        size = 0
        with open(src, "wb") as fh:
            while chunk := file.file.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, f"File is larger than {MAX_UPLOAD // (1024 * 1024)} MB.")
                fh.write(chunk)
        with HEAVY_JOBS:
            try:
                a = an.analyze(str(src))
            except ValueError as e:
                raise HTTPException(422, str(e))
            infos = {i.index: i for i in a.infos}
            skip = a.outliers | a.invalid | a.degenerate
            prev = [(m.copy().apply_scale(a.unit_mm), TYPE_COLORS.get(infos[i].type, [190, 190, 190, 255]))
                    for i, m in a.meshes.items() if i not in skip]
            if prev:
                _preview_glb(prev, d / "original.glb")
            del prev
            has_lines = _write_lines(a, skip, np.zeros(3), d / "original-lines.bin")
            print_dims = _print_dims(a, a.outliers | a.invalid | a.degenerate | a.duplicates)
            summary = an.summary(a)
    except HTTPException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    except Exception:
        log.exception("upload failed")
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(422, "Could not read this model. Is it a valid Rhino .3dm file?")

    with LOCK:
        JOBS[job_id] = {"analysis": a, "dir": d, "stem": _safe_stem(name), "touched": time.time(), "busy": 0}
    _sweep_jobs()  # enforce the memory cap now that the new session is counted
    return {
        "job": job_id,
        "name": name,
        "summary": summary,
        "print_dims_mm": print_dims,
        "preview": f"/api/{job_id}/file/original.glb" if (d / "original.glb").exists() else None,
        "lines": f"/api/{job_id}/file/original-lines.bin" if has_lines else None,
        "printers": pp.PRINTERS,
    }


class CleanBody(BaseModel):
    origin: Literal["base_center", "center", "min_corner", "none"] = "base_center"
    remove_outliers: bool = True
    remove_invalid: bool = True
    remove_degenerate: bool = True
    remove_duplicates: bool = True
    remove_curves: bool = False
    repair_meshes: bool = True
    decimate_meshes: bool = True
    max_mesh_faces: int = Field(200_000, ge=5_000, le=5_000_000)


@app.post("/api/{job_id}/clean")
def clean(job_id: str, body: CleanBody):
    with _use_job(job_id) as job:
        a: an.Analysis = job["analysis"]
        name = f"{job['stem']}-fixed.3dm"
        d = job["dir"]
        try:
            with HEAVY_JOBS:
                rep = cl.clean(a, cl.CleanOptions(**body.model_dump()), str(d / name))
                # Preview of the cleaned model (meshes moved by the same shift).
                infos = {i.index: i for i in a.infos}
                removed = set(rep["removed"])
                shift = np.array(rep["shift"]) + a.offset
                prev = []
                for i, m in a.meshes.items():
                    if i in removed:
                        continue
                    mm = m.copy()
                    mm.apply_translation(shift)
                    mm.apply_scale(a.unit_mm)
                    prev.append((mm, TYPE_COLORS.get(infos[i].type, [190, 190, 190, 255])))
                if prev:
                    _preview_glb(prev, d / "cleaned.glb")
                has_lines = _write_lines(a, removed, shift, d / "cleaned-lines.bin")
        except Exception:
            log.exception("clean failed")
            raise HTTPException(500, "Clean-up failed on this model.")
    rep["download"] = f"/api/{job_id}/file/{name}"
    rep["preview"] = f"/api/{job_id}/file/cleaned.glb" if prev else None
    rep["lines"] = f"/api/{job_id}/file/cleaned-lines.bin" if has_lines else None
    rep["filename"] = name
    return rep


class PrintBody(BaseModel):
    # Bounds keep a single request from asking for an unbounded voxel grid.
    scale: float = Field(1.0, gt=0, le=10_000)
    printer: Literal[tuple(pp.PRINTERS)] = "x1c"  # type: ignore[valid-type]
    resolution: Literal["draft", "standard", "fine"] = "standard"
    min_wall_mm: float = Field(0.8, ge=0, le=10)
    close_gaps_mm: float = Field(1.0, ge=0, le=20)
    min_part_mm3: float = Field(1.0, ge=0, le=1_000_000)
    force_remesh: bool = False
    smooth: bool = True


@app.post("/api/{job_id}/print")
def make_printable(job_id: str, body: PrintBody):
    with _use_job(job_id) as job:
        d = job["dir"]
        stem = f"{job['stem']}-print"
        try:
            with HEAVY_JOBS:
                mesh, rep = pp.prepare(job["analysis"], pp.PrintOptions(**body.model_dump()))
                mesh.export(d / f"{stem}.3mf", file_type="3mf")
                mesh.export(d / f"{stem}.stl", file_type="stl")
                _preview_glb([(mesh, [235, 140, 60, 255])], d / "print.glb")
                del mesh
        except ValueError as e:
            raise HTTPException(422, str(e))
        except MemoryError:
            raise HTTPException(503, "Ran out of memory - try the Draft detail level or a smaller print size.")
        except Exception:
            log.exception("print preparation failed")
            raise HTTPException(500, "Could not make this model printable.")
    rep["downloads"] = {"3mf": f"/api/{job_id}/file/{stem}.3mf", "stl": f"/api/{job_id}/file/{stem}.stl"}
    rep["preview"] = f"/api/{job_id}/file/print.glb"
    return rep


@app.get("/api/{job_id}/file/{name}")
def get_file(job_id: str, name: str):
    if not JOB_ID_RE.match(job_id) or not FILE_NAME_RE.match(name) or name == "input.3dm":
        raise HTTPException(404, "File not found.")
    d = JOBS_DIR / job_id
    p = (d / name).resolve()
    if p.parent != d or not p.is_file():
        raise HTTPException(404, "File not found.")
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id]["touched"] = time.time()
    headers = {"Cache-Control": "no-store"}
    if p.suffix == ".glb":
        return FileResponse(p, media_type="model/gltf-binary", headers=headers)
    if p.suffix == ".bin":
        return FileResponse(p, media_type="application/octet-stream", headers=headers)
    return FileResponse(p, filename=name, media_type="application/octet-stream", headers=headers)


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8765"))
    host = os.environ.get("HOST", "127.0.0.1")  # the container sets 0.0.0.0
    uvicorn.run(app, host=host, port=port)
