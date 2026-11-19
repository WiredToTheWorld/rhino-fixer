import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ viewer
THREE.Object3D.DEFAULT_UP.set(0, 0, 1); // Rhino and printers are Z-up
const host = $("viewer");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x8890a0, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(1, -1.5, 2);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff, 0.5);
fill.position.set(-1, 1, 0.5);
scene.add(fill);

let modelGroup = null;
let helpers = new THREE.Group();
scene.add(helpers);
const loader = new GLTFLoader();

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + "px";
  renderer.domElement.style.height = h + "px";
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);
(function loop() {
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
})();

function clearHelpers() {
  scene.remove(helpers);
  helpers = new THREE.Group();
  scene.add(helpers);
}

function frame(box) {
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const r = Math.max(size.length() / 2, 1);
  const dist = r / Math.sin((camera.fov * Math.PI) / 360) * 1.05;
  camera.near = dist / 1000;
  camera.far = dist * 100;
  camera.updateProjectionMatrix();
  camera.position.copy(center).add(new THREE.Vector3(0.9, -1.4, 0.9).normalize().multiplyScalar(dist));
  controls.target.copy(center);
}

function groundGrid(box) {
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y) * 1.6;
  const step = Math.pow(10, Math.floor(Math.log10(span / 10)));
  const n = Math.ceil(span / step);
  const grid = new THREE.GridHelper(n * step, n, 0x9aa2b2, 0xc9ced8);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(0, 0, box.min.z);
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  helpers.add(grid);
  const axes = new THREE.AxesHelper(step * 2);
  helpers.add(axes);
}

function bedHelper(bed) {
  const [w, d, h] = bed;
  const plate = new THREE.Mesh(
    new THREE.PlaneGeometry(w, d),
    new THREE.MeshStandardMaterial({ color: 0x2b2f36, roughness: 0.9, transparent: true, opacity: 0.85 })
  );
  plate.position.set(w / 2, d / 2, -0.05);
  helpers.add(plate);
  const grid = new THREE.GridHelper(Math.max(w, d), Math.max(w, d) / 16, 0x555b66, 0x40454e);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(w / 2, d / 2, 0);
  helpers.add(grid);
  const vol = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(w, d, h)),
    new THREE.LineBasicMaterial({ color: 0xe8702a, transparent: true, opacity: 0.45 })
  );
  vol.position.set(w / 2, d / 2, h / 2);
  helpers.add(vol);
}

let edgeLines = null;
const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x15171c });

async function loadLines(url) {
  const buf = await (await fetch(url + "?t=" + Date.now())).arrayBuffer();
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buf), 3));
  return new THREE.LineSegments(geom, edgeMaterial);
}

async function showModel(url, label, { bed = null, lines = null } = {}) {
  if (modelGroup) scene.remove(modelGroup);
  clearHelpers();
  edgeLines = null;
  $("edgeToggle").hidden = true;
  if (!url) return;
  const [gltf, edges] = await Promise.all([
    loader.loadAsync(url + "?t=" + Date.now()),
    lines ? loadLines(lines).catch(() => null) : null,
  ]);
  modelGroup = gltf.scene;
  // trimesh writes our Z-up coordinates unchanged, and the camera is Z-up too.
  modelGroup.traverse((o) => {
    if (o.isMesh) {
      o.material = new THREE.MeshStandardMaterial({
        vertexColors: !!o.geometry.attributes.color, color: 0xffffff,
        roughness: 0.65, metalness: 0.05, side: THREE.DoubleSide, flatShading: false,
        // push faces back a touch so edge lines draw cleanly on top (like Rhino's Shaded view)
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      });
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    }
  });
  if (edges) {
    edgeLines = edges;
    edgeLines.visible = $("showEdges").checked;
    modelGroup.add(edgeLines);
    $("edgeToggle").hidden = false;
  }
  scene.add(modelGroup);
  modelGroup.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(modelGroup);
  if (bed) {
    bedHelper(bed);
    frame(box.clone().union(new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(bed[0], bed[1], box.max.z))));
  } else {
    groundGrid(box);
    frame(box);
  }
  $("viewerLabel").textContent = label;
  $("viewerLabel").hidden = false;
}

// ------------------------------------------------------------------ state
let job = null;        // upload response
let baseDims = null;   // [x,y,z] mm of the model at 100%
let printers = {};
let resolution = "standard";
let maxUploadMb = null;

const fmtMM = (v) => (v >= 1000 ? (v / 1000).toLocaleString(undefined, { maximumFractionDigits: 2 }) + " m"
  : v.toLocaleString(undefined, { maximumFractionDigits: 1 }) + " mm");
const fmtDist = (v) => (v >= 1e6 ? (v / 1e6).toLocaleString(undefined, { maximumFractionDigits: 2 }) + " km" : fmtMM(v));
const fmtBytes = (b) => (b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : (b / 1e3).toFixed(0) + " KB");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function busy(on, text = "Working…") {
  $("busy").hidden = !on;
  $("busyText").textContent = text;
  for (const b of [$("cleanBtn"), $("printBtn")]) b.disabled = on;
}

function showError(msg) {
  const e = $("error");
  e.hidden = !msg;
  e.textContent = msg || "";
}

async function api(url, opts) {
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch { /* not json */ }
  if (!res.ok) throw new Error((data && data.detail) || `Request failed (${res.status})`);
  return data;
}

// ------------------------------------------------------------------ upload
async function upload(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".3dm")) { showError("Please choose a Rhino .3dm file."); return; }
  if (maxUploadMb && file.size > maxUploadMb * 1024 * 1024) {
    showError(`This file is ${fmtBytes(file.size)}; the online version accepts up to ${maxUploadMb} MB. ` +
      `In Rhino, use File > Export Selected to save just the parts you need (keep Shaded view on so display meshes ` +
      `are included), or delete unused objects, then upload again.`);
    return;
  }
  showError(null);
  busy(true, `Uploading and analysing ${file.name}…`);
  try {
    const fd = new FormData();
    fd.append("file", file);
    job = await api("/api/upload", { method: "POST", body: fd });
    printers = job.printers;
    renderAnalysis();
    $("viewerEmpty").hidden = true;
    if (job.preview) await showModel(job.preview, "Original model (shown relative to its own centre)", { lines: job.lines });
  } catch (e) {
    showError(e.message);
  } finally {
    busy(false);
  }
}

function renderAnalysis() {
  const s = job.summary;
  $("intro").hidden = true;
  $("fileCard").hidden = false;
  $("modes").hidden = false;
  $("fileName").textContent = job.name;
  $("fileMeta").textContent = `${fmtBytes(s.file_size)} · units: ${s.units} · tolerance ${s.tolerance}`;
  const counts = Object.entries(s.counts).map(([k, v]) => `${v} ${k === "InstanceReference" ? "block" : k}${v > 1 ? "s" : ""}`).join(", ");
  const d = s.dims_mm;
  $("stats").innerHTML = `
    <div class="stat full"><div class="k">Objects</div><div class="v">${esc(counts || "none")}</div></div>
    <div class="stat full"><div class="k">Size</div><div class="v">${fmtMM(d[0])} × ${fmtMM(d[1])} × ${fmtMM(d[2])}</div></div>
    <div class="stat"><div class="k">Distance from origin</div><div class="v">${fmtDist(s.distance_from_origin_mm)}</div></div>
    <div class="stat"><div class="k">Display triangles</div><div class="v">${s.triangles.toLocaleString()}</div></div>`;
  const items = s.issues.length ? s.issues : [{ severity: "ok", title: "No problems found", detail: "The model looks healthy." }];
  $("issues").innerHTML = items.map((i) =>
    `<li class="${i.severity}"><b>${esc(i.title)}${i.count ? ` (${i.count})` : ""}</b><span>${esc(i.detail)}</span></li>`).join("");

  $("cleanResult").hidden = true;
  $("printResult").hidden = true;

  // print sizing
  baseDims = job.print_dims_mm;
  $("pPrinter").innerHTML = Object.entries(printers)
    .map(([k, p]) => `<option value="${k}">${esc(p.name)} - ${p.bed.join(" × ")} mm</option>`).join("");
  $("pPrinter").value = "x1c";
  $("printBtn").disabled = !baseDims;
  if (baseDims) {
    $("origSize").textContent = `Model at 100%: ${baseDims.map((v) => v.toFixed(1)).join(" × ")} mm`;
    setScale(fitsAt(1) ? 1 : fitScale());
  } else {
    $("origSize").textContent = "No printable geometry";
  }
}

// ------------------------------------------------------------------ sizing
let scale = 1;
const bed = () => printers[$("pPrinter").value].bed;
const fitsAt = (s) => baseDims.every((v, i) => v * s <= bed()[i] + 1e-6);
const fitScale = () => Math.min(...baseDims.map((v, i) => (bed()[i] - 2) / v)); // 1 mm margin each side

function setScale(s, except = null) {
  if (!baseDims || !isFinite(s) || s <= 0) return;
  scale = s;
  const ids = ["dX", "dY", "dZ"];
  ids.forEach((id, i) => { if (id !== except) $(id).value = (baseDims[i] * s).toFixed(1); });
  if (except !== "pScale") $("pScale").value = +(s * 100).toFixed(2);
  const ok = fitsAt(s);
  const fm = $("fitMsg");
  fm.className = "fit " + (ok ? "ok" : "bad");
  fm.textContent = ok ? `✓ Fits the ${printers[$("pPrinter").value].name} build volume`
    : `✗ Too big for this printer's ${bed().join(" × ")} mm build volume`;
}
["dX", "dY", "dZ"].forEach((id, i) => $(id).addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  if (v > 0) setScale(v / baseDims[i], id);
}));
$("pScale").addEventListener("input", (e) => { const v = parseFloat(e.target.value); if (v > 0) setScale(v / 100, "pScale"); });
$("pPrinter").addEventListener("change", () => setScale(scale));
$("fitBtn").addEventListener("click", () => setScale(fitScale()));
$("realBtn").addEventListener("click", () => setScale(1));
$("pRes").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  e.preventDefault();
  resolution = b.dataset.v;
  [...$("pRes").children].forEach((c) => c.classList.toggle("on", c === b));
});

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
  $("tab-clean").hidden = t.dataset.tab !== "clean";
  $("tab-print").hidden = t.dataset.tab !== "print";
}));

// ------------------------------------------------------------------ clean
$("cleanBtn").addEventListener("click", async () => {
  showError(null);
  busy(true, "Cleaning model…");
  try {
    const body = {
      origin: $("cOrigin").value,
      remove_outliers: $("cOutliers").checked,
      remove_invalid: $("cInvalid").checked,
      remove_degenerate: $("cDegenerate").checked,
      remove_duplicates: $("cDuplicates").checked,
      remove_curves: $("cCurves").checked,
      repair_meshes: $("cRepair").checked,
      decimate_meshes: $("cDecimate").checked,
      max_mesh_faces: parseInt($("cMaxFaces").value, 10) || 200000,
    };
    const r = await api(`/api/${job.job}/clean`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const el = $("cleanResult");
    el.hidden = false;
    el.innerHTML = `
      <h3>Done</h3>
      <div class="badges">
        <span class="badge">${r.objects_before} → ${r.objects_after} objects</span>
        <span class="badge">${fmtBytes(r.file_size_before)} → ${fmtBytes(r.file_size_after)}</span>
      </div>
      <ul>${r.actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>
      <a class="btn primary wide" href="${r.download}" download="${esc(r.filename)}">⬇ Download ${esc(r.filename)}</a>`;
    if (r.preview) await showModel(r.preview, "Cleaned model, now at the origin", { lines: r.lines });
  } catch (e) {
    showError(e.message);
  } finally {
    busy(false);
  }
});

// ------------------------------------------------------------------ print
$("printBtn").addEventListener("click", async () => {
  showError(null);
  if (!fitsAt(scale) && !confirm("The model is larger than the selected printer's build volume. Continue anyway?")) return;
  busy(true, "Building a watertight solid… this can take up to a minute");
  try {
    const body = {
      scale,
      printer: $("pPrinter").value,
      resolution,
      min_wall_mm: parseFloat($("pWall").value) || 0,
      close_gaps_mm: parseFloat($("pGaps").value) || 0,
      min_part_mm3: parseFloat($("pCrumbs").value) || 0,
      force_remesh: $("pForce").checked,
      smooth: $("pSmooth").checked,
    };
    const r = await api(`/api/${job.job}/print`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const d = r.dims_mm;
    const el = $("printResult");
    el.hidden = false;
    el.innerHTML = `
      <h3>Print-ready model</h3>
      <div class="badges">
        <span class="badge ${r.watertight ? "ok" : "bad"}">${r.watertight ? "✓ Watertight" : "✗ Not watertight"}</span>
        <span class="badge ${r.fits_bed ? "ok" : "bad"}">${r.fits_bed ? "✓ Fits bed" : "✗ Exceeds bed"}</span>
        <span class="badge">${d.map((v) => v.toFixed(1)).join(" × ")} mm</span>
        ${r.volume_cm3 != null ? `<span class="badge">${r.volume_cm3.toFixed(1)} cm³ solid</span>` : ""}
        <span class="badge">${r.triangles.toLocaleString()} triangles</span>
        ${r.bodies ? `<span class="badge">${r.bodies} part${r.bodies > 1 ? "s" : ""}</span>` : ""}
      </div>
      <ul>${r.steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>
      <div class="downloads">
        <a class="btn primary" href="${r.downloads["3mf"]}" download>⬇ 3MF (Bambu Studio)</a>
        <a class="btn" href="${r.downloads.stl}" download>⬇ STL</a>
      </div>`;
    await showModel(r.preview, `On the ${r.printer} build plate`, { bed: r.bed_mm });
  } catch (e) {
    showError(e.message);
  } finally {
    busy(false);
  }
});

fetch("/api/config").then((r) => r.json()).then((c) => {
  maxUploadMb = c.max_upload_mb;
  if (c.hosted) {
    $("privacyNote").textContent = `Files are processed on this app's server and deleted automatically after ` +
      `${+c.job_ttl_hours.toFixed(1)} hours. Maximum upload ${c.max_upload_mb} MB.`;
  }
}).catch(() => {});

$("showEdges").addEventListener("change", (e) => { if (edgeLines) edgeLines.visible = e.target.checked; });

// ------------------------------------------------------------------ drag & drop
const dz = $("dropZone");
for (const inp of [$("fileInput"), $("fileInput2")]) inp.addEventListener("change", (e) => upload(e.target.files[0]));
const wrap = document.querySelector(".viewer-wrap");
wrap.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("hover"); });
wrap.addEventListener("dragleave", () => dz.classList.remove("hover"));
wrap.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("hover");
  upload(e.dataTransfer.files[0]);
});
