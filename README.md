# Rhino Fixer

A web app that repairs Rhino `.3dm` files and makes them 3D-printable.

## Run it on your own computer

```powershell
.\run.ps1
```

This opens http://localhost:8765. The first run sets up a Python 3.13 environment in `.venv`.

## Host it online (Google Cloud Run)

The app runs on Google's servers from the `Dockerfile`. Cloud Run watches the GitHub repository and rebuilds and redeploys on every push to `main`. It scales to zero when nobody is using it, so normal use stays inside the monthly free tier: 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests. Each upload or print job takes roughly 5-20 seconds. Storing the built container image may cost a few pence a month.

**One-time setup** (all in the browser):

1. **GitHub:** create a repository and push this folder to it.
2. **Google Cloud:** sign in at https://console.cloud.google.com and create a project (e.g. `rhino-fixer`). Add a billing account when asked; Cloud Run requires one even for free-tier use.
3. **Budget alert (recommended):** go to **Billing → Budgets & alerts → Create budget**. Set it to e.g. £1 so you're emailed if anything is ever charged.
4. **Create the service:** go to **Cloud Run → Deploy container → Service**.
   - Choose **Continuously deploy from a repository**, then **Set up with Cloud Build**. Connect GitHub and pick this repository.
   - Branch `^main$`, build type **Dockerfile**, source location `/Dockerfile`.
   - **Region:** `europe-west1` (Belgium). It is a low-cost "Tier 1" region; London `europe-west2` uses up the free allowance faster.
   - **Authentication:** *Allow public access*.
   - **Billing:** *Request-based* (CPU only allocated while handling a request). This is what keeps it free.
   - **Scaling:** minimum instances `0`, **maximum instances `1`**. The app keeps each upload's session in memory, so it must run as a single copy.
   - **Container settings:** port `8080`, **memory `4 GiB`**, **CPU `2`**, request timeout `300` seconds, maximum concurrent requests `8`.
5. Click **Create**. The first build takes about 5 minutes. The app's address (`https://rhino-fixer-….run.app`) is shown at the top of the service page.

After that, every `git push` to `main` redeploys automatically. GitHub also runs the tests on each push (`.github/workflows/tests.yml`).

**Limits when hosted:**

- **Uploads are capped at 32 MB.** That's Cloud Run's request size limit. The page explains how to slim larger files.
- **Files are deleted after 2 hours, or when the service goes idle**, which is roughly 15 minutes after the last visitor.
- **The first visit after an idle spell takes a few seconds** while the service starts.

All of these are set in the `Dockerfile`.

## What it does

**On upload**, it analyses the file and reports problems:

- the model sitting far from the origin (precision loss → jitter, failed booleans, crashes)
- stray objects thousands of model-sizes away
- invalid, zero-size and duplicate objects
- damaged meshes (degenerate faces) and absurdly dense meshes
- open surfaces that can't be printed directly

**1 · Clean up model** writes a repaired `.3dm`:

- moves the model to the origin (centred, sitting on Z=0, or other options)
- deletes stray / invalid / zero-size / duplicate objects (each optional)
- repairs meshes and reduces over-dense ones
- NURBS, SubD and Brep geometry is kept exactly. It is only moved, and layers and materials are preserved.

**2 · 3D print** writes a watertight `.3mf` (for Bambu Studio) and `.stl`:

- pick a Bambu printer (A1 mini, A1, P1S/P1P, P2S, X1C/X1E, H2D, H2S), then set the size by any dimension, by scale %, or with **Fit to bed**
- if all bodies are already closed solids they are unioned exactly
- otherwise it voxel-remeshes: it seals gaps, fills enclosed volumes, thickens open surfaces to a minimum wall, removes loose crumbs and smooths the result. Output is watertight by construction, then rescaled to the requested size.
- the result is placed centred on the build plate

## Limitations

- `rhino3dm` has no meshing kernel, so Breps and Extrusions are meshed from the **render meshes saved in the file**. If faces come out missing, open the file in Rhino, switch to Shaded view and save again.
- SubDs are approximated by subdividing their control net.
- Voxel remeshing softens detail finer than the chosen resolution (Draft 0.8 mm / Standard 0.4 mm / Fine 0.25 mm, coarsened automatically for very large prints to bound memory).
- Thickening open sheets grows them slightly outward. The longest side is rescaled to exactly what you asked for, so the other sides can differ by a fraction of a millimetre.

## Layout

```
server.py              FastAPI app (upload / clean / print / downloads)
rhinofixer/
  model_io.py          .3dm reading, units, Rhino geometry -> triangle meshes
  analyze.py           problem detection
  clean.py             mode 1 - repaired .3dm
  printprep.py         mode 2 - watertight print mesh
  meshtools.py         repair / decimation helpers
web/                   front end (three.js viewer)
tests/test_pipeline.py end-to-end checks:  .venv\Scripts\python tests\test_pipeline.py
jobs/                  temporary per-upload working files (auto-deleted after 6 h)
```
