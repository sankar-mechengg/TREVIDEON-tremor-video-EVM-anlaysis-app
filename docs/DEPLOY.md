# Hosting TREVIDEON (Vercel + Railway + Modal)

```
browser ──► Vercel (static website, frontend/)
   │
   └──────► Railway (API container, backend/Dockerfile): sessions, chunked uploads (≤1 GB), scans, queue
                │  TREVIDEON_EXECUTOR=modal
                └──► Modal (GPU worker, backend/modal_app.py): hand isolation + EVM + measurement
```

Nothing is stored permanently: uploads and results live in the API container's temporary folder under a
random session id and are deleted after `TREVIDEON_TTL_MIN` minutes of inactivity, on "Delete my data", or
when the container restarts. Job files on the Modal Volume are deleted as soon as the results are fetched.

> Status: the local mode is tested, and the hosted mode was tested on one machine
> (`TREVIDEON_MODE=hosted`, local executor): sessions, chunked uploads, the 1080p clamp, session-scoped file
> access, the ZIP bundle and "Delete my data". The Docker image, Railway, Vercel and the Modal worker
> (modal_app.py, evm/modal_exec.py) have not been deployed yet; expect to adjust details (e.g. Modal API
> versions, Railway volume / memory limits) on the first deployment.

## 1. GPU worker on Modal

```bash
pip install modal
modal token new                      # once
cd backend
modal deploy modal_app.py            # builds the CUDA image (PyTorch cu128, MediaPipe, SAM 2.1) and deploys
```

`GPU = "L4"` in `modal_app.py` (24 GB; T4 cheaper, A10G / L40S faster). Billing is per second while a job
runs; containers scale to zero when idle.

## 2. API on Railway

1. New project → *Deploy from GitHub repo* → set the service **root directory** to `backend/`
   (Railway picks up `railway.json` and the `Dockerfile`).
2. Variables (see `backend/.env.example`):
   `TREVIDEON_MODE=hosted`, `TREVIDEON_EXECUTOR=modal`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`,
   `TREVIDEON_CORS=https://<your-vercel-domain>`, optional limits (`TREVIDEON_MAX_UPLOAD_MB=1024`,
   `TREVIDEON_MAX_SEGMENT_S=60`, `TREVIDEON_MAX_HEIGHT=1080`, `TREVIDEON_TTL_MIN=60`).
3. Give the service enough memory (≥ 4 GB) and ephemeral disk for uploads (several GB).
4. Health check: `GET /api/health`.

Without Modal (`TREVIDEON_EXECUTOR=local`) the jobs run on the container's CPU: it works, but a 60 s
1080p segment at 100 fps takes a long time.

## 3. Website on Vercel

1. New project → import the repo → **root directory** `frontend/` (framework: Vite; `vercel.json` sets the
   build and SPA rewrites).
2. Environment variable `VITE_API_BASE=https://<your-railway-domain>` (no trailing slash).
3. Deploy. Large uploads go directly from the browser to Railway in 8 MB chunks (never through Vercel).

## 3D Kinematics on the hosted site

The Modal worker builds the WiLoR code into its image; the checkpoint and the MANO model go on a separate Volume
(see the top of `backend/modal_app.py`). Without them the 3D tab runs MediaPipe. The hosted site never serves the
MANO mesh (only joints / rotations / series) and never offers the VibeSkin hand model.

**Licences to clear first:** MANO is licensed for non-commercial research only and "may not be made available in
any form to any third party"; WiLoR's weights are CC BY-NC-ND 4.0. Running them on a public web service for
other people may need permission from the Max Planck Institute (ps-license@tue.mpg.de) and the WiLoR authors.
Until then, deploy the hosted 3D tab with MediaPipe only (simply leave the `trevideon-models` Volume empty).

## 4. Before going public

* Review `PRIVACY.md` / the Privacy and Terms pages with IIT Delhi.
* Publish the repository (Apache-2.0) and update `APP.repo` in `frontend/src/site/content.js`.
* Optionally mint a Zenodo DOI for the release and add it to `CITATION.cff` and the Cite page.

## GPU hosting options (October 2026)

Railway has no GPUs (company statement, January 2026), so GPU processing needs a second provider or a
different host. Prices are approximate list prices from third-party comparisons; check before committing.

| Option | How TREVIDEON runs | Cost model |
|---|---|---|
| **Railway + Modal** (configured) | API on Railway, jobs as Modal GPU functions | L4 ≈ $0.80/h, per second, scales to zero |
| Railway + RunPod Serverless | same pattern with the Docker image | L4-class ≈ $0.68/h, per second |
| Google Cloud Run + L4 | whole backend in one GPU container | per second, scales to zero |
| RunPod GPU Pod (always on) | the Docker image 24/7 | ≈ $0.4–1.1/h, billed while idle |
| Hugging Face Spaces (Docker, GPU) | one Space | per minute while running |

Only the magnification (PyTorch) and SAM 2.1 segmentation use the GPU; optical flow (OpenCV DIS) stays on
the CPU, so give the GPU container several CPU cores too.
