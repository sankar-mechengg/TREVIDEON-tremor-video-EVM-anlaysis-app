# TREVIDEON

**TRemor Eulerian VIDEO magnificatiON** — open-source tremor identification from video.

TREVIDEON magnifies and measures hand tremor in ordinary videos. Upload one or many MP4 recordings of a hand
(optionally with a marker CSV of what happened when), choose a segment, and get:

* the segment, a **magnified** video (phase-based Eulerian Video Magnification, Riesz pyramid, ×15) and a
  **tremor-envelope overlay** (heatmap, micro-vectors, fingertip trails), frame-synchronised;
* **tremor metrics** per finger and for the whole hand: dominant frequency, RMS / P95 amplitude, tremor-band
  power, % of hand size; spectra, sliding-RMS envelopes, spectrograms, fingertip micro-trajectories;
* **hand isolation** (optional): the dominant hand is segmented (SAM 2.1 on a GPU, MediaPipe on CPU), the
  background removed, the hand scaled to a standard size and the wrist centre (+) pinned, so that voluntary
  arm movement disappears and amplitudes are comparable between people;
* **frame-timing repair**: repeated frames (screen / VR recordings, variable-frame-rate phones) are replaced
  by motion-interpolated frames; dropped frames are detected and reported;
* **marker statistics**: tremor metrics for every marked segment / event, for stress–tremor correlation;
* one **ZIP** with videos, `metrics.csv`, `marker_metrics.csv`, `analysis.json` and the settings.

All numbers come from optical flow (DIS) on the *un-magnified* video; magnification is only a visual aid.

Developed at the RISE Lab (Rehab, Instrumentation & Sensory Engineering Lab), Centre for Biomedical
Engineering, IIT Delhi. Made with ♥ by [Sankar Balasubramanian](https://www.linkedin.com/in/sankar4).
PI: Prof. Biswarup Mukherjee.

> TREVIDEON is a research tool, **not a medical device**. Do not use it for diagnosis or treatment decisions.

## Run it on your computer (Windows)

Requirements: Python 3.12 (3.11 / 3.13 also work), Node.js 18+, internet for the first run. An NVIDIA GPU is
optional (SAM 2.1 hand segmentation and GPU magnification; RTX 50-series needs the CUDA 12.8 build, which
`run.ps1` installs).

```powershell
.\run.ps1                                  # set-up on the first run, then opens http://127.0.0.1:8000/analyze
.\run.ps1 -DataDir "D:\study\data\raw"     # open a specific video folder
.\run.ps1 -Dev                             # website with hot reload (development)
```

If scripts are blocked: `powershell -ExecutionPolicy Bypass -File .\run.ps1`.
In the app, **Browse…** chooses the video folder; results go to the sibling `results` folder
(`…\data\raw` → `…\results`) and are reused as a cache. Locally there are no limits on file size or
segment length, and nothing leaves your computer.

Command line (no web app):

```powershell
backend\.venv\Scripts\python -m evm.pipeline video.mp4 --result-dir results\video --set start_s=40 duration_s=20 hand_isolation=true
```

## Recording

At least **100 fps** (tremor lies at 8–12 Hz; a video only resolves rhythms below half its frame rate),
constant frame rate, bright flicker-free light (beware 100 Hz lamp flicker on 50 Hz mains), the whole hand in
view for 20 s or more. Videos from 24 to 99 fps are accepted with a warning. See the *Recording guide* page.

## Marker CSV

```
time_s,end_s,marker,value,notes
5.0,65.0,BASELINE,,hand held still
70.0,130.0,CONDITION,Low,low-difficulty block
75.4,,FAULT,,ring touched the wire
131.0,,STRESS_RATING,3,self-report 1-10
```

`time_s` (required, seconds or m:ss), `end_s` (segment end), `marker` (required), `value`, `notes`. A template
is downloadable in the app.

## Layout

```
backend/            FastAPI server + processing (evm/), Dockerfile (Railway), modal_app.py (GPU worker)
  evm/pipeline.py   two-stage cached pipeline (analysis / overlay)
  evm/isolate.py    hand isolation (segmentation + wrist anchoring)
  evm/segment.py    SAM 2.1 / MediaPipe / landmark hand segmenters
  evm/magnify*.py   Riesz-pyramid EVM (NumPy CPU and PyTorch CUDA) and linear EVM
  evm/markers.py    marker CSV parsing and per-marker metrics
  evm/storage.py    local library vs. ephemeral hosted sessions
frontend/           React + Vite website (home with three.js / GSAP, analysis portal, guide, legal pages)
scripts/            helpers (e.g. packaging the home-page example)
docs/DEPLOY.md      hosting on Vercel (website) + Railway (API) + Modal (GPU)
```

## Hosting

The website is a static build (Vercel); the API runs in a container (Railway); with
`TREVIDEON_EXECUTOR=modal` the processing runs on a Modal GPU. Uploads are kept in temporary storage only and
deleted 60 min after the last activity (or on request). See [docs/DEPLOY.md](docs/DEPLOY.md).

## Citing

```bibtex
@software{trevideon2026,
  author    = {Sankar, B. and Mukherjee, Biswarup},
  title     = {{TREVIDEON}: {TRemor Eulerian VIDEO magnificatiON}},
  year      = {2026},
  version   = {1.0.0},
  publisher = {RISE Lab, Centre for Biomedical Engineering, Indian Institute of Technology Delhi},
  url       = {https://github.com/sankar-mechengg/TREVIDEON-tremor-video-EVM-anlaysis-app},
  license   = {Apache-2.0}
}
```

Please also cite Wu et al. 2012 (EVM), Wadhwa et al. 2014 (Riesz pyramids) and Kroeger et al. 2016 (DIS
optical flow); see `CITATION.cff`.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The RISE Lab and IIT Delhi names and logos
are trademarks of their owners and are not covered by the license.
