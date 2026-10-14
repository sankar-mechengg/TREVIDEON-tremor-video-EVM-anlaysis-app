"""TREVIDEON API server (FastAPI).

Local mode  (default)                 uvicorn app:app --port 8000        (run.ps1 does this)
Hosted mode TREVIDEON_MODE=hosted     ephemeral per-browser sessions, auto-deleted, upload limits

Environment
  TREVIDEON_MODE            local | hosted                                   (default local)
  TREVIDEON_DATA_DIR        local: folder with the raw videos                (default ../../data/raw if present)
  TREVIDEON_RESULTS_DIR     local: results folder                            (default: sibling results/)
  TREVIDEON_TMP             hosted: temporary storage root                   (default <tmp>/trevideon)
  TREVIDEON_TTL_MIN         hosted: minutes of inactivity before deletion    (default 60)
  TREVIDEON_MAX_UPLOAD_MB   hosted: per-video upload limit                   (default 1024)
  TREVIDEON_MAX_SEGMENT_S   hosted: longest analysed segment                 (default 60)
  TREVIDEON_MAX_HEIGHT      hosted: highest working resolution               (default 1080)
  TREVIDEON_EXECUTOR        local | modal                                    (default local)
  TREVIDEON_CORS            comma-separated allowed origins for the hosted frontend (default *)
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

from evm import config, markers as MK
from evm.jobs import EXECUTOR_KIND, JobManager, make_executor
from evm.kin3d import config3d as C3
from evm.kin3d.pipeline import keys_for3d, paths3d
from evm.kin3d.wilor import available as wilor_available
from evm.pipeline import keys_for, paths
from evm.scan import ScanManager
from evm.segment import available as seg_available
from evm.storage import LocalLibrary, SessionStore, safe_name, video_meta
from evm.video_io import make_thumbnail, probe

APP_NAME = "TREVIDEON"
APP_VERSION = "1.0.0"
HERE = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(HERE)
DIST_DIR = os.path.join(APP_ROOT, "frontend", "dist")
MODE = os.environ.get("TREVIDEON_MODE", "local").lower()
HOSTED = MODE == "hosted"
LIMITS = {
    "max_upload_mb": float(os.environ.get("TREVIDEON_MAX_UPLOAD_MB", 1024)) if HOSTED else None,
    "max_segment_s": float(os.environ.get("TREVIDEON_MAX_SEGMENT_S", 60)) if HOSTED else None,
    "max_height": int(os.environ.get("TREVIDEON_MAX_HEIGHT", 1080)) if HOSTED else None,
    "min_fps": 24.0, "recommended_fps": 100.0,
}
CHUNK = 8 * 1024 * 1024

if HOSTED:
    STORE = SessionStore(os.environ.get("TREVIDEON_TMP", os.path.join(tempfile.gettempdir(), "trevideon")),
                         float(os.environ.get("TREVIDEON_TTL_MIN", 60)))
else:
    _default_data = os.path.normpath(os.path.join(APP_ROOT, "..", "data", "raw"))
    _data = os.environ.get("TREVIDEON_DATA_DIR") or (_default_data if os.path.isdir(_default_data) else os.getcwd())
    STORE = LocalLibrary(_data, os.environ.get("TREVIDEON_RESULTS_DIR") or None)

app = FastAPI(title=f"{APP_NAME} API", version=APP_VERSION)
_origins = [o.strip() for o in os.environ.get("TREVIDEON_CORS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["Content-Disposition"])

JOBS = JobManager(make_executor(EXECUTOR_KIND)) if __name__ != "__mp_main__" else None
_scans: dict[str, ScanManager] = {}
_probe_cache: dict[str, tuple[float, dict]] = {}
_uploads: dict[str, dict] = {}


# ------------------------------------------------------------------------------------------------ context
def ctx(request: Request) -> str | None:
    """Session id (hosted) or None (local). Hosted requests must carry a valid session."""
    if not HOSTED:
        return None
    sid = request.headers.get("x-session") or request.query_params.get("s")
    if not STORE.valid(sid):
        raise HTTPException(401, "Session expired or missing - reload the page to start a new session.")
    STORE.touch(sid)
    return sid


def _scan_mgr(sid) -> ScanManager:
    key = sid or "_local"
    if key not in _scans:
        _scans[key] = ScanManager()
    return _scans[key]


def _video(name: str, sid) -> str:
    p = STORE.video_path(name, sid)
    if not p:
        raise HTTPException(404, f"Unknown video: {name}")
    return p


def _info(path: str) -> dict:
    mt = os.path.getmtime(path)
    c = _probe_cache.get(path)
    if c and c[0] == mt:
        return c[1]
    info = probe(path)
    info["size_bytes"] = os.path.getsize(path)
    _probe_cache[path] = (mt, info)
    return info


def fps_status(fps: float) -> str:
    if fps < LIMITS["min_fps"]:
        return "blocked"
    return "ok" if fps >= LIMITS["recommended_fps"] - 0.5 else "low"


def _json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    return obj


def _results_for(name: str, sid) -> list[dict]:
    rd = STORE.result_dir(name, sid)
    out = []
    for key_a in os.listdir(rd):
        pa = paths(rd, key_a)
        if not os.path.exists(pa["analysis"]):
            continue
        try:
            with open(pa["params"]) as fh:
                P = json.load(fh)
        except OSError:
            continue
        P.setdefault("repair_duplicates", False)
        P.setdefault("hand_isolation", False)
        overlays = []
        if os.path.isdir(pa["overlays"]):
            for f in os.listdir(pa["overlays"]):
                if f.endswith(".json") and os.path.exists(os.path.join(pa["overlays"], f[:-5] + ".mp4")):
                    try:
                        with open(os.path.join(pa["overlays"], f)) as fh:
                            oj = json.load(fh)
                        overlays.append({"key_b": f[:-5], "overlay": oj["overlay"], "created": oj.get("created")})
                    except (OSError, ValueError, KeyError):
                        continue
        out.append({"key_a": key_a, "params": P,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(pa["analysis"]))),
                    "overlays": sorted(overlays, key=lambda o: o.get("created") or "")})
    return sorted(out, key=lambda r: r["created"], reverse=True)


# ------------------------------------------------------------------------------------------------ config
@app.get("/api/health")
def health():
    return {"ok": True, "app": APP_NAME, "version": APP_VERSION, "mode": MODE}


@app.get("/api/config")
def get_config():
    seg = seg_available()
    return {"app": {"name": APP_NAME, "version": APP_VERSION}, "mode": MODE, "executor": EXECUTOR_KIND,
            "limits": LIMITS, "schema": config.SCHEMA, "defaults": config.DEFAULTS,
            "process_keys": config.PROCESS_KEYS, "overlay_keys": config.OVERLAY_KEYS,
            "version": config.PIPELINE_VERSION, "compute": {"gpu": seg["cuda"], "segmenters": seg},
            "ttl_min": STORE.ttl / 60 if HOSTED else None,
            "kin3d": {"schema": C3.SCHEMA3D, "defaults": C3.DEFAULTS3D, "process_keys": C3.PROCESS3D,
                      "display_keys": C3.DISPLAY3D, "version": C3.K3D_VERSION,
                      "models": {"wilor": wilor_available() if EXECUTOR_KIND == "local" else {"ready": True, "remote": True},
                                 "mediapipe": True},
                      "hand_model": {"vibeskin": (not HOSTED) and _hand_fbx("right") is not None},
                      "mesh": not HOSTED}}


# ------------------------------------------------------------------------------------------------ local library
class LibraryReq(BaseModel):
    data_dir: str
    results_dir: str | None = None


@app.get("/api/library")
def library(request: Request):
    if HOSTED:
        raise HTTPException(404, "Not available in hosted mode")
    return STORE.info()


@app.post("/api/library")
def set_library(req: LibraryReq):
    if HOSTED:
        raise HTTPException(404, "Not available in hosted mode")
    try:
        STORE.set_data_dir(req.data_dir, req.results_dir)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    return STORE.info()


_DIALOG = r"""
import sys, tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True); root.update()
p = filedialog.askdirectory(title=sys.argv[1], initialdir=sys.argv[2], mustexist=True)
root.destroy(); print(p or '')
"""


@app.post("/api/library/browse")
def browse(request: Request, kind: str = "data"):
    """Opens the operating system's folder dialog on this computer (local mode only)."""
    if HOSTED:
        raise HTTPException(404, "Not available in hosted mode")
    if request.client and request.client.host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(403, "The folder dialog can only be opened from this computer")
    title = "Select the folder with the raw videos" if kind == "data" else "Select the results folder"
    start = STORE.data_dir if kind == "data" else STORE.results_dir
    try:
        r = subprocess.run([sys.executable, "-c", _DIALOG, title, start], capture_output=True, text=True,
                           timeout=600, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise HTTPException(408, "No folder selected")
    path = r.stdout.strip()
    if not path:
        return {"cancelled": True, **STORE.info()}
    if kind == "data":
        STORE.set_data_dir(path)
    else:
        STORE.set_data_dir(STORE.data_dir, path)
    return {"cancelled": False, **STORE.info()}


# ------------------------------------------------------------------------------------------------ hosted sessions
@app.post("/api/session")
def new_session():
    if not HOSTED:
        return {"session": None, "mode": MODE}
    sid = STORE.create()
    return {"session": sid, "mode": MODE, "ttl_min": STORE.ttl / 60}


@app.get("/api/session")
def session_info(request: Request):
    sid = ctx(request)
    return STORE.info(sid) if HOSTED else STORE.info()


@app.delete("/api/session")
def delete_session(request: Request):
    """'Delete my data': removes every upload and result of this session immediately."""
    if not HOSTED:
        raise HTTPException(404, "Not available in local mode")
    sid = ctx(request)
    JOBS.drop_owner(sid)
    STORE.delete(sid)
    return {"deleted": True}


@app.post("/api/session/end")
async def end_session(request: Request):
    """Sent by the browser (navigator.sendBeacon) when the page is hidden for good. Browsers cannot tell a
    tab close from a reload, so the session is set to expire in ~2 minutes instead of being deleted at
    once: a reload touches it again and keeps it, a closed tab lets it expire."""
    if HOSTED:
        sid = request.query_params.get("s")
        if STORE.valid(sid):
            STORE.last[sid] = time.time() - STORE.ttl + 120
    return {"ok": True}


# ------------------------------------------------------------------------------------------------ uploads (hosted)
class UploadInit(BaseModel):
    filename: str
    size: int


@app.post("/api/uploads")
def upload_init(req: UploadInit, request: Request):
    sid = ctx(request)
    if not HOSTED:
        raise HTTPException(404, "Local mode reads videos from the selected folder")
    name = safe_name(req.filename)
    if not name.lower().endswith((".mp4", ".mov", ".m4v", ".mkv", ".avi")):
        raise HTTPException(400, "Please upload an MP4 (or MOV / MKV / AVI) video")
    if req.size > LIMITS["max_upload_mb"] * 1024 * 1024:
        raise HTTPException(413, f"Video is larger than {LIMITS['max_upload_mb']:.0f} MB")
    uid = os.urandom(8).hex()
    path = os.path.join(STORE.upload_dir(sid), uid + ".part")
    open(path, "wb").close()
    _uploads[uid] = {"sid": sid, "name": name, "size": req.size, "path": path, "received": 0}
    return {"upload_id": uid, "chunk_size": CHUNK, "name": name}


@app.put("/api/uploads/{uid}")
async def upload_chunk(uid: str, request: Request, offset: int = 0):
    sid = ctx(request)
    u = _uploads.get(uid)
    if not u or u["sid"] != sid:
        raise HTTPException(404, "Unknown upload")
    if offset != u["received"]:
        return JSONResponse({"received": u["received"]}, status_code=409)
    body = await request.body()
    if u["received"] + len(body) > u["size"]:
        raise HTTPException(400, "More data than announced")
    with open(u["path"], "ab") as fh:
        fh.write(body)
    u["received"] += len(body)
    return {"received": u["received"]}


@app.post("/api/uploads/{uid}/complete")
def upload_complete(uid: str, request: Request):
    sid = ctx(request)
    u = _uploads.pop(uid, None)
    if not u or u["sid"] != sid:
        raise HTTPException(404, "Unknown upload")
    if u["received"] != u["size"]:
        os.remove(u["path"])
        raise HTTPException(400, "Upload incomplete")
    name, existing = u["name"], set(STORE.videos(sid))
    stem, ext = os.path.splitext(name)
    k = 2
    while name in existing:
        name = f"{stem}_{k}{ext}"
        k += 1
    dst = os.path.join(STORE.base(sid), "videos", name)
    os.replace(u["path"], dst)
    try:
        info = probe(dst)
    except Exception:  # noqa: BLE001
        os.remove(dst)
        raise HTTPException(400, "This file could not be read as a video")
    return {"video": name, **info, "fps_status": fps_status(info["fps"])}


@app.delete("/api/videos/{name}")
def delete_video(name: str, request: Request):
    sid = ctx(request)
    if not HOSTED:
        raise HTTPException(403, "Videos are not deleted from your folder by the app")
    p = _video(name, sid)
    os.remove(p)
    import shutil
    shutil.rmtree(STORE.result_dir(name, sid), ignore_errors=True)
    return {"deleted": name}


# ------------------------------------------------------------------------------------------------ videos
@app.get("/api/videos")
def list_videos(request: Request):
    sid = ctx(request)
    items = []
    for name in STORE.videos(sid):
        p = STORE.video_path(name, sid)
        try:
            info = _info(p)
        except Exception:  # noqa: BLE001
            continue
        items.append({"name": name, "stem": os.path.splitext(name)[0], **info, "meta": video_meta(name),
                      "fps_status": fps_status(info["fps"]),
                      "max_band_hz": config.max_band_hz(info["fps"]),
                      "has_markers": os.path.exists(STORE.markers_path(name, sid)),
                      "n_results": len(_results_for(name, sid)), "n_k3d": len(_k3d_results(name, sid))})
    return items


@app.get("/api/videos/{name}/thumb")
def thumb(name: str, request: Request):
    sid = ctx(request)
    src = _video(name, sid)
    out = os.path.join(STORE.thumbs_dir(sid), os.path.splitext(name)[0] + ".jpg")
    if not os.path.exists(out):
        make_thumbnail(src, out)
    return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/videos/{name}/raw")
def raw(name: str, request: Request):
    return FileResponse(_video(name, ctx(request)), media_type="video/mp4")


@app.get("/api/videos/{name}/scan")
def scan(name: str, request: Request, duration_s: float = 20.0):
    sid = ctx(request)
    return _scan_mgr(sid).get(name, _video(name, sid), os.path.join(STORE.result_dir(name, sid), "scan.json"),
                              duration_s)


@app.get("/api/videos/{name}/results")
def results(name: str, request: Request):
    sid = ctx(request)
    _video(name, sid)
    return _results_for(name, sid)


# ------------------------------------------------------------------------------------------------ markers
@app.get("/api/markers/template")
def marker_template():
    return PlainTextResponse(MK.TEMPLATE, media_type="text/csv",
                             headers={"Content-Disposition": 'attachment; filename="trevideon_markers_template.csv"'})


@app.post("/api/videos/{name}/markers")
async def upload_markers(name: str, request: Request):
    sid = ctx(request)
    p = _video(name, sid)
    raw_bytes = await request.body()
    if len(raw_bytes) > 2 * 1024 * 1024:
        raise HTTPException(413, "Marker file is larger than 2 MB")
    text = raw_bytes.decode("utf-8-sig", errors="replace")
    parsed = MK.parse(text, _info(p)["duration"])
    if not parsed["errors"]:
        with open(STORE.markers_path(name, sid), "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    return parsed


@app.get("/api/videos/{name}/markers")
def get_markers(name: str, request: Request):
    sid = ctx(request)
    p = _video(name, sid)
    mp = STORE.markers_path(name, sid)
    if not os.path.exists(mp):
        return {"rows": [], "errors": [], "warnings": [], "present": False}
    with open(mp, encoding="utf-8") as fh:
        return {**MK.parse(fh.read(), _info(p)["duration"]), "present": True}


@app.delete("/api/videos/{name}/markers")
def delete_markers(name: str, request: Request):
    sid = ctx(request)
    _video(name, sid)
    mp = STORE.markers_path(name, sid)
    if os.path.exists(mp):
        os.remove(mp)
    return {"deleted": True}


# ------------------------------------------------------------------------------------------------ processing
class ProcessRequest(BaseModel):
    video: str
    settings: dict = {}


class ProcessAll(BaseModel):
    items: list[ProcessRequest]


def _prepare(video: str, settings: dict, sid):
    src = _video(video, sid)
    info = _info(src)
    if fps_status(info["fps"]) == "blocked":
        raise HTTPException(400, f"{video}: {info['fps']:.0f} fps is below the minimum of {LIMITS['min_fps']:.0f} fps")
    s = config.normalize(settings, info["fps"])
    s.update(config.runtime(settings))
    notes = []
    if HOSTED:
        rem = max(0.0, info["duration"] - s["start_s"])
        want = rem if s["duration_s"] <= 0 else min(s["duration_s"], rem)
        if want > LIMITS["max_segment_s"]:
            s["duration_s"] = LIMITS["max_segment_s"]
            notes.append(f"Segment shortened to {LIMITS['max_segment_s']:.0f} s (hosted limit)")
        if s["work_height"] == 0 or s["work_height"] > LIMITS["max_height"]:
            s["work_height"] = LIMITS["max_height"]
            notes.append(f"Working resolution limited to {LIMITS['max_height']}p (hosted limit)")
    P, O, ka, kb = keys_for(src, s, info["fps"])
    rd = STORE.result_dir(video, sid)
    return src, s, info, ka, kb, rd, notes


@app.post("/api/check")
def check(req: ProcessRequest, request: Request):
    sid = ctx(request)
    src, s, info, ka, kb, rd, notes = _prepare(req.video, req.settings, sid)
    p = paths(rd, ka, kb)
    return {"key_a": ka, "key_b": kb, "stage_a_cached": os.path.exists(p["analysis"]),
            "stage_b_cached": os.path.exists(p["overlay"]), "notes": notes, "effective": s}


def _submit(req: ProcessRequest, sid):
    auto = bool(req.settings.get("auto_start"))
    src, s, info, ka, kb, rd, notes = _prepare(req.video, req.settings, sid)
    p = paths(rd, ka, kb)
    if not auto and os.path.exists(p["analysis"]) and os.path.exists(p["overlay"]):
        return {"video": req.video, "cached": True, "job": None, "key_a": ka, "key_b": kb, "notes": notes}
    spec = {"video_path": src, "result_dir": rd, "settings": s, "fps": info["fps"], "video": req.video,
            "auto_start": auto, "scan_path": os.path.join(rd, "scan.json"),
            "max_segment_s": LIMITS["max_segment_s"]}
    job = JOBS.submit(sid, req.video, spec, ka, kb)
    return {"video": req.video, "cached": False, "job": JOBS.public(job), "key_a": ka, "key_b": kb, "notes": notes}


@app.post("/api/process")
def process(req: ProcessRequest, request: Request):
    return _submit(req, ctx(request))


@app.post("/api/process_all")
def process_all(req: ProcessAll, request: Request):
    """'Run all': queue every listed video (each with its own settings / segment)."""
    sid = ctx(request)
    out = []
    for it in req.items:
        try:
            out.append(_submit(it, sid))
        except HTTPException as e:
            out.append({"video": it.video, "error": e.detail})
    return out


@app.get("/api/jobs")
def jobs(request: Request):
    return JOBS.for_owner(ctx(request))


@app.get("/api/jobs/{jid}")
def job(jid: str, request: Request):
    sid = ctx(request)
    j = JOBS.jobs.get(jid)
    if not j or j["owner"] != sid:
        raise HTTPException(404, "Unknown job")
    return JOBS.public(j)


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str, request: Request):
    sid = ctx(request)
    j = JOBS.jobs.get(jid)
    if not j or j["owner"] != sid:
        raise HTTPException(404, "Unknown job")
    JOBS.cancel(jid)
    return {"ok": True}


# ------------------------------------------------------------------------------------------------ results
_KEY = re.compile(r"[0-9a-f]{12}")


def _result_paths(name, key_a, key_b, sid):
    _video(name, sid)
    if not _KEY.fullmatch(key_a) or (key_b and not _KEY.fullmatch(key_b)):
        raise HTTPException(400, "Bad key")
    p = paths(STORE.result_dir(name, sid), key_a, key_b)
    if not os.path.exists(p["analysis"]):
        raise HTTPException(404, "Result not found")
    return p


def _file_url(name, key_a, fname, sid):
    q = f"?s={sid}" if sid else ""
    return f"/files/{name}/{key_a}/{fname}{q}"


def _marker_rows(name, analysis, p, sid):
    mp = STORE.markers_path(name, sid)
    if not os.path.exists(mp):
        return None
    with open(mp, encoding="utf-8") as fh:
        parsed = MK.parse(fh.read())
    if parsed["errors"]:
        return None
    Db = np.load(p["meta"])["Db"]
    return MK.segment_metrics(parsed["rows"], analysis, Db)


@app.get("/api/result/{name}/{key_a}")
def result(name: str, key_a: str, request: Request, key_b: str | None = None):
    sid = ctx(request)
    p = _result_paths(name, key_a, key_b, sid)
    with open(p["analysis"]) as fh:
        analysis = json.load(fh)
    ov = None
    if key_b and os.path.exists(p.get("overlay_json", "")):
        with open(p["overlay_json"]) as fh:
            ov = json.load(fh)
    urls = {"original": _file_url(name, key_a, "original.mp4", sid),
            "magnified": _file_url(name, key_a, "magnified.mp4", sid),
            "bundle": f"/api/result/{name}/{key_a}/bundle.zip?{'key_b=' + key_b + '&' if key_b else ''}"
                      f"{'s=' + sid if sid else ''}"}
    if key_b and os.path.exists(p["overlay"]):
        urls["overlay"] = _file_url(name, key_a, f"overlays/{key_b}.mp4", sid)
    if os.path.exists(p["isolated"]):
        urls["isolated"] = _file_url(name, key_a, "isolated_input.mp4", sid)
    return JSONResponse(_json_safe({"key_a": key_a, "key_b": key_b, "analysis": analysis, "overlay": ov,
                                    "urls": urls, "marker_metrics": _marker_rows(name, analysis, p, sid)}))


_ALLOWED = re.compile(r"^(original\.mp4|magnified\.mp4|isolated_input\.mp4|overlays/[0-9a-f]{12}\.mp4)$")


@app.get("/files/{name}/{key_a}/{fname:path}")
def files(name: str, key_a: str, fname: str, request: Request):
    sid = ctx(request)
    if not _ALLOWED.match(fname):
        raise HTTPException(404, "Not found")
    p = _result_paths(name, key_a, None, sid)
    f = os.path.join(p["dir"], *fname.split("/"))
    if not os.path.exists(f):
        raise HTTPException(404, "Not found")
    return FileResponse(f, media_type="video/mp4")


def metrics_csv(analysis: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    cols = ["peak_hz", "rms_px", "p95_px", "band_power_pct", "rms_rel_hand_pct", "rms_pct_hand_len"]
    w.writerow(["video", "mode", "start_s", "duration_s", "fps", "band_lo_hz", "band_hi_hz", "point"] + cols)
    for point, m in analysis["metrics"].items():
        w.writerow([analysis["video"], analysis.get("mode", "standard"), analysis["start_s"], analysis["duration_s"],
                    analysis["fps"], analysis["band"][0], analysis["band"][1], point] + [m.get(c, "") for c in cols])
    return buf.getvalue()


@app.get("/api/result/{name}/{key_a}/bundle.zip")
def bundle(name: str, key_a: str, request: Request, key_b: str | None = None, videos: bool = True):
    sid = ctx(request)
    p = _result_paths(name, key_a, key_b, sid)
    with open(p["analysis"]) as fh:
        analysis = json.load(fh)
    stem = os.path.splitext(name)[0]
    tag = "isolated" if analysis.get("mode") == "hand_isolation" else "standard"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w") as z:
        z.write(p["analysis"], "analysis.json", compress_type=zipfile.ZIP_DEFLATED)
        z.write(p["params"], "settings.json", compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("metrics.csv", metrics_csv(analysis), compress_type=zipfile.ZIP_DEFLATED)
        mm = _marker_rows(name, analysis, p, sid)
        if mm:
            z.writestr("marker_metrics.csv", MK.to_csv(mm), compress_type=zipfile.ZIP_DEFLATED)
            z.write(STORE.markers_path(name, sid), "markers.csv", compress_type=zipfile.ZIP_DEFLATED)
        if videos:
            for f in ("original.mp4", "magnified.mp4"):
                z.write(os.path.join(p["dir"], f), f, compress_type=zipfile.ZIP_STORED)
            if key_b and os.path.exists(p["overlay"]):
                z.write(p["overlay"], "overlay.mp4", compress_type=zipfile.ZIP_STORED)
        z.writestr("README.txt", f"{APP_NAME} {APP_VERSION} result for {name}\n"
                                 f"Mode: {tag}. Settings: settings.json. Per-point tremor metrics: metrics.csv.\n"
                                 "Please cite TREVIDEON if you use these results (see CITATION in the app).\n")
    return FileResponse(tmp.name, media_type="application/zip", filename=f"{stem}_{tag}_{key_a}.zip",
                        background=BackgroundTask(lambda: os.remove(tmp.name)))


# ------------------------------------------------------------------------------------------------ 3D kinematics
HAND_FBX_DIR = os.environ.get("TREVIDEON_HAND_FBX_DIR") or os.path.normpath(
    os.path.join(APP_ROOT, "..", "..", "..", "Application", "VibeSkinVR", "Assets", "Animated Hands", "Models"))


def _hand_fbx(side: str) -> str | None:
    """The VibeSkin VR 'Animated Hands' mesh (local use only; not part of TREVIDEON's repository)."""
    f = os.path.join(HAND_FBX_DIR, f"{'r' if side == 'right' else 'l'}_hand_skeletal_lowres.fbx")
    return f if os.path.isfile(f) else None


def _k3d_results(name: str, sid) -> list[dict]:
    root = os.path.join(STORE.result_dir(name, sid), "k3d")
    out = []
    if not os.path.isdir(root):
        return out
    for kk in os.listdir(root):
        p = paths3d(os.path.dirname(root), kk)
        if not os.path.exists(p["json"]):
            continue
        try:
            with open(p["params"]) as fh:
                K = json.load(fh)
        except OSError:
            continue
        displays = []
        if os.path.isdir(p["display_dir"]):
            for f in os.listdir(p["display_dir"]):
                if f.endswith(".json") and os.path.exists(os.path.join(p["display_dir"], f[:-5] + ".mp4")):
                    try:
                        with open(os.path.join(p["display_dir"], f)) as fh:
                            dj = json.load(fh)
                        displays.append({"key_v": f[:-5], "display": dj.get("display"), "created": dj.get("created")})
                    except (OSError, ValueError):
                        continue
        out.append({"key": kk, "params": K, "displays": displays,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p["json"])))})
    return sorted(out, key=lambda r: r["created"], reverse=True)


def _prepare3d(video: str, settings: dict, sid):
    src = _video(video, sid)
    info = _info(src)
    if fps_status(info["fps"]) == "blocked":
        raise HTTPException(400, f"{video}: {info['fps']:.0f} fps is below the minimum of {LIMITS['min_fps']:.0f} fps")
    s = C3.normalize3d(settings, info["fps"])
    notes = []
    if HOSTED:
        rem = max(0.0, info["duration"] - s["start_s"])
        if min(s["duration_s"], rem) > LIMITS["max_segment_s"]:
            s["duration_s"] = LIMITS["max_segment_s"]
            notes.append(f"Segment shortened to {LIMITS['max_segment_s']:.0f} s (hosted limit)")
    K, V, R, kk, kv = keys_for3d(src, s, info["fps"])
    s.update(R)
    return src, s, info, kk, kv, STORE.result_dir(video, sid), notes


@app.post("/api/k3d/check")
def k3d_check(req: ProcessRequest, request: Request):
    sid = ctx(request)
    src, s, info, kk, kv, rd, notes = _prepare3d(req.video, req.settings, sid)
    p = paths3d(rd, kk, kv)
    return {"key": kk, "key_v": kv, "k_cached": os.path.exists(p["json"]), "v_cached": os.path.exists(p["display"]),
            "notes": notes, "effective": s}


def _submit3d(req: ProcessRequest, sid):
    src, s, info, kk, kv, rd, notes = _prepare3d(req.video, req.settings, sid)
    p = paths3d(rd, kk, kv)
    if os.path.exists(p["json"]) and os.path.exists(p["display"]):
        return {"video": req.video, "cached": True, "job": None, "key": kk, "key_v": kv, "notes": notes}
    spec = {"kind": "kin3d", "video_path": src, "result_dir": rd, "settings": s, "fps": info["fps"],
            "video": req.video, "export_mesh": not HOSTED}
    job = JOBS.submit(sid, req.video, spec, kk, kv)
    return {"video": req.video, "cached": False, "job": JOBS.public(job), "key": kk, "key_v": kv, "notes": notes}


@app.post("/api/k3d/process")
def k3d_process(req: ProcessRequest, request: Request):
    return _submit3d(req, ctx(request))


@app.post("/api/k3d/process_all")
def k3d_process_all(req: ProcessAll, request: Request):
    sid = ctx(request)
    out = []
    for it in req.items:
        try:
            out.append(_submit3d(it, sid))
        except HTTPException as e:
            out.append({"video": it.video, "error": e.detail})
    return out


@app.get("/api/videos/{name}/k3d")
def k3d_results(name: str, request: Request):
    sid = ctx(request)
    _video(name, sid)
    return _k3d_results(name, sid)


def _k3d_paths(name, kk, kv, sid):
    _video(name, sid)
    if not _KEY.fullmatch(kk) or (kv and not _KEY.fullmatch(kv)):
        raise HTTPException(400, "Bad key")
    p = paths3d(STORE.result_dir(name, sid), kk, kv)
    if not os.path.exists(p["json"]):
        raise HTTPException(404, "Result not found")
    return p


def _url3d(name, kk, fname, sid):
    return f"/files3d/{name}/{kk}/{fname}" + (f"?s={sid}" if sid else "")


@app.get("/api/k3d/{name}/{kk}")
def k3d_result(name: str, kk: str, request: Request, key_v: str | None = None):
    sid = ctx(request)
    p = _k3d_paths(name, kk, key_v, sid)
    figs = sorted(f for f in os.listdir(p["figures"]) if f.endswith(".png")) if os.path.isdir(p["figures"]) else []
    exps = sorted(os.listdir(p["exports"])) if os.path.isdir(p["exports"]) else []
    urls = {"data": _url3d(name, kk, "kinematics.json", sid),
            "figures": [{"name": f, "url": _url3d(name, kk, f"figures/{f}", sid)} for f in figs],
            "exports": [{"name": f, "url": _url3d(name, kk, f"exports/{f}", sid)} for f in exps],
            "bundle": f"/api/k3d/{name}/{kk}/bundle.zip?{'key_v=' + key_v + '&' if key_v else ''}"
                      f"{'s=' + sid if sid else ''}"}
    if key_v and os.path.exists(p["display"]):
        urls["display"] = _url3d(name, kk, f"display/{key_v}.mp4", sid)
    if os.path.exists(p["mesh"]) and not HOSTED:
        urls["mesh"] = _url3d(name, kk, "mesh.bin", sid)
    disp = None
    if key_v and os.path.exists(p.get("display_json", "")):
        with open(p["display_json"]) as fh:
            disp = json.load(fh)
    with open(p["params"]) as fh:
        K = json.load(fh)
    return {"key": kk, "key_v": key_v, "params": K, "display": disp, "urls": urls}


_ALLOWED3D = re.compile(r"^(kinematics\.json|mesh\.bin|display/[0-9a-f]{12}\.mp4|figures/[\w.-]+\.png|"
                        r"exports/[\w.-]+\.(csv|mat|bvh|glb))$")
_MEDIA = {".json": "application/json", ".bin": "application/octet-stream", ".mp4": "video/mp4", ".png": "image/png",
          ".csv": "text/csv", ".mat": "application/octet-stream", ".bvh": "text/plain", ".glb": "model/gltf-binary"}


@app.get("/files3d/{name}/{kk}/{fname:path}")
def files3d(name: str, kk: str, fname: str, request: Request):
    sid = ctx(request)
    if not _ALLOWED3D.match(fname) or (HOSTED and fname == "mesh.bin"):
        raise HTTPException(404, "Not found")
    p = _k3d_paths(name, kk, None, sid)
    f = os.path.join(p["dir"], *fname.split("/"))
    if not os.path.exists(f):
        raise HTTPException(404, "Not found")
    ext = os.path.splitext(f)[1]
    headers = {"Content-Disposition": f'attachment; filename="{os.path.basename(f)}"'} \
        if fname.startswith("exports/") else None
    return FileResponse(f, media_type=_MEDIA.get(ext, "application/octet-stream"), headers=headers)


@app.get("/api/k3d/{name}/{kk}/bundle.zip")
def k3d_bundle(name: str, kk: str, request: Request, key_v: str | None = None, videos: bool = True):
    sid = ctx(request)
    p = _k3d_paths(name, kk, key_v, sid)
    stem = os.path.splitext(name)[0]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w") as z:
        z.write(p["json"], "kinematics.json", compress_type=zipfile.ZIP_DEFLATED)
        z.write(p["params"], "settings.json", compress_type=zipfile.ZIP_DEFLATED)
        for sub in ("exports", "figures"):
            d = p[sub]
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    z.write(os.path.join(d, f), f"{sub}/{f}",
                            compress_type=zipfile.ZIP_STORED if f.endswith(".png") else zipfile.ZIP_DEFLATED)
        if videos and key_v and os.path.exists(p["display"]):
            z.write(p["display"], "segmented_hand.mp4", compress_type=zipfile.ZIP_STORED)
        z.writestr("README.txt", f"{APP_NAME} {APP_VERSION} 3D hand kinematics for {name}\n"
                                 "Frames: world (head motion removed), wrist (palm: Kabsch fit of the wrist and the "
                                 "knuckles), rigid fingers (knuckle -> tip).\n"
                                 "exports/: CSV tables (transforms, joints, series, angles, metrics, coherence), "
                                 "kinematics.mat, hand.bvh, hand_animation.glb.\n"
                                 "figures/: PNG 300 dpi. Units: mm, degrees, seconds.\n"
                                 "Please cite TREVIDEON, WiLoR (Potamias et al. 2025) and MANO (Romero et al. 2017).\n")
    return FileResponse(tmp.name, media_type="application/zip", filename=f"{stem}_3d_{kk}.zip",
                        background=BackgroundTask(lambda: os.remove(tmp.name)))


@app.get("/api/assets/hand/{side}.fbx")
def hand_fbx(side: str):
    if HOSTED or side not in ("right", "left"):
        raise HTTPException(404, "Not available")
    f = _hand_fbx(side)
    if not f:
        raise HTTPException(404, "VibeSkin hand model not found")
    return FileResponse(f, media_type="application/octet-stream", headers={"Cache-Control": "max-age=86400"})


# ------------------------------------------------------------------------------------------------ frontend (local)
if os.path.isdir(DIST_DIR) and not HOSTED:
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets") \
        if os.path.isdir(os.path.join(DIST_DIR, "assets")) else None

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        f = os.path.join(DIST_DIR, full_path)
        if full_path and os.path.isfile(f) and os.path.commonpath([DIST_DIR, os.path.abspath(f)]) == DIST_DIR:
            return FileResponse(f)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
