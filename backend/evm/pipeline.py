"""Two-stage cached processing pipeline.

Stage A (process settings)  -> <result_dir>/<keyA>/
    original.mp4, magnified.mp4, disp_field.npy, meta.npz, analysis.json, params.json
    (+ isolated_input.mp4, isolation.json when the dominant hand is isolated first)
Stage B (overlay settings)  -> <result_dir>/<keyA>/overlays/<keyB>.mp4 (+ .json)

<result_dir> is one folder per video (chosen by the storage layer). A stage is considered cached when its
final artefact exists (analysis.json / overlay mp4). Cache keys depend on the video content (size, mtime)
and the settings, not on the file name, so renaming a video keeps its results.
"""
import hashlib
import json
import os
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config
from .analysis import analyse, displacement_fields, envelope_fields, sample_flow
from .hands import ANCHORS, HandDetector, anchor_points, hand_mask, hand_size, smooth_landmarks, transform_landmarks
from .isolate import build_isolated_video, draw_anchor
from .magnify import bandpass_sos, make_magnifier
from .overlay import OverlayRenderer
from .stabilize import MotionEstimator, compute_correction, stabilization_region, warp
from .video_io import SegmentReader, VideoWriter, probe

GRID = 8  # motion-field decimation factor


class Cancelled(Exception):
    pass


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def stage_a_key(video_path: str, P: dict) -> str:
    st = os.stat(video_path)
    return _hash({"v": config.PIPELINE_VERSION, "size": st.st_size, "mtime": int(st.st_mtime), "P": P})


def stage_b_key(key_a: str, O: dict) -> str:
    return _hash({"v": config.OVERLAY_VERSION, "a": key_a, "O": O})


def paths(result_dir: str, key_a: str, key_b: str | None = None) -> dict:
    d = os.path.join(result_dir, key_a)
    p = {"dir": d, "analysis": os.path.join(d, "analysis.json"), "params": os.path.join(d, "params.json"),
         "original": os.path.join(d, "original.mp4"), "magnified": os.path.join(d, "magnified.mp4"),
         "disp": os.path.join(d, "disp_field.npy"), "meta": os.path.join(d, "meta.npz"),
         "overlays": os.path.join(d, "overlays"), "isolated": os.path.join(d, "isolated_input.mp4"),
         "isolation": os.path.join(d, "isolation.json")}
    if key_b:
        p["overlay"] = os.path.join(d, "overlays", f"{key_b}.mp4")
        p["overlay_json"] = os.path.join(d, "overlays", f"{key_b}.json")
    return p


class Progress:
    """Maps sub-stage progress into overall [0,1] and throttles reports."""

    def __init__(self, report, spans: dict):
        self.report = report or (lambda d: None)
        self.spans = spans
        self.t0 = time.time()
        self.last = 0.0
        self.cancel_check = lambda: False

    def __call__(self, stage: str, frac: float, msg: str = "", force: bool = False):
        a, b = self.spans[stage]
        overall = a + (b - a) * min(max(frac, 0.0), 1.0)
        now = time.time()
        if force or now - self.last > 0.4:
            self.last = now
            el = now - self.t0
            eta = el / overall * (1 - overall) if overall > 0.02 else None
            self.report({"state": "running", "stage": stage, "progress": round(overall, 4), "message": msg,
                         "elapsed_s": round(el, 1), "eta_s": None if eta is None else round(eta, 1)})
        if self.cancel_check():
            raise Cancelled()


# ---------------------------------------------------------------------------------------------- stage A
def inner_params(P: dict) -> dict:
    """Settings used to analyse the wrist-anchored canvas video of the hand-isolation mode."""
    return dict(P, start_s=0.0, duration_s=0, work_height=int(P["canvas_px"]), stabilization="none",
                output_view="original", num_hands=1, magnify_region="hand", repair_duplicates=False)


def run_stage_a(video_path: str, P: dict, out: dict, prog: Progress, device: str = "auto",
                anchor_xy=None, outer: dict | None = None):
    """outer: {"P", "video_path", "start_s", "isolation"} when video_path is the isolated canvas video."""
    os.makedirs(out["dir"], exist_ok=True)
    for k in ("analysis",):
        if os.path.exists(out[k]):
            os.remove(out[k])
    reader = SegmentReader(video_path, P["start_s"], P["duration_s"], P["work_height"],
                           repair=bool(P.get("repair_duplicates")))
    fps = reader.fps
    Wd, Hd = reader.size
    n_est = reader.n
    stab = P["stabilization"]
    t_start = time.time()

    # ---- pass 1: hand landmarks + camera motion
    det = HandDetector(P["num_hands"], P["hand_confidence"])
    est = MotionEstimator()
    lms = np.full((n_est, 2, 21, 2), np.nan, np.float32)
    A = [np.eye(3)]
    prev_gray = prev_region = None
    T = 0
    try:
        for t, f in enumerate(reader.frames()):
            lms[t] = det.detect(f, int(round(t * 1000.0 / fps)))
            if stab != "none":
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    A.append(est.step(prev_gray, gray, prev_region))
                prev_region = stabilization_region((Hd, Wd), lms[t], stab, hand_mask)
                prev_gray = gray
            T = t + 1
            prog("pass1", T / n_est, f"Tracking hand & camera motion  {T}/{n_est}")
    finally:
        det.close()
    if T < max(10, int(2 * fps)):
        raise RuntimeError(f"Segment too short ({T} frames). Choose a longer duration / earlier start.")
    lms = lms[:T]
    detected = ~np.isnan(lms[:, 0, 0, 0])
    lms_s = smooth_landmarks(lms, fps)
    if stab != "none":
        W, stab_stats = compute_correction(A[:T], fps, P["stab_cutoff_hz"])
    else:
        W, stab_stats = np.repeat(np.eye(3)[None], T, axis=0), {}
    stab_lms = transform_landmarks(lms_s, W).astype(np.float32)
    anchors = np.full((T, 6, 2), np.nan, np.float32)
    hs = np.full(T, np.nan, np.float32)
    for t in range(T):
        if not np.isnan(stab_lms[t, 0, 0, 0]):
            anchors[t] = anchor_points(stab_lms[t, 0])
            hs[t] = hand_size(stab_lms[t, 0])
    hand_px = float(np.nanmedian(hs)) if np.isfinite(hs).any() else float(min(Wd, Hd) / 4)

    # ---- pass 2: stabilise, magnify, dense flow
    mag, levels, device_used = make_magnifier(P, fps, (Hd, Wd), device)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gh, gw = Hd // GRID, Wd // GRID
    flow_tmp = os.path.join(out["dir"], "flow_grid.tmp.npy")
    Vg = np.lib.format.open_memmap(flow_tmp, mode="w+", dtype=np.float32, shape=(T, gh, gw, 2))
    Vg[0] = 0
    anchor_v = np.zeros((T, 6, 2), np.float32)
    w_orig = VideoWriter(out["original"], (Wd, Hd), fps, crf=18)
    w_mag = VideoWriter(out["magnified"], (Wd, Hd), fps, crf=16)
    view_orig = P["output_view"] == "original" and stab != "none"
    prev_g = prev_mask = None
    radius = max(3, int(0.07 * hand_px))
    flow_pool = ThreadPoolExecutor(max_workers=1)
    flow_job = None

    def flow_step(t, a, b, m):
        flow = dis.calc(a, b, None)
        Vg[t] = cv2.resize(flow, (gw, gh), interpolation=cv2.INTER_AREA)
        if np.isfinite(anchors[t - 1]).all():
            anchor_v[t] = sample_flow(flow, anchors[t - 1], m, radius)

    try:
        for t, f in enumerate(reader.frames(limit=T)):
            fs = f if stab == "none" else warp(f, W[t], (Wd, Hd))
            ycc_s = cv2.cvtColor(fs, cv2.COLOR_BGR2YCrCb)
            Ys = ycc_s[..., 0].astype(np.float32) * (1.0 / 255.0)
            Ym = mag.process(Ys)
            delta = Ym - Ys
            mask = hand_mask((Hd, Wd), stab_lms[t], margin=float(P["mask_margin"]))
            if P["magnify_region"] == "hand":
                sig = max(3.0, 0.06 * hand_px)
                soft = cv2.GaussianBlur(cv2.dilate(mask, None, iterations=2).astype(np.float32) * (1 / 255.0), (0, 0), sig)
                delta *= soft
            if view_orig:
                delta = cv2.warpAffine(delta, np.linalg.inv(W[t])[:2], (Wd, Hd), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                base = f
                ycc = cv2.cvtColor(base, cv2.COLOR_BGR2YCrCb)
            else:
                base = fs
                ycc = ycc_s.copy()
            ycc[..., 0] = np.clip(ycc[..., 0].astype(np.float32) + delta * 255.0, 0, 255).astype(np.uint8)
            magf = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
            if anchor_xy is not None:     # the static wrist anchor is drawn on the outputs only
                w_orig.write(draw_anchor(base.copy(), anchor_xy))
                w_mag.write(draw_anchor(magf, anchor_xy))
            else:
                w_orig.write(base)
                w_mag.write(magf)

            g = ycc_s[..., 0].copy()
            if flow_job is not None:
                flow_job.result()
            if prev_g is not None:
                # dense flow runs in a worker thread, overlapping with the next frame's magnification
                flow_job = flow_pool.submit(flow_step, t, prev_g, g, prev_mask)
            prev_g, prev_mask = g, mask
            prog("pass2", (t + 1) / T, f"Magnifying & measuring motion  {t + 1}/{T}")
        if flow_job is not None:
            flow_job.result()
        w_orig.close()
        w_mag.close()
    except BaseException:
        w_orig.abort()
        w_mag.abort()
        del Vg
        raise
    finally:
        flow_pool.shutdown(wait=True, cancel_futures=True)

    # ---- motion fields & analysis
    prog("analysis", 0.05, "Band-passing motion fields", force=True)
    sos = bandpass_sos(P["freq_lo"], P["freq_hi"], fps, P["filter_order"])
    D = np.lib.format.open_memmap(out["disp"] + ".tmp", mode="w+", dtype=np.float16, shape=(T, gh, gw, 2))
    displacement_fields(Vg, D, sos)
    D.flush()
    del Vg
    try:
        os.remove(flow_tmp)
    except OSError:
        pass
    prog("analysis", 0.5, "Computing spectra & envelopes", force=True)
    anchor_valid = np.isfinite(anchors).all(axis=(1, 2))
    res, Db = analyse(anchor_v, anchor_valid, D, stab_lms, fps, P, (gw / Wd, gh / Hd), hand_px)
    del D
    os.replace(out["disp"] + ".tmp", out["disp"])
    start_s = outer["start_s"] if outer else reader.start_s
    np.savez_compressed(out["meta"], W=W.astype(np.float32), stab_lms=stab_lms, anchors=anchors, Db=Db,
                        detected=detected, fps=fps, size=np.array([Wd, Hd]), start_s=start_s,
                        anchor_xy=np.array(anchor_xy if anchor_xy is not None else [np.nan, np.nan], np.float32))
    src_path = outer["video_path"] if outer else video_path
    P_public = outer["P"] if outer else P
    res.update({
        "video": os.path.basename(src_path), "fps": fps, "n_frames": T, "start_s": round(start_s, 3),
        "duration_s": round(T / fps, 3), "size": [Wd, Hd], "grid": GRID, "pyr_levels": levels,
        "band": [P["freq_lo"], P["freq_hi"]], "hand_size_px": round(hand_px, 1),
        "hand_detected_pct": round(100 * float(detected.mean()), 1), "stabilization": stab_stats,
        "params": P_public, "processing_s": round(time.time() - t_start + (outer["isolation"]["processing_s"]
                                                                           if outer else 0.0), 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "source": probe(src_path), "device": device_used,
        "mode": "hand_isolation" if outer else "standard",
    })
    if outer:
        res["frame_repair"] = outer["isolation"].get("frame_repair")
    elif P.get("repair_duplicates"):
        res["frame_repair"] = {"repaired_frames": len(reader.repaired),
                               "repaired_pct": round(100 * len(reader.repaired) / max(T, 1), 2),
                               "suspected_dropped_frames": len(reader.suspected_drops),
                               "suspected_dropped_pct": round(100 * len(reader.suspected_drops) / max(T, 1), 2),
                               "first_indices": reader.repaired[:50]}
    if outer:
        iso = {k: v for k, v in outer["isolation"].items() if k != "wrist_slow_path_px"}
        res["isolation"] = iso
        # amplitudes in the canvas are in normalised-hand pixels: also give them as % of hand length
        for m in res["metrics"].values():
            m["rms_pct_hand_len"] = round(100 * m["rms_px"] / float(iso["hand_len_px"]), 4)
    with open(out["params"], "w") as fh:
        json.dump(P_public, fh, indent=2)
    with open(out["analysis"] + ".tmp", "w") as fh:
        json.dump(res, fh)
    os.replace(out["analysis"] + ".tmp", out["analysis"])
    prog("analysis", 1.0, "Analysis complete", force=True)


# ---------------------------------------------------------------------------------------------- stage B
def run_stage_b(P: dict, O: dict, out: dict, prog: Progress):
    os.makedirs(out["overlays"], exist_ok=True)
    with open(out["analysis"]) as fh:
        res = json.load(fh)
    meta = np.load(out["meta"])
    W, stab_lms, anchors, Db = meta["W"], meta["stab_lms"], meta["anchors"], meta["Db"]
    fps = float(meta["fps"])
    Wd, Hd = (int(v) for v in meta["size"])
    D = np.load(out["disp"], mmap_mode="r")
    T, gh, gw, _ = D.shape
    prog("overlay", 0.0, "Computing tremor envelope", force=True)
    E = envelope_fields(D, int(round(float(O["envelope_window_s"]) * fps)))

    # global normalisation from hand cells
    idx = np.arange(0, T, max(1, T // 240))
    ev, dv = [], []
    for t in idx:
        m = hand_mask((gh, gw), stab_lms[t], margin=float(P["mask_margin"]), scale=gw / Wd) > 0
        if m.any():
            ev.append(E[t][m].astype(np.float32))
            dv.append(np.linalg.norm(np.asarray(D[t], np.float32)[m], axis=-1))
    ev = np.concatenate(ev) if ev else np.array([1.0])
    dv = np.concatenate(dv) if dv else np.array([1.0])
    scale_E = float(np.percentile(ev, 99))
    scale_D = float(np.percentile(dv, 97))
    tips = np.linalg.norm(Db[:, :5], axis=-1)
    tip_p95 = float(np.percentile(tips, 95)) if tips.size else 1.0
    hand_px = float(res.get("hand_size_px", 100))
    trail_gain = 0.16 * hand_px / max(tip_p95, 1e-6)

    info = {"start_s": float(meta["start_s"]), "peak_hz": res["metrics"]["hand"]["peak_hz"]}
    anchor_xy = meta["anchor_xy"] if "anchor_xy" in meta.files and np.isfinite(meta["anchor_xy"]).all() else None
    rend = OverlayRenderer(O, P, (Wd, Hd), (gh, gw), fps, scale_E, scale_D, trail_gain, info)
    hand_env = np.asarray(res["envelope"]["series"]["hand"], np.float32)
    env_t = np.asarray(res["envelope"]["t"], np.float32)

    base_path = out["magnified"] if O["overlay_base"] == "magnified" else out["original"]
    cap = cv2.VideoCapture(base_path)
    writer = VideoWriter(out["overlay"], (Wd, Hd), fps, crf=18)
    view_orig = P["output_view"] == "original" and P["stabilization"] != "none"
    def render(t, base):
        col8, a8 = rend.render_layer(t, np.asarray(E[t], np.float32), np.asarray(D[t], np.float32),
                                     stab_lms[t], anchors[t], Db)
        frame = rend.composite(base, col8, a8, np.linalg.inv(W[t]) if view_orig else None)
        if O["show_hud"]:
            env_now = float(np.interp(t / fps, env_t, hand_env)) if len(env_t) else 0.0
            rend.hud(frame, t, env_now)
        if anchor_xy is not None:
            draw_anchor(frame, anchor_xy)
        return frame

    # OpenCV/numpy release the GIL, so a small thread pool renders frames in parallel (written in order)
    pool = ThreadPoolExecutor(max_workers=max(2, min(6, (os.cpu_count() or 4) - 2)))
    pending = deque()
    done = 0
    try:
        for t in range(T):
            ok, base = cap.read()
            if not ok:
                break
            pending.append(pool.submit(render, t, base))
            while len(pending) > 12 or (pending and pending[0].done()):
                writer.write(pending.popleft().result())
                done += 1
                prog("overlay", done / T, f"Rendering tremor-envelope overlay  {done}/{T}")
        while pending:
            writer.write(pending.popleft().result())
            done += 1
            prog("overlay", done / T, f"Rendering tremor-envelope overlay  {done}/{T}")
        writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        cap.release()
    with open(out["overlay_json"], "w") as fh:
        json.dump({"overlay": O, "scale_E_px": scale_E, "scale_D_px": scale_D, "trail_gain": trail_gain,
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)


# ---------------------------------------------------------------------------------------------- job entry
def keys_for(video_path: str, settings: dict, fps: float | None = None):
    P, O = config.split(settings, fps)
    key_a = stage_a_key(video_path, P)
    return P, O, key_a, stage_b_key(key_a, O)


def _auto_start(spec: dict, report) -> dict:
    """Run all: pick the steadiest segment with the hand visible (cached 2 fps scan)."""
    from .scan import recommend, scan_video
    settings = dict(spec["settings"])
    sp = spec.get("scan_path")
    if sp and os.path.exists(sp):
        with open(sp) as fh:
            sc = json.load(fh)
    else:
        if report:
            report({"state": "running", "stage": "scan", "progress": 0.0, "message": "Scanning for the best segment"})
        sc = scan_video(spec["video_path"])
        if sp:
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            with open(sp, "w") as fh:
                json.dump(sc, fh)
    rec = recommend(sc, float(settings.get("duration_s") or 20.0))
    settings["start_s"] = rec["start_s"]
    return settings


def run_job(spec: dict, report=None, cancel_check=None) -> dict:
    """spec: {video_path, result_dir, settings, fps[, auto_start, scan_path]}. Runs the stages not yet cached."""
    if spec.get("auto_start"):
        spec = dict(spec, settings=_auto_start(spec, report))
    P, O, key_a, key_b = keys_for(spec["video_path"], spec["settings"], spec.get("fps"))
    device = config.runtime(spec["settings"])["device"]
    out = paths(spec["result_dir"], key_a, key_b)
    iso = bool(P.get("hand_isolation"))
    need_a = not os.path.exists(out["analysis"])
    need_b = not os.path.exists(out["overlay"])
    if need_a and iso:
        spans = {"isolate1": (0.0, 0.08), "isolate2": (0.08, 0.32), "pass1": (0.32, 0.42), "pass2": (0.42, 0.8),
                 "analysis": (0.8, 0.84), "overlay": (0.84, 1.0)}
    elif need_a:
        spans = {"pass1": (0.0, 0.22), "pass2": (0.22, 0.75), "analysis": (0.75, 0.8), "overlay": (0.8, 1.0)}
    else:
        spans = {"overlay": (0.0, 1.0)}
    for k in ("isolate1", "isolate2", "pass1", "pass2", "analysis", "overlay"):
        spans.setdefault(k, (0, 0))
    prog = Progress(report, spans)
    if cancel_check:
        prog.cancel_check = cancel_check
    P_eff = inner_params(P) if iso else P
    if need_a:
        if iso:
            if os.path.exists(out["isolated"]) and os.path.exists(out["isolation"]):
                with open(out["isolation"]) as fh:
                    info = json.load(fh)
            else:
                info = build_isolated_video(spec["video_path"], P, out["dir"], prog)
            run_stage_a(out["isolated"], P_eff, out, prog, device, anchor_xy=info["anchor_xy"],
                        outer={"P": P, "video_path": spec["video_path"], "start_s": info["start_s"], "isolation": info})
        else:
            run_stage_a(spec["video_path"], P, out, prog, device)
    if need_b:
        run_stage_b(P_eff, O, out, prog)
    return {"key_a": key_a, "key_b": key_b}


def worker_main(spec: dict, queue, cancel_event):
    """Entry point for a spawned worker process."""
    def report(d):
        queue.put(d)
    try:
        if spec.get("kind") == "kin3d":
            from .kin3d.pipeline import run_job3d
            r = run_job3d(spec, report, cancel_event.is_set)
        else:
            r = run_job(spec, report, cancel_event.is_set)
        queue.put({"state": "done", "progress": 1.0, "stage": "done", "message": "Done", **r})
    except Cancelled:
        queue.put({"state": "cancelled", "message": "Cancelled"})
    except Exception as e:  # noqa: BLE001
        queue.put({"state": "error", "message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Process one video without the web app.")
    ap.add_argument("video")
    ap.add_argument("--result-dir", required=True, help="folder for this video's results")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides, e.g. hand_isolation=true start_s=40")
    a = ap.parse_args()
    s = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        s[k] = (v.lower() == "true") if v.lower() in ("true", "false") else \
            (float(v) if v.replace(".", "", 1).isdigit() else v)
    print(run_job({"video_path": a.video, "result_dir": os.path.abspath(a.result_dir), "settings": s,
                   "fps": probe(a.video)["fps"]},
                  report=lambda d: print(f"\r{d['progress'] * 100:5.1f}%  {d['message']:<60}", end="", flush=True)))
