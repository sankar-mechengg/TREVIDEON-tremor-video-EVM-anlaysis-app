"""Fast whole-video scan (2 fps) to find the steadiest segment with the hand fully visible.

Used to pick the recommended default segment and to draw the hand-presence / motion timeline in the UI.
"""
import json
import os
import subprocess
import threading

import cv2
import numpy as np

from .hands import MODEL_PATH
from .video_io import FFMPEG, _NO_WINDOW, probe

SCAN_FPS = 2.0
SCAN_W = 480


def scan_video(path: str, progress=None) -> dict:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    info = probe(path)
    w = SCAN_W
    h = int(round(info["height"] * w / info["width"] / 2)) * 2
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
           "-vf", f"fps={SCAN_FPS},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                                        running_mode=vision.RunningMode.IMAGE, num_hands=2,
                                        min_hand_detection_confidence=0.4)
    lm = vision.HandLandmarker.create_from_options(opts)
    n_exp = max(1, int(info["duration"] * SCAN_FPS))
    present, full, size, pos, cam = [], [], [], [], []
    prev = None
    fsz = w * h * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(fsz)
            if len(buf) < fsz:
                break
            f = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            hands = [np.array([[p.x, p.y] for p in hl], np.float32) for hl in res.hand_landmarks]
            if hands:
                a = max(hands, key=lambda a: np.ptp(a[:, 0]) * np.ptp(a[:, 1]))
                present.append(1)
                full.append(int(a.min() > 0.01 and a.max() < 0.99))
                size.append(float(np.linalg.norm((a[0] - a[9]) * [w / h, 1])))  # relative to frame height
                pos.append(a[[0, 5, 9, 13, 17]].mean(0).tolist())
            else:
                present.append(0)
                full.append(0)
                size.append(0.0)
                pos.append([np.nan, np.nan])
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                (dx, dy), _ = cv2.phaseCorrelate(prev, g, cv2.createHanningWindow((w, h), cv2.CV_32F))
                cam.append(float(np.hypot(dx, dy)) / w)
            else:
                cam.append(0.0)
            prev = g
            i += 1
            if progress and i % 10 == 0:
                progress(min(0.99, i / n_exp))
    finally:
        proc.stdout.close()
        proc.kill()
        lm.close()

    n = len(present)
    t = (np.arange(n) / SCAN_FPS).tolist()
    pos = np.array(pos, np.float64).reshape(-1, 2)
    hand_mv = np.zeros(n)
    if n > 1:
        d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        hand_mv[1:] = np.nan_to_num(d, nan=0.2)
    size_a = np.array(size)
    good = (np.array(present) == 1) & (size_a > 0.08)
    motion = np.array(cam) + hand_mv  # fraction of frame per 0.5 s
    return {"fps": SCAN_FPS, "t": t, "present": present, "full": full, "size": np.round(size_a, 4).tolist(),
            "motion": np.round(motion, 5).tolist(), "good": good.astype(int).tolist(), "duration": info["duration"]}


def recommend(scan: dict, duration_s: float = 20.0) -> dict:
    t = np.array(scan["t"])
    good = np.array(scan["good"], bool)
    motion = np.array(scan["motion"])
    n = len(t)
    win = max(1, int(round(duration_s * scan["fps"])))
    if n == 0:
        return {"start_s": 0.0, "duration_s": duration_s, "score": None}
    if n <= win:
        return {"start_s": 0.0, "duration_s": float(scan["duration"]), "score": float(good.mean())}
    full = np.array(scan["full"], bool) & good
    cg = np.concatenate([[0], np.cumsum(good)])
    cf = np.concatenate([[0], np.cumsum(full)])
    cm = np.concatenate([[0], np.cumsum(np.minimum(motion, 0.2))])
    best, best_s = None, 0
    for s in range(0, n - win + 1):
        frac = (cg[s + win] - cg[s]) / win
        mv = (cm[s + win] - cm[s]) / win
        # visibility dominates; then steadiness; slight preference for earlier segments
        ffrac = (cf[s + win] - cf[s]) / win
        cost = (1 - frac) * 10 + (1 - ffrac) * 2 + mv * 50 + s / n * 0.05
        if best is None or cost < best:
            best, best_s = cost, s
    return {"start_s": float(t[best_s]), "duration_s": float(duration_s),
            "visible_pct": round(100 * float((cg[best_s + win] - cg[best_s]) / win), 1)}


class ScanManager:
    """Runs (once) and caches the 2 fps scan of a video in its results folder (scan.json)."""

    def __init__(self):
        self.state: dict[str, dict] = {}
        self.lock = threading.Lock()

    def get(self, name: str, video_path: str, scan_path: str, duration_s: float = 20.0) -> dict:
        if os.path.exists(scan_path):
            with open(scan_path) as fh:
                sc = json.load(fh)
            return {"state": "done", "scan": sc, "recommended": recommend(sc, duration_s)}
        key = scan_path
        with self.lock:
            st = self.state.get(key)
            if st and st["state"] == "running":
                return dict(st)
            if st and st["state"] == "error":
                return self.state.pop(key)  # report once, retry on next request
            self.state[key] = {"state": "running", "progress": 0.0}
        threading.Thread(target=self._run, args=(key, video_path, scan_path), daemon=True).start()
        return dict(self.state[key])

    def _run(self, key: str, video_path: str, scan_path: str):
        try:
            def prog(f):
                self.state[key]["progress"] = round(f, 3)
            sc = scan_video(video_path, prog)
            os.makedirs(os.path.dirname(scan_path), exist_ok=True)
            with open(scan_path, "w") as fh:
                json.dump(sc, fh)
            self.state[key] = {"state": "done"}
        except Exception as e:  # noqa: BLE001
            self.state[key] = {"state": "error", "message": f"{type(e).__name__}: {e}"}
