"""Hand isolation: segment the dominant hand and pin its wrist centre to a fixed canvas point.

Output: a square, background-free video ("isolated_input.mp4") in which
  * only the dominant hand (plus a short forearm stub) is visible, on a black background,
  * the wrist centre follows its slow path only: wrist motion below `anchor_cutoff_hz` (voluntary arm /
    hand movement) is removed, faster motion (tremor, including tremor of the wrist itself) is kept,
  * the hand is scaled so that wrist -> middle-finger MCP = `hand_len_px` (amplitudes comparable across
    people); orientation is NOT normalised, so rotational tremor about the wrist is preserved.
The canvas anchor (+) sits opposite the median finger direction so that the whole hand fits.
The standard EVM / optical-flow pipeline then runs on this video.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .hands import MODEL_PATH, hand_mask, hand_size
from .segment import make_segmenter
from .video_io import SegmentReader, VideoWriter


class DominantHandTracker:
    """MediaPipe hand landmarks with handedness; keeps following the chosen hand over time."""

    def __init__(self, prefer: str = "auto", mirrored: bool = False, confidence: float = 0.4):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                                            running_mode=vision.RunningMode.VIDEO, num_hands=2,
                                            min_hand_detection_confidence=confidence,
                                            min_hand_presence_confidence=confidence,
                                            min_tracking_confidence=confidence)
        self.lm = vision.HandLandmarker.create_from_options(opts)
        # MediaPipe labels handedness as seen in a MIRRORED (selfie) image
        self.want = None
        if prefer in ("right", "left"):
            phys = prefer.capitalize()
            self.want = phys if mirrored else ("Left" if phys == "Right" else "Right")
        self.prev = None

    def detect(self, bgr, t_ms):
        h, w = bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(img, int(t_ms))
        hands = []
        for hl, hd in zip(res.hand_landmarks, res.handedness):
            hands.append((np.array([[p.x * w, p.y * h] for p in hl], np.float32), hd[0].category_name))
        nan = np.full((21, 2), np.nan, np.float32)
        if not hands:
            return nan, nan
        cand = [i for i, (_, lab) in enumerate(hands) if self.want is None or lab == self.want]
        if not cand:
            cand = list(range(len(hands)))
            if self.prev is None:
                return nan, (hands[0][0] if hands else nan)
        if self.prev is not None:
            d = [float(np.linalg.norm(hands[i][0][0] - self.prev[0])) for i in cand]
            i = cand[int(np.argmin(d))]
            if min(d) > 0.3 * max(w, h):
                i = max(cand, key=lambda k: np.ptp(hands[k][0][:, 0]) * np.ptp(hands[k][0][:, 1]))
        else:
            i = max(cand, key=lambda k: np.ptp(hands[k][0][:, 0]) * np.ptp(hands[k][0][:, 1]))
        self.prev = hands[i][0]
        other = next((hands[j][0] for j in range(len(hands)) if j != i), nan)
        return hands[i][0], other

    def close(self):
        self.lm.close()


def _fill(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN rows (edges held). x: (T, ...)"""
    T = x.shape[0]
    flat = x.reshape(T, -1).astype(np.float64)
    ok = np.isfinite(flat[:, 0])
    if not ok.any():
        return flat.reshape(x.shape)
    t = np.arange(T)
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(t, t[ok], flat[ok, c])
    return flat.reshape(x.shape)


def wrist_path(wrist: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """Slow wrist path (zero-phase low-pass); cutoff 0 = the raw wrist position (exact pin)."""
    w = _fill(wrist)
    if cutoff_hz <= 0 or len(w) < 16:
        return w
    sos = butter(2, min(cutoff_hz, 0.45 * fps), btype="low", fs=fps, output="sos")
    return sosfiltfilt(sos, w, axis=0)


def build_isolated_video(video_path: str, P: dict, out_dir: str, prog) -> dict:
    """Writes out_dir/isolated_input.mp4 + isolation.json and returns the isolation info."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    reader = SegmentReader(video_path, P["start_s"], P["duration_s"], P["work_height"],
                           repair=bool(P.get("repair_duplicates")))
    fps, n = reader.fps, reader.n
    C = int(P["canvas_px"])
    L = float(P["hand_len_px"])

    # ---- pass 1: dominant-hand landmarks
    trk = DominantHandTracker(P["dominant_hand"], bool(P["mirrored"]), float(P["hand_confidence"]))
    lm = np.full((n, 21, 2), np.nan, np.float32)
    oth = np.full((n, 21, 2), np.nan, np.float32)
    T = 0
    try:
        for t, f in enumerate(reader.frames()):
            lm[t], oth[t] = trk.detect(f, int(round(t * 1000.0 / fps)))
            T = t + 1
            prog("isolate1", T / n, f"Finding the dominant hand  {T}/{n}")
    finally:
        trk.close()
    lm, oth = lm[:T], oth[:T]
    found = np.isfinite(lm[:, 0, 0])
    if found.mean() < 0.5:
        raise RuntimeError(f"The dominant hand was found in only {100 * found.mean():.0f} % of the frames; "
                           "choose another segment or hand.")
    lm_f = _fill(lm)
    path = wrist_path(lm_f[:, 0], fps, float(P["anchor_cutoff_hz"]))
    sizes = np.linalg.norm(lm[:, 0] - lm[:, 9], axis=1)
    s = L / float(np.nanmedian(sizes))
    d = np.nanmedian(lm[:, 9] - lm[:, 0], axis=0)
    d = d / max(np.linalg.norm(d), 1e-6)
    anchor = np.array([C / 2.0, C / 2.0]) - d * 0.22 * C       # fingers point away from the anchor
    M = np.zeros((T, 2, 3), np.float64)
    M[:, 0, 0] = M[:, 1, 1] = s
    M[:, :, 2] = anchor[None] - s * path

    # ---- pass 2: segment, warp into the canvas, temporally smooth the mask, composite
    seg, used = make_segmenter(P["segmenter"])
    writer = VideoWriter(os.path.join(out_dir, "isolated_input.mp4"), (C, C), fps, crf=10)
    feather = max(1.0, 0.0022 * C)          # ~1.6 px at 720: soft edge without background glow
    taps = {-2: 1.0, -1: 4.0, 0: 6.0, 1: 4.0, 2: 1.0}          # centred temporal smoothing of the mask
    frames, masks = {}, {}                                      # frame index -> canvas image / soft mask
    coverage = []
    state = {"next": 0}

    def emit(i):
        num = sum(taps[k] * masks[i + k] for k in taps if (i + k) in masks)
        den = sum(taps[k] for k in taps if (i + k) in masks)
        mk = cv2.erode(num / den, None, iterations=1)
        mk = cv2.GaussianBlur(mk, (0, 0), feather)
        writer.write(np.clip(frames[i].astype(np.float32) * mk[..., None], 0, 255).astype(np.uint8))
        coverage.append(float(mk.mean()))
        for j in [j for j in frames if j < i - 1]:              # keep the two frames the next window needs
            del frames[j], masks[j]

    try:
        for t, f in enumerate(reader.frames(limit=T)):
            if found[t]:
                m = seg(f, lm[t], oth[t] if np.isfinite(oth[t]).all() else None, int(round(t * 1000.0 / fps)))
            else:   # short detection gap: polygon from the interpolated landmarks
                m = hand_mask(f.shape[:2], lm_f[t][None].astype(np.float32), margin=1.15).astype(np.float32) / 255
            frames[t] = cv2.warpAffine(f, M[t], (C, C), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            masks[t] = cv2.warpAffine(m, M[t], (C, C), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
            while state["next"] <= t - 2:                       # centred window of frame `next` is complete
                emit(state["next"])
                state["next"] += 1
            prog("isolate2", (t + 1) / T, f"Segmenting the hand ({used}) & pinning the wrist  {t + 1}/{T}")
        while state["next"] < T:                                # tail: shorter windows
            emit(state["next"])
            state["next"] += 1
        writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        seg.close()

    info = {"segmenter": used, "requested_segmenter": P["segmenter"], "anchor_xy": [round(float(v), 2) for v in anchor],
            "canvas_px": C, "scale": round(float(s), 5), "hand_len_px": L,
            "hand_len_src_px": round(float(np.nanmedian(sizes)), 2), "finger_direction": [round(float(v), 4) for v in d],
            "anchor_cutoff_hz": float(P["anchor_cutoff_hz"]), "hand_found_pct": round(100 * float(found.mean()), 1),
            "mask_coverage_pct": round(100 * float(np.mean(coverage)), 2) if coverage else 0.0,
            "fallback_frames": int(getattr(seg, "n_fallback", 0)), "fps": fps, "n_frames": T,
            "start_s": round(reader.start_s, 3), "source_size": list(reader.size),
            "wrist_slow_path_px": np.round(path, 2).tolist(),
            "wrist_removed_motion_rms_px": round(float(np.sqrt(np.mean(np.sum((_fill(lm[:, 0]) - path) ** 2, 1)))), 3)
            if float(P["anchor_cutoff_hz"]) > 0 else 0.0,
            "processing_s": round(time.time() - t0, 1),
            "frame_repair": {"repaired_frames": len(reader.repaired),
                             "repaired_pct": round(100 * len(reader.repaired) / max(T, 1), 2),
                             "suspected_dropped_frames": len(reader.suspected_drops),
                             "suspected_dropped_pct": round(100 * len(reader.suspected_drops) / max(T, 1), 2)}
            if P.get("repair_duplicates") else None}
    with open(os.path.join(out_dir, "isolation.json"), "w") as fh:
        json.dump(info, fh)
    return info


def draw_anchor(img: np.ndarray, xy, size: int = 14):
    """Draw the static wrist anchor (+) on an output frame (never on frames that are measured)."""
    x, y = int(round(xy[0])), int(round(xy[1]))
    for col, th in (((0, 0, 0), 4), ((60, 230, 255), 2)):
        cv2.line(img, (x - size, y), (x + size, y), col, th, cv2.LINE_AA)
        cv2.line(img, (x, y - size), (x, y + size), col, th, cv2.LINE_AA)
    return img
