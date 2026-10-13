"""MediaPipe hand landmark tracking, landmark smoothing and hand-mask rasterisation."""
import os

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
                          "hand_landmarker.task")

CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
PALM = [0, 1, 2, 5, 9, 13, 17]
TIPS = [4, 8, 12, 16, 20]
DIPS = [3, 7, 11, 15, 19]
ANCHORS = ["thumb", "index", "middle", "ring", "pinky", "dorsum"]


class HandDetector:
    def __init__(self, num_hands: int = 2, confidence: float = 0.4):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(num_hands),
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.prev_primary = None

    def detect(self, bgr: np.ndarray, t_ms: int) -> np.ndarray:
        """Returns (2, 21, 2) pixel landmarks; slot 0 = primary hand, NaN where absent."""
        h, w = bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(img, int(t_ms))
        out = np.full((2, 21, 2), np.nan, np.float32)
        hands = [np.array([[p.x * w, p.y * h] for p in hl], np.float32) for hl in res.hand_landmarks]
        if not hands:
            return out

        def area(a):
            return float(np.ptp(a[:, 0]) * np.ptp(a[:, 1]))

        if self.prev_primary is not None:
            d = [float(np.linalg.norm(a[0] - self.prev_primary[0])) for a in hands]
            i = int(np.argmin(d))
            if d[i] > 0.25 * max(w, h):
                i = int(np.argmax([area(a) for a in hands]))
        else:
            i = int(np.argmax([area(a) for a in hands]))
        out[0] = hands[i]
        others = [a for j, a in enumerate(hands) if j != i]
        if others:
            out[1] = others[0]
        self.prev_primary = hands[i]
        return out

    def close(self):
        self.lm.close()


def _runs(mask: np.ndarray):
    """Contiguous True runs as (start, stop) pairs."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def smooth_landmarks(L: np.ndarray, fps: float, cutoff: float = 5.0, max_gap_s: float = 0.5) -> np.ndarray:
    """Fill short detection gaps and zero-phase low-pass the landmark tracks (removes MediaPipe jitter).

    L: (T, 2, 21, 2). Returns same shape, NaN where the hand is absent for longer than max_gap_s.
    """
    T = L.shape[0]
    out = np.full_like(L, np.nan)
    t = np.arange(T)
    sos = butter(2, min(cutoff, 0.45 * fps), btype="low", fs=fps, output="sos")
    max_gap = int(round(max_gap_s * fps))
    for slot in range(L.shape[1]):
        valid = ~np.isnan(L[:, slot, 0, 0])
        vi = np.where(valid)[0]
        if len(vi) < 2:
            continue
        series = L[:, slot].reshape(T, -1).astype(np.float64)
        filled = np.empty_like(series)
        for c in range(series.shape[1]):
            filled[:, c] = np.interp(t, vi, series[vi, c])
        pos = np.searchsorted(vi, t)
        left = vi[np.clip(pos - 1, 0, len(vi) - 1)]
        right = vi[np.clip(pos, 0, len(vi) - 1)]
        dist = np.minimum(np.abs(t - left), np.abs(right - t))
        ok = dist <= max_gap
        for a, b in _runs(ok):
            seg = filled[a:b]
            if b - a > 20:
                seg = sosfiltfilt(sos, seg, axis=0)
            out[a:b, slot] = seg.reshape(b - a, 21, 2)
    return out


def transform_landmarks(L: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Apply per-frame 3x3 affine W (T,3,3) to landmarks (T, H, 21, 2)."""
    A = W[:, None, :2, :2]
    b = W[:, None, None, :2, 2]
    return np.einsum("thij,thkj->thki", np.broadcast_to(A, (L.shape[0], L.shape[1], 2, 2)), L) + b


def hand_size(lm: np.ndarray) -> float:
    """Wrist to middle-finger MCP distance (pixels)."""
    return float(np.linalg.norm(lm[0] - lm[9]))


def anchor_points(lm: np.ndarray) -> np.ndarray:
    """(6,2): fingertip pads (slightly inside the tip) + distal dorsum just behind the knuckles
    (kept away from the wrist-worn device, whose screen/cable would contaminate the flow)."""
    tips = lm[TIPS] * 0.7 + lm[DIPS] * 0.3
    knuckles = lm[[5, 9, 13, 17]].mean(axis=0)
    dorsum = (knuckles * 0.8 + lm[0] * 0.2)[None]
    return np.concatenate([tips, dorsum], axis=0)


def hand_mask(shape_hw, hands: np.ndarray, margin: float = 1.0, scale: float = 1.0,
              fingers: bool = True, palm_only_shrink: float = 1.0) -> np.ndarray:
    """Rasterise a hand mask from landmarks. hands: (H, 21, 2) possibly with NaN rows."""
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    for lm in hands:
        if np.isnan(lm[0, 0]):
            continue
        p = lm * scale
        s = hand_size(p)
        if s < 2:
            continue
        thick = max(1, int(round(0.26 * s * margin)))
        if fingers:
            palm = p[PALM]
        else:
            c = p[PALM].mean(axis=0)
            palm = c + (p[PALM] - c) * palm_only_shrink
        hull = cv2.convexHull(palm.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_8)
        if fingers:
            for chain in CHAINS:
                pts = p[chain].astype(np.int32)
                cv2.polylines(mask, [pts], False, 255, thick, lineType=cv2.LINE_8)
                cv2.circle(mask, tuple(int(v) for v in p[chain[-1]]), max(1, thick // 2), 255, -1)
            cv2.polylines(mask, [hull], True, 255, max(1, thick // 2))
        # wrist / wearable region
        wrist_end = p[0] + (p[0] - p[9]) * 0.45
        cv2.line(mask, tuple(int(v) for v in p[0]), tuple(int(v) for v in wrist_end), 255,
                 max(1, int(round(0.75 * s * margin))))
    return mask
