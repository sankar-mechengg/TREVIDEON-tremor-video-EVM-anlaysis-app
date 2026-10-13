"""Camera-motion estimation and low-pass trajectory stabilisation.

Frame-to-frame similarity transforms are estimated from sparse LK feature tracks restricted to a region
(background = everything except the hand; hand = the dorsum/wrist). The accumulated camera trajectory is
zero-phase low-passed; the difference between the smooth and the raw trajectory is the per-frame correction
W_t, which removes all motion above the cut-off (incl. the tremor band) while keeping slow framing drift.
"""
import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

I3 = np.eye(3)


class MotionEstimator:
    def __init__(self, scale: float = 0.5):
        self.scale = scale

    def step(self, prev_gray: np.ndarray, cur_gray: np.ndarray, region: np.ndarray | None) -> np.ndarray:
        """3x3 similarity mapping prev-frame coords -> current-frame coords."""
        s = self.scale
        if region is not None and cv2.countNonZero(region) < 500:
            return I3.copy()
        ps = cv2.resize(prev_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cs = cv2.resize(cur_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ms = None if region is None else cv2.resize(region, (ps.shape[1], ps.shape[0]), interpolation=cv2.INTER_NEAREST)
        pts = cv2.goodFeaturesToTrack(ps, maxCorners=600, qualityLevel=0.004, minDistance=6, mask=ms, blockSize=7)
        if pts is None or len(pts) < 12:
            return I3.copy()
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(ps, cs, pts, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(cs, ps, nxt, None, **lk)
        fb = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)
        if good.sum() < 10:
            return I3.copy()
        M, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                             ransacReprojThreshold=0.75, maxIters=3000, confidence=0.995)
        if M is None or inl is None or inl.sum() < 8:
            return I3.copy()
        M = M.astype(np.float64)
        M[:, 2] /= s
        return np.vstack([M, [0, 0, 1]])


def stabilization_region(shape_hw, hands_lm, mode: str, hand_mask_fn) -> np.ndarray | None:
    """Feature mask for motion estimation."""
    if mode == "background":
        m = hand_mask_fn(shape_hw, hands_lm, margin=1.8)
        region = cv2.bitwise_not(m)
        # ignore the extreme border (rolling-shutter/lens distortion)
        b = max(4, shape_hw[0] // 40)
        region[:b] = 0
        region[-b:] = 0
        region[:, :b] = 0
        region[:, -b:] = 0
        return region
    if mode == "hand":
        primary = hands_lm[:1]
        if np.isnan(primary[0, 0, 0]):
            return np.zeros(shape_hw, np.uint8)
        return hand_mask_fn(shape_hw, primary, margin=1.0, fingers=False, palm_only_shrink=0.85)
    return None


def compute_correction(A: list[np.ndarray], fps: float, cutoff_hz: float) -> tuple[np.ndarray, dict]:
    """A[t] maps frame t-1 -> t (A[0] = I). Returns W (T,3,3) mapping frame t -> stabilised frame t."""
    T = len(A)
    C = np.empty((T, 3, 3))
    C[0] = I3
    for t in range(1, T):
        C[t] = A[t] @ C[t - 1]
    # similarity params [[a,-b,tx],[b,a,ty]]
    params = np.stack([C[:, 0, 0], C[:, 1, 0], C[:, 0, 2], C[:, 1, 2]], axis=1)
    if T > 30:
        sos = butter(2, min(cutoff_hz, 0.45 * fps), btype="low", fs=fps, output="sos")
        smooth = sosfiltfilt(sos, params, axis=0)
    else:
        smooth = np.repeat(params.mean(axis=0, keepdims=True), T, axis=0)
    S = np.zeros((T, 3, 3))
    S[:, 0, 0] = smooth[:, 0]
    S[:, 0, 1] = -smooth[:, 1]
    S[:, 1, 0] = smooth[:, 1]
    S[:, 1, 1] = smooth[:, 0]
    S[:, 0, 2] = smooth[:, 2]
    S[:, 1, 2] = smooth[:, 3]
    S[:, 2, 2] = 1
    W = S @ np.linalg.inv(C)
    corr = np.linalg.norm(W[:, :2, 2], axis=1)
    stats = {
        "camera_path_px": float(np.linalg.norm(params[-1, 2:] - params[0, 2:])),
        "correction_rms_px": float(np.sqrt(np.mean(corr ** 2))),
        "correction_max_px": float(corr.max()) if T else 0.0,
    }
    return W, stats


def warp(img: np.ndarray, W: np.ndarray, size: tuple[int, int], interp=cv2.INTER_CUBIC, border=cv2.BORDER_REFLECT):
    return cv2.warpAffine(img, W[:2], size, flags=interp, borderMode=border)
