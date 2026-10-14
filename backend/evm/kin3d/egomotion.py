"""Head-mounted (moving) camera motion from the static background.

1. BackgroundTracker: KLT feature tracks on the background (both hands masked out) relative to a keyframe;
   a RANSAC homography per frame maps first-frame pixels to frame-t pixels (H_t). The dominant plane in
   egocentric recordings of the hands is the floor / table, so H_t is the floor-plane homography.
2. camera_motion: the head ROTATION R_t is the polar part of K^-1 H_t K (exact for a rotating camera and any
   scene). The head translation is NOT estimated: with an assumed field of view, any focal-length error turns
   pure rotation into a spurious translation, which in tests moved the hand by centimetres. The floor normal n
   (H_t = K (R_t + t_t n^T / d) K^-1, n shared by all frames, solved by alternating polar decomposition and
   SVD of the stacked rank-1 residuals) is used only to turn the world axes upright when it is well determined.
   The rotation is then low-passed (smooth_motion): background tracking noise is larger than hand tremor inside
   the tremor band, so only slow head rotation is removed.
3. world_frame: world = first-frame camera; if the floor normal is reliable the axes are turned so that Z points
   up (away from the floor), X follows the camera's x and the origin is on the floor below the first camera
   (at the given camera height). Otherwise X = camera x, Y = camera forward, Z = camera up.
"""
from __future__ import annotations

import cv2
import numpy as np


def intrinsics(w: int, h: int, hfov_deg: float) -> np.ndarray:
    f = (w / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, (w - 1) / 2.0], [0, f, (h - 1) / 2.0], [0, 0, 1.0]])


class BackgroundTracker:
    def __init__(self, scale: float = 0.5):
        self.s = scale
        self.prev = None
        self.kf_pts = self.cur_pts = None
        self.H_kf = np.eye(3)
        self.H = []
        self.inliers = []
        self.n0 = 0
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def _detect(self, g, mask):
        pts = cv2.goodFeaturesToTrack(g, maxCorners=1500, qualityLevel=0.002, minDistance=5, mask=mask, blockSize=7)
        return None if pts is None else pts.reshape(-1, 2)

    def step(self, gray: np.ndarray, bg_mask: np.ndarray | None):
        """gray: full-resolution grayscale; bg_mask: uint8, 255 where the static background may be tracked."""
        s = self.s
        g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = None if bg_mask is None else cv2.resize(bg_mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        S = np.diag([s, s, 1.0])
        if self.prev is None:
            self.kf_pts = self._detect(g, m)
            self.cur_pts = None if self.kf_pts is None else self.kf_pts.copy()
            self.n0 = 0 if self.kf_pts is None else len(self.kf_pts)
            self.H.append(np.eye(3))
            self.inliers.append(self.n0)
            self.prev = g
            return
        H_t = self.H[-1]
        n_in = 0
        if self.cur_pts is not None and len(self.cur_pts) >= 12:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, self.cur_pts.astype(np.float32), None, **self.lk)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(g, self.prev, p1, None, **self.lk)
            fb = np.linalg.norm(p0r.reshape(-1, 2) - self.cur_pts, axis=1)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.8)
            if m is not None:
                xi = np.clip(p1.reshape(-1, 2)[:, 0].astype(int), 0, g.shape[1] - 1)
                yi = np.clip(p1.reshape(-1, 2)[:, 1].astype(int), 0, g.shape[0] - 1)
                ok &= m[yi, xi] > 0
            self.kf_pts, self.cur_pts = self.kf_pts[ok], p1.reshape(-1, 2)[ok]
            if len(self.cur_pts) >= 12:
                Hs, inl = cv2.findHomography(self.kf_pts, self.cur_pts, cv2.RANSAC, 1.0, maxIters=3000,
                                             confidence=0.999)
                if Hs is not None:
                    n_in = int(inl.sum())
                    H_t = np.linalg.inv(S) @ Hs @ S @ self.H_kf
        self.H.append(H_t / H_t[2, 2])
        self.inliers.append(n_in)
        if self.cur_pts is None or len(self.cur_pts) < max(60, 0.35 * self.n0):
            # new keyframe at the current frame
            self.H_kf = self.H[-1].copy()
            self.kf_pts = self._detect(g, m)
            self.cur_pts = None if self.kf_pts is None else self.kf_pts.copy()
            self.n0 = 0 if self.kf_pts is None else len(self.kf_pts)
        self.prev = g


def _polar(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def camera_motion(H: np.ndarray, K: np.ndarray, plane_distance: float, iters: int = 8) -> dict:
    """H: (T,3,3) first-frame -> frame-t homographies. Returns R (T,3,3), t (T,3) with X_t = R X_0 + t."""
    T = len(H)
    Ki = np.linalg.inv(K)
    G = np.empty((T, 3, 3))
    for i in range(T):
        g = Ki @ H[i] @ K
        g /= np.linalg.svd(g, compute_uv=False)[1]
        if np.linalg.det(g) < 0:
            g = -g
        G[i] = g
    R = np.stack([_polar(g) for g in G])
    n = np.array([0.0, 1.0, 0.0])
    a = np.zeros((T, 3))
    sv = np.array([1.0, 1.0, 1.0])
    for _ in range(iters):
        M = (G - R).reshape(-1, 3)
        _, sv, Vt = np.linalg.svd(M, full_matrices=False)
        n = Vt[0] / np.linalg.norm(Vt[0])
        if n[1] < 0:                               # floor below the camera (image y points down)
            n = -n
        a = np.einsum("tij,j->ti", G - R, n)
        R = np.stack([_polar(G[i] - np.outer(a[i], n)) for i in range(T)])
    ratio = float(sv[0] / max(sv[1], 1e-12))
    trans = float(np.median(np.linalg.norm(a, axis=1)))
    reliable = bool(ratio > 4.0 and trans > 0.004 and n[1] + n[2] > 0.3)     # plane below / in front
    R = np.stack([_polar(g) for g in G])                                     # head rotation only
    return {"R": R, "t": np.zeros((T, 3)), "n": n, "reliable": reliable, "sv_ratio": round(ratio, 2),
            "median_translation_rel": round(trans, 5), "plane_distance_m": plane_distance,
            "rotation_fit": round(float(np.mean([np.linalg.svd(g, compute_uv=False)[0] /
                                                 np.linalg.svd(g, compute_uv=False)[2] - 1 for g in G[::3]])), 4)}


def world_frame(cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Camera-to-world transforms per frame: P_w = Rcw[t] @ X_t + pcw[t]. Also returns R_W0 (world <- cam0)."""
    R, t = cam["R"], cam["t"]
    if cam["reliable"]:
        Z = -cam["n"]
        X = np.array([1.0, 0, 0]) - Z[0] * Z
        X /= np.linalg.norm(X)
        Y = np.cross(Z, X)
        R_W0 = np.stack([X, Y, Z])
        origin = np.array([0, 0, cam["plane_distance_m"]])
        info = {"axes": "Z up from the floor plane, X along the first camera's x, origin on the floor below the "
                        "first camera", "floor_aligned": True}
    else:
        R_W0 = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
        origin = np.zeros(3)
        info = {"axes": "X = first camera right, Y = first camera forward, Z = first camera up; origin at the first "
                        "camera", "floor_aligned": False}
    Rt = np.transpose(R, (0, 2, 1))
    Rcw = np.einsum("ij,tjk->tik", R_W0, Rt)
    pcw = -np.einsum("tij,tj->ti", Rcw, t) + origin
    return Rcw, pcw, R_W0, info


def smooth_motion(cam: dict, fps: float, cutoff_hz: float) -> dict:
    """Zero-phase low-pass of the camera trajectory. Background tracking is far noisier than the hand's tremor in
    the tremor band, so only the slow head motion (below cutoff_hz) is removed from the world frame."""
    from scipy.signal import butter, sosfiltfilt
    from .frames import orthonormalize
    T = len(cam["R"])
    if cutoff_hz <= 0 or T < 30:
        return cam
    sos = butter(2, min(cutoff_hz, 0.45 * fps), btype="low", fs=fps, output="sos")
    R = orthonormalize(sosfiltfilt(sos, cam["R"].reshape(T, 9), axis=0).reshape(T, 3, 3))
    t = sosfiltfilt(sos, cam["t"], axis=0)
    return dict(cam, R=R, t=t)


def static_camera(T: int) -> dict:
    return {"R": np.repeat(np.eye(3)[None], T, 0), "t": np.zeros((T, 3)), "n": np.array([0, 1.0, 0]),
            "reliable": False, "sv_ratio": 0.0, "median_translation_rel": 0.0, "plane_distance_m": 0.0}
