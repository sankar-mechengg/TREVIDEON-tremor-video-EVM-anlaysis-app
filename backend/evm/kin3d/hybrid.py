"""Hybrid finger vibration: sub-pixel optical flow, with the wrist's rigid motion removed, in the wrist frame.

Per frame pair (DIS optical flow on the un-magnified frames):
  * palm points (just distal of the knuckles + the distal dorsum; the wrist-worn device is avoided) give the
    rigid 2D motion of the back of the hand (least-squares similarity transform),
  * each fingertip pad's flow minus the flow the palm's rigid motion predicts at that point = the finger's
    own motion relative to the wrist, in pixels,
  * the per-frame increments are scaled to millimetres with the 3D model's metric depth (mm / px at the hand)
    and rotated into the (slowly varying, low-passed) wrist frame, then accumulated.
Optical flow sees only the motion across the image: the component along the camera's line of sight is not
measured. `observability` says how well each wrist-frame axis is seen (1 = in the image plane, 0 = along the
line of sight). The residuals of the rigid palm fit (skin texture at the hand's depth) give the flow noise.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..analysis import sample_flow
from ..hands import DIPS, TIPS


def tip_points(lm: np.ndarray) -> np.ndarray:
    return lm[TIPS] * 0.7 + lm[DIPS] * 0.3


def palm_points(lm: np.ndarray) -> np.ndarray:
    k = lm[[5, 9, 13, 17]]
    dist = k * 0.8 + lm[0] * 0.2
    dorsum = (k.mean(0) * 0.75 + lm[0] * 0.25)[None]
    mid = ((lm[5] + lm[9]) / 2 * 0.7 + lm[0] * 0.3)[None]
    mid2 = ((lm[13] + lm[17]) / 2 * 0.7 + lm[0] * 0.3)[None]
    return np.concatenate([dist, dorsum, mid, mid2], 0)


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 similarity (scale, rotation, translation) mapping src -> dst in the least-squares sense."""
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    if ok.sum() < 2:
        return np.array([[1.0, 0, 0], [0, 1.0, 0]])
    s, d = src[ok].astype(np.float64), dst[ok].astype(np.float64)
    sc, dc = s.mean(0), d.mean(0)
    a, b = s - sc, d - dc
    den = float((a ** 2).sum()) or 1e-12
    p = float((a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum()) / den
    q = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()) / den
    A = np.array([[p, -q], [q, p]])
    return np.hstack([A, (dc - A @ sc)[:, None]])


class FlowVibration:
    def __init__(self, hand_px: float):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.r_tip = max(3, int(0.06 * hand_px))
        self.r_palm = max(4, int(0.08 * hand_px))
        self.r_bg = max(6, int(0.10 * hand_px))

    def step(self, g0: np.ndarray, g1: np.ndarray, lm0: np.ndarray, mask0: np.ndarray):
        """Returns (tip residual (5,2) px, palm similarity (2,3), per-component variance of the palm-fit
        residuals (px^2, flow-noise estimate))."""
        flow = self.dis.calc(g0, g1, None)
        tips = tip_points(lm0)
        pp = palm_points(lm0)
        v_t = sample_flow(flow, tips, mask0, self.r_tip)
        v_p = sample_flow(flow, pp, mask0, self.r_palm)
        A = fit_similarity(pp, pp + v_p)
        pred = tips @ A[:, :2].T + A[:, 2] - tips
        rp = v_p - (pp @ A[:, :2].T + A[:, 2] - pp)
        ok = np.isfinite(rp).all(1)
        dof = 2 * int(ok.sum()) - 4
        pvar = float((rp[ok] ** 2).sum() / dof) if dof > 0 else np.nan
        return (v_t - pred).astype(np.float32), A, pvar


def to_palm_mm(resid_px: np.ndarray, mm_per_px: np.ndarray, R_cp: np.ndarray) -> np.ndarray:
    """Per-frame image-plane increments (T,5,2) px -> accumulated displacement in the wrist frame (T,5,3) mm.

    R_cp: (T,3,3) palm axes in camera coordinates (columns). An image displacement (du, dv) at the hand is the
    camera-frame displacement (du, dv, 0) * mm/px; it is rotated into the palm frame by R_cp^T."""
    inc_c = np.concatenate([resid_px, np.zeros(resid_px.shape[:2] + (1,), np.float32)], -1) * mm_per_px[:, None, None]
    inc_p = np.einsum("tji,tfj->tfi", R_cp, inc_c)
    inc_p[~np.isfinite(inc_p)] = 0.0
    return np.cumsum(inc_p, axis=0)


def observability(R_cp: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """(3,) median visibility of each palm axis: |component perpendicular to the viewing ray|."""
    r = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    dots = np.einsum("tji,tj->ti", R_cp, r)                # palm axis . ray
    vis = np.sqrt(np.clip(1 - dots ** 2, 0, 1))
    return np.nanmedian(vis, axis=0)
