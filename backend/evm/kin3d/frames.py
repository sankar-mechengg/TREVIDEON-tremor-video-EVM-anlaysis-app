"""Body frames of the hand and relative poses.

Joint order (MediaPipe / OpenPose, also WiLoR's output): 0 wrist; thumb 1 CMC, 2 MCP, 3 IP, 4 tip;
index 5-8, middle 9-12, ring 13-16, pinky 17-20 (MCP, PIP, DIP, tip).

Wrist (palm) frame W_p  - one rigid body: wrist + the four finger knuckles (MCPs) fitted with Kabsch to a
                          per-clip palm template, so a single noisy keypoint cannot tilt the frame.
                          origin = wrist joint, y = distal (towards the middle knuckle), z = dorsal (back of
                          the hand), x = y x z (right-handed; ulnar side for a right hand, radial for a left).
Finger frame F_f        - the finger as ONE rigid body: origin = knuckle (MCP; thumb: its MCP), x = knuckle ->
                          fingertip, z = the palm normal made perpendicular to x, y = z x x.
Relative pose           - F_f expressed in W_p: R_pf = R_p^T R_f, o_pf = R_p^T (o_f - o_p). The world (and any
                          camera / head motion) cancels in this product.
Finger angles           - of the finger axis in the palm frame: flexion (out of the palm plane, + towards the
                          palm), abduction (in the palm plane, + towards the thumb side); twist about the finger
                          axis comes from WiLoR's MANO joint rotations when available.
"""
from __future__ import annotations

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
CHAIN = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16],
         "pinky": [17, 18, 19, 20]}
RIGID = {"thumb": (2, 4), "index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}
TIP = {f: CHAIN[f][-1] for f in FINGERS}
PALM_IDX = [0, 5, 9, 13, 17]
JOINT_NAMES = {"thumb": ["cmc", "mcp", "ip"], "index": ["mcp", "pip", "dip"], "middle": ["mcp", "pip", "dip"],
               "ring": ["mcp", "pip", "dip"], "pinky": ["mcp", "pip", "dip"]}


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def palm_basis(J: np.ndarray, right: bool) -> np.ndarray:
    """Three-point palm basis per frame (columns x, y, z). J: (T,21,3)."""
    y = _unit(J[:, 9] - J[:, 0])
    a, b = J[:, 17] - J[:, 0], J[:, 5] - J[:, 0]
    z = np.cross(a, b) if right else np.cross(b, a)        # dorsal normal for either hand
    z = _unit(z - np.sum(z * y, -1, keepdims=True) * y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def kabsch(Q: np.ndarray, P: np.ndarray):
    """R, t minimising |R Q + t - P| (rows are points)."""
    qc, pc = Q.mean(0), P.mean(0)
    Hm = (Q - qc).T @ (P - pc)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, pc - R @ qc


def palm_frames(J: np.ndarray, right: bool):
    """Kabsch palm frames. Returns R (T,3,3) [columns = palm axes in J's frame], o (T,3), template (5,3)."""
    B = palm_basis(J, right)
    local = np.einsum("tji,tkj->tki", B, J[:, PALM_IDX] - J[:, :1])     # B^T (p - wrist)
    ok = np.isfinite(local).all(axis=(1, 2))
    Q = np.median(local[ok], axis=0) if ok.any() else local[0]
    T = len(J)
    R = np.full((T, 3, 3), np.nan)
    o = np.full((T, 3), np.nan)
    for t in np.where(ok)[0]:
        R[t], o[t] = kabsch(Q, J[t, PALM_IDX])
    return R, o, Q


def finger_frames(J: np.ndarray, Rp: np.ndarray) -> dict:
    out = {}
    zp = Rp[:, :, 2]
    for f in FINGERS:
        m, tip = RIGID[f]
        x = _unit(J[:, tip] - J[:, m])
        z = _unit(zp - np.sum(zp * x, -1, keepdims=True) * x)
        y = np.cross(z, x)
        out[f] = (np.stack([x, y, z], axis=-1), J[:, m].copy())
    return out


def to_palm(Rp, op, P):
    """Points P (T,...,3) in the palm frame."""
    return np.einsum("tji,t...j->t...i", Rp, P - op.reshape(len(op), *([1] * (P.ndim - 2)), 3))


def finger_angles(J: np.ndarray, Rp: np.ndarray, right: bool) -> dict:
    """Flexion / abduction (deg) of each rigid finger relative to the palm."""
    radial = -1.0 if right else 1.0              # palm x is ulnar for a right hand
    out = {}
    for f in FINGERS:
        m, tip = RIGID[f]
        d = np.einsum("tji,tj->ti", Rp, _unit(J[:, tip] - J[:, m]))
        flex = np.degrees(np.arcsin(np.clip(-d[:, 2], -1, 1)))
        abd = np.degrees(np.arctan2(radial * d[:, 0], d[:, 1]))
        out[f] = {"flexion": flex, "abduction": abd}
    return out


def _angle(a, b):
    c = np.sum(_unit(a) * _unit(b), -1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def joint_angles(J: np.ndarray) -> dict:
    """Flexion angle (deg) at each finger joint: angle between consecutive bone vectors."""
    out = {}
    for f in FINGERS:
        c = CHAIN[f]
        pts = [0] + c
        names = JOINT_NAMES[f]
        out[f] = {names[i]: _angle(J[:, pts[i + 1]] - J[:, pts[i]], J[:, pts[i + 2]] - J[:, pts[i + 1]])
                  for i in range(3)}
    return out


def rotvec(R: np.ndarray) -> np.ndarray:
    """(T,3,3) -> rotation vectors (T,3) in radians."""
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(tr)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    s = np.sin(th)
    k = np.where(s > 1e-8, th / (2 * np.maximum(s, 1e-12)), 0.5)
    return v * k[:, None]


def mean_rotation(R: np.ndarray) -> np.ndarray:
    ok = np.isfinite(R).all(axis=(1, 2))
    M = R[ok].sum(0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def orthonormalize(R: np.ndarray) -> np.ndarray:
    out = np.empty_like(R)
    for t in range(len(R)):
        U, _, Vt = np.linalg.svd(R[t])
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        out[t] = U @ D @ Vt
    return out


def fill_nan(x: np.ndarray, max_gap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation over NaN frames (axis 0). Returns (filled, valid-mask before filling)."""
    T = x.shape[0]
    flat = x.reshape(T, -1).astype(np.float64)
    ok = np.isfinite(flat).all(axis=1)
    if ok.all() or not ok.any():
        return flat.reshape(x.shape), ok
    t = np.arange(T)
    out = flat.copy()
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t, t[ok], flat[ok, c])
    return out.reshape(x.shape), ok
