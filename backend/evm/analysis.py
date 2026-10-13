"""Tremor quantification from optical-flow motion (Eulerian fields + Lagrangian fingertip anchors)."""
import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, spectrogram, welch

from .hands import ANCHORS, hand_mask
from .magnify import bandpass_sos


def sample_flow(flow: np.ndarray, pts: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Median flow in a disk (restricted to the hand mask) around each anchor point. -> (N,2)"""
    h, w = flow.shape[:2]
    out = np.zeros((len(pts), 2), np.float32)
    r = max(2, int(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy) <= r * r
    for i, (x, y) in enumerate(pts):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - r), min(w, xi + r + 1)
        y0, y1 = max(0, yi - r), min(h, yi + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        d = disk[y0 - (yi - r):y1 - (yi - r), x0 - (xi - r):x1 - (xi - r)]
        m = d & (mask[y0:y1, x0:x1] > 0)
        if m.sum() < 5:
            m = d
        f = flow[y0:y1, x0:x1][m]
        if len(f):
            out[i] = np.median(f, axis=0)
    return out


def displacement_fields(Vg: np.ndarray, out: np.ndarray, sos: np.ndarray, rows_per_chunk: int = 8):
    """out[:] = bandpass(cumsum_t(Vg)). Chunked over rows to bound memory."""
    gh = Vg.shape[1]
    for r0 in range(0, gh, rows_per_chunk):
        r1 = min(gh, r0 + rows_per_chunk)
        c = np.cumsum(np.asarray(Vg[:, r0:r1], np.float64), axis=0)
        out[:, r0:r1] = sosfiltfilt(sos, c, axis=0).astype(out.dtype)


def envelope_fields(D: np.ndarray, win: int, rows_per_chunk: int = 8) -> np.ndarray:
    """Sliding-RMS tremor envelope magnitude (T, gh, gw) float16."""
    T, gh, gw, _ = D.shape
    E = np.empty((T, gh, gw), np.float16)
    for r0 in range(0, gh, rows_per_chunk):
        r1 = min(gh, r0 + rows_per_chunk)
        d = np.asarray(D[:, r0:r1], np.float32)
        p = (d * d).sum(-1)
        E[:, r0:r1] = np.sqrt(uniform_filter1d(p, size=max(1, win), axis=0, mode="nearest")).astype(np.float16)
    return E


def _sliding_rms(x: np.ndarray, win: int) -> np.ndarray:
    return np.sqrt(uniform_filter1d((x * x).sum(-1), size=max(1, win), axis=0, mode="nearest"))


def _decimate_idx(T: int, max_pts: int) -> np.ndarray:
    step = max(1, int(np.ceil(T / max_pts)))
    return np.arange(0, T, step)


def _r(a, nd=4):
    a = np.nan_to_num(np.asarray(a, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.round(a, nd).tolist()


def _peak_hz(f, vel, band):
    if not band.any() or not np.isfinite(vel[band]).any():
        return None
    return round(float(f[band][np.nanargmax(vel[band])]), 2)


def analyse(anchor_v: np.ndarray, anchor_valid: np.ndarray, D: np.ndarray, stab_lms: np.ndarray,
            fps: float, P: dict, grid_scale: tuple[float, float], hand_px: float) -> tuple[dict, np.ndarray]:
    """Compute metrics, spectra, envelopes and spectrograms.

    anchor_v: (T,6,2) Lagrangian anchor velocities (px/frame). Returns (analysis dict, Db (T,6,2)).
    """
    T = anchor_v.shape[0]
    lo, hi = float(P["freq_lo"]), float(P["freq_hi"])
    sos_b = bandpass_sos(lo, hi, fps, P["filter_order"])
    sos_h = butter(2, 1.0, btype="highpass", fs=fps, output="sos")

    Dc = np.cumsum(np.nan_to_num(anchor_v.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0), axis=0)
    Db = np.nan_to_num(sosfiltfilt(sos_b, Dc, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    Dh = np.nan_to_num(sosfiltfilt(sos_h, Dc, axis=0), nan=0.0, posinf=0.0, neginf=0.0)

    nper = int(min(T, round(4 * fps)))
    f, _ = welch(Dh[:, 0, 0], fs=fps, nperseg=nper, detrend="linear")
    psd = {}
    for k, name in enumerate(ANCHORS):
        _, px = welch(Dh[:, k, 0], fs=fps, nperseg=nper, detrend="linear")
        _, py = welch(Dh[:, k, 1], fs=fps, nperseg=nper, detrend="linear")
        psd[name] = px + py
    band = (f >= lo) & (f <= hi)
    tot = (f >= 1.0)

    win = int(round(0.5 * fps))
    env_idx = _decimate_idx(T, 1500)
    env_series = {}
    metrics = {}
    for k, name in enumerate(ANCHORS):
        p = psd[name]
        vel = p * (2 * np.pi * f) ** 2  # velocity spectrum emphasises the tremor peak over 1/f^2 drift
        pk = _peak_hz(f, vel, band)
        mag = np.linalg.norm(Db[:, k], axis=1)
        env = _sliding_rms(Db[:, k], win)
        env_series[name] = _r(env[env_idx])
        metrics[name] = {
            "peak_hz": pk,
            "rms_px": round(float(np.sqrt(np.mean(mag ** 2))), 4),
            "p95_px": round(float(np.percentile(mag, 95)), 4),
            "band_power_pct": round(float(100 * p[band].sum() / max(p[tot].sum(), 1e-12)), 1),
            "rms_rel_hand_pct": round(float(100 * np.sqrt(np.mean(mag ** 2)) / max(hand_px, 1e-6)), 4),
        }

    # Eulerian whole-hand envelope from the displacement field
    gh, gw = D.shape[1:3]
    sx, sy = grid_scale
    hand_env = np.zeros(T, np.float32)
    for t in range(T):
        m = hand_mask((gh, gw), stab_lms[t], margin=float(P["mask_margin"]), scale=sx)
        if m.any():
            d = np.asarray(D[t], np.float32)[m > 0]
            hand_env[t] = float(np.sqrt(np.mean((d * d).sum(-1))))
    hand_env = np.nan_to_num(hand_env, nan=0.0, posinf=0.0, neginf=0.0)
    hand_env_s = np.sqrt(uniform_filter1d(hand_env ** 2, size=win, mode="nearest"))
    env_series["hand"] = _r(hand_env_s[env_idx])
    mean_psd = np.mean([psd[n] for n in ANCHORS[:5]], axis=0)
    vel = mean_psd * (2 * np.pi * f) ** 2
    metrics["hand"] = {
        "peak_hz": _peak_hz(f, vel, band),
        "rms_px": round(float(np.sqrt(np.mean(hand_env ** 2))), 4),
        "p95_px": round(float(np.percentile(hand_env, 95)), 4),
        "band_power_pct": round(float(100 * mean_psd[band].sum() / max(mean_psd[tot].sum(), 1e-12)), 1),
        "rms_rel_hand_pct": round(float(100 * np.sqrt(np.mean(hand_env ** 2)) / max(hand_px, 1e-6)), 4),
    }

    # spectrograms (displacement power, dB)
    nps = int(min(T, round(2 * fps)))
    hop = max(int(round(0.25 * fps)), int(np.ceil(max(T - nps, 1) / 800)))
    spec = {}
    sf = st = None
    for k, name in enumerate(ANCHORS):
        sf, st, Sx = spectrogram(Dh[:, k, 0], fs=fps, nperseg=nps, noverlap=max(0, nps - hop), detrend="linear")
        _, _, Sy = spectrogram(Dh[:, k, 1], fs=fps, nperseg=nps, noverlap=max(0, nps - hop), detrend="linear")
        spec[name] = np.round(10 * np.log10(Sx + Sy + 1e-12), 1).tolist()

    tr_idx = _decimate_idx(T, 6000)
    analysis = {
        "anchors": ANCHORS,
        "metrics": metrics,
        "spectrum": {"freqs": _r(f, 3), "psd": {k: np.asarray(v).tolist() for k, v in psd.items()}},
        "envelope": {"t": _r(env_idx / fps, 3), "series": env_series, "window_s": 0.5},
        "spectrogram": {"f": _r(sf, 3), "t": _r(st, 3), "db": spec},
        "trace": {"t": _r(tr_idx / fps, 3),
                  "xy": {name: np.round(Db[tr_idx, k], 4).tolist() for k, name in enumerate(ANCHORS)}},
        "detection_rate": round(float(np.mean(anchor_valid)), 3),
    }
    return analysis, Db.astype(np.float32)
