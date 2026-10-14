"""Signal processing for the kinematic series (positions in mm, angles in degrees)."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, coherence, sosfiltfilt, spectrogram, welch


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 2) -> np.ndarray:
    sos = butter(int(order), [lo, min(hi, 0.49 * fs)], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0) if len(x) > 3 * (2 * order + 1) * 2 else np.zeros_like(x)


def lowpass(x: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    sos = butter(int(order), min(fc, 0.45 * fs), btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0) if len(x) > 30 else x.copy()


def deriv(x: np.ndarray, fs: float) -> np.ndarray:
    return np.gradient(x, axis=0) * fs


def nperseg(T: int, fs: float, seconds: float = 4.0) -> int:
    return int(min(T, max(16, round(seconds * fs))))


def psd(x: np.ndarray, fs: float, seg: int | None = None):
    seg = seg or nperseg(len(x), fs)
    f, P = welch(x, fs=fs, nperseg=seg, noverlap=seg // 2, detrend="linear", axis=0)
    return f, P


def band_stats(f: np.ndarray, P: np.ndarray, lo: float, hi: float, P_noise: np.ndarray | None = None) -> dict:
    df = float(f[1] - f[0]) if len(f) > 1 else 1.0
    band = (f >= lo) & (f <= hi)
    full = f >= 0.5
    bp = float(P[band].sum() * df)
    tot = float(P[full].sum() * df)
    out = {"peak_hz": round(float(f[band][np.argmax(P[band])]), 3) if band.any() else None,
           "band_rms": float(np.sqrt(max(bp, 0.0))), "total_rms": float(np.sqrt(max(tot, 0.0))),
           "band_power_pct": round(100 * bp / tot, 2) if tot > 0 else None}
    if P_noise is not None:
        nb = float(P_noise[band].sum() * df)
        out["noise_rms"] = float(np.sqrt(max(nb, 0.0)))
        out["snr_db"] = round(10 * np.log10(bp / nb), 2) if nb > 0 and bp > 0 else None
    return out


def spectro(x: np.ndarray, fs: float, fmax: float):
    seg = int(min(len(x), max(16, round(2.0 * fs))))
    f, t, S = spectrogram(x, fs=fs, nperseg=seg, noverlap=int(seg * 0.85), detrend="linear", scaling="density")
    keep = f <= fmax
    db = 10 * np.log10(np.maximum(S[keep], 1e-12))
    step = max(1, db.shape[1] // 160)
    return t[::step], f[keep], db[:, ::step]


def coherence_band(x: np.ndarray, y: np.ndarray, fs: float, lo: float, hi: float) -> float:
    seg = nperseg(len(x), fs)
    f, C = coherence(x, y, fs=fs, nperseg=seg, noverlap=seg // 2, detrend="linear")
    band = (f >= lo) & (f <= hi)
    return float(np.mean(C[band])) if band.any() else float("nan")


def principal(xyz_band: np.ndarray):
    """First principal direction of a band-passed 3D signal and the projection onto it."""
    C = np.cov(xyz_band.T)
    w, V = np.linalg.eigh(C)
    d = V[:, -1]
    if d[np.argmax(np.abs(d))] < 0:
        d = -d
    return xyz_band @ d, d


def ellipsoid(xyz_band: np.ndarray) -> dict:
    """2-sigma tremor ellipsoid: semi-axes (mm) and directions (columns), largest first."""
    C = np.cov(xyz_band.T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    return {"axes": [float(2 * np.sqrt(max(w[i], 0.0))) for i in order], "dirs": V[:, order].T.tolist()}


def rnd(a, nd: int = 4):
    """Round to `nd` significant-ish decimals for compact JSON (NaN -> None)."""
    a = np.asarray(a, dtype=np.float64)
    out = np.round(a, nd)
    if np.isnan(out).any():
        return [None if not np.isfinite(v) else float(v) for v in out.ravel()] if out.ndim == 1 else \
            [rnd(r, nd) for r in out]
    return out.tolist()
