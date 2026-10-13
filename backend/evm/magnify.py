"""Streaming Eulerian video magnification of the luminance channel.

* RieszMagnifier  - phase-based motion magnification with the Riesz pyramid
                    (Wadhwa, Rubinstein, Durand, Freeman, ICCP 2014).
* LinearMagnifier - classic linear EVM on a Laplacian pyramid (Wu et al., SIGGRAPH 2012).

Both use a causal Butterworth band-pass applied per pixel in a streaming fashion (constant memory).
"""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.signal import butter, sosfilt_zi

EPS = 1e-6
KX = np.array([[-0.5, 0.0, 0.5]], np.float32)
KY = KX.T.copy()


def auto_levels(h: int, w: int) -> int:
    return int(max(2, min(6, np.floor(np.log2(min(h, w) / 20.0)))))


def build_laplacian(img: np.ndarray, levels: int):
    lap = []
    cur = img
    for _ in range(levels):
        down = cv2.pyrDown(cur)
        up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
        lap.append(cur - up)
        cur = down
    return lap, cur


def collapse_laplacian(lap, resid):
    cur = resid
    for L in reversed(lap):
        cur = cv2.pyrUp(cur, dstsize=(L.shape[1], L.shape[0])) + L
    return cur


def bandpass_sos(lo: float, hi: float, fps: float, order: int):
    nyq = 0.5 * fps
    hi = min(hi, 0.95 * nyq)
    lo = min(max(lo, 0.05), hi - 0.25)
    return butter(int(order), [lo, hi], btype="bandpass", fs=fps, output="sos")


class StreamingSOS:
    """Per-element IIR (second-order sections, direct form II transposed) with persistent state."""

    def __init__(self, sos: np.ndarray):
        self.sos = [tuple(float(c) for c in row) for row in sos]
        self.zi = sosfilt_zi(sos)
        self.z = None

    def reset_steady(self, x0: np.ndarray):
        self.z = [[(self.zi[s, 0] * x0).astype(np.float32), (self.zi[s, 1] * x0).astype(np.float32)]
                  for s in range(len(self.sos))]

    def reset_zero(self, like: np.ndarray):
        self.z = [[np.zeros_like(like), np.zeros_like(like)] for _ in self.sos]

    def apply_rows(self, x: np.ndarray, sl: slice) -> np.ndarray:
        """Filter a row band of the signal, updating only that band of the state (thread-safe per band)."""
        y = x
        for s, (b0, b1, b2, _a0, a1, a2) in enumerate(self.sos):
            z0 = self.z[s][0][sl]
            z1 = self.z[s][1][sl]
            out = b0 * y + z0
            n0 = b1 * y - a1 * out + z1
            z1[...] = b2 * y - a2 * out
            z0[...] = n0
            y = out
        return y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.z is None:
            self.z = [[np.zeros_like(x), np.zeros_like(x)] for _ in self.sos]
        y = x
        for s, (b0, b1, b2, _a0, a1, a2) in enumerate(self.sos):
            z0, z1 = self.z[s]
            out = b0 * y + z0
            self.z[s][0] = b1 * y - a1 * out + z1
            self.z[s][1] = b2 * y - a2 * out
            y = out
        return y


class RieszMagnifier:
    def __init__(self, fps, lo, hi, order, alpha, levels, sigma, phase_clip=np.pi / 2, motion_limit_px=0.0,
                 fine_gain=1.0):
        self.fine = float(fine_gain)  # amplification multiplier for the finest level (texture / noise)
        self.sos = bandpass_sos(lo, hi, fps, order)
        self.alpha = float(alpha)
        self.levels = int(levels)
        self.sigma = float(sigma)
        self.phase_clip = float(phase_clip)
        self.motion_limit = float(motion_limit_px)
        self.prev = None
        self.phase_c = None
        self.phase_s = None
        self.fc = self.fs = None
        self.n_threads = max(1, min(8, (os.cpu_count() or 4) // 2))
        self.pool = ThreadPoolExecutor(max_workers=self.n_threads)

    @staticmethod
    def _riesz(L):
        return (cv2.filter2D(L, -1, KX, borderType=cv2.BORDER_REFLECT),
                cv2.filter2D(L, -1, KY, borderType=cv2.BORDER_REFLECT))

    def _bands(self, h: int, w: int):
        n = 1 if h * w < 60000 else min(self.n_threads, max(1, h // 16))
        edges = np.linspace(0, h, n + 1).astype(int)
        return [slice(edges[k], edges[k + 1]) for k in range(n)]

    def _run(self, fn, bands):
        if len(bands) == 1:
            fn(bands[0])
        else:
            list(self.pool.map(fn, bands))

    def process(self, Y: np.ndarray) -> np.ndarray:
        lap, resid = build_laplacian(Y, self.levels)
        cur = [(L,) + self._riesz(L) for L in lap]
        if self.prev is None:
            self.prev = cur
            self.phase_c = [np.zeros_like(L) for L in lap]
            self.phase_s = [np.zeros_like(L) for L in lap]
            self.fc = [StreamingSOS(self.sos) for _ in lap]
            self.fs = [StreamingSOS(self.sos) for _ in lap]
            for i, L in enumerate(lap):
                self.fc[i].reset_zero(L)
                self.fs[i].reset_zero(L)
            return Y.copy()
        out_levels = []
        for i, (L, R1, R2) in enumerate(cur):
            pL, pR1, pR2 = self.prev[i]
            h, w = L.shape
            bands = self._bands(h, w)
            fc = np.empty_like(L)
            fs = np.empty_like(L)
            amp = np.empty_like(L)
            pc, ps = self.phase_c[i], self.phase_s[i]
            flt_c, flt_s = self.fc[i], self.fs[i]

            # 1) quaternion phase difference, phase accumulation, temporal band-pass (row bands in parallel)
            def part1(sl):
                l, r1, r2, a_, b_, c_ = L[sl], R1[sl], R2[sl], pL[sl], pR1[sl], pR2[sl]
                qr = l * a_ + r1 * b_ + r2 * c_
                qx = a_ * r1 - l * b_
                qy = a_ * r2 - l * c_
                qxy = np.sqrt(qx * qx + qy * qy) + EPS
                dphi = np.arctan2(qxy, qr)
                pc[sl] += dphi * (qx / qxy)
                ps[sl] += dphi * (qy / qxy)
                fc[sl] = flt_c.apply_rows(pc[sl], sl)
                fs[sl] = flt_s.apply_rows(ps[sl], sl)
                amp[sl] = np.sqrt(np.sqrt(qr * qr + qxy * qxy)) + EPS
            self._run(part1, bands)

            # 2) amplitude-weighted spatial smoothing of the filtered phase
            if self.sigma > 0:
                wsum = cv2.GaussianBlur(amp, (0, 0), self.sigma)
                fc = cv2.GaussianBlur(fc * amp, (0, 0), self.sigma) / wsum
                fs = cv2.GaussianBlur(fs * amp, (0, 0), self.sigma) / wsum

            # 3) amplify phase (with large-motion suppression + clipping) and shift the coefficients
            out = np.empty_like(L)
            a0 = self.alpha * (self.fine if i == 0 else 1.0)
            dscale = (2.0 ** i) / (np.pi / 2)  # phase -> px (~pi/2 rad/px at level 0)

            def part2(sl):
                c, s = fc[sl], fs[sl]
                a = a0
                if self.motion_limit > 0:
                    # in-band motion larger than the limit (voluntary movement leaking into the band) is not magnified
                    disp = np.sqrt(c * c + s * s) * (dscale / self.motion_limit)
                    d2 = disp * disp
                    a = a0 / (1.0 + d2 * d2)
                mc = a * c
                ms = a * s
                mag = np.sqrt(mc * mc + ms * ms) + EPS
                if self.phase_clip > 0:
                    k = np.minimum(1.0, self.phase_clip / mag)
                    mc *= k
                    ms *= k
                    mag = np.minimum(mag, self.phase_clip)
                sinc = np.sin(mag) / mag
                out[sl] = np.cos(mag) * L[sl] - (mc * sinc) * R1[sl] - (ms * sinc) * R2[sl]
            self._run(part2, bands)
            out_levels.append(out)
        self.prev = cur
        return collapse_laplacian(out_levels, resid)


class LinearMagnifier:
    def __init__(self, fps, lo, hi, order, alpha, levels, lambda_c, exaggeration=2.0):
        self.sos = bandpass_sos(lo, hi, fps, order)
        self.alpha = float(alpha)
        self.levels = int(levels)
        self.lambda_c = float(lambda_c)
        self.exaggeration = exaggeration
        self.filters = None

    def process(self, Y: np.ndarray) -> np.ndarray:
        lap, resid = build_laplacian(Y, self.levels)
        if self.filters is None:
            self.filters = []
            for L in lap:
                f = StreamingSOS(self.sos)
                f.reset_steady(L)
                self.filters.append(f)
        h, w = Y.shape
        lam = np.sqrt(h * h + w * w) / 3.0
        delta = self.lambda_c / 8.0 / (1.0 + self.alpha)
        out = []
        for i, L in enumerate(lap):
            filt = self.filters[i](L)
            cur_alpha = (lam / delta / 8.0 - 1.0) * self.exaggeration
            a = 0.0 if i == 0 else float(min(self.alpha, max(cur_alpha, 0.0)))
            out.append(L + a * filt)
            lam /= 2.0
        return collapse_laplacian(out, resid)


def make_magnifier(P: dict, fps: float, shape_hw, device: str = "cpu"):
    """Returns (magnifier, levels, device_used). device: cpu | gpu | auto (Riesz only runs on the GPU)."""
    levels = int(P["pyr_levels"]) or auto_levels(*shape_hw)
    if P["method"] == "linear":
        return (LinearMagnifier(fps, P["freq_lo"], P["freq_hi"], P["filter_order"], P["alpha"], levels,
                                P["lambda_c"]), levels, "cpu")
    if device in ("gpu", "auto"):
        try:
            from .magnify_torch import TorchRieszMagnifier, cuda_ok
            if cuda_ok():
                return (TorchRieszMagnifier(fps, P["freq_lo"], P["freq_hi"], P["filter_order"], P["alpha"], levels,
                                            P["phase_sigma"], motion_limit_px=P["motion_limit_px"], fine_gain=0.5),
                        levels, "gpu")
        except Exception as e:  # noqa: BLE001
            print(f"[magnify] GPU unavailable, using CPU: {e}", flush=True)
    return (RieszMagnifier(fps, P["freq_lo"], P["freq_hi"], P["filter_order"], P["alpha"], levels, P["phase_sigma"],
                           motion_limit_px=P["motion_limit_px"], fine_gain=0.5), levels, "cpu")
