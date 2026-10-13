"""CUDA (PyTorch) port of the streaming phase-based Riesz-pyramid magnifier in magnify.py.

Same algorithm and parameters as RieszMagnifier (Wadhwa et al. 2014): Laplacian pyramid, Riesz transform,
quaternion phase difference, streaming Butterworth band-pass of the accumulated phase, amplitude-weighted
spatial smoothing, large-motion suppression, phase clipping.  Pyramid filters reproduce OpenCV's
pyrDown / pyrUp (5-tap binomial kernel, reflect-101 borders), so CPU and GPU outputs agree closely.
Used only for the magnified (visual) video; tremor numbers always come from optical flow.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .magnify import bandpass_sos

EPS = 1e-6


def cuda_ok() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class _Ops:
    def __init__(self, device):
        self.dev = device
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device) / 16.0
        self.k_h = k.view(1, 1, 1, 5)
        self.k_v = k.view(1, 1, 5, 1)
        self.kx = torch.tensor([-0.5, 0.0, 0.5], device=device).view(1, 1, 1, 3)
        self.ky = self.kx.view(1, 1, 3, 1)
        self._gauss = {}

    @staticmethod
    def _pad(x, l, r, t, b):
        return F.pad(x, (l, r, t, b), mode="reflect")

    def blur5(self, x):
        x = F.conv2d(self._pad(x, 2, 2, 0, 0), self.k_h)
        return F.conv2d(self._pad(x, 0, 0, 2, 2), self.k_v)

    def pyr_down(self, x):
        h, w = x.shape[-2:]
        y = self.blur5(x)[..., ::2, ::2]
        return y[..., :(h + 1) // 2, :(w + 1) // 2]

    def pyr_up(self, x, size):
        h, w = size
        up = torch.zeros((1, 1, x.shape[-2] * 2, x.shape[-1] * 2), device=x.device, dtype=x.dtype)
        up[..., ::2, ::2] = x
        up = up[..., :h, :w]
        if up.shape[-2] < h or up.shape[-1] < w:
            up = F.pad(up, (0, w - up.shape[-1], 0, h - up.shape[-2]))
        return self.blur5(up) * 4.0

    def riesz(self, L):
        return (F.conv2d(self._pad(L, 1, 1, 0, 0), self.kx), F.conv2d(self._pad(L, 0, 0, 1, 1), self.ky))

    def gauss(self, x, sigma):
        if sigma not in self._gauss:
            r = max(1, int(math.ceil(3 * sigma)))
            t = torch.arange(-r, r + 1, device=self.dev, dtype=torch.float32)
            g = torch.exp(-0.5 * (t / sigma) ** 2)
            g = g / g.sum()
            self._gauss[sigma] = (g.view(1, 1, 1, -1), g.view(1, 1, -1, 1), r)
        gh, gv, r = self._gauss[sigma]
        rh = min(r, x.shape[-1] - 1)
        rv = min(r, x.shape[-2] - 1)
        if rh < r or rv < r:     # tiny level: fall back to a cropped kernel
            gh = gh[..., r - rh:r + rh + 1] / gh[..., r - rh:r + rh + 1].sum()
            gv = gv[..., r - rv:r + rv + 1, :] / gv[..., r - rv:r + rv + 1, :].sum()
        x = F.conv2d(self._pad(x, rh, rh, 0, 0), gh)
        return F.conv2d(self._pad(x, 0, 0, rv, rv), gv)


class _SOS:
    """Element-wise streaming IIR (DF-II transposed) on tensors."""

    def __init__(self, sos, like):
        self.c = [tuple(float(v) for v in row) for row in sos]
        self.z = [[torch.zeros_like(like), torch.zeros_like(like)] for _ in self.c]

    def __call__(self, x):
        y = x
        for s, (b0, b1, b2, _a0, a1, a2) in enumerate(self.c):
            z0, z1 = self.z[s]
            out = b0 * y + z0
            self.z[s][0] = b1 * y - a1 * out + z1
            self.z[s][1] = b2 * y - a2 * out
            y = out
        return y


class TorchRieszMagnifier:
    def __init__(self, fps, lo, hi, order, alpha, levels, sigma, phase_clip=np.pi / 2, motion_limit_px=0.0,
                 fine_gain=1.0, device="cuda"):
        self.dev = torch.device(device)
        self.ops = _Ops(self.dev)
        self.sos = bandpass_sos(lo, hi, fps, order)
        self.alpha = float(alpha)
        self.levels = int(levels)
        self.sigma = float(sigma)
        self.phase_clip = float(phase_clip)
        self.motion_limit = float(motion_limit_px)
        self.fine = float(fine_gain)
        self.prev = None

    def _laplacian(self, x):
        lap, cur = [], x
        for _ in range(self.levels):
            down = self.ops.pyr_down(cur)
            up = self.ops.pyr_up(down, cur.shape[-2:])
            lap.append(cur - up)
            cur = down
        return lap, cur

    def _collapse(self, lap, resid):
        cur = resid
        for L in reversed(lap):
            cur = self.ops.pyr_up(cur, L.shape[-2:]) + L
        return cur

    @torch.inference_mode()
    def process(self, Y: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)).to(self.dev).view(1, 1, *Y.shape)
        lap, resid = self._laplacian(x)
        cur = [(L,) + self.ops.riesz(L) for L in lap]
        if self.prev is None:
            self.prev = cur
            self.pc = [torch.zeros_like(L) for L in lap]
            self.ps = [torch.zeros_like(L) for L in lap]
            self.fc = [_SOS(self.sos, L) for L in lap]
            self.fs = [_SOS(self.sos, L) for L in lap]
            return Y.copy()
        out_levels = []
        for i, (L, R1, R2) in enumerate(cur):
            pL, pR1, pR2 = self.prev[i]
            qr = L * pL + R1 * pR1 + R2 * pR2
            qx = pL * R1 - L * pR1
            qy = pL * R2 - L * pR2
            qxy = torch.sqrt(qx * qx + qy * qy) + EPS
            dphi = torch.atan2(qxy, qr)
            self.pc[i] += dphi * (qx / qxy)
            self.ps[i] += dphi * (qy / qxy)
            fc = self.fc[i](self.pc[i])
            fs = self.fs[i](self.ps[i])
            amp = torch.sqrt(torch.sqrt(qr * qr + qxy * qxy)) + EPS
            if self.sigma > 0:
                wsum = self.ops.gauss(amp, self.sigma)
                fc = self.ops.gauss(fc * amp, self.sigma) / wsum
                fs = self.ops.gauss(fs * amp, self.sigma) / wsum
            a = self.alpha * (self.fine if i == 0 else 1.0)
            if self.motion_limit > 0:
                dscale = (2.0 ** i) / (np.pi / 2)
                disp = torch.sqrt(fc * fc + fs * fs) * (dscale / self.motion_limit)
                d2 = disp * disp
                a = a / (1.0 + d2 * d2)
            mc = a * fc
            ms = a * fs
            mag = torch.sqrt(mc * mc + ms * ms) + EPS
            if self.phase_clip > 0:
                k = torch.clamp(self.phase_clip / mag, max=1.0)
                mc = mc * k
                ms = ms * k
                mag = torch.clamp(mag, max=self.phase_clip)
            sinc = torch.sin(mag) / mag
            out_levels.append(torch.cos(mag) * L - (mc * sinc) * R1 - (ms * sinc) * R2)
        self.prev = cur
        return self._collapse(out_levels, resid)[0, 0].float().cpu().numpy()
