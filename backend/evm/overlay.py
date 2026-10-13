"""Rendering of the micro-tremor envelope overlay (heatmap + iso-contours, micro-vectors, halo, fingertip trails)."""
import cv2
import numpy as np

from .hands import ANCHORS, hand_mask, hand_size

CMAPS = {
    "turbo": cv2.COLORMAP_TURBO, "inferno": cv2.COLORMAP_INFERNO, "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS, "jet": cv2.COLORMAP_JET, "hot": cv2.COLORMAP_HOT,
}
# BGR colours for fingertip trails (thumb, index, middle, ring, pinky, dorsum)
FINGER_COLORS = [(107, 107, 255), (61, 217, 255), (119, 203, 107), (255, 150, 77), (255, 125, 199), (240, 240, 240)]


def lut(name: str) -> np.ndarray:
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), CMAPS.get(name, cv2.COLORMAP_TURBO))[:, 0, :]


def _text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    # outline via offset copies (a thicker stroke would also widen glyph advances and misalign the shadow)
    x, y = org
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1)):
        cv2.putText(img, s, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class OverlayRenderer:
    def __init__(self, O: dict, P: dict, size: tuple[int, int], grid_shape: tuple[int, int], fps: float,
                 scale_E: float, scale_D: float, trail_gain: float, info: dict):
        self.O, self.P = O, P
        self.W, self.H = size
        self.gh, self.gw = grid_shape
        self.fps = fps
        self.scale_E = max(scale_E, 1e-6)
        self.scale_D = max(scale_D, 1e-6)
        self.trail_gain = trail_gain * float(O["trail_scale"])
        self.lut = lut(O["colormap"])
        self.cmap = CMAPS.get(O["colormap"], cv2.COLORMAP_TURBO)
        self.info = info
        sp = int(O["vector_spacing"])
        xs = np.arange(sp // 2, self.W, sp)
        ys = np.arange(sp // 2, self.H, sp)
        self.gx, self.gy = [a.astype(np.float32) for a in np.meshgrid(xs, ys)]
        self.sx = self.gw / self.W
        self.sy = self.gh / self.H
        self.vec_len = sp * 1.1 * float(O["vector_scale"])

    # ------------------------------------------------------------------ helpers
    def _up(self, g: np.ndarray, interp=cv2.INTER_CUBIC) -> np.ndarray:
        return cv2.resize(g, (self.W, self.H), interpolation=interp)

    def _colors(self, e01: np.ndarray) -> np.ndarray:
        return cv2.applyColorMap(np.clip(e01 * 255, 0, 255).astype(np.uint8), self.cmap)

    # ------------------------------------------------------------------ main
    def render_layer(self, t: int, E_t: np.ndarray, D_t: np.ndarray, hands: np.ndarray, anchors: np.ndarray,
                     Db: np.ndarray):
        """Returns (colour uint8 HxWx3, alpha uint8 HxW) in stabilised coordinates."""
        O = self.O
        H, W = self.H, self.W
        col8 = np.zeros((H, W, 3), np.uint8)
        a8 = np.zeros((H, W), np.uint8)
        margin = float(self.P["mask_margin"])
        mask = hand_mask((H, W), hands, margin=margin)
        has_hand = mask.any()
        prim = hands[0]
        hs = hand_size(prim) if not np.isnan(prim[0, 0]) else 0.0

        if has_hand:
            # all per-pixel float work happens inside the hand bounding box (+ halo margin)
            bx, by, bw, bh = cv2.boundingRect(mask)
            pad = int(O["halo_width"]) + 8
            x0, y0 = max(0, bx - pad), max(0, by - pad)
            x1, y1 = min(W, bx + bw + pad), min(H, by + bh + pad)
            m = mask[y0:y1, x0:x1]
            rh, rw = m.shape
            e = np.clip(self._up(E_t.astype(np.float32))[y0:y1, x0:x1] * (1.0 / self.scale_E), 0, 1)
            col = np.zeros((rh, rw, 3), np.float32)
            alpha = np.zeros((rh, rw), np.float32)

            if O["show_halo"]:
                q = 4
                hq, wq = max(1, rh // q), max(1, rw // q)
                mq = cv2.resize(m, (wq, hq), interpolation=cv2.INTER_AREA)
                eq = cv2.resize(e, (wq, hq), interpolation=cv2.INTER_AREA) * (mq > 127)
                k = max(1, int(O["halo_width"]) // q)
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                spread = cv2.GaussianBlur(cv2.dilate(eq, ker), (0, 0), max(1.0, k / 3))
                dist = cv2.distanceTransform((mq < 128).astype(np.uint8), cv2.DIST_L2, 3) * q
                spread = cv2.resize(spread, (rw, rh), interpolation=cv2.INTER_LINEAR)
                dist = cv2.resize(dist, (rw, rh), interpolation=cv2.INTER_LINEAR)
                reach = float(O["halo_width"]) * (0.2 + 0.8 * spread)
                ha = np.clip(1.0 - dist / np.maximum(reach, 1.0), 0, 1)
                ha *= ha
                ha *= (dist > 0.5) * (0.25 + 0.75 * spread) * 0.95
                col = self._colors(spread).astype(np.float32)
                alpha = ha

            if O["show_heatmap"]:
                soft = cv2.GaussianBlur(m.astype(np.float32) * (1 / 255.0), (0, 0), max(1.5, 0.03 * hs))
                a = float(O["heatmap_opacity"]) * soft * (0.3 + 0.7 * e)
                c = self._colors(e).astype(np.float32)
                new_a = a + alpha * (1 - a)
                col = (c * a[..., None] + col * (alpha * (1 - a))[..., None]) / np.maximum(new_a, 1e-6)[..., None]
                alpha = new_a

            col8[y0:y1, x0:x1] = np.clip(col, 0, 255).astype(np.uint8)
            a8[y0:y1, x0:x1] = np.clip(alpha * 255, 0, 255).astype(np.uint8)

        if has_hand and O["show_heatmap"] and int(O["contour_levels"]) > 0:
            n = int(O["contour_levels"])
            es = cv2.GaussianBlur(e, (0, 0), max(3.0, 0.04 * hs))
            inside = m > 0
            for i in range(1, n + 1):
                lvl = i / (n + 1)
                mm = ((es >= lvl) & inside).astype(np.uint8)
                cnts, _ = cv2.findContours(mm, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(x0, y0))
                cnts = [c for c in cnts if cv2.contourArea(c) > 20]
                if not cnts:
                    continue
                c = tuple(int(v) for v in self.lut[int(lvl * 255)])
                thick = 2 if i == n else 1
                cv2.drawContours(col8, cnts, -1, (20, 20, 20), thick + 2, cv2.LINE_AA)
                cv2.drawContours(a8, cnts, -1, 200, thick + 2, cv2.LINE_AA)
                cv2.drawContours(col8, cnts, -1, c, thick, cv2.LINE_AA)
                cv2.drawContours(a8, cnts, -1, 255, thick, cv2.LINE_AA)

        if has_hand and O["show_vectors"]:
            inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            sel = inner[self.gy.astype(int), self.gx.astype(int)] > 0
            px, py = self.gx[sel], self.gy[sel]
            if len(px):
                mx = ((px + 0.5) * self.sx - 0.5).reshape(-1, 1).astype(np.float32)
                my = ((py + 0.5) * self.sy - 0.5).reshape(-1, 1).astype(np.float32)
                d = cv2.remap(D_t.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE).reshape(-1, 2)
                mag = np.linalg.norm(d, axis=1)
                cols = self.lut[np.clip(mag / self.scale_D * 255, 0, 255).astype(np.uint8)]
                # arrow length ∝ amplitude^0.6 (compressed so small tremor stays visible)
                lens = self.vec_len * np.power(mag / self.scale_D, 0.6)
                starts = np.stack([px, py], 1)
                ends = starts + d / np.maximum(mag, 1e-9)[:, None] * lens[:, None]
                s4 = np.round(starts * 4).astype(int).tolist()
                e4 = np.round(ends * 4).astype(int).tolist()
                tips = np.minimum(0.45, 6.0 / np.maximum(lens, 1e-6)).tolist()
                for p0, p1, c, L, tip in zip(s4, e4, cols.tolist(), lens.tolist(), tips):
                    p0, p1 = tuple(p0), tuple(p1)
                    if L < 1.5:
                        cv2.circle(col8, p0, 4, c, -1, cv2.LINE_AA, shift=2)
                        cv2.circle(a8, p0, 4, 200, -1, cv2.LINE_AA, shift=2)
                        continue
                    cc = tuple(c)
                    cv2.arrowedLine(col8, p0, p1, (15, 15, 15), 3, cv2.LINE_AA, shift=2, tipLength=tip)
                    cv2.arrowedLine(a8, p0, p1, 230, 3, cv2.LINE_AA, shift=2, tipLength=tip)
                    cv2.arrowedLine(col8, p0, p1, cc, 1, cv2.LINE_AA, shift=2, tipLength=tip)
                    cv2.arrowedLine(a8, p0, p1, 255, 1, cv2.LINE_AA, shift=2, tipLength=tip)

        if O["show_trails"] and np.isfinite(anchors).all():
            L = max(2, int(round(float(O["trail_seconds"]) * self.fps)))
            t0 = max(0, t - L + 1)
            for k in range(5):
                seg = anchors[k][None, :] + Db[t0:t + 1, k] * self.trail_gain
                n = len(seg)
                base = FINGER_COLORS[k]
                ptsi = np.round(seg * 4).astype(np.int32)
                for j in range(1, n):
                    f = j / max(n - 1, 1)
                    c = tuple(int(v * (0.35 + 0.65 * f)) for v in base)
                    th = 1 + int(round(2 * f))
                    cv2.line(col8, tuple(ptsi[j - 1]), tuple(ptsi[j]), (10, 10, 10), th + 2, cv2.LINE_AA, shift=2)
                    cv2.line(a8, tuple(ptsi[j - 1]), tuple(ptsi[j]), int(255 * (0.3 + 0.7 * f)), th + 2, cv2.LINE_AA, shift=2)
                    cv2.line(col8, tuple(ptsi[j - 1]), tuple(ptsi[j]), c, th, cv2.LINE_AA, shift=2)
                    cv2.line(a8, tuple(ptsi[j - 1]), tuple(ptsi[j]), 255, th, cv2.LINE_AA, shift=2)
                hx, hy = int(round(seg[-1, 0])), int(round(seg[-1, 1]))
                cv2.circle(col8, (hx, hy), 4, (10, 10, 10), -1, cv2.LINE_AA)
                cv2.circle(col8, (hx, hy), 3, base, -1, cv2.LINE_AA)
                cv2.circle(a8, (hx, hy), 4, 255, -1, cv2.LINE_AA)
                ax, ay = int(round(anchors[k][0])), int(round(anchors[k][1]))
                cv2.drawMarker(col8, (ax, ay), base, cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)
                cv2.drawMarker(a8, (ax, ay), 160, cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)

        return col8, a8

    def composite(self, base: np.ndarray, col8: np.ndarray, a8: np.ndarray, Winv: np.ndarray | None) -> np.ndarray:
        if Winv is not None:
            col8 = cv2.warpAffine(col8, Winv[:2], (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=0)
            a8 = cv2.warpAffine(a8, Winv[:2], (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=0)
        nz = cv2.findNonZero(a8)
        if nz is None:
            return base
        x, y, w, h = cv2.boundingRect(nz)
        out = base.copy()
        a = a8[y:y + h, x:x + w].astype(np.float32)[..., None] * (1.0 / 255.0)
        b = base[y:y + h, x:x + w].astype(np.float32)
        out[y:y + h, x:x + w] = (b + (col8[y:y + h, x:x + w].astype(np.float32) - b) * a).astype(np.uint8)
        return out

    def hud(self, img: np.ndarray, t: int, env_now: float):
        O, P, info = self.O, self.P, self.info
        H, W = img.shape[:2]
        s = H / 720.0
        fs = 0.5 * s
        th = max(1, int(round(s)))
        method = "Phase EVM" if P["method"] == "riesz" else "Linear EVM"
        line1 = f"{method}  x{P['alpha']:g}   band {P['freq_lo']:g}-{P['freq_hi']:g} Hz   t={info['start_s'] + t / self.fps:6.2f}s"
        pk = info.get("peak_hz")
        line2 = f"hand tremor RMS {env_now:5.2f}px   peak {pk:.1f} Hz" if pk == pk and pk is not None else ""
        box = img[int(8 * s):int(62 * s), int(8 * s):int(440 * s)]
        img[int(8 * s):int(62 * s), int(8 * s):int(440 * s)] = (box * 0.45).astype(np.uint8)
        _text(img, line1, (int(16 * s), int(30 * s)), fs, (255, 255, 255), th)
        if line2:
            _text(img, line2, (int(16 * s), int(52 * s)), fs, (200, 230, 255), th)

        # colour bar
        bh, bw = int(0.38 * H), int(12 * s)
        x0, y0 = W - bw - int(58 * s), int(20 * s)
        bar = self.lut[np.linspace(255, 0, bh).astype(np.uint8)][:, None, :].repeat(bw, axis=1)
        img[y0:y0 + bh, x0:x0 + bw] = bar
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + bw, y0 + bh), (230, 230, 230), 1)
        _text(img, f"{self.scale_E:.2f}px", (x0 - int(4 * s), y0 + bh + int(18 * s)), 0.42 * s, (255, 255, 255), th)
        _text(img, "0", (x0 + bw + int(4 * s), y0 + bh), 0.42 * s, (255, 255, 255), th)
        _text(img, "RMS", (x0 + bw + int(4 * s), y0 + int(12 * s)), 0.42 * s, (255, 255, 255), th)

        if O["show_trails"]:
            yy = y0 + bh + int(44 * s)
            for k in range(5):
                c = FINGER_COLORS[k]
                cv2.circle(img, (x0, yy + int(k * 18 * s)), int(5 * s), c, -1, cv2.LINE_AA)
                _text(img, ANCHORS[k], (x0 + int(10 * s), yy + int(k * 18 * s) + int(5 * s)), 0.42 * s, c, th)
