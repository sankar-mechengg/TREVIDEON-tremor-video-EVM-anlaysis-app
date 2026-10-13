"""Video reading (segment + resize) and browser-friendly H.264 writing via ffmpeg."""
import os
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def probe(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": float(fps), "frames": n, "width": w, "height": h, "duration": n / fps if fps else 0.0}


def work_size(w: int, h: int, target_h: int) -> tuple[int, int]:
    target_h = min(int(target_h), h)
    scale = target_h / h
    return int(round(w * scale / 2)) * 2, int(round(target_h / 2)) * 2


def _midpoint_frame(prev: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Motion-interpolated frame halfway between prev and nxt (bidirectional DIS flow, half-step warps)."""
    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    f01 = dis.calc(g0, g1, None)
    f10 = dis.calc(g1, g0, None)
    h, w = g0.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    a = cv2.remap(prev, gx - 0.5 * f01[..., 0], gy - 0.5 * f01[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    b = cv2.remap(nxt, gx - 0.5 * f10[..., 0], gy - 0.5 * f10[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(a, 0.5, b, 0.5, 0)


class DuplicateDetector:
    """Flags frames that repeat the previous frame (screen / VR recordings, variable-frame-rate phones).

    A repeated frame makes the motion stall for one frame and jump double on the next; at a regular rate
    this produces a comb of spectral lines (e.g. 2.5, 5, 10, 12.5 Hz for one repeat every 12 frames at
    30 fps) that can be mistaken for tremor."""

    def __init__(self, rel: float = 0.15, floor: float = 0.02, drop_ratio: float = 1.75):
        self.rel, self.floor, self.drop_ratio = rel, floor, drop_ratio
        self.hist: list[float] = []
        self.prev_small = None
        self.prev_dup = False
        self.last_jump = False      # frame moved ~2x the typical amount without a repeat before it (dropped frame)

    def __call__(self, frame: np.ndarray) -> bool:
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32)
        if self.prev_small is None:
            self.prev_small = small
            return False
        d = float(np.mean(np.abs(small - self.prev_small)))
        self.prev_small = small
        ref = float(np.median(self.hist)) if len(self.hist) >= 5 else None
        dup = d <= self.floor or (ref is not None and d < self.rel * ref)
        self.last_jump = bool(ref is not None and not dup and not self.prev_dup and d > self.drop_ratio * ref)
        self.prev_dup = dup
        if not dup:
            self.hist = (self.hist + [d])[-60:]
        return dup


class SegmentReader:
    """Yields resized BGR frames of [start_s, start_s + duration_s).

    repair=True replaces exact / near-exact repeated frames by a motion-interpolated midpoint of their
    neighbours (see DuplicateDetector); the indices of repaired frames are kept in `self.repaired`."""

    def __init__(self, path: str, start_s: float, duration_s: float, target_h: int, repair: bool = False):
        self.path = path
        info = probe(path)
        self.info = info
        self.fps = info["fps"]
        total = max(info["frames"], 1)
        start = int(round(max(0.0, start_s) * self.fps))
        start = min(start, max(0, total - int(2 * self.fps)))
        avail = total - start
        n = avail if duration_s <= 0 else min(avail, int(round(duration_s * self.fps)))
        self.start_frame = start
        self.n = max(n, 1)
        self.size = work_size(info["width"], info["height"], target_h or info["height"])
        self.repair = repair
        self.repaired: list[int] = []
        self.suspected_drops: list[int] = []

    @property
    def start_s(self) -> float:
        return self.start_frame / self.fps

    def _raw(self, n):
        cap = cv2.VideoCapture(self.path)
        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        i = 0
        try:
            while i < n:
                ok, f = cap.read()
                if not ok:
                    break
                if (f.shape[1], f.shape[0]) != self.size:
                    f = cv2.resize(f, self.size, interpolation=cv2.INTER_AREA)
                yield f
                i += 1
        finally:
            cap.release()

    def frames(self, limit: int | None = None):
        n = self.n if limit is None else min(limit, self.n)
        if not self.repair:
            yield from self._raw(n)
            return
        self.repaired = []
        self.suspected_drops = []
        det = DuplicateDetector()
        prev = pend = None          # pend: a duplicate frame waiting for its successor
        for i, f in enumerate(self._raw(n)):
            dup = det(f)
            if det.last_jump:
                self.suspected_drops.append(i)
            if pend is not None:
                if dup:             # two repeats in a row: keep them (rare; nothing reliable to interpolate)
                    yield pend
                    pend = None
                else:
                    yield _midpoint_frame(prev, f)
                    self.repaired.append(i - 1)
                    pend = None
            if dup and prev is not None and pend is None and i < n - 1:
                pend = f
                continue
            yield f
            prev = f
        if pend is not None:
            yield pend


class VideoWriter:
    """Pipes raw BGR frames into ffmpeg/libx264 (yuv420p, faststart) so browsers can play the result."""

    def __init__(self, path: str, size: tuple[int, int], fps: float, crf: int = 18, preset: str = "veryfast"):
        self.path = path
        self.tmp = path[:-4] + ".part.mp4"
        self.log_path = path + ".log"
        w, h = size
        cmd = [FFMPEG, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
               "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.tmp]
        self._log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self._log, creationflags=_NO_WINDOW)

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        rc = self.proc.wait()
        self._log.close()
        if rc != 0:
            with open(self.log_path, "r", errors="replace") as fh:
                msg = fh.read()[-2000:]
            raise RuntimeError(f"ffmpeg failed ({rc}): {msg}")
        os.replace(self.tmp, self.path)
        try:
            os.remove(self.log_path)
        except OSError:
            pass

    def abort(self):
        try:
            self.proc.kill()
            self.proc.wait()
        except Exception:
            pass
        self._log.close()
        for p in (self.tmp, self.log_path):
            try:
                os.remove(p)
            except OSError:
                pass


def make_thumbnail(path: str, out_path: str, width: int = 360):
    info = probe(path)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(info["fps"] * 5), max(info["frames"] - 1, 0)))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise IOError("Could not read frame for thumbnail")
    h = int(round(f.shape[0] * width / f.shape[1]))
    f = cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, f, [cv2.IMWRITE_JPEG_QUALITY, 82])
