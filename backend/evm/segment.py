"""Dominant-hand segmentation for the hand-isolation mode.

Three interchangeable segmenters return a soft mask (float32, 0..1, frame size) of ONE hand, given that
hand's 21 MediaPipe landmarks (pixel coordinates) and, optionally, the other hand's landmarks:

  * Sam2Segmenter       SAM 2.1 image predictor prompted with landmark points + box (GPU recommended)
  * MediaPipeSegmenter  MediaPipe multiclass selfie segmenter, "body-skin" class, restricted to a region
                        around the landmarks (CPU)
  * LandmarkSegmenter   rasterised landmark polygon (no model; fallback)

Every mask is restricted to a region of interest grown from the landmarks (hand + a short wrist / forearm
stub) so that the other hand, the face or the background can never leak in.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .hands import CHAINS, PALM, TIPS, DIPS, hand_mask, hand_size

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MP_SEG_MODEL = os.path.join(MODELS, "selfie_multiclass_256x256.tflite")
SAM2_CKPT = os.path.join(MODELS, "sam2.1_hiera_small.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
MP_SKIN = 2           # selfie_multiclass categories: 0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 other


def roi_mask(shape_hw, lm: np.ndarray, grow: float = 0.25, stub_len: float = 0.45, stub_w: float = 0.75) -> np.ndarray:
    """Region around one hand: landmark mask grown by `grow` x hand size, plus a short wrist / forearm stub."""
    m = hand_mask(shape_hw, lm[None], margin=1.4)
    s = max(hand_size(lm), 4.0)
    k = max(3, int(round(grow * s)) | 1)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    wrist_end = lm[0] + (lm[0] - lm[9]) * stub_len
    cv2.line(m, tuple(int(v) for v in lm[0]), tuple(int(v) for v in wrist_end), 255, max(2, int(stub_w * s)))
    return m


def _largest_component_touching(mask: np.ndarray, seed_xy) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 2:
        return mask
    x, y = int(np.clip(seed_xy[0], 0, mask.shape[1] - 1)), int(np.clip(seed_xy[1], 0, mask.shape[0] - 1))
    keep = lab[y, x]
    if keep == 0:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == keep, 255, 0).astype(np.uint8)


def _clean(binary: np.ndarray, lm: np.ndarray) -> np.ndarray:
    s = max(hand_size(lm), 4.0)
    k = max(3, int(round(0.06 * s)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
    palm_c = lm[PALM].mean(axis=0)
    m = _largest_component_touching(m, palm_c)
    # fill holes: pad with background first, so that regions open to the image border are never "holes"
    h, w = m.shape
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = cv2.bitwise_not(pad)
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 0)
    return cv2.bitwise_or(m, ff[1:-1, 1:-1])


class LandmarkSegmenter:
    name = "landmarks"

    def __call__(self, bgr, lm, other=None, t_ms=0):
        m = hand_mask(bgr.shape[:2], lm[None], margin=1.15)
        return m.astype(np.float32) / 255.0

    def close(self):
        pass


class MediaPipeSegmenter:
    name = "mediapipe"

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        opts = vision.ImageSegmenterOptions(base_options=BaseOptions(model_asset_path=MP_SEG_MODEL),
                                            running_mode=vision.RunningMode.VIDEO,
                                            output_category_mask=True, output_confidence_masks=False)
        self.seg = vision.ImageSegmenter.create_from_options(opts)
        self.fallback = LandmarkSegmenter()
        self.n_fallback = 0
        self.last_ts = -1

    def __call__(self, bgr, lm, other=None, t_ms=0):
        h, w = bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ts = max(int(t_ms), self.last_ts + 1)          # VIDEO mode needs strictly increasing timestamps
        self.last_ts = ts
        cat = self.seg.segment_for_video(img, ts).category_mask.numpy_view()
        if cat.shape != (h, w):
            cat = cv2.resize(cat, (w, h), interpolation=cv2.INTER_NEAREST)
        roi = roi_mask((h, w), lm)
        if other is not None and np.isfinite(other).all():
            roi[hand_mask((h, w), other[None], margin=1.3) > 0] = 0
        skin = np.where((cat == MP_SKIN) & (roi > 0), 255, 0).astype(np.uint8)
        ref = hand_mask((h, w), lm[None], margin=1.0)
        if skin.sum() < 0.35 * ref.sum():      # segmenter missed the hand -> landmark polygon
            self.n_fallback += 1
            return self.fallback(bgr, lm)
        # skin pixels refine the outline; a slightly thinner landmark polygon guarantees that shadowed
        # fingers (often the thumb) are not lost
        core = hand_mask((h, w), lm[None], margin=0.85)
        return _clean(cv2.bitwise_or(skin, core), lm).astype(np.float32) / 255.0

    def close(self):
        self.seg.close()


class Sam2Segmenter:
    name = "sam2"

    def __init__(self, device: str = "cuda"):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        self.device = device
        model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
        self.pred = SAM2ImagePredictor(model)
        self.n_fallback = 0

    def _prompts(self, lm, other, shape_hw):
        h, w = shape_hw
        tips = lm[TIPS] * 0.7 + lm[DIPS] * 0.3
        palm_c = lm[PALM].mean(axis=0)
        mids = np.array([lm[c[1]] * 0.5 + lm[c[2]] * 0.5 for c in CHAINS])
        pos = np.vstack([palm_c[None], lm[[0, 5, 9, 13, 17]], mids, tips])
        neg = [lm[0] + (lm[0] - lm[9]) * 1.1]                   # far forearm: stop the mask at the wrist
        if other is not None and np.isfinite(other).all():
            neg.append(other[PALM].mean(axis=0))
        pts = np.vstack([pos] + ([np.array(neg)] if neg else []))
        labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)
        s = hand_size(lm)
        stub = lm[0] + (lm[0] - lm[9]) * 0.5
        allp = np.vstack([lm, stub[None]])
        x0, y0 = allp.min(axis=0) - 0.18 * s
        x1, y1 = allp.max(axis=0) + 0.18 * s
        box = np.array([max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)], np.float32)
        return pts.astype(np.float32), labels, box

    def __call__(self, bgr, lm, other=None, t_ms=0):
        torch = self.torch
        h, w = bgr.shape[:2]
        pts, labels, box = self._prompts(lm, other, (h, w))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            self.pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            masks, scores, _ = self.pred.predict(point_coords=pts, point_labels=labels, box=box[None],
                                                 multimask_output=True)
        roi = roi_mask((h, w), lm) > 0
        ref_area = float((hand_mask((h, w), lm[None], margin=1.0) > 0).sum())
        # plausibility is judged on the finger points only (mid-phalanx + fingertip pads): palm, wrist and
        # knuckle prompts are often hidden under a wrist-worn device or strap that SAM rightly excludes
        finger_pts = pts[6:16]
        best, best_key = None, None
        for m, sc in zip(masks, scores):
            mb = (m > 0.5) & roi
            area = float(mb.sum()) / max(ref_area, 1.0)
            xi = np.clip(finger_pts[:, 0].astype(int), 0, w - 1)
            yi = np.clip(finger_pts[:, 1].astype(int), 0, h - 1)
            cover = float(mb[yi, xi].mean())
            plausible = cover >= 0.9 and 0.4 <= area <= 2.2
            key = (plausible, cover if not plausible else float(sc))
            if best_key is None or key > best_key:
                best, best_key = mb, key
        if not best_key[0]:
            self.n_fallback += 1
        m = best.astype(np.uint8) * 255
        # SAM can split a finger off at an occluding strap / glove edge: a thin landmark polygon guarantees
        # every finger is present, SAM provides the outline everywhere else
        m = cv2.bitwise_or(m, hand_mask((h, w), lm[None], margin=0.85))
        return _clean(m, lm).astype(np.float32) / 255.0

    def close(self):
        try:
            del self.pred
            self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def available() -> dict:
    out = {"landmarks": True, "mediapipe": os.path.exists(MP_SEG_MODEL), "sam2": False, "cuda": False}
    try:
        import torch
        out["cuda"] = bool(torch.cuda.is_available())
        import sam2  # noqa: F401
        out["sam2"] = out["cuda"] and os.path.exists(SAM2_CKPT)
    except Exception:  # noqa: BLE001
        pass
    return out


def make_segmenter(kind: str):
    """kind: auto | sam2 | mediapipe | landmarks.  Falls back gracefully; returns (segmenter, used_name)."""
    av = available()
    order = {"auto": ["sam2", "mediapipe", "landmarks"], "sam2": ["sam2", "mediapipe", "landmarks"],
             "mediapipe": ["mediapipe", "landmarks"], "landmarks": ["landmarks"]}.get(kind, ["landmarks"])
    for k in order:
        if not av.get(k):
            continue
        try:
            if k == "sam2":
                return Sam2Segmenter("cuda"), k
            if k == "mediapipe":
                return MediaPipeSegmenter(), k
            return LandmarkSegmenter(), k
        except Exception as e:  # noqa: BLE001
            print(f"[segment] {k} unavailable: {e}", flush=True)
    return LandmarkSegmenter(), "landmarks"
