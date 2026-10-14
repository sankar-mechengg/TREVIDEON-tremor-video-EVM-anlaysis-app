"""Dominant-hand tracking with MediaPipe: 2D landmarks, metric 3D world landmarks and handedness."""
from __future__ import annotations

import cv2
import numpy as np

from ..isolate import DominantHandTracker


class HandTracker3D(DominantHandTracker):
    """Like DominantHandTracker, but also returns MediaPipe's 3D world landmarks (metres, origin near the hand
    centre, camera-aligned axes), the handedness label of the followed hand and a position vote.

    MediaPipe's handedness classifier is unreliable for the back of the hand seen from a head-mounted camera,
    so a requested side is chosen by image position when both hands are visible (`rightmost_is_right`: in a
    non-mirrored egocentric view the right hand is the right-most one; in a third-person view facing the
    person it is the left-most one), and every frame with two hands votes on the side of the followed hand."""

    def __init__(self, prefer: str = "auto", mirrored: bool = False, confidence: float = 0.4,
                 rightmost_is_right: bool = True):
        super().__init__("auto", mirrored, confidence)
        self.prefer = prefer
        self.rmr = rightmost_is_right
        self.votes: list = []
        self.arm_votes: list = []

    def detect3d(self, bgr: np.ndarray, t_ms: int):
        lm2, w3, lab, other = self._detect3d(bgr, t_ms)
        if np.isfinite(lm2).all() and np.isfinite(other).all():
            is_rightmost = float(np.mean(lm2[:, 0])) > float(np.mean(other[:, 0]))
            self.votes.append(is_rightmost == self.rmr)          # True = physical right hand
        if np.isfinite(lm2).all():
            # forearm direction (knuckles -> wrist): the right forearm runs towards the right of an egocentric image
            d = lm2[0] - lm2[9]
            if abs(float(d[0])) > 0.25 * float(np.linalg.norm(d)):
                self.arm_votes.append((float(d[0]) > 0) == self.rmr)
        return lm2, w3, lab, other

    def _detect3d(self, bgr: np.ndarray, t_ms: int):
        h, w = bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(img, int(t_ms))
        nan2 = np.full((21, 2), np.nan, np.float32)
        nan3 = np.full((21, 3), np.nan, np.float32)
        hands = []
        for i, (hl, hd) in enumerate(zip(res.hand_landmarks, res.handedness)):
            wl = res.hand_world_landmarks[i] if i < len(res.hand_world_landmarks) else None
            hands.append((np.array([[p.x * w, p.y * h] for p in hl], np.float32),
                          np.array([[p.x, p.y, p.z] for p in wl], np.float32) if wl else nan3,
                          hd[0].category_name, float(hd[0].score)))
        if not hands:
            return nan2, nan3, None, nan2
        cand = list(range(len(hands)))

        def area(k):
            return float(np.ptp(hands[k][0][:, 0]) * np.ptp(hands[k][0][:, 1]))

        if self.prefer in ("right", "left") and len(hands) >= 2:
            # requested side by image position (two hands visible)
            want_rightmost = (self.prefer == "right") == self.rmr
            xs = [float(np.mean(hh[0][:, 0])) for hh in hands]
            i = int(np.argmax(xs)) if want_rightmost else int(np.argmin(xs))
        elif self.prev is not None:
            d = [float(np.linalg.norm(hands[i][0][0] - self.prev[0])) for i in cand]
            i = cand[int(np.argmin(d))]
            if min(d) > 0.3 * max(w, h):
                i = max(cand, key=area)
        else:
            i = max(cand, key=area)
        self.prev = hands[i][0]
        other = next((hands[j][0] for j in range(len(hands)) if j != i), nan2)
        return hands[i][0], hands[i][1], hands[i][2], other


class MPCropLandmarker:
    """MediaPipe in IMAGE mode on a hand crop: world landmarks of the largest hand (noise-floor re-runs)."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from ..hands import MODEL_PATH
        self._mp = mp
        opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                                            running_mode=vision.RunningMode.IMAGE, num_hands=1,
                                            min_hand_detection_confidence=0.3)
        self.lm = vision.HandLandmarker.create_from_options(opts)

    def world(self, crop_bgr: np.ndarray) -> np.ndarray:
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
        res = self.lm.detect(img)
        if not res.hand_world_landmarks:
            return np.full((21, 3), np.nan, np.float32)
        return np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]], np.float32)

    def close(self):
        self.lm.close()


def appears_right(labels: list) -> bool | None:
    """Whether the followed hand LOOKS like a right hand in the image (what WiLoR needs). MediaPipe labels
    handedness for mirrored (selfie) images, so a hand labelled 'Left' looks like a right hand."""
    votes = [lab for lab in labels if lab]
    if not votes:
        return None
    return sum(lab == "Left" for lab in votes) >= 0.5 * len(votes)


def handedness(position_votes: list, labels: list, prefer: str, mirrored: bool, min_votes: int = 10,
               arm_votes: list | None = None):
    """-> (appears_right, physical_right, how). Order of evidence: a requested side, the image-position votes
    of frames with both hands visible, the forearm direction (single hand), MediaPipe's handedness label."""
    arm_votes = arm_votes or []
    if prefer in ("right", "left"):
        phys = prefer == "right"
        how = "requested"
    elif len(position_votes) >= min_votes:
        phys = sum(position_votes) >= 0.5 * len(position_votes)
        how = f"image position of the two hands ({len(position_votes)} frames)"
    elif len(arm_votes) >= min_votes and abs(sum(arm_votes) / len(arm_votes) - 0.5) > 0.2:
        phys = sum(arm_votes) >= 0.5 * len(arm_votes)
        how = f"forearm direction ({len(arm_votes)} frames)"
    else:
        a = appears_right(labels)
        if a is None:
            a = True
        return a, (a if not mirrored else not a), "MediaPipe handedness label"
    return (phys if not mirrored else not phys), phys, how


def mediapipe_to_camera(lm2d: np.ndarray, world: np.ndarray, K: np.ndarray):
    """Place MediaPipe world landmarks in the pinhole camera: px-per-metre from the least-squares fit of the
    (camera-aligned) world x/y to the image landmarks, depth = f / scale, centre on the viewing ray."""
    if not (np.isfinite(lm2d).all() and np.isfinite(world).all()):
        return None, np.nan
    q = world[:, :2] - world[:, :2].mean(0)
    p = lm2d - lm2d.mean(0)
    s = float(np.sum(p * q) / max(np.sum(q * q), 1e-12))
    if s <= 0:
        return None, np.nan
    f = float(K[0, 0])
    z = f / s
    uc = lm2d.mean(0)
    ray = np.array([(uc[0] - K[0, 2]) / f, (uc[1] - K[1, 2]) / K[1, 1], 1.0])
    from .wilor import rot_between
    R_ray = rot_between(np.array([0, 0, 1.0]), ray)
    rel = world.astype(np.float64) - world.mean(0)
    return rel @ R_ray.T + ray * z, s
