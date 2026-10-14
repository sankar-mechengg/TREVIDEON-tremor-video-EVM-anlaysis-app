"""WiLoR hand-mesh model (Potamias et al., CVPR 2025) on hand crops.

* The WiLoR code (CC BY-NC-ND 4.0) is downloaded unchanged into backend/third_party/WiLoR and only imported;
  its rendering helpers (pyrender) and its YOLO hand detector (AGPL) are not used - TREVIDEON supplies the
  hand box from the MediaPipe landmarks.
* The MANO hand model (research licence, mano.is.tue.mpg.de) must be downloaded by the user; it is looked for
  in $TREVIDEON_MANO_DIR, backend/models/mano, and <Video-EVM>/data/mano_data[/models].
* WiLoR predicts a right hand; left hands are mirrored into the crop and mirrored back (as in the WiLoR demo).
* WiLoR's camera has a 5000 px focal length per 256 px crop (near-orthographic). `to_camera` re-places the
  predicted hand in a real pinhole camera: depth from the apparent hand size, orientation rotated onto the
  viewing ray through the hand (perspective correction), so that hands away from the image centre keep the
  orientation they appear to have.
"""
from __future__ import annotations

import os
import sys
import types

import cv2
import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WILOR_DIR = os.environ.get("TREVIDEON_WILOR_DIR") or os.path.join(BACKEND, "third_party", "WiLoR")
CKPT = os.environ.get("TREVIDEON_WILOR_CKPT") or os.path.join(BACKEND, "models", "wilor_final.ckpt")
IMG = 256
MEAN = 255.0 * np.array([0.485, 0.456, 0.406], np.float32)
STD = 255.0 * np.array([0.229, 0.224, 0.225], np.float32)
# MANO joint index of the local rotation that moves each finger's proximal phalanx (hand_pose[k-1])
MANO_PROX = {"thumb": (13, 14), "index": (1,), "middle": (4,), "ring": (10,), "pinky": (7,)}
MANO_PROX_CHILD = {"thumb": 15, "index": 2, "middle": 5, "ring": 11, "pinky": 8}
MANO_PROX_JOINT = {"thumb": 14, "index": 1, "middle": 4, "ring": 10, "pinky": 7}


def mano_dir() -> str | None:
    cands = [os.environ.get("TREVIDEON_MANO_DIR"), os.path.join(BACKEND, "models", "mano"),
             os.path.normpath(os.path.join(BACKEND, "..", "..", "data", "mano_data", "models")),
             os.path.normpath(os.path.join(BACKEND, "..", "..", "data", "mano_data"))]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "MANO_RIGHT.pkl")):
            return c
    return None


def available() -> dict:
    code = os.path.isfile(os.path.join(WILOR_DIR, "wilor", "models", "wilor.py"))
    out = {"code": code, "checkpoint": os.path.isfile(CKPT), "mano": mano_dir() is not None}
    out["ready"] = all(out.values())
    return out


class _ChumpyStub:
    """Stand-in for chumpy arrays inside the original MANO .pkl files (chumpy does not install on Python 3.12)."""

    def __setstate__(self, st):
        if isinstance(st, dict):
            self.__dict__.update(st)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.r, dtype=dtype)

    @property
    def r(self):
        d = self.__dict__
        if "x" in d:                                   # chumpy.ch.Ch: the array itself
            return np.asarray(d["x"])
        if "a" in d and "idxs" in d:                   # chumpy.reordering.Select: a.ravel()[idxs] reshaped
            v = np.asarray(d["a"]).ravel()[np.asarray(d["idxs"])]
            shp = d.get("preferred_shape")
            return v.reshape(shp) if shp is not None else v
        raise ValueError("Unsupported chumpy object in the MANO file")


def _load_mano_pkl(fh, encoding="latin1"):
    """pickle.load for the MANO .pkl that maps chumpy objects to plain numpy arrays."""
    import pickle

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("chumpy"):
                return _ChumpyStub
            return super().find_class(module, name)

    data = U(fh, encoding=encoding).load()
    if isinstance(data, dict):
        data = {k: (np.asarray(v) if isinstance(v, _ChumpyStub) else v) for k, v in data.items()}
    return data


def _with_chumpy_stub(fn):
    """Run fn while smplx reads MANO .pkl files with _load_mano_pkl (chumpy is not needed)."""
    import smplx.body_models as bm
    orig = bm.pickle
    shim = types.SimpleNamespace(load=_load_mano_pkl, dump=orig.dump, loads=orig.loads, dumps=orig.dumps)
    bm.pickle = shim
    try:
        return fn()
    finally:
        bm.pickle = orig


def _import_wilor():
    if WILOR_DIR not in sys.path:
        sys.path.insert(0, WILOR_DIR)
    if "wilor.utils" not in sys.modules:
        # import the WiLoR sub-modules without wilor/utils/__init__.py (it pulls in pyrender for rendering)
        utils = types.ModuleType("wilor.utils")
        utils.__path__ = [os.path.join(WILOR_DIR, "wilor", "utils")]
        utils.SkeletonRenderer = utils.MeshRenderer = None
        sys.modules["wilor.utils"] = utils
    from wilor.configs import get_config
    from wilor.models.wilor import WiLoR
    return get_config, WiLoR


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        ax = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1.0, 0])
        ax /= np.linalg.norm(ax)
        return 2 * np.outer(ax, ax) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def make_crop(img: np.ndarray, center, size: float, flip: bool) -> np.ndarray:
    """256x256 RGB crop of the square box (center, size), mirrored for left hands (WiLoR's ViTDetDataset)."""
    h, w = img.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    if flip:
        img = img[:, ::-1]
        cx = w - cx - 1
    s = IMG / float(size)
    r = int(0.75 * size) + 8
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
    x1, y1 = min(w, int(cx + r)), min(h, int(cy + r))
    roi = np.ascontiguousarray(img[y0:y1, x0:x1])
    ds = size / IMG / 2.0
    if ds > 1.1:                                   # anti-aliasing blur before downsampling (as WiLoR)
        roi = cv2.GaussianBlur(roi, (0, 0), (ds - 1) / 2)
    M = np.array([[s, 0, IMG / 2 - s * (cx - x0)], [0, s, IMG / 2 - s * (cy - y0)]], np.float32)
    patch = cv2.warpAffine(roi, M, (IMG, IMG), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return patch[:, :, ::-1]


class WiLoRModel:
    def __init__(self, device: str = "cuda", fp16: bool = True):
        import torch
        st = available()
        if not st["ready"]:
            missing = [k for k, v in st.items() if k != "ready" and not v]
            raise RuntimeError("WiLoR is not installed (missing: " + ", ".join(missing) + ")")
        self.torch = torch
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        get_config, WiLoR = _import_wilor()
        cfg = get_config(os.path.join(WILOR_DIR, "pretrained_models", "model_config.yaml"), update_cachedir=True)
        cfg.defrost()
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
        md = mano_dir()
        cfg.MANO.DATA_DIR = md
        cfg.MANO.MODEL_PATH = md
        cfg.MANO.MEAN_PARAMS = os.path.join(WILOR_DIR, "mano_data", "mano_mean_params.npz")
        cfg.freeze()
        self.cfg = cfg
        model = _with_chumpy_stub(lambda: WiLoR(cfg, init_renderer=False))
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k: v for k, v in sd.items() if not k.startswith("discriminator.")}
        model.load_state_dict(sd, strict=False)
        del sd
        model.eval().to(self.device)
        self.model = model
        self.focal = float(cfg.EXTRA.FOCAL_LENGTH)
        self.faces = np.asarray(model.mano.faces, np.int32)
        with torch.no_grad():          # MANO rest-pose joints (bone axes for finger twist)
            z = torch.zeros
            out = model.mano(global_orient=torch.eye(3)[None, None].to(self.device),
                             hand_pose=torch.eye(3)[None, None].repeat(1, 15, 1, 1).to(self.device),
                             betas=z(1, 10, device=self.device), pose2rot=False)
            self.rest_joints_mano = model.mano.J_regressor.to(self.device).matmul(out.vertices[0]).cpu().numpy()

    def predict(self, crops: list[np.ndarray]) -> dict:
        """crops: list of 256x256 RGB uint8 (already mirrored for left hands). Returns numpy arrays."""
        torch = self.torch
        x = np.stack(crops).astype(np.float32)
        x = (x - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                                    enabled=self.fp16):
            out = self.model.forward_step({"img": x}, train=False)
        mp = out["pred_mano_params"]
        f = lambda t: t.float().cpu().numpy()
        return {"cam": f(out["pred_cam"]), "joints": f(out["pred_keypoints_3d"]), "verts": f(out["pred_vertices"]),
                "global_orient": f(mp["global_orient"]).reshape(-1, 3, 3),
                "hand_pose": f(mp["hand_pose"]).reshape(-1, 15, 3, 3), "betas": f(mp["betas"]).reshape(-1, 10)}


def to_camera(pts: np.ndarray, cam: np.ndarray, center, size: float, right: bool, img_wh, K: np.ndarray,
              focal_crop: float = 5000.0, extra: np.ndarray | None = None):
    """Place WiLoR joints (MANO space, right-hand) in the real camera.

    Returns (joints_cam (N,3) metres, R_ray (3,3), px_per_m at the hand, 2D projection (N,2) under WiLoR's
    camera, extra points (e.g. mesh vertices) moved with the same transform or None)."""
    m = 1.0 if right else -1.0
    P = pts.astype(np.float64).copy()
    P[:, 0] *= m
    E = None
    if extra is not None:
        E = extra.astype(np.float64).copy()
        E[:, 0] *= m
    s, tx, ty = float(cam[0]), float(cam[1]) * m, float(cam[2])
    W, H = img_wh
    f_s = focal_crop / IMG * max(W, H)
    bs = size * s + 1e-9
    t_full = np.array([2 * (center[0] - W / 2) / bs + tx, 2 * (center[1] - H / 2) / bs + ty, 2 * f_s / bs])
    Pc = P + t_full
    uv = f_s * Pc[:, :2] / Pc[:, 2:3] + np.array([W / 2, H / 2])
    px_per_m = f_s / float(np.mean(Pc[:, 2]))
    # real pinhole camera: the hand keeps its apparent size and position, viewed along its own ray
    cen = P.mean(axis=0)
    uc = f_s * (cen[:2] + t_full[:2]) / (cen[2] + t_full[2]) + np.array([W / 2, H / 2])
    f_real = float(K[0, 0])
    z = f_real / px_per_m
    ray = np.array([(uc[0] - K[0, 2]) / f_real, (uc[1] - K[1, 2]) / K[1, 1], 1.0])
    R_ray = rot_between(np.array([0, 0, 1.0]), ray)
    out = (P - cen) @ R_ray.T + ray * z
    if E is not None:
        E = (E - cen) @ R_ray.T + ray * z
    return out, R_ray, px_per_m, uv, E


def mano_twist(hand_pose: np.ndarray, rest_joints: np.ndarray, right: bool) -> dict:
    """Twist (deg) of each finger's proximal phalanx about its own axis relative to the palm (MANO root),
    from the swing-twist decomposition of the local MANO rotation(s) of the MCP joint."""
    out = {}
    for fname, ks in MANO_PROX.items():
        R = np.eye(3)
        for k in ks:
            R = R @ hand_pose[k - 1]
        j, c = MANO_PROX_JOINT[fname], MANO_PROX_CHILD[fname]
        axis = rest_joints[c] - rest_joints[j]
        axis /= np.linalg.norm(axis)
        q = _quat(R)
        proj = np.dot(q[1:], axis)
        tw = 2 * np.degrees(np.arctan2(proj, q[0]))
        tw = (tw + 180) % 360 - 180
        out[fname] = float(tw if right else -tw)
    return out


def _quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)
