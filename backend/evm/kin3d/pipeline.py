"""3D Kinematics job.

Stage K (process settings) -> <result_dir>/k3d/<keyK>/
    kinematics.json (everything the 3D tab shows), kinematics.npz (arrays), params.json, mesh.bin (local only),
    exports/ (CSV, MATLAB, BVH, glTF), figures/ (PNG 300 dpi), acquire.npz (raw results of the two video passes,
    so that a crash or a code change in the analysis step does not repeat the slow GPU passes)
Stage V (display settings) -> <result_dir>/k3d/<keyK>/display/<keyV>.mp4 : segmented dominant hand, cropped,
    original camera geometry (no anchoring / rescaling), optional wrist-frame axes.

Pass 1  MediaPipe dominant-hand tracking (2D + 3D world landmarks, handedness) and background KLT tracking.
Pass 2  WiLoR on a fixed-size crop that follows the smoothed hand path (+ shifted crops for the noise floor),
        and DIS optical flow for the hybrid vibration (in a worker thread, overlapping the GPU work).
Then    camera motion -> world frame; palm / finger frames; model and hybrid series; spectra, coherence,
        ellipsoids, noise floors; display arrays; exports.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ..hands import hand_mask
from ..pipeline import Cancelled, Progress, _hash
from ..video_io import SegmentReader, VideoWriter, probe
from . import config3d as C
from . import egomotion as EG
from . import frames as FR
from . import hybrid as HY
from . import signals as SG
from .tracking import HandTracker3D, MPCropLandmarker, handedness, mediapipe_to_camera

TIPS = [FR.TIP[f] for f in FR.FINGERS]
BATCH = 32


# ---------------------------------------------------------------------------------------------- keys / paths
def key_k(video_path: str, K: dict) -> str:
    st = os.stat(video_path)
    return _hash({"v": C.K3D_VERSION, "size": st.st_size, "mtime": int(st.st_mtime), "K": K})


def key_v(kk: str, V: dict) -> str:
    return _hash({"v": C.DISPLAY_VERSION, "k": kk, "V": V})


def keys_for3d(video_path: str, settings: dict, fps: float | None = None):
    K, V, R = C.split3d(settings, fps)
    kk = key_k(video_path, K)
    return K, V, R, kk, key_v(kk, V)


def paths3d(result_dir: str, kk: str, kv: str | None = None) -> dict:
    d = os.path.join(result_dir, "k3d", kk)
    p = {"dir": d, "json": os.path.join(d, "kinematics.json"), "npz": os.path.join(d, "kinematics.npz"),
         "params": os.path.join(d, "params.json"), "mesh": os.path.join(d, "mesh.bin"),
         "exports": os.path.join(d, "exports"), "figures": os.path.join(d, "figures"),
         "display_dir": os.path.join(d, "display")}
    if kv:
        p["display"] = os.path.join(d, "display", f"{kv}.mp4")
        p["display_json"] = os.path.join(d, "display", f"{kv}.json")
    return p


# ---------------------------------------------------------------------------------------------- helpers
def _crop_boxes(lm: np.ndarray, fps: float, scale: float):
    """Smoothed crop centres and one fixed crop size for the clip (a jittering crop would jitter the model)."""
    lm_f, _ = FR.fill_nan(lm)
    lo, hi = np.nanmin(lm_f, axis=1), np.nanmax(lm_f, axis=1)
    centre = SG.lowpass((lo + hi) / 2, fps, 2.0)
    ext = np.nanmax(hi - lo, axis=1)
    size = float(scale * np.nanmedian(ext))
    return centre, size


def _shifts(n_aug: int, size: float):
    d = 0.03 * size
    base = [(d, 0, 1.0), (-d, 0, 1.0), (0, d, 1.0), (0, -d, 1.0), (0, 0, 1.03), (0, 0, 0.97), (d, d, 1.0),
            (-d, -d, 1.0)]
    return base[:n_aug]


def _series_entry(x: np.ndarray, fps: float, lo: float, hi: float, order: int, unit: str, vel: bool = True):
    x = np.asarray(x, np.float64)
    b = SG.bandpass(x, fps, lo, hi, order)
    f, P = SG.psd(x, fps)
    e = {"raw": x, "band": b, "psd": P, "unit": unit}
    if vel:
        v = SG.deriv(b, fps)
        e["vel"] = v
        e["acc"] = SG.deriv(v, fps)
    return e, f


def _model_signals(Jw: np.ndarray, right: bool, twist: dict | None):
    """Signals of one 3D-joint sequence (world, mm, NaN-free): wrist pose and rigid-finger poses in the palm."""
    Rp, op, _ = FR.palm_frames(Jw, right)
    Rp, _ = FR.fill_nan(Rp)
    Rp = FR.orthonormalize(Rp)
    op, _ = FR.fill_nan(op)
    tipP = FR.to_palm(Rp, op, Jw[:, TIPS])
    ang = FR.finger_angles(Jw, Rp, right)
    Rm = FR.mean_rotation(Rp)
    rv = np.degrees(FR.rotvec(np.einsum("ji,tjk->tik", Rm, Rp)))
    sig = {"wrist": {"pos": {"x": op[:, 0], "y": op[:, 1], "z": op[:, 2]},
                     "rot": {"x": rv[:, 0], "y": rv[:, 1], "z": rv[:, 2]}},
           "fingers": {}}
    for i, f in enumerate(FR.FINGERS):
        a = {"flexion": ang[f]["flexion"], "abduction": ang[f]["abduction"]}
        if twist is not None:
            a["twist"] = twist[f]
        sig["fingers"][f] = {"tip": {"x": tipP[:, i, 0], "y": tipP[:, i, 1], "z": tipP[:, i, 2]}, "ang": a}
    return sig, Rp, op, Rm


def _iter_signals(sig: dict):
    for grp, d in sig["wrist"].items():
        for ax, x in d.items():
            yield ("wrist", grp, ax, x)
    for f, d in sig["fingers"].items():
        for grp, dd in d.items():
            for ax, x in dd.items():
                yield (f, grp, ax, x)


def _unit(grp: str, ax: str) -> str:
    return "deg" if grp in ("rot", "ang") else "mm"


# ---------------------------------------------------------------------------------------------- acquisition
def _acquire(reader, K: dict, Kc: np.ndarray, stabilised: bool, prog: Progress, device: str) -> dict:
    """Passes 1 and 2 over the video (tracking, background, 3D model, optical flow) -> raw arrays."""
    fps, n = reader.fps, reader.n
    Wd, Hd = reader.size
    # ---- pass 1: MediaPipe tracking + background tracking
    trk = HandTracker3D(K["dominant_hand"], bool(K["mirrored"]), 0.4,
                        rightmost_is_right=(K["view"] == "egocentric") == (not K["mirrored"]))
    bgt = EG.BackgroundTracker()
    lm = np.full((n, 21, 2), np.nan, np.float32)
    oth = np.full((n, 21, 2), np.nan, np.float32)
    wl = np.full((n, 21, 3), np.nan, np.float32)
    labels = [None] * n
    T = 0
    try:
        for t, f in enumerate(reader.frames()):
            lm[t], wl[t], labels[t], oth[t] = trk.detect3d(f, int(round(t * 1000.0 / fps)))
            if stabilised:
                hands = np.stack([lm[t], oth[t]])
                hm = hand_mask((Hd, Wd), hands, margin=1.4)
                bg = 255 - cv2.dilate(hm, None, iterations=10)
                bgt.step(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), bg)
            T = t + 1
            prog("track", T / n, f"Tracking the hand and the background  {T}/{n}")
    finally:
        trk.close()
    if T < max(10, int(2 * fps)):
        raise RuntimeError(f"Segment too short ({T} frames).")
    lm, oth, wl, labels = lm[:T], oth[:T], wl[:T], labels[:T]
    found = np.isfinite(lm[:, 0, 0])
    if found.mean() < 0.5:
        raise RuntimeError(f"The hand was found in only {100 * found.mean():.0f} % of the frames; choose another "
                           "segment or hand.")
    right, physical_right, hand_how = handedness(trk.votes, labels, K["dominant_hand"], bool(K["mirrored"]),
                                                 arm_votes=trk.arm_votes)
    repaired = list(reader.repaired)
    drops = list(reader.suspected_drops)
    centre, box = _crop_boxes(lm, fps, float(K["box_scale"]))
    lm_s, _ = FR.fill_nan(lm)
    lm_s = SG.lowpass(lm_s.reshape(T, -1), fps, 6.0).reshape(T, 21, 2).astype(np.float32)
    hand_px = float(np.nanmedian(np.linalg.norm(lm[:, 0] - lm[:, 9], axis=1)))

    # ---- camera motion -> world frame
    if stabilised:
        H = np.array(bgt.H[:T])
        cam = EG.camera_motion(H, Kc, float(K["floor_distance_m"]))
    else:
        cam = EG.static_camera(T)
    bg_inliers = np.array(bgt.inliers[:T]) if stabilised else None

    # ---- pass 2: 3D model + optical flow
    from .wilor import available as wilor_available
    want = K["model"]
    use_wilor = want in ("auto", "wilor") and wilor_available()["ready"] and device != "cpu"
    if want == "wilor" and not use_wilor:
        if not wilor_available()["ready"]:
            raise RuntimeError("WiLoR is not installed (code, checkpoint or MANO missing) - choose MediaPipe")
        use_wilor = True          # explicitly requested on the CPU: slow but allowed
    model = None
    if use_wilor:
        prog("pose", 0.0, "Loading WiLoR", force=True)
        from .wilor import WiLoRModel
        try:
            model = WiLoRModel("cpu" if device == "cpu" else "cuda", fp16=True)
        except Exception as e:  # noqa: BLE001
            if want == "wilor":
                raise
            print(f"WiLoR unavailable ({e}); using MediaPipe", flush=True)
            model = None
    primary = "wilor" if model is not None else "mediapipe"
    n_aug = int(K["n_aug"]) if K["noise_floor"] else 0
    shifts = _shifts(n_aug, box)
    W_j = np.full((T, 21, 3), np.nan, np.float32)        # WiLoR joints (MANO space)
    W_cam = np.full((T, 3), np.nan, np.float32)
    W_aug_j = np.full((n_aug, T, 21, 3), np.nan, np.float32)
    W_aug_cam = np.full((n_aug, T, 3), np.nan, np.float32)
    W_v = np.full((T, 778, 3), np.nan, np.float32) if model is not None else None
    W_pose = np.full((T, 15, 3, 3), np.nan, np.float32) if model is not None else None
    W_go = np.full((T, 3, 3), np.nan, np.float32) if model is not None else None
    mp_aug = np.full((n_aug, T, 21, 3), np.nan, np.float32) if (model is None and n_aug) else None
    mpc = MPCropLandmarker() if mp_aug is not None else None
    pending = []                          # (t, k, crop, centre, size)
    flow = HY.FlowVibration(hand_px)
    resid = np.zeros((T, 5, 2), np.float32)
    palmA = np.tile(np.array([[1.0, 0, 0], [0, 1.0, 0]], np.float32), (T, 1, 1))
    palm_var = np.full(T, np.nan, np.float32)            # per-component flow variance of the palm fit residuals
    pool = ThreadPoolExecutor(max_workers=1)
    fjob = None

    def run_batch():
        if not pending:
            return
        o = model.predict([p[2] for p in pending])
        for i, (t, k, _, _, _) in enumerate(pending):
            if k == 0:
                W_j[t], W_cam[t] = o["joints"][i], o["cam"][i]
                W_v[t], W_pose[t], W_go[t] = o["verts"][i], o["hand_pose"][i], o["global_orient"][i]
            else:
                W_aug_j[k - 1, t], W_aug_cam[k - 1, t] = o["joints"][i], o["cam"][i]
        pending.clear()

    def flow_job(t, g0, g1, l0, m0):
        r, A, pv = flow.step(g0, g1, l0, m0)
        resid[t], palmA[t], palm_var[t] = r, A, pv

    prev_g = prev_m = None
    try:
        from .wilor import make_crop
        for t, f in enumerate(reader.frames(limit=T)):
            if model is not None:
                pending.append((t, 0, make_crop(f, centre[t], box, flip=not right), centre[t], box))
                for k, (dx, dy, sc) in enumerate(shifts, start=1):
                    c = centre[t] + np.array([dx, dy])
                    pending.append((t, k, make_crop(f, c, box * sc, flip=not right), c, box * sc))
                if len(pending) >= BATCH:
                    run_batch()
            elif mpc is not None:
                for k, (dx, dy, sc) in enumerate(shifts):
                    c = centre[t] + np.array([dx, dy])
                    s2 = box * sc / 2
                    x0, y0 = int(max(0, c[0] - s2)), int(max(0, c[1] - s2))
                    x1, y1 = int(min(Wd, c[0] + s2)), int(min(Hd, c[1] + s2))
                    if x1 - x0 > 32 and y1 - y0 > 32:
                        mp_aug[k, t] = mpc.world(f[y0:y1, x0:x1])
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            m = hand_mask((Hd, Wd), lm_s[t][None], margin=1.15)
            if fjob is not None:
                fjob.result()
                fjob = None
            if prev_g is not None:
                fjob = pool.submit(flow_job, t, prev_g, g, lm_s[t - 1], prev_m)
            prev_g, prev_m = g, m
            prog("pose", (t + 1) / T, f"3D hand pose ({'WiLoR' if model is not None else 'MediaPipe'}) & "
                                      f"optical flow  {t + 1}/{T}")
        if model is not None:
            run_batch()
        if fjob is not None:
            fjob.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if mpc is not None:
            mpc.close()

    focal = rest_mano = faces_all = None
    if model is not None:
        focal, rest_mano, faces_all = model.focal, model.rest_joints_mano, model.faces
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return dict(T=T, lm=lm, oth=oth, wl=wl, right=bool(right), found=found, repaired=np.array(repaired, int),
                drops=np.array(drops, int), centre=centre, box=box, lm_s=lm_s, hand_px=hand_px,
                cam_R=cam["R"], cam_t=cam["t"], cam_n=cam["n"], cam_reliable=cam["reliable"],
                cam_sv=cam["sv_ratio"], cam_trans=cam["median_translation_rel"],
                cam_d=cam["plane_distance_m"], H=np.array(bgt.H[:T]) if stabilised else np.zeros(0), bg_inliers=bg_inliers if bg_inliers is not None else np.zeros(0),
                primary=primary, n_aug=n_aug, shifts=np.array(shifts, float).reshape(-1, 3),
                W_j=W_j, W_cam=W_cam, W_aug_j=W_aug_j, W_aug_cam=W_aug_cam,
                W_v=W_v if W_v is not None else np.zeros(0), W_pose=W_pose if W_pose is not None else np.zeros(0),
                mp_aug=mp_aug if mp_aug is not None else np.zeros(0), resid=resid, palmA=palmA, palm_var=palm_var,
                physical_right=bool(physical_right), hand_how=hand_how,
                focal=focal if focal is not None else np.nan,
                rest_mano=rest_mano if rest_mano is not None else np.zeros(0),
                faces_all=faces_all if faces_all is not None else np.zeros(0), want=want)


# ---------------------------------------------------------------------------------------------- stage K
def run_stage_k(video_path: str, K: dict, out: dict, prog: Progress, device: str = "auto",
                export_mesh: bool = True):
    t_start = time.time()
    os.makedirs(out["dir"], exist_ok=True)
    if os.path.exists(out["json"]):
        os.remove(out["json"])
    reader = SegmentReader(video_path, K["start_s"], K["duration_s"], 0, repair=bool(K["repair_duplicates"]))
    fps, n = reader.fps, reader.n
    Wd, Hd = reader.size
    Kc = EG.intrinsics(Wd, Hd, float(K["camera_hfov_deg"]))
    lo, hi, order = float(K["freq_lo"]), float(K["freq_hi"]), int(K["filter_order"])
    stabilised = K["world_frame"] == "stabilised"

    acq_path = os.path.join(out["dir"], "acquire.npz")
    A = None
    if os.path.exists(acq_path):
        try:
            z = np.load(acq_path, allow_pickle=False)
            A = {k: z[k] for k in z.files}
            for k in ("T", "box", "hand_px", "right", "found_dummy", "primary", "n_aug", "focal", "cam_reliable",
                      "cam_sv", "cam_trans", "cam_d", "want"):
                if k in A and A[k].ndim == 0:
                    A[k] = A[k].item()
            prog("analysis", 0.0, "Re-using the cached pose / flow passes", force=True)
        except Exception:  # noqa: BLE001
            A = None
    if A is None:
        A = _acquire(reader, K, Kc, stabilised, prog, device)
        np.savez_compressed(acq_path, **A)
    T, lm, oth, wl, right, found = int(A["T"]), A["lm"], A["oth"], A["wl"], bool(A["right"]), A["found"]
    repaired, drops, centre, box, lm_s = list(A["repaired"]), list(A["drops"]), A["centre"], float(A["box"]), A["lm_s"]
    hand_px, primary, n_aug, want = float(A["hand_px"]), str(A["primary"]), int(A["n_aug"]), str(A["want"])
    shifts = [tuple(r) for r in A["shifts"]]
    W_j, W_cam, W_aug_j, W_aug_cam = A["W_j"], A["W_cam"], A["W_aug_j"], A["W_aug_cam"]
    W_v, W_pose, mp_aug = A["W_v"], A["W_pose"], (A["mp_aug"] if A["mp_aug"].size else None)
    resid, palmA, palm_var = A["resid"], A["palmA"], A["palm_var"]
    hand_how = str(A["hand_how"])
    bg_inliers = A["bg_inliers"] if A["bg_inliers"].size else None
    cam = {"R": A["cam_R"], "t": A["cam_t"], "n": A["cam_n"], "reliable": bool(A["cam_reliable"]),
           "sv_ratio": float(A["cam_sv"]), "median_translation_rel": float(A["cam_trans"]),
           "plane_distance_m": float(A["cam_d"])}
    physical_right = bool(A["physical_right"])
    cam_raw = cam
    if stabilised:
        cam = EG.smooth_motion(cam, fps, float(K["camera_cutoff_hz"]))
    Rcw, pcw, R_W0, winfo = EG.world_frame(cam)
    is_wilor = primary == "wilor"
    prog("analysis", 0.05, "Building coordinate frames", force=True)
    # ---- 3D joints in the camera
    J_mp = np.full((T, 21, 3), np.nan)
    s_mp = np.full(T, np.nan)
    for t in range(T):
        if np.isfinite(lm[t]).all():
            p, s = mediapipe_to_camera(lm[t], wl[t], Kc)
            if p is not None:
                J_mp[t], s_mp[t] = p, s
    faces = None
    twist = None
    V_cam = None
    if is_wilor:
        from .wilor import mano_twist, to_camera
        J_w = np.full((T, 21, 3), np.nan)
        ppm = np.full(T, np.nan)
        V_cam = np.full((T, 778, 3), np.nan, np.float32) if export_mesh else None
        tw = {f: np.full(T, np.nan) for f in FR.FINGERS}
        for t in range(T):
            if np.isfinite(W_j[t]).all():
                J_w[t], _, ppm[t], _, E = to_camera(W_j[t], W_cam[t], centre[t], box, right, (Wd, Hd), Kc,
                                                     float(A["focal"]), W_v[t] if export_mesh else None)
                if E is not None:
                    V_cam[t] = E
                for f_, v in mano_twist(W_pose[t], A["rest_mano"], right).items():
                    tw[f_][t] = v
        J_aug = np.full((n_aug, T, 21, 3), np.nan)
        for k, (dx, dy, sc) in enumerate(shifts):
            for t in range(T):
                if np.isfinite(W_aug_j[k, t]).all():
                    J_aug[k, t] = to_camera(W_aug_j[k, t], W_aug_cam[k, t], centre[t] + np.array([dx, dy]),
                                            box * sc, right, (Wd, Hd), Kc, float(A["focal"]))[0]
        faces = A["faces_all"]
        twist = {f_: FR.fill_nan(v)[0] for f_, v in tw.items()}
        J_cam, ppm_used = J_w, ppm
    else:
        J_cam, ppm_used = J_mp, s_mp
        J_aug = np.full((n_aug, T, 21, 3), np.nan)
        if mp_aug is not None:
            for k in range(n_aug):
                for t in range(T):
                    if np.isfinite(mp_aug[k, t]).all() and np.isfinite(lm[t]).all():
                        J_aug[k, t] = mediapipe_to_camera(lm[t], mp_aug[k, t], Kc)[0]
    valid = np.isfinite(J_cam).all(axis=(1, 2))
    if valid.mean() < 0.5:
        raise RuntimeError(f"The 3D model produced a pose in only {100 * valid.mean():.0f} % of the frames.")

    # ---- metric scale (optional measured hand length) - uniform scaling about the camera keeps the projection
    def chain_len(J):
        c = [0, 9, 10, 11, 12]
        return np.nanmedian(sum(np.linalg.norm(J[:, c[i + 1]] - J[:, c[i]], axis=1) for i in range(4))) * 1000
    model_len = float(chain_len(J_cam))
    factor = float(K["hand_length_mm"]) / model_len if K["hand_length_mm"] > 0 and model_len > 0 else 1.0
    J_cam = J_cam * factor
    if primary == "mediapipe":
        factor_mp = factor
    else:
        mp_len = float(chain_len(J_mp)) if np.isfinite(J_mp).any() else 0.0
        factor_mp = float(K["hand_length_mm"]) / mp_len if K["hand_length_mm"] > 0 and mp_len > 0 else 1.0
    J_mp_s = J_mp * factor_mp
    J_aug = J_aug * factor
    if V_cam is not None:
        V_cam = V_cam * factor
    mm_px = SG.lowpass(FR.fill_nan(1000.0 / ppm_used)[0], fps, 1.0) * factor

    def world_mm(Jc):
        Jf, _ = FR.fill_nan(Jc)
        return (np.einsum("tij,tkj->tki", Rcw, Jf) + pcw[:, None]) * 1000.0

    Jw = world_mm(J_cam)
    sig_model, Rp, op, Rm = _model_signals(Jw, right, twist)
    sig_mp = _model_signals(world_mm(J_mp_s), right, None)[0] if primary == "wilor" and \
        np.isfinite(J_mp).all(axis=(1, 2)).mean() > 0.5 else None

    # ---- hybrid (optical flow) in the low-passed wrist frame
    prog("analysis", 0.25, "Hybrid optical-flow vibration", force=True)
    R_cp = np.einsum("tji,tjk->tik", Rcw, Rp)              # palm axes in camera coordinates
    R_cp_s = FR.orthonormalize(SG.lowpass(R_cp.reshape(T, 9), fps, float(K["pose_cutoff_hz"])).reshape(T, 3, 3))
    tip_h = HY.to_palm_mm(resid, mm_px, R_cp_s)
    J_cam_f = FR.fill_nan(J_cam)[0]
    rays = J_cam_f[:, [0, 5, 9, 13, 17]].mean(axis=1)
    obs = HY.observability(R_cp_s, rays)
    radial = -1.0 if right else 1.0
    J_palm = FR.to_palm(Rp, op, Jw)
    L_f, e_ab, e_fl = {}, {}, {}
    for f in FR.FINGERS:
        m_, tp = FR.RIGID[f]
        d = np.nanmedian(J_palm[:, tp] - J_palm[:, m_], axis=0)
        L_f[f] = float(np.linalg.norm(d))
        d = d / max(L_f[f], 1e-9)
        a = radial * np.array([1.0, 0, 0])
        a = a - np.dot(a, d) * d
        e_ab[f] = a / max(np.linalg.norm(a), 1e-9)
        fl = np.array([0, 0, -1.0])                      # towards the palm, perpendicular to the finger
        fl = fl - np.dot(fl, d) * d
        e_fl[f] = fl / max(np.linalg.norm(fl), 1e-9)
    # wrist (hybrid): palm translation / roll in the image, minus the camera rotation's own image motion
    cen = np.array([HY.palm_points(lm_s[max(t - 1, 0)]).mean(0) for t in range(T)])
    dxy = np.zeros((T, 2))
    droll = np.zeros(T)
    Kinv = np.linalg.inv(Kc)
    for t in range(1, T):
        A = palmA[t]
        d = A[:, :2] @ cen[t] + A[:, 2] - cen[t]
        rl = np.arctan2(A[1, 0], A[0, 0])
        if stabilised:
            dR = cam["R"][t] @ cam["R"][t - 1].T
            ph = Kc @ dR @ Kinv @ np.array([cen[t][0], cen[t][1], 1.0])
            d = d - (ph[:2] / ph[2] - cen[t])
            rl -= FR.rotvec(dR[None])[0][2]
        dxy[t], droll[t] = d, rl
    wrist_h = np.cumsum(dxy * mm_px[:, None], axis=0)
    sig_h = {"wrist": {"pos": {"x_img": wrist_h[:, 0], "y_img": -wrist_h[:, 1]},
                       "rot": {"roll": np.degrees(np.cumsum(droll))}}, "fingers": {}}
    for i, f in enumerate(FR.FINGERS):
        sig_h["fingers"][f] = {"tip": {"x": tip_h[:, i, 0], "y": tip_h[:, i, 1], "z": tip_h[:, i, 2]},
                               "ang": {"flexion": np.degrees(tip_h[:, i] @ e_fl[f] / L_f[f]),
                                       "abduction": np.degrees(tip_h[:, i] @ e_ab[f] / L_f[f])}}

    # ---- noise floors
    prog("analysis", 0.4, "Noise floors, spectra and coherence", force=True)
    noise_model = {}
    if n_aug and np.isfinite(J_aug).any():
        diffs = {}
        for k in range(n_aug):
            if np.isfinite(J_aug[k]).all(axis=(1, 2)).mean() < 0.5:
                continue
            sk = _model_signals(world_mm(J_aug[k]), right, None)[0]
            for (b, g, a, x) in _iter_signals(sk):
                base = _get(sig_model, b, g, a)
                if base is None:
                    continue
                f_, P = SG.psd(np.asarray(x) - base, fps)
                diffs.setdefault((b, g, a), []).append(P / 2.0)
        noise_model = {key: np.mean(v, axis=0) for key, v in diffs.items()}
    noise_h_mm, noise_h_px, flow_sigma_px = None, None, None
    pv = palm_var[np.isfinite(palm_var)]
    if K["noise_floor"] and len(pv) > 10:
        # optical-flow noise from the residuals of the rigid palm fit (skin texture at the hand's depth): white
        # per-frame increments of variance s2 accumulate to a random walk, PSD = 2 s2 / (fs 4 sin^2(pi f / fs));
        # x1.5 for the uncertainty of the palm-motion prediction at the fingertip
        s2 = float(np.median(pv)) * 1.5
        flow_sigma_px = float(np.sqrt(s2))
        f0 = SG.psd(np.zeros(T), fps)[0]
        den = fps * 4 * np.sin(np.pi * np.maximum(f0, f0[1] if len(f0) > 1 else 1.0) / fps) ** 2
        noise_h_px = 2 * s2 / den
        noise_h_mm = noise_h_px * float(np.median(mm_px)) ** 2

    # ---- series, spectra, metrics
    methods = {"model": sig_model, "hybrid": sig_h}
    if sig_mp is not None:
        methods["mediapipe"] = sig_mp
    series, metrics = {}, {}
    f_axis = None
    for meth, sig in methods.items():
        series[meth], metrics[meth] = {}, {}
        for (b, g, a, x) in _iter_signals(sig):
            unit = _unit(g, a)
            e, f_axis = _series_entry(x, fps, lo, hi, order, unit, vel=meth != "mediapipe")
            Pn = None
            if meth == "model":
                Pn = noise_model.get((b, g, a))
            elif meth == "hybrid" and noise_h_mm is not None:
                if b == "wrist" and g == "rot":
                    Pn = None
                elif g == "ang":
                    Pn = noise_h_mm * (180 / np.pi / L_f[b]) ** 2
                else:
                    Pn = noise_h_mm
            st = SG.band_stats(f_axis, e["psd"], lo, hi, Pn)
            if meth == "hybrid" and b != "wrist" and g == "tip":
                st["observability"] = round(float(obs["xyz".index(a)]), 3)
            e["noise"] = Pn
            series[meth].setdefault(b, {}).setdefault(g, {})[a] = e
            metrics[meth].setdefault(b, {}).setdefault(g, {})[a] = {"unit": unit, **st}

    # principal-axis projections -> spectrograms, coherence, ellipsoids
    bodies = ["wrist_pos", "wrist_rot"] + FR.FINGERS
    spectrograms, coh, ellipsoids, principal_dirs = {}, {}, {}, {}
    for meth in ("model", "hybrid"):
        proj = {}
        for b in bodies:
            if b.startswith("wrist"):
                g = "pos" if b == "wrist_pos" else "rot"
                d = series[meth]["wrist"][g]
            else:
                d = series[meth][b]["tip"]
            axes = list(d.keys())
            band = np.stack([d[a]["band"] for a in axes], 1)
            raw = np.stack([d[a]["raw"] for a in axes], 1)
            if band.shape[1] == 1:
                pb, dirv = band[:, 0], np.array([1.0])
            else:
                pb, dirv = SG.principal(band)
            proj[b] = pb
            principal_dirs.setdefault(meth, {})[b] = {"axes": axes, "dir": dirv.tolist()}
            tS, fS, dB = SG.spectro(raw @ dirv if band.shape[1] > 1 else raw[:, 0], fps, min(hi + 3, fps / 2))
            spectrograms.setdefault(meth, {})[b] = {"t": tS, "f": fS, "db": dB}
            if not b.startswith("wrist") and band.shape[1] == 3:
                ellipsoids.setdefault(meth, {})[b] = SG.ellipsoid(band)
        M = np.eye(len(bodies))
        for i in range(len(bodies)):
            for j in range(i + 1, len(bodies)):
                M[i, j] = M[j, i] = SG.coherence_band(proj[bodies[i]], proj[bodies[j]], fps, lo, hi)
        coh[meth] = M

    # ---- display arrays for the 3D view
    prog("analysis", 0.6, "3D view data", force=True)
    J_palm_low = SG.lowpass(J_palm.reshape(T, -1), fps, float(K["pose_cutoff_hz"])).reshape(T, 21, 3)
    band_model = SG.bandpass(J_palm.reshape(T, -1), fps, lo, hi, order).reshape(T, 21, 3)
    tip_h_band = SG.bandpass(tip_h.reshape(T, -1), fps, lo, hi, order).reshape(T, 5, 3)
    band_hyb = np.zeros((T, 21, 3))
    for i, f in enumerate(FR.FINGERS):
        m_, tp = FR.RIGID[f]
        Lt = np.nanmedian(np.linalg.norm(J_palm[:, tp] - J_palm[:, m_], axis=1))
        for j in FR.CHAIN[f]:
            if j == m_ or (f == "thumb" and j == 1):
                continue
            fac = np.nanmedian(np.linalg.norm(J_palm[:, j] - J_palm[:, m_], axis=1)) / max(Lt, 1e-9)
            band_hyb[:, j] = fac * tip_h_band[:, i]
    Rp_slow = FR.orthonormalize(SG.lowpass(Rp.reshape(T, 9), fps, float(K["pose_cutoff_hz"])).reshape(T, 3, 3))
    op_slow = SG.lowpass(op, fps, float(K["pose_cutoff_hz"]))
    ff = FR.finger_frames(Jw, Rp)
    jang = FR.joint_angles(Jw)
    t_axis = np.arange(T) / fps
    if V_cam is not None:
        Vf, _ = FR.fill_nan(V_cam.reshape(T, -1))
        Vw = (np.einsum("tij,tkj->tki", Rcw, Vf.reshape(T, 778, 3)) + pcw[:, None]).astype(np.float16)
        with open(out["mesh"], "wb") as fh:
            fh.write(Vw.tobytes())
    elif os.path.exists(out["mesh"]):
        os.remove(out["mesh"])

    def ser_json(e):
        d = {k: SG.rnd(v, 5) for k, v in e.items() if k in ("raw", "band", "vel", "acc")}
        d["psd"] = SG.rnd(e["psd"], 9)
        d["noise"] = None if e.get("noise") is None else SG.rnd(e["noise"], 9)
        d["unit"] = e["unit"]
        return d

    res = {
        "version": C.K3D_VERSION, "video": os.path.basename(video_path), "fps": fps, "n_frames": T,
        "start_s": round(reader.start_s, 3), "duration_s": round(T / fps, 3), "size": [Wd, Hd],
        "model": primary, "model_requested": want, "comparison": "mediapipe" if sig_mp is not None else None,
        "hand": {"appears_right": bool(right), "physical_side": "right" if physical_right else "left",
                 "side_from": hand_how, "view": K["view"],
                 "mirrored": bool(K["mirrored"]), "found_pct": round(100 * float(found.mean()), 1),
                 "pose_pct": round(100 * float(valid.mean()), 1), "hand_px": round(hand_px, 1),
                 "crop_px": round(box, 1)},
        "band": [lo, hi], "intrinsics": {"f_px": round(float(Kc[0, 0]), 2), "cx": float(Kc[0, 2]),
                                          "cy": float(Kc[1, 2]), "hfov_deg": float(K["camera_hfov_deg"])},
        "world": {"mode": K["world_frame"], **winfo, "reliable_plane": bool(cam["reliable"]),
                  "sv_ratio": cam["sv_ratio"], "median_translation_rel": cam["median_translation_rel"],
                  "floor_normal_cam": SG.rnd(cam["n"], 4), "floor_distance_m": float(K["floor_distance_m"]),
                  "bg_inliers_median": int(np.median(bg_inliers)) if bg_inliers is not None else None,
                  "camera_cutoff_hz": float(K["camera_cutoff_hz"]),
                  "head_rotation_range_deg": round(float(np.degrees(np.max(np.linalg.norm(FR.rotvec(cam_raw["R"]), axis=1)))), 2),
                  "head_translation": "not estimated", "rotation_fit": cam_raw.get("rotation_fit")},
        "scale": {"model_hand_length_mm": round(model_len, 1), "measured_hand_length_mm": K["hand_length_mm"] or None,
                  "factor": round(factor, 4), "mm_per_px_median": round(float(np.median(mm_px)), 4),
                  "finger_length_mm": {f: round(L_f[f], 1) for f in FR.FINGERS}},
        "observability": {"x": round(float(obs[0]), 3), "y": round(float(obs[1]), 3), "z": round(float(obs[2]), 3)},
        "noise": {"model_reruns": n_aug if noise_model else 0,
                  "flow_sigma_px_per_frame": round(flow_sigma_px, 5) if flow_sigma_px else None,
                  "flow_noise_band_rms_px": round(float(np.sqrt(SG.band_stats(f_axis, noise_h_px, lo, hi)["band_rms"] ** 2)), 5)
                  if noise_h_px is not None else None},
        "frame_repair": {"repaired_frames": len(repaired), "repaired_pct": round(100 * len(repaired) / T, 2),
                         "suspected_dropped_frames": len(drops), "suspected_dropped_pct": round(100 * len(drops) / T, 2)}
        if K["repair_duplicates"] else None,
        "t": SG.rnd(t_axis, 4), "f": SG.rnd(f_axis, 4),
        "series": {m: {b: {g: {a: ser_json(e) for a, e in gd.items()} for g, gd in bd.items()}
                       for b, bd in md.items()} for m, md in series.items()},
        "metrics": metrics,
        "spectrograms": {m: {b: {"t": SG.rnd(v["t"], 3), "f": SG.rnd(v["f"], 3), "db": SG.rnd(v["db"], 1)}
                             for b, v in md.items()} for m, md in spectrograms.items()},
        "coherence": {"bodies": bodies, **{m: SG.rnd(v, 3) for m, v in coh.items()}},
        "ellipsoids": {m: {b: {"axes": SG.rnd(v["axes"], 5), "dirs": SG.rnd(v["dirs"], 4)} for b, v in md.items()}
                       for m, md in ellipsoids.items()},
        "principal": principal_dirs,
        "joint_angles": {f: {j: SG.rnd(v, 2) for j, v in d.items()} for f, d in jang.items()},
        "display": {
            "joints_world": SG.rnd(Jw, 2), "joints_palm_low": SG.rnd(J_palm_low, 3),
            "band_model": SG.rnd(band_model, 5), "band_hybrid": SG.rnd(band_hyb, 5),
            "palm": {"R": SG.rnd(Rp.reshape(T, 9), 5), "o": SG.rnd(op, 2)},
            "palm_slow": {"R": SG.rnd(Rp_slow.reshape(T, 9), 5), "o": SG.rnd(op_slow, 2)},
            "palm_mean": {"R": SG.rnd(Rm.reshape(9), 5), "o": SG.rnd(np.nanmean(op, 0), 2)},
            "fingers": {f: {"R": SG.rnd(R.reshape(T, 9), 4), "o": SG.rnd(o_, 2)} for f, (R, o_) in ff.items()},
            "camera": {"R": SG.rnd(Rcw.reshape(T, 9), 5), "o": SG.rnd(pcw * 1000, 2)},
        },
        "mesh": {"available": V_cam is not None, "n_verts": 778, "faces": faces.tolist() if (faces is not None and V_cam is not None) else None},
        "params": K, "processing_s": None, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "source": probe(video_path),
    }

    # arrays for the display stage and the exports
    np.savez_compressed(out["npz"], lm=lm, lm_s=lm_s, oth=oth, found=found, centre=centre, box=box, Kc=Kc,
                        R_cp_s=R_cp_s, op_cam=FR.fill_nan(J_cam)[0][:, 0], Jw=Jw, Rp=Rp, op=op, Rcw=Rcw, pcw=pcw,
                        right=right, fps=fps, start_s=reader.start_s, size=np.array([Wd, Hd]))
    prog("analysis", 0.7, "Writing exports and figures", force=True)
    from .exports import write_all
    files = write_all(out, res, series, {"Jw": Jw, "Rp": Rp, "op": op, "ff": ff, "Rcw": Rcw, "pcw": pcw * 1000,
                                         "jang": jang, "t": t_axis, "fps": fps, "right": right,
                                         "J_palm": J_palm, "coh": coh, "bodies": bodies, "f": f_axis,
                                         "ellipsoids": ellipsoids, "spectrograms": spectrograms},
                      lambda fr, msg: prog("analysis", 0.7 + 0.28 * fr, msg))
    res["files"] = files
    res["processing_s"] = round(time.time() - t_start, 1)
    with open(out["params"], "w") as fh:
        json.dump(K, fh, indent=2)
    with open(out["json"] + ".tmp", "w") as fh:
        json.dump(_safe(res), fh, separators=(",", ":"))
    os.replace(out["json"] + ".tmp", out["json"])
    prog("analysis", 1.0, "3D kinematics complete", force=True)


def _get(sig, b, g, a):
    try:
        return np.asarray(sig["wrist"][g][a]) if b == "wrist" else np.asarray(sig["fingers"][b][g][a])
    except KeyError:
        return None


def _safe(o):
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _safe(o.tolist())
    if isinstance(o, np.generic):
        return _safe(o.item())
    return o


# ---------------------------------------------------------------------------------------------- stage V
def run_stage_v(video_path: str, K: dict, V: dict, out: dict, prog: Progress):
    from ..segment import make_segmenter
    os.makedirs(out["display_dir"], exist_ok=True)
    z = np.load(out["npz"])
    lm, lm_s, oth, found = z["lm"], z["lm_s"], z["oth"], z["found"]
    centre, box, Kc, R_cp, o_cam = z["centre"], float(z["box"]), z["Kc"], z["R_cp_s"], z["op_cam"]
    fps = float(z["fps"])
    T = len(lm)
    reader = SegmentReader(video_path, K["start_s"], K["duration_s"], 0, repair=bool(K["repair_duplicates"]))
    Wd, Hd = reader.size
    side = 0.8 * box                                      # crop around the hand (box = WiLoR crop, 2x the hand)
    D = int(V["display_px"])
    seg, used = make_segmenter(V["segmenter"])
    writer = VideoWriter(out["display"], (D, D), fps, crf=17)
    bgc = np.array([20, 15, 11], np.float32)              # dark background (BGR)
    try:
        for t, f in enumerate(reader.frames(limit=T)):
            if found[t]:
                m = seg(f, lm[t], oth[t] if np.isfinite(oth[t]).all() else None, int(round(t * 1000.0 / fps)))
            else:
                m = hand_mask(f.shape[:2], lm_s[t][None], margin=1.15).astype(np.float32) / 255
            m = cv2.GaussianBlur(m, (0, 0), 1.2)
            comp = f.astype(np.float32) * m[..., None] + bgc * (1 - m[..., None])
            s = D / side
            M = np.array([[s, 0, D / 2 - s * centre[t][0]], [0, s, D / 2 - s * centre[t][1]]], np.float32)
            img = cv2.warpAffine(comp.astype(np.uint8), M, (D, D), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=bgc.tolist())
            if V["draw_axes"] and np.isfinite(R_cp[t]).all() and np.isfinite(o_cam[t]).all():
                o = o_cam[t]
                L = 0.04
                pts = [o] + [o + L * R_cp[t][:, i] for i in range(3)]
                uv = []
                for p in pts:
                    q = Kc @ p
                    uv.append((M[:, :2] @ (q[:2] / q[2]) + M[:, 2]))
                for i, col in enumerate(((60, 60, 255), (80, 220, 80), (255, 140, 40))):
                    a, b = tuple(int(v) for v in uv[0]), tuple(int(v) for v in uv[i + 1])
                    cv2.line(img, a, b, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.line(img, a, b, col, 3, cv2.LINE_AA)
                cv2.circle(img, tuple(int(v) for v in uv[0]), 5, (255, 255, 255), -1, cv2.LINE_AA)
            writer.write(img)
            prog("display", (t + 1) / T, f"Segmented hand video ({used})  {t + 1}/{T}")
        writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        seg.close()
    with open(out["display_json"], "w") as fh:
        json.dump({"display": V, "segmenter": used, "crop_side_px": round(side, 1),
                   "created": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=2)


# ---------------------------------------------------------------------------------------------- job entry
def run_job3d(spec: dict, report=None, cancel_check=None) -> dict:
    """spec: {kind: 'kin3d', video_path, result_dir, settings, fps[, export_mesh]}."""
    K, V, R, kk, kv = keys_for3d(spec["video_path"], spec["settings"], spec.get("fps"))
    out = paths3d(spec["result_dir"], kk, kv)
    need_k = not os.path.exists(out["json"])
    need_v = not os.path.exists(out["display"])
    spans = {"track": (0.0, 0.2), "pose": (0.2, 0.72), "analysis": (0.72, 0.85), "display": (0.85, 1.0)} if need_k \
        else {"track": (0, 0), "pose": (0, 0), "analysis": (0, 0), "display": (0.0, 1.0)}
    prog = Progress(report, spans)
    if cancel_check:
        prog.cancel_check = cancel_check
    device = {"gpu": "cuda", "cpu": "cpu"}.get(R.get("device", "auto"), "auto")
    if need_k:
        run_stage_k(spec["video_path"], K, out, prog, device, bool(spec.get("export_mesh", True)))
    if need_v:
        run_stage_v(spec["video_path"], K, V, out, prog)
    return {"key_a": kk, "key_b": kv}
