"""Exports of a 3D Kinematics result: CSV tables, MATLAB .mat, BVH + glTF animation, PNG figures (300 dpi)."""
from __future__ import annotations

import csv
import json
import os
import struct

import numpy as np

from . import frames as FR
from .wilor import rot_between

M_YUP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])     # world (Z up) -> Y-up animation space
PARENT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 5, 7: 6, 8: 7, 9: 0, 10: 9, 11: 10, 12: 11, 13: 0, 14: 13, 15: 14,
          16: 15, 17: 0, 18: 17, 19: 18, 20: 19}
NAMES = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip",
         "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip", "ring_dip",
         "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]
BODIES = ["wrist"] + FR.FINGERS


def _w(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _f(v, nd=5):
    return "" if v is None or not np.isfinite(v) else f"{float(v):.{nd}f}"


# ---------------------------------------------------------------------------------------------- skeleton anim
def _children(j):
    return [c for c, p in PARENT.items() if p == j]


def _local_rotations(J_palm: np.ndarray):
    """Swing-only local joint rotations (T,21,3,3) of the skeleton expressed in the palm frame, rest pose =
    clip-median joint positions in the palm frame (all rest joint frames aligned with the palm)."""
    T = len(J_palm)
    rest = np.nanmedian(J_palm, axis=0)
    L = np.repeat(np.eye(3)[None, None], T, 0).repeat(21, 1)
    G = np.repeat(np.eye(3)[None, None], T, 0).repeat(21, 1)
    order = [j for j in range(1, 21)]
    for j in order:
        ch = _children(j)
        p = PARENT[j]
        if not ch:
            G[:, j] = G[:, p]
            continue
        c = ch[0]
        d_rest = rest[c] - rest[j]
        for t in range(T):
            d_cur = J_palm[t, c] - J_palm[t, j]
            if not np.isfinite(d_cur).all() or np.linalg.norm(d_cur) < 1e-9:
                continue
            L[t, j] = rot_between(d_rest, G[t, p].T @ d_cur)
            G[t, j] = G[t, p] @ L[t, j]
    return rest, L


def _euler_zxy(R):
    a = np.arcsin(np.clip(R[..., 2, 1], -1, 1))
    b = np.arctan2(-R[..., 2, 0], R[..., 2, 2])
    g = np.arctan2(-R[..., 0, 1], R[..., 1, 1])
    return np.degrees(np.unwrap(np.stack([g, a, b], -1), axis=0))       # Z, X, Y


def write_bvh(path, ctx):
    J_palm, Rp, op, fps = ctx["J_palm"], ctx["Rp"], ctx["op"], ctx["fps"]
    rest, L = _local_rotations(J_palm)
    T = len(J_palm)
    lines = ["HIERARCHY"]

    def emit(j, depth):
        ind = "  " * depth
        off = (rest[j] - rest[PARENT[j]]) * 0.1 if j else np.zeros(3)
        ch = _children(j)
        kind = "ROOT" if j == 0 else "JOINT"
        lines.append(f"{ind}{kind} {NAMES[j]}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}  OFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}")
        lines.append(f"{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation" if j == 0 else
                     f"{ind}  CHANNELS 3 Zrotation Xrotation Yrotation")
        for c in ch:
            if _children(c):
                emit(c, depth + 1)
            else:
                e = (rest[c] - rest[j]) * 0.1
                lines.append(f"{ind}  End Site")
                lines.append(f"{ind}  {{")
                lines.append(f"{ind}    OFFSET {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}")
                lines.append(f"{ind}  }}")
        lines.append(f"{ind}}}")

    emit(0, 0)
    joints = [0]

    def collect(j):
        for c in _children(j):
            if _children(c):
                joints.append(c)
                collect(c)
    collect(0)
    root_R = np.einsum("ij,tjk->tik", M_YUP, Rp)
    root_p = np.einsum("ij,tj->ti", M_YUP, op) * 0.1
    eul = {0: _euler_zxy(root_R)}
    for j in joints[1:]:
        eul[j] = _euler_zxy(L[:, j])
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"]
    for t in range(T):
        vals = list(root_p[t]) + list(eul[0][t])
        for j in joints[1:]:
            vals += list(eul[j][t])
        lines.append(" ".join(f"{v:.4f}" for v in vals))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _quat_xyzw(R):
    from .wilor import _quat
    w, x, y, z = _quat(R)
    return [x, y, z, w]


def write_glb(path, ctx):
    """Animated joint hierarchy (no mesh) in glTF 2.0 binary: Y-up, metres. Opens in Blender / Unity (glTFast)."""
    J_palm, Rp, op, fps = ctx["J_palm"], ctx["Rp"], ctx["op"], ctx["fps"]
    rest, L = _local_rotations(J_palm)
    T = len(J_palm)
    times = (np.arange(T) / fps).astype(np.float32)
    nodes, buf, views, accs, channels, samplers = [], bytearray(), [], [], [], []

    def add_acc(arr, typ, minmax=False):
        arr = np.ascontiguousarray(arr, np.float32)
        while len(buf) % 4:
            buf.append(0)
        off = len(buf)
        buf.extend(arr.tobytes())
        views.append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes})
        a = {"bufferView": len(views) - 1, "componentType": 5126, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            a["min"], a["max"] = [float(arr.min())], [float(arr.max())]
        accs.append(a)
        return len(accs) - 1

    t_acc = add_acc(times, "SCALAR", True)
    idx = {}
    for j in range(21):
        off = (rest[j] - rest[PARENT[j]]) * 0.001 if j else np.zeros(3)
        idx[j] = len(nodes)
        nodes.append({"name": NAMES[j], "translation": [float(v) for v in off]})
    for j in range(21):
        ch = [idx[c] for c in _children(j)]
        if ch:
            nodes[idx[j]]["children"] = ch
    root_R = np.einsum("ij,tjk->tik", M_YUP, Rp)
    root_p = np.einsum("ij,tj->ti", M_YUP, op) * 0.001

    def anim(node, path_, data, typ):
        a = add_acc(data, typ)
        samplers.append({"input": t_acc, "output": a, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path_}})

    anim(idx[0], "translation", root_p, "VEC3")
    anim(idx[0], "rotation", np.array([_quat_xyzw(R) for R in root_R]), "VEC4")
    for j in range(1, 21):
        if _children(j):
            anim(idx[j], "rotation", np.array([_quat_xyzw(R) for R in L[:, j]]), "VEC4")
    gltf = {"asset": {"version": "2.0", "generator": "TREVIDEON"}, "scene": 0,
            "scenes": [{"nodes": [idx[0]]}], "nodes": nodes,
            "animations": [{"name": "hand", "samplers": samplers, "channels": channels}],
            "buffers": [{"byteLength": len(buf)}], "bufferViews": views, "accessors": accs}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    while len(buf) % 4:
        buf.append(0)
    total = 12 + 8 + len(js) + 8 + len(buf)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(buf), 0x004E4942) + bytes(buf))


# ---------------------------------------------------------------------------------------------- tables
def write_tables(d, res, series, ctx):
    T, t = len(ctx["t"]), ctx["t"]
    files = []
    # transforms
    rows = []
    bodies = [("camera", ctx["Rcw"], ctx["pcw"]), ("wrist", ctx["Rp"], ctx["op"])] + \
             [(f, ctx["ff"][f][0], ctx["ff"][f][1]) for f in FR.FINGERS]
    for i in range(T):
        for name, R, o in bodies:
            rows.append([i, _f(t[i], 4), name] + [_f(v, 6) for v in R[i].ravel()] + [_f(v, 3) for v in o[i]])
    p = os.path.join(d, "transforms_world.csv")
    _w(p, ["frame", "t_s", "body"] + [f"r{a}{b}" for a in (1, 2, 3) for b in (1, 2, 3)] + ["x_mm", "y_mm", "z_mm"],
       rows)
    files.append(p)
    # relative finger poses in the wrist frame
    rows = []
    Rp, op = ctx["Rp"], ctx["op"]
    for f in FR.FINGERS:
        R, o = ctx["ff"][f]
        Rr = np.einsum("tji,tjk->tik", Rp, R)
        orr = np.einsum("tji,tj->ti", Rp, o - op)
        for i in range(T):
            rows.append([i, _f(t[i], 4), f] + [_f(v, 6) for v in Rr[i].ravel()] + [_f(v, 3) for v in orr[i]])
    p = os.path.join(d, "fingers_in_wrist_frame.csv")
    _w(p, ["frame", "t_s", "finger"] + [f"r{a}{b}" for a in (1, 2, 3) for b in (1, 2, 3)] +
       ["origin_x_mm", "origin_y_mm", "origin_z_mm"], rows)
    files.append(p)
    # joints
    Jw = ctx["Jw"]
    p = os.path.join(d, "joints_world.csv")
    _w(p, ["frame", "t_s"] + [f"{n}_{a}_mm" for n in NAMES for a in "xyz"],
       [[i, _f(t[i], 4)] + [_f(v, 3) for v in Jw[i].ravel()] for i in range(T)])
    files.append(p)
    # all series (long format)
    rows = []
    for meth, md in series.items():
        for b, bd in md.items():
            for g, gd in bd.items():
                for a, e in gd.items():
                    raw, band = e["raw"], e["band"]
                    vel, acc = e.get("vel"), e.get("acc")
                    for i in range(T):
                        rows.append([meth, b, g, a, e["unit"], i, _f(t[i], 4), _f(raw[i], 5), _f(band[i], 6),
                                     _f(vel[i], 5) if vel is not None else "", _f(acc[i], 4) if acc is not None else ""])
    p = os.path.join(d, "series_long.csv")
    _w(p, ["method", "body", "group", "axis", "unit", "frame", "t_s", "value", "band", "band_velocity",
           "band_acceleration"], rows)
    files.append(p)
    # joint angles
    rows = []
    for i in range(T):
        r = [i, _f(t[i], 4)]
        for f in FR.FINGERS:
            for jn in FR.JOINT_NAMES[f]:
                r.append(_f(ctx["jang"][f][jn][i], 3))
        rows.append(r)
    p = os.path.join(d, "joint_angles_deg.csv")
    _w(p, ["frame", "t_s"] + [f"{f}_{jn}" for f in FR.FINGERS for jn in FR.JOINT_NAMES[f]], rows)
    files.append(p)
    # metrics
    rows = []
    for meth, md in res["metrics"].items():
        for b, bd in md.items():
            for g, gd in bd.items():
                for a, m in gd.items():
                    rows.append([meth, b, g, a, m["unit"], m.get("peak_hz"), _f(m.get("band_rms"), 6),
                                 _f(m.get("total_rms"), 6), m.get("band_power_pct"), _f(m.get("noise_rms"), 6),
                                 m.get("snr_db"), m.get("observability", "")])
    p = os.path.join(d, "metrics.csv")
    _w(p, ["method", "body", "group", "axis", "unit", "peak_hz", "band_rms", "total_rms", "band_power_pct",
           "noise_band_rms", "snr_db", "observability"], rows)
    files.append(p)
    # coherence
    rows = []
    for meth, M in ctx["coh"].items():
        for i, bi in enumerate(ctx["bodies"]):
            rows.append([meth, bi] + [_f(v, 4) for v in M[i]])
    p = os.path.join(d, "coherence_band.csv")
    _w(p, ["method", "body"] + ctx["bodies"], rows)
    files.append(p)
    return files


def write_mat(path, res, series, ctx):
    from scipy.io import savemat
    m = {"t": ctx["t"], "fps": ctx["fps"], "joints_world_mm": ctx["Jw"], "wrist_R": ctx["Rp"], "wrist_o_mm": ctx["op"],
         "camera_R": ctx["Rcw"], "camera_o_mm": ctx["pcw"], "joint_names": np.array(NAMES, dtype=object),
         "band_hz": np.array(res["band"]), "f_hz": ctx["f"]}
    for f in FR.FINGERS:
        m[f"{f}_R"], m[f"{f}_o_mm"] = ctx["ff"][f]
        for jn, v in ctx["jang"][f].items():
            m[f"angle_{f}_{jn}_deg"] = v
    s = {}
    for meth, md in series.items():
        s[meth] = {b: {g: {a: {k: np.asarray(v) for k, v in e.items() if k in ("raw", "band", "vel", "acc", "psd")
                               and v is not None} for a, e in gd.items()} for g, gd in bd.items()}
                   for b, bd in md.items()}
    m["series"] = s
    m["coherence_bodies"] = np.array(ctx["bodies"], dtype=object)
    for meth, M in ctx["coh"].items():
        m[f"coherence_{meth}"] = M
    savemat(path, m, do_compression=True, long_field_names=True)


# ---------------------------------------------------------------------------------------------- figures
COL = {"thumb": "#e8590c", "index": "#1c7ed6", "middle": "#2f9e44", "ring": "#9c36b5", "pinky": "#c2255c",
       "wrist": "#495057"}


def write_figures(outdir, res, series, ctx, cb=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    t = ctx["t"]
    lo, hi = res["band"]
    files = []
    title = f"{res['video']}  {res['start_s']:.1f}-{res['start_s'] + res['duration_s']:.1f} s  ({res['model']})"

    def save(fig, name):
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        files.append(p)

    # 1. frames overview (3D)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    i = len(t) // 2
    J = ctx["Jw"][i]
    for f in FR.FINGERS:
        ch = [0] + FR.CHAIN[f]
        ax.plot(J[ch, 0], J[ch, 1], J[ch, 2], color=COL[f], lw=2)
    ax.plot(J[[5, 9, 13, 17], 0], J[[5, 9, 13, 17], 1], J[[5, 9, 13, 17], 2], color="#868e96", lw=1.5)

    def triad(R, o, L, lw=2):
        for k, c in enumerate(("#e03131", "#2f9e44", "#1971c2")):
            e = o + L * R[:, k]
            ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]], color=c, lw=lw)
    triad(ctx["Rp"][i], ctx["op"][i], 40, 3)
    for f in FR.FINGERS:
        R, o = ctx["ff"][f]
        triad(R[i], o[i], 18, 1.4)
    c = J.mean(0)
    r = 110
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("X world (mm)")
    ax.set_ylabel("Y world (mm)")
    ax.set_zlabel("Z world (mm)")
    ax.set_title("Wrist frame (thick) and rigid-finger frames  x red, y green, z blue\n" + title, fontsize=7)
    save(fig, "01_frames_overview.png")
    if cb:
        cb(0.15, "Figures: fingertip motion")

    # 2-3. fingertip motion in the wrist frame (band) - model vs hybrid
    for k, meth in enumerate(("model", "hybrid")):
        fig, axs = plt.subplots(5, 3, figsize=(9, 8), sharex=True)
        for r_, f in enumerate(FR.FINGERS):
            for c_, a in enumerate("xyz"):
                e = series[meth][f]["tip"][a]
                axs[r_, c_].plot(t, e["band"], color=COL[f], lw=0.6)
                if r_ == 0:
                    axs[r_, c_].set_title(f"wrist-frame {a}")
                if c_ == 0:
                    axs[r_, c_].set_ylabel(f"{f}\n(mm)")
        for c_ in range(3):
            axs[-1, c_].set_xlabel("time (s)")
        fig.suptitle(f"Fingertip motion relative to the wrist, {lo:g}-{hi:g} Hz band ({meth})\n{title}", fontsize=8)
        save(fig, f"0{2 + k}_fingertip_band_{meth}.png")
    if cb:
        cb(0.35, "Figures: angles and spectra")

    # 4. finger angles (model)
    fig, axs = plt.subplots(5, 3, figsize=(9, 8), sharex=True)
    for r_, f in enumerate(FR.FINGERS):
        for c_, a in enumerate(("flexion", "abduction", "twist")):
            e = series["model"][f]["ang"].get(a)
            if e is not None:
                axs[r_, c_].plot(t, e["raw"], color=COL[f], lw=0.6)
            if r_ == 0:
                axs[r_, c_].set_title(a)
            if c_ == 0:
                axs[r_, c_].set_ylabel(f"{f}\n(deg)")
    fig.suptitle("Rigid-finger orientation relative to the wrist frame (model)\n" + title, fontsize=8)
    save(fig, "04_finger_angles.png")

    # 5. spectra with noise floors
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    f_ax = np.asarray(ctx["f"])
    for k, b in enumerate(["wrist"] + FR.FINGERS):
        ax = axs.ravel()[k]
        for meth, ls in (("model", "-"), ("hybrid", "-")):
            if b == "wrist":
                g = "pos"
            else:
                g = "tip"
            d = series[meth][b][g]
            P = sum(np.asarray(e["psd"]) for e in d.values())
            Pn = [np.asarray(e["noise"]) for e in d.values() if e.get("noise") is not None]
            col = "#1c7ed6" if meth == "model" else "#e8590c"
            ax.semilogy(f_ax, P, color=col, lw=1.1, label=meth)
            if Pn:
                ax.semilogy(f_ax, sum(Pn), color=col, lw=0.8, ls="--", label=f"{meth} noise")
        ax.axvspan(lo, hi, color="#ffd43b", alpha=0.18)
        ax.set_title(b)
        ax.set_xlabel("Hz")
        ax.set_ylabel("PSD (mm²/Hz)")
        ax.set_xlim(0, f_ax[-1])
        if k == 0:
            ax.legend(fontsize=6, frameon=False)
    fig.suptitle("Power spectra (sum over axes); dashed = noise floor; shaded = tremor band\n" + title, fontsize=8)
    fig.tight_layout()
    save(fig, "05_spectra_noise_floor.png")
    if cb:
        cb(0.6, "Figures: spectrograms and coherence")

    # 6. spectrograms (hybrid + model, principal axis)
    sp = ctx["spectrograms"]
    fig, axs = plt.subplots(2, 6, figsize=(13, 4.4), sharey=True)
    for r_, meth in enumerate(("model", "hybrid")):
        for c_, b in enumerate(["wrist_pos"] + FR.FINGERS):
            v = sp[meth][b]
            ax = axs[r_, c_]
            ax.pcolormesh(v["t"], v["f"], v["db"], shading="auto", cmap="magma")
            ax.axhline(lo, color="w", lw=0.4, ls=":")
            ax.axhline(hi, color="w", lw=0.4, ls=":")
            ax.set_title(f"{b} ({meth})", fontsize=7)
            if c_ == 0:
                ax.set_ylabel("Hz")
            if r_ == 1:
                ax.set_xlabel("s")
    fig.suptitle("Spectrograms of the principal tremor axis\n" + title, fontsize=8)
    fig.tight_layout()
    save(fig, "06_spectrograms.png")

    # 7. coherence
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for k, meth in enumerate(("model", "hybrid")):
        M = ctx["coh"][meth]
        im = axs[k].imshow(M, vmin=0, vmax=1, cmap="viridis")
        axs[k].set_xticks(range(len(ctx["bodies"])), ctx["bodies"], rotation=45, ha="right")
        axs[k].set_yticks(range(len(ctx["bodies"])), ctx["bodies"])
        for i_ in range(len(M)):
            for j_ in range(len(M)):
                axs[k].text(j_, i_, f"{M[i_, j_]:.2f}", ha="center", va="center", fontsize=6,
                            color="w" if M[i_, j_] < 0.6 else "k")
        axs[k].set_title(f"{meth}: mean coherence {lo:g}-{hi:g} Hz")
    fig.colorbar(im, ax=axs, shrink=0.8)
    save(fig, "07_coherence.png")

    # 8. ellipsoids (projections in the wrist frame)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    planes = [(0, 1, "x", "y"), (1, 2, "y", "z"), (0, 2, "x", "z")]
    th = np.linspace(0, 2 * np.pi, 100)
    for k, (a, b, na, nb) in enumerate(planes):
        ax = axs[k]
        for f in FR.FINGERS:
            for meth, ls in (("model", "-"), ("hybrid", "--")):
                e = series[meth][f]["tip"]
                X = np.stack([np.asarray(e[x]["band"]) for x in "xyz"], 1)[:, [a, b]]
                C2 = np.cov(X.T)
                w, V = np.linalg.eigh(C2)
                pts = (V @ (2 * np.sqrt(np.maximum(w, 0))[:, None] * np.stack([np.cos(th), np.sin(th)])))
                ax.plot(pts[0], pts[1], color=COL[f], ls=ls, lw=0.9, label=f"{f}" if (k == 0 and meth == "model") else None)
        ax.set_aspect("equal")
        ax.set_xlabel(f"wrist-frame {na} (mm)")
        ax.set_ylabel(f"{nb} (mm)")
    axs[0].legend(fontsize=6, frameon=False)
    fig.suptitle("2-sigma tremor ellipses of the fingertips in the wrist frame (solid = model, dashed = hybrid)\n" +
                 title, fontsize=8)
    fig.tight_layout()
    save(fig, "08_tremor_ellipses.png")
    if cb:
        cb(0.85, "Figures: velocity, wrist, joints")

    # 9. velocity / acceleration (hybrid, principal axis per finger)
    fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for f in FR.FINGERS:
        e = series["hybrid"][f]["tip"]
        v = np.sqrt(sum(np.asarray(e[a]["vel"]) ** 2 for a in "xyz"))
        acc = np.sqrt(sum(np.asarray(e[a]["acc"]) ** 2 for a in "xyz"))
        axs[0].plot(t, v, color=COL[f], lw=0.6, label=f)
        axs[1].plot(t, acc / 1000.0, color=COL[f], lw=0.6)
    axs[0].set_ylabel("tremor speed (mm/s)")
    axs[1].set_ylabel("tremor acceleration (m/s²)")
    axs[1].set_xlabel("time (s)")
    axs[0].legend(fontsize=6, ncol=5, frameon=False)
    fig.suptitle("Fingertip tremor velocity and acceleration relative to the wrist (hybrid, band)\n" + title, fontsize=8)
    save(fig, "09_velocity_acceleration.png")

    # 10. wrist motion in the world
    fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for a, c in zip("xyz", ("#e03131", "#2f9e44", "#1971c2")):
        e = series["model"]["wrist"]["pos"][a]
        axs[0].plot(t, np.asarray(e["raw"]) - np.mean(e["raw"]), color=c, lw=0.7, label=a)
        r = series["model"]["wrist"]["rot"][a]
        axs[1].plot(t, r["raw"], color=c, lw=0.7, label=f"about palm {a}")
    axs[0].set_ylabel("wrist position - mean (mm, world)")
    axs[1].set_ylabel("wrist rotation (deg)")
    axs[1].set_xlabel("time (s)")
    axs[0].legend(fontsize=6, frameon=False, ncol=3)
    axs[1].legend(fontsize=6, frameon=False, ncol=3)
    fig.suptitle("Wrist (hand as a whole) in the world frame (model)\n" + title, fontsize=8)
    save(fig, "10_wrist_world.png")

    # 11. joint angles
    fig, axs = plt.subplots(5, 1, figsize=(9, 8), sharex=True)
    for r_, f in enumerate(FR.FINGERS):
        for jn, ls in zip(FR.JOINT_NAMES[f], ("-", "--", ":")):
            axs[r_].plot(t, ctx["jang"][f][jn], color=COL[f], ls=ls, lw=0.8, label=jn.upper())
        axs[r_].set_ylabel(f"{f}\n(deg)")
        axs[r_].legend(fontsize=6, frameon=False, ncol=3, loc="upper right")
    axs[-1].set_xlabel("time (s)")
    fig.suptitle("Joint flexion angles (model)\n" + title, fontsize=8)
    save(fig, "11_joint_angles.png")
    return files


def write_all(out: dict, res: dict, series: dict, ctx: dict, cb=None) -> list[str]:
    os.makedirs(out["exports"], exist_ok=True)
    os.makedirs(out["figures"], exist_ok=True)
    files = []
    cb and cb(0.0, "Writing CSV tables")
    files += write_tables(out["exports"], res, series, ctx)
    cb and cb(0.15, "Writing MATLAB, BVH and glTF")
    try:
        write_mat(os.path.join(out["exports"], "kinematics.mat"), res, series, ctx)
        files.append(os.path.join(out["exports"], "kinematics.mat"))
    except Exception as e:  # noqa: BLE001
        print("MAT export failed:", e, flush=True)
    write_bvh(os.path.join(out["exports"], "hand.bvh"), ctx)
    write_glb(os.path.join(out["exports"], "hand_animation.glb"), ctx)
    files += [os.path.join(out["exports"], "hand.bvh"), os.path.join(out["exports"], "hand_animation.glb")]
    files += write_figures(out["figures"], res, series, ctx,
                           (lambda fr, msg: cb(0.25 + 0.75 * fr, msg)) if cb else None)
    return [os.path.relpath(f, out["dir"]).replace("\\", "/") for f in files]
