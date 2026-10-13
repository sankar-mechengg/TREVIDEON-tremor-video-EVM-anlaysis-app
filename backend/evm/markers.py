"""Marker CSV: what happened when during a video (for later stress / tremor correlation).

Format (UTF-8 CSV, header required; one row per event):

    time_s,end_s,marker,value,notes
    0.0,,SESSION_START,,
    12.5,72.5,CONDITION,High,graded stress block
    30.2,,FAULT,,ring touched the wire
    45.0,,STRESS_RATING,7,self-report 1-10

  time_s   required  event time from the START OF THE VIDEO: seconds (12.5) or m:ss(.ms) / h:mm:ss(.ms)
  end_s    optional  end time (same formats); a row with end_s is a SEGMENT, without it an EVENT
  marker   required  label; suggested: SESSION_START, SESSION_END, CONDITION, BASELINE, TASK, FAULT,
                     STRESS_RATING, NOTE (any other label is accepted)
  value    optional  free value: condition name, rating, ...
  notes    optional  free text
"""
from __future__ import annotations

import csv
import io
import re

import numpy as np
from scipy.signal import welch

TEMPLATE = """time_s,end_s,marker,value,notes
0.0,,SESSION_START,,video starts
5.0,65.0,BASELINE,,hand held still - no task
70.0,130.0,CONDITION,Low,low-difficulty block
75.4,,FAULT,,ring touched the wire
131.0,,STRESS_RATING,3,self-report 1-10
140.0,200.0,CONDITION,High,high-difficulty block
152.8,,FAULT,,ring touched the wire
201.0,,STRESS_RATING,8,self-report 1-10
205.0,,SESSION_END,,
"""
COLUMNS = ["time_s", "end_s", "marker", "value", "notes"]
SUGGESTED = ["SESSION_START", "SESSION_END", "CONDITION", "BASELINE", "TASK", "FAULT", "STRESS_RATING", "NOTE"]
_TIME = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def parse_time(v: str) -> float | None:
    v = (v or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        pass
    m = _TIME.match(v)
    if not m:
        raise ValueError(f"'{v}' is not a time (use seconds like 12.5 or m:ss like 1:12.5)")
    h, mi, s = m.groups()
    return (int(h) if h else 0) * 3600 + int(mi) * 60 + float(s)


def parse(text: str, duration_s: float | None = None) -> dict:
    """Returns {"rows": [...], "errors": [...], "warnings": [...]}; rows are sorted by time."""
    errors, warnings, rows = [], [], []
    text = text.lstrip("﻿")
    try:
        reader = csv.DictReader(io.StringIO(text))
        header = [h.strip().lower() for h in (reader.fieldnames or [])]
    except csv.Error as e:
        return {"rows": [], "errors": [f"Not a valid CSV file: {e}"], "warnings": []}
    for need in ("time_s", "marker"):
        if need not in header:
            errors.append(f"Missing required column '{need}'. Header must be: {','.join(COLUMNS)}")
    if errors:
        return {"rows": [], "errors": errors, "warnings": []}
    extra = [h for h in header if h not in COLUMNS]
    if extra:
        warnings.append(f"Ignored unknown column(s): {', '.join(extra)}")
    for i, raw in enumerate(reader, start=2):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        if not any(r.values()):
            continue
        try:
            t = parse_time(r.get("time_s", ""))
            e = parse_time(r.get("end_s", ""))
        except ValueError as ex:
            errors.append(f"Line {i}: {ex}")
            continue
        if t is None:
            errors.append(f"Line {i}: time_s is empty")
            continue
        if t < 0:
            errors.append(f"Line {i}: time_s must not be negative")
            continue
        if e is not None and e <= t:
            errors.append(f"Line {i}: end_s ({e}) must be after time_s ({t})")
            continue
        if not r.get("marker"):
            errors.append(f"Line {i}: marker is empty")
            continue
        if duration_s and t > duration_s + 0.5:
            warnings.append(f"Line {i}: {r['marker']} at {t:.2f} s is after the end of the video ({duration_s:.1f} s)")
        rows.append({"line": i, "time_s": round(t, 4), "end_s": None if e is None else round(e, 4),
                     "marker": r["marker"], "value": r.get("value", ""), "notes": r.get("notes", "")})
    rows.sort(key=lambda x: x["time_s"])
    if not rows and not errors:
        errors.append("No marker rows found.")
    return {"rows": rows, "errors": errors, "warnings": warnings}


def segment_metrics(rows: list, analysis: dict, Db: np.ndarray, event_window_s: float = 2.0) -> list[dict]:
    """Tremor metrics inside every marker segment (rows with end_s) and in a window centred on every event,
    restricted to the analysed part of the video."""
    fps = float(analysis["fps"])
    a0 = float(analysis["start_s"])
    T = Db.shape[0]
    a1 = a0 + T / fps
    lo, hi = analysis["band"]
    hand_len = float((analysis.get("isolation") or {}).get("hand_len_px") or analysis.get("hand_size_px") or 1.0)
    env_t = np.asarray(analysis["envelope"]["t"], float) + a0
    env_h = np.asarray(analysis["envelope"]["series"]["hand"], float)
    tips = Db[:, :5]                                    # fingertip band-passed displacement (T, 5, 2)
    out = []
    for r in rows:
        if r["end_s"] is not None:
            s0, s1, kind = r["time_s"], r["end_s"], "segment"
        else:
            s0, s1, kind = r["time_s"] - event_window_s / 2, r["time_s"] + event_window_s / 2, "event"
        c0, c1 = max(s0, a0), min(s1, a1)
        row = {"marker": r["marker"], "value": r["value"], "kind": kind, "time_s": r["time_s"], "end_s": r["end_s"],
               "window_start_s": round(s0, 3), "window_end_s": round(s1, 3), "analysed_s": round(max(0.0, c1 - c0), 3),
               "fingertip_rms_px": None, "hand_envelope_px": None, "rms_pct_hand_len": None, "peak_hz": None}
        if c1 - c0 >= 0.5:
            i0, i1 = int((c0 - a0) * fps), int((c1 - a0) * fps)
            seg = tips[i0:i1]
            mag = np.linalg.norm(seg, axis=-1)
            rms = float(np.sqrt(np.mean(mag ** 2)))
            row["fingertip_rms_px"] = round(rms, 4)
            row["rms_pct_hand_len"] = round(100 * rms / hand_len, 4)
            m = (env_t >= c0) & (env_t <= c1)
            if m.any():
                row["hand_envelope_px"] = round(float(np.mean(env_h[m])), 4)
            n = seg.shape[0]
            if n >= int(1.5 * fps):
                x = seg.reshape(n, -1)
                f, p = welch(x, fs=fps, nperseg=min(n, int(2 * fps)), axis=0)
                p = p.sum(axis=1) * (2 * np.pi * f) ** 2
                band = (f >= lo) & (f <= hi)
                if band.any():
                    row["peak_hz"] = round(float(f[band][np.argmax(p[band])]), 2)
        out.append(row)
    return out


def to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
