"""Where videos come from and where results go.

LocalLibrary  (TREVIDEON_MODE=local)   a folder of videos on this computer (chosen with a folder dialog or
              -DataDir); results are kept in a sibling `results/` folder and reused as a cache.
              Videos named P##_B#_<Level>.mp4 get results/P##/B#/<Level>/, others results/<stem>/.
SessionStore  (TREVIDEON_MODE=hosted)  uploads live in a temporary folder per browser session
              (random 128-bit id); everything is deleted after `ttl_min` minutes without activity,
              on request, or when the server restarts. Nothing is written anywhere else.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
PID_RE = re.compile(r"^(?P<pid>P\d{2})_(?P<session>B\d+)_(?P<level>[A-Za-z]+)$")
SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_name(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/")).strip()
    base = SAFE_NAME.sub("_", base)[:120]
    return base or "video.mp4"


def result_rel(video_name: str) -> str:
    stem = os.path.splitext(video_name)[0]
    m = PID_RE.match(stem)
    if m:
        return os.path.join(m["pid"], m["session"], m["level"])
    return stem


def video_meta(video_name: str) -> dict | None:
    m = PID_RE.match(os.path.splitext(video_name)[0])
    return m.groupdict() if m else None


def default_results_for(data_dir: str) -> str:
    """Sibling results folder: .../data/raw -> .../results ; any other folder -> <folder>/../results."""
    p = Path(data_dir).resolve()
    if p.name.lower() == "raw" and p.parent.name.lower() == "data":
        return str(p.parent.parent / "results")
    return str(p.parent / "results")


class LocalLibrary:
    mode = "local"

    def __init__(self, data_dir: str, results_dir: str | None = None):
        self.prefs_path = Path.home() / ".trevideon" / "local.json"
        self.lock = threading.Lock()
        prefs = self._load_prefs()
        self.recent: list[str] = prefs.get("recent", [])
        self.custom_results: dict[str, str] = prefs.get("results", {})
        self.set_data_dir(data_dir)
        if results_dir:            # start-up override (run.ps1 -ResultsDir / env): this session only, not saved
            self.results_dir = str(Path(results_dir).resolve())
            os.makedirs(self.results_dir, exist_ok=True)

    # ---- preferences (only on this computer)
    def _load_prefs(self) -> dict:
        try:
            return json.loads(self.prefs_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _save_prefs(self):
        try:
            self.prefs_path.parent.mkdir(parents=True, exist_ok=True)
            self.prefs_path.write_text(json.dumps({"recent": self.recent[:10], "results": self.custom_results}),
                                       encoding="utf-8")
        except OSError:
            pass

    def set_data_dir(self, data_dir: str, results_dir: str | None = None):
        d = str(Path(data_dir).resolve())
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Folder not found: {d}")
        with self.lock:
            self.data_dir = d
            if results_dir:
                self.custom_results[d] = str(Path(results_dir).resolve())
            self.results_dir = self.custom_results.get(d) or default_results_for(d)
            os.makedirs(self.results_dir, exist_ok=True)
            self.recent = [d] + [r for r in self.recent if r != d]
            self._save_prefs()

    def info(self, ctx=None) -> dict:
        return {"mode": self.mode, "data_dir": self.data_dir, "results_dir": self.results_dir, "recent": self.recent[:10]}

    # ---- videos
    def videos(self, ctx=None) -> list[str]:
        try:
            return sorted(f for f in os.listdir(self.data_dir)
                          if f.lower().endswith(VIDEO_EXT) and os.path.isfile(os.path.join(self.data_dir, f)))
        except OSError:
            return []

    def video_path(self, name: str, ctx=None) -> str | None:
        if name not in self.videos():
            return None
        return os.path.join(self.data_dir, name)

    def result_dir(self, name: str, ctx=None) -> str:
        d = os.path.join(self.results_dir, result_rel(name))
        os.makedirs(d, exist_ok=True)
        return d

    def thumbs_dir(self, ctx=None) -> str:
        d = os.path.join(self.results_dir, "_thumbs")
        os.makedirs(d, exist_ok=True)
        return d

    def markers_path(self, name: str, ctx=None) -> str:
        return os.path.join(self.result_dir(name), "markers.csv")

    def touch(self, ctx=None):
        pass


class SessionStore:
    mode = "hosted"

    def __init__(self, root: str, ttl_min: float = 60.0):
        self.root = root
        self.ttl = ttl_min * 60.0
        os.makedirs(root, exist_ok=True)
        # nothing survives a restart
        for d in os.listdir(root):
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)
        self.last: dict[str, float] = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._reaper, daemon=True).start()

    def create(self) -> str:
        sid = secrets.token_hex(16)
        for sub in ("videos", "results", "uploads"):
            os.makedirs(os.path.join(self.root, sid, sub), exist_ok=True)
        self.last[sid] = time.time()
        return sid

    def valid(self, sid: str | None) -> bool:
        return bool(sid) and re.fullmatch(r"[0-9a-f]{32}", sid or "") is not None and sid in self.last

    def touch(self, sid):
        if sid in self.last:
            self.last[sid] = time.time()

    def delete(self, sid: str):
        with self.lock:
            self.last.pop(sid, None)
        shutil.rmtree(os.path.join(self.root, sid), ignore_errors=True)

    def _reaper(self):
        while True:
            time.sleep(60)
            now = time.time()
            for sid, t in list(self.last.items()):
                if now - t > self.ttl:
                    self.delete(sid)

    def expires_in(self, sid) -> float:
        return max(0.0, self.ttl - (time.time() - self.last.get(sid, 0)))

    def info(self, sid=None) -> dict:
        return {"mode": self.mode, "ttl_min": self.ttl / 60.0,
                "expires_in_s": round(self.expires_in(sid)) if sid else None}

    def base(self, sid) -> str:
        return os.path.join(self.root, sid)

    def videos(self, sid) -> list[str]:
        d = os.path.join(self.base(sid), "videos")
        try:
            return sorted(f for f in os.listdir(d) if f.lower().endswith(VIDEO_EXT))
        except OSError:
            return []

    def video_path(self, name: str, sid=None) -> str | None:
        if name not in self.videos(sid):
            return None
        return os.path.join(self.base(sid), "videos", name)

    def result_dir(self, name: str, sid=None) -> str:
        d = os.path.join(self.base(sid), "results", os.path.splitext(name)[0])
        os.makedirs(d, exist_ok=True)
        return d

    def thumbs_dir(self, sid=None) -> str:
        d = os.path.join(self.base(sid), "results", "_thumbs")
        os.makedirs(d, exist_ok=True)
        return d

    def markers_path(self, name: str, sid=None) -> str:
        return os.path.join(self.result_dir(name, sid), "markers.csv")

    def upload_dir(self, sid) -> str:
        return os.path.join(self.base(sid), "uploads")
