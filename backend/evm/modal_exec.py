"""Executor that runs jobs on the Modal GPU worker (modal_app.py) instead of this machine.

Used by the hosted API when TREVIDEON_EXECUTOR=modal (needs `pip install modal` and the MODAL_TOKEN_ID /
MODAL_TOKEN_SECRET environment variables). Flow per job:
  1. upload the video to the Modal Volume under /<job id>/
  2. spawn modal_app.process on a GPU; mirror its progress (Modal Dict) into the job record
  3. copy /<job id>/results back into the session's result folder, then delete /<job id> from the Volume

Status: written against the Modal Python API; not yet run on a Modal account.
"""
from __future__ import annotations

import os
import threading
import time

APP_NAME = "trevideon"


class ModalExecutor:
    name = "modal"

    def __init__(self):
        import modal
        self.modal = modal
        self.fn = modal.Function.from_name(APP_NAME, "process")
        self.vol = modal.Volume.from_name("trevideon-jobs", create_if_missing=True)
        self.progress = modal.Dict.from_name("trevideon-progress", create_if_missing=True)

    def make_event(self):
        return threading.Event()

    def _download(self, remote_dir: str, local_dir: str):
        for entry in self.vol.listdir(remote_dir, recursive=True):
            if entry.type != self.modal.volume.FileEntryType.FILE:
                continue
            rel = os.path.relpath(entry.path, remote_dir.lstrip("/"))
            dst = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst + ".part", "wb") as fh:
                for chunk in self.vol.read_file(entry.path):
                    fh.write(chunk)
            os.replace(dst + ".part", dst)

    def run(self, job: dict, update) -> str:
        spec = job["spec"]
        jid = job["id"]
        name = os.path.basename(spec["video_path"])
        update({"stage": "upload", "message": "Sending the video to the GPU worker", "progress": 0.0})
        with self.vol.batch_upload(force=True) as batch:
            batch.put_file(spec["video_path"], f"/{jid}/{name}")
        call = self.fn.spawn(jid, name, spec["settings"], spec["fps"], spec.get("kind", "evm"))
        result = None
        try:
            while True:
                try:
                    result = call.get(timeout=1.0)
                    break
                except TimeoutError:
                    pass
                d = self.progress.get(jid)
                if d:
                    update({k: v for k, v in d.items() if k != "state"})
                if job["cancel"].is_set():
                    self.progress[f"{jid}:cancel"] = True
                    call.cancel()
                    update({"state": "cancelled", "message": "Cancelled"})
                    return "cancelled"
            update({"stage": "download", "message": "Fetching the results", "progress": 0.99})
            self._download(f"/{jid}/results", spec["result_dir"])
            update({"state": "done", "progress": 1.0, "stage": "done", "message": "Done", **(result or {})})
            return "done"
        except Exception as e:  # noqa: BLE001
            update({"state": "error", "message": f"{type(e).__name__}: {e}"})
            return "error"
        finally:
            for k in (jid, f"{jid}:cancel"):
                try:
                    self.progress.pop(k)
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.vol.remove_file(f"/{jid}", recursive=True)
            except Exception:  # noqa: BLE001
                time.sleep(0)
