"""Job queue: one analysis at a time, visible queue position, cancellation.

Executors
  local  - the pipeline runs in a spawned worker process on this machine (CPU and/or local GPU)
  modal  - the pipeline runs remotely on a Modal GPU container (see modal_app.py); the API server only
           ships the video there and fetches the results back
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
import uuid

from .pipeline import worker_main


class LocalExecutor:
    name = "local"

    def __init__(self):
        self.ctx = mp.get_context("spawn")

    def make_event(self):
        return self.ctx.Event()

    def run(self, job: dict, update) -> str:
        q = self.ctx.Queue()
        p = self.ctx.Process(target=worker_main, args=(job["spec"], q, job["cancel"]), daemon=True)
        p.start()
        final = None
        while True:
            try:
                msg = q.get(timeout=0.5)
                if "trace" in msg:
                    print(msg["trace"], flush=True)
                update({k: v for k, v in msg.items() if k != "trace"})
                if msg.get("state") in ("done", "error", "cancelled"):
                    final = msg["state"]
                    break
            except queue.Empty:
                if not p.is_alive():
                    break
        p.join(timeout=10)
        if final is None:
            final = "cancelled" if job["cancel"].is_set() else "error"
            update({"state": final, "message": "Cancelled" if final == "cancelled"
                    else f"Worker exited (code {p.exitcode})"})
        return final


class JobManager:
    def __init__(self, executor):
        self.ex = executor
        self.jobs: dict[str, dict] = {}
        self.order: list[str] = []
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, owner: str | None, video: str, spec: dict, key_a: str, key_b: str) -> dict:
        with self.lock:
            for j in self.jobs.values():
                if j["owner"] == owner and j["key_a"] == key_a and j["key_b"] == key_b \
                        and j["state"] in ("queued", "running"):
                    return j
            jid = uuid.uuid4().hex[:12]
            job = {"id": jid, "owner": owner, "video": video, "key_a": key_a, "key_b": key_b, "spec": spec,
                   "state": "queued", "progress": 0.0, "stage": "queued", "message": "Waiting in queue",
                   "created": time.time(), "cancel": self.ex.make_event(),
                   "isolation": bool(spec["settings"].get("hand_isolation")), "kind": spec.get("kind", "evm")}
            self.jobs[jid] = job
            self.order.append(jid)
        self.q.put(jid)
        return job

    def cancel(self, jid: str):
        j = self.jobs.get(jid)
        if not j:
            return False
        j["cancel"].set()
        if j["state"] == "queued":
            j.update(state="cancelled", message="Cancelled")
        return True

    def drop_owner(self, owner: str):
        for j in self.jobs.values():
            if j["owner"] == owner and j["state"] in ("queued", "running"):
                self.cancel(j["id"])

    def position(self, jid: str) -> int:
        waiting = [k for k in self.order if self.jobs.get(k, {}).get("state") == "queued"]
        return waiting.index(jid) + 1 if jid in waiting else 0

    def public(self, j: dict) -> dict:
        out = {k: v for k, v in j.items() if k not in ("cancel", "spec", "owner")}
        out["queue_position"] = self.position(j["id"])
        return out

    def for_owner(self, owner: str | None) -> list[dict]:
        return [self.public(j) for j in sorted(self.jobs.values(), key=lambda j: j["created"], reverse=True)
                if j["owner"] == owner]

    def _loop(self):
        while True:
            jid = self.q.get()
            job = self.jobs.get(jid)
            if not job or job["state"] != "queued":
                continue
            job.update(state="running", message="Starting worker", started=time.time())
            try:
                self.ex.run(job, job.update)
            except Exception as e:  # noqa: BLE001
                job.update(state="error", message=f"{type(e).__name__}: {e}")
            job["finished"] = time.time()


def make_executor(kind: str):
    if kind == "modal":
        from .modal_exec import ModalExecutor
        return ModalExecutor()
    return LocalExecutor()


EXECUTOR_KIND = os.environ.get("TREVIDEON_EXECUTOR", "local")
