"""TREVIDEON GPU worker on Modal (https://modal.com).

The hosted API (Railway, CPU) keeps uploads and sessions; with TREVIDEON_EXECUTOR=modal every analysis job
is executed here on a GPU and the results are copied back. Nothing is kept: the job folder on the Modal
Volume is deleted as soon as the API has fetched the results.

Deploy (from the backend folder, after `pip install modal` and `modal token new`):
    modal deploy modal_app.py

3D Kinematics with WiLoR on the worker (optional): the WiLoR code is fetched while the image is built; the
checkpoint and the MANO model are NOT in the image. Put them on the "trevideon-models" Volume yourself:
    modal volume create trevideon-models
    modal volume put trevideon-models models/wilor_final.ckpt /wilor_final.ckpt
    modal volume put trevideon-models <path>/MANO_RIGHT.pkl /mano/MANO_RIGHT.pkl
MANO is licensed for non-commercial research only and may not be made available to third parties: check its
licence (and WiLoR's CC BY-NC-ND licence) before offering this on a public service. The worker returns joints
and rotations only - never the MANO mesh. Without the Volume the 3D tab falls back to MediaPipe.

Status: written against the Modal Python API (Functions, Volumes, Dicts); not yet run on a Modal account.
"""
import modal

APP_NAME = "trevideon"
GPU = "L4"                     # 24 GB; "T4" is cheaper, "A10G" / "L40S" faster

app = modal.App(APP_NAME)
jobs_volume = modal.Volume.from_name("trevideon-jobs", create_if_missing=True)
models_volume = modal.Volume.from_name("trevideon-models", create_if_missing=True)
progress = modal.Dict.from_name("trevideon-progress", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "curl")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .env({"SAM2_BUILD_CUDA": "0"})
    .pip_install_from_requirements("requirements.txt")
    .pip_install_from_requirements("requirements-wilor.txt")
    .run_commands(
        "mkdir -p /root/third_party && curl -sSL -o /tmp/wilor.zip https://github.com/rolpotamias/WiLoR/archive/refs/heads/main.zip"
        " && cd /root/third_party && python -c \"import zipfile; zipfile.ZipFile('/tmp/wilor.zip').extractall('.')\""
        " && mv WiLoR-main WiLoR && rm /tmp/wilor.zip",
        "mkdir -p /root/models",
        "curl -sSL -o /root/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "curl -sSL -o /root/models/selfie_multiclass_256x256.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        "curl -sSL -o /root/models/sam2.1_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    )
    .env({"TREVIDEON_WILOR_DIR": "/root/third_party/WiLoR", "TREVIDEON_WILOR_CKPT": "/models/wilor_final.ckpt",
          "TREVIDEON_MANO_DIR": "/models/mano"})
    .add_local_dir("evm", "/root/evm")
)


@app.function(gpu=GPU, image=image, volumes={"/jobs": jobs_volume, "/models": models_volume}, timeout=3 * 3600,
              max_containers=4)
def process(job_id: str, video_name: str, settings: dict, fps: float, kind: str = "evm") -> dict:
    import sys

    sys.path.insert(0, "/root")
    from evm.pipeline import run_job
    from evm.kin3d.pipeline import run_job3d

    jobs_volume.reload()
    spec = {"video_path": f"/jobs/{job_id}/{video_name}", "result_dir": f"/jobs/{job_id}/results",
            "settings": settings, "fps": fps, "kind": kind, "export_mesh": False}

    def report(d):
        progress[job_id] = d

    def cancelled():
        return bool(progress.get(f"{job_id}:cancel", False))

    try:
        out = (run_job3d if kind == "kin3d" else run_job)(spec, report, cancelled)
    finally:
        jobs_volume.commit()
    return out
