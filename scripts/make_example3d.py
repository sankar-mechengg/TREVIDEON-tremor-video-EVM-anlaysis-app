"""Package a 3D-kinematics result as the home-page 3D example (web-sized segmented video + slim JSON).

    backend/.venv/Scripts/python scripts/make_example3d.py <result_dir>/k3d/<key> --name p06-high-3d [--crf 28]

Writes frontend/public/examples/<name>/{segmented.mp4, kinematics.json}. The MANO mesh is never included.
Only use recordings of people who agreed to their hands being shown publicly.
"""
import argparse
import json
import os
import subprocess

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "..", "frontend", "public", "examples")
KEEP = ["version", "video", "fps", "n_frames", "start_s", "duration_s", "model", "hand", "band", "world", "scale",
        "observability", "noise", "frame_repair", "display", "ellipsoids", "metrics", "t"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="folder of one 3D result (contains kinematics.json and display/)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--crf", type=int, default=28)
    a = ap.parse_args()
    out = os.path.join(OUT_ROOT, a.name)
    os.makedirs(out, exist_ok=True)
    disp = sorted(f for f in os.listdir(os.path.join(a.result, "display")) if f.endswith(".mp4"))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i",
                    os.path.join(a.result, "display", disp[-1]), "-an", "-vf", "scale=540:540", "-c:v", "libx264",
                    "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    os.path.join(out, "segmented.mp4")], check=True)
    with open(os.path.join(a.result, "kinematics.json")) as fh:
        res = json.load(fh)
    slim = {k: res[k] for k in KEEP if k in res}
    slim["metrics"] = {m: v for m, v in res["metrics"].items() if m in ("hybrid", "model")}
    slim["mesh"] = {"available": False}
    slim["series"] = {"hybrid": {f: {"tip": {ax: {"band": res["series"]["hybrid"][f]["tip"][ax]["band"]}
                                            for ax in ("x", "y")}}
                                 for f in ("thumb", "index", "middle", "ring", "pinky")}}
    with open(os.path.join(out, "kinematics.json"), "w") as fh:
        json.dump(slim, fh, separators=(",", ":"))
    for f in sorted(os.listdir(out)):
        print(f"{f:18s} {os.path.getsize(os.path.join(out, f)) / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
