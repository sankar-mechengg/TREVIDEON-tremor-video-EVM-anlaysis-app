"""Package a processed result as the home-page example (web-sized videos + analysis JSON).

    backend/.venv/Scripts/python scripts/make_example.py <result_dir>/<key_a> --name p06-high [--crf 27]

Writes frontend/public/examples/<name>/{original,magnified,overlay}.mp4 and analysis.json.
Only use recordings of people who agreed to their hands being shown publicly.
"""
import argparse
import json
import os
import shutil
import subprocess

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "..", "frontend", "public", "examples")


def encode(src, dst, crf, height=720):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, "-an",
           "-vf", f"scale=-2:'min({height},ih)'", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="folder of one processed result (contains analysis.json)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--crf", type=int, default=27)
    a = ap.parse_args()
    out = os.path.join(OUT_ROOT, a.name)
    os.makedirs(out, exist_ok=True)
    encode(os.path.join(a.result, "original.mp4"), os.path.join(out, "original.mp4"), a.crf)
    encode(os.path.join(a.result, "magnified.mp4"), os.path.join(out, "magnified.mp4"), a.crf)
    ov = [f for f in os.listdir(os.path.join(a.result, "overlays")) if f.endswith(".mp4")]
    if ov:
        encode(os.path.join(a.result, "overlays", sorted(ov)[-1]), os.path.join(out, "overlay.mp4"), a.crf)
    with open(os.path.join(a.result, "analysis.json")) as fh:
        res = json.load(fh)
    res.pop("trace", None)            # fingertip micro-trajectories are not shown on the home page
    with open(os.path.join(out, "analysis.json"), "w") as fh:
        json.dump(res, fh, separators=(",", ":"))
    for f in sorted(os.listdir(out)):
        print(f"{f:16s} {os.path.getsize(os.path.join(out, f)) / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
