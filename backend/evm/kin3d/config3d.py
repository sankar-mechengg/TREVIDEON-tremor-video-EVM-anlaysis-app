"""Settings of the 3D Kinematics analysis (separate from the EVM settings).

Stages
  "process" -> stage K: 3D pose, frames, vibration, spectra, exports      (cached in <result>/k3d/<keyK>/)
  "display" -> stage V: the segmented-hand display video                  (<keyK>/display/<keyV>.mp4)
  "runtime" -> device; never part of a cache key
"""

K3D_VERSION = "1.2"
DISPLAY_VERSION = "1.0"

SCHEMA3D = [
    {
        "group": "Segment",
        "fields": [
            {"key": "start_s", "label": "Start time", "type": "number", "default": 5.0, "min": 0, "max": 100000,
             "step": 0.5, "unit": "s", "stage": "process", "help": "Where the analysed segment starts."},
            {"key": "duration_s", "label": "Duration", "type": "number", "default": 20.0, "min": 1, "max": 100000,
             "step": 1, "unit": "s", "stage": "process",
             "help": "Length of the analysed segment (3D pose is estimated in every frame)."},
            {"key": "repair_duplicates", "label": "Repair duplicated frames", "type": "bool", "default": True,
             "stage": "process",
             "help": "Replace repeated frames (screen / VR recordings) by motion-interpolated frames; repeated "
                     "frames otherwise create a comb of false spectral peaks."},
        ],
    },
    {
        "group": "3D hand model",
        "fields": [
            {"key": "model", "label": "3D pose model", "type": "select", "default": "auto",
             "options": [["auto", "Auto (WiLoR if installed, else MediaPipe)"], ["wilor", "WiLoR + MANO (GPU)"],
                         ["mediapipe", "MediaPipe 3D landmarks (CPU)"]], "stage": "process",
             "help": "WiLoR (CVPR 2025) fits the MANO hand model to every frame: joints, mesh and joint rotations. "
                     "MediaPipe gives 21 keypoints with metric depth (no mesh, no finger twist). The other model is "
                     "always run as well for comparison."},
            {"key": "dominant_hand", "label": "Hand", "type": "select", "default": "auto",
             "options": [["auto", "Auto (largest, most persistent)"], ["right", "Right"], ["left", "Left"]],
             "stage": "process",
             "help": "Auto follows the largest hand and works out whether it is the right or left hand from the "
                     "positions of the two hands, or, with one hand visible, from the direction of its forearm "
                     "(MediaPipe's own label is a last resort: it is unreliable for the back of the hand)."},
            {"key": "view", "label": "Camera view", "type": "select", "default": "egocentric",
             "options": [["egocentric", "Egocentric (head / chest camera)"],
                         ["third_person", "Third person (camera facing the person)"]], "stage": "process",
             "help": "Used to tell the right hand from the left by image position: in an egocentric view the right "
                     "hand is on the right of the image; facing the person it is on the left."},
            {"key": "mirrored", "label": "Video is mirrored (selfie camera)", "type": "bool", "default": False,
             "stage": "process",
             "help": "Front / selfie cameras mirror the image; egocentric (head / chest) cameras do not."},
            {"key": "box_scale", "label": "Crop size", "type": "number", "default": 2.0, "min": 1.4, "max": 3.0,
             "step": 0.1, "unit": "x", "stage": "process",
             "help": "Hand crop given to WiLoR, relative to the landmark bounding box (WiLoR default 2.0). The crop "
                     "follows a smoothed hand path with a fixed size, so the crop itself does not jitter."},
            {"key": "hand_length_mm", "label": "Measured hand length", "type": "number", "default": 0, "min": 0,
             "max": 260, "step": 1, "unit": "mm", "stage": "process",
             "help": "Wrist crease to middle-finger tip. 0 = use the model's metric hand size (about 10 % "
                     "uncertainty). When given, all millimetre values are rescaled to it."},
        ],
    },
    {
        "group": "Coordinate frames",
        "fields": [
            {"key": "world_frame", "label": "World frame", "type": "select", "default": "stabilised",
             "options": [["stabilised", "Background-stabilised (head motion removed)"],
                         ["camera", "Camera (moves with a head-mounted camera)"]], "stage": "process",
             "help": "Stabilised: the static floor / background is tracked to remove camera (head) rotation; "
                     "world = the first-frame camera, turned so that Z points up from the floor when the floor plane "
                     "can be estimated. Head translation is not removed (it cannot be measured reliably without the "
                     "exact camera calibration). Finger motion relative to the wrist does not depend on this."},
            {"key": "camera_hfov_deg", "label": "Camera horizontal field of view", "type": "number", "default": 90,
             "min": 30, "max": 150, "step": 1, "unit": "deg", "stage": "process",
             "help": "Used to build the pinhole camera (focal length). VR passthrough / action cameras: 90-110; "
                     "phones (1x): about 65-70; webcams: about 60-78."},
            {"key": "floor_distance_m", "label": "Camera height above the floor", "type": "number", "default": 1.4,
             "min": 0.3, "max": 3.0, "step": 0.05, "unit": "m", "stage": "process",
             "help": "Only places the world origin on the floor below the first camera position (eye height when "
                     "standing, about 1.5 m; sitting, about 1.2 m)."},
            {"key": "camera_cutoff_hz", "label": "Head-rotation cut-off", "type": "number", "default": 2.0,
             "min": 0, "max": 10, "step": 0.5, "unit": "Hz", "stage": "process",
             "help": "Camera (head) rotation estimated from the background is low-passed at this frequency before it "
                     "is removed: faster background estimates are dominated by tracking noise. 0 = no smoothing."},
            {"key": "pose_cutoff_hz", "label": "Frame smoothing cut-off", "type": "number", "default": 3.0,
             "min": 0.5, "max": 8, "step": 0.5, "unit": "Hz", "stage": "process",
             "help": "The wrist frame used to express the optical-flow vibration, and the 'slow pose' of the 3D view, "
                     "are low-passed at this frequency (below the tremor band)."},
        ],
    },
    {
        "group": "Tremor band & noise floor",
        "fields": [
            {"key": "freq_lo", "label": "Band low cut", "type": "number", "default": 4.0, "min": 0.5, "max": 40,
             "step": 0.5, "unit": "Hz", "stage": "process", "fps_limit": True},
            {"key": "freq_hi", "label": "Band high cut", "type": "number", "default": 12.0, "min": 1, "max": 45,
             "step": 0.5, "unit": "Hz", "stage": "process", "fps_limit": True,
             "help": "Must stay below half the frame rate (15 Hz at 30 fps)."},
            {"key": "filter_order", "label": "Filter order", "type": "select", "default": 2,
             "options": [[1, "1"], [2, "2 (recommended)"], [3, "3"], [4, "4"]], "stage": "process"},
            {"key": "noise_floor", "label": "Measure the noise floor", "type": "bool", "default": True,
             "stage": "process",
             "help": "Re-runs the 3D model on slightly shifted crops of the same frames (model jitter) and measures "
                     "optical flow on static floor patches (flow noise). Shown as dashed lines on the spectra."},
            {"key": "n_aug", "label": "Noise-floor re-runs", "type": "number", "default": 4, "min": 2, "max": 8,
             "step": 1, "stage": "process", "requires": "noise_floor"},
        ],
    },
    {
        "group": "Display video",
        "fields": [
            {"key": "segmenter", "label": "Hand segmentation", "type": "select", "default": "auto",
             "options": [["auto", "Auto (SAM 2 on GPU, MediaPipe on CPU)"], ["sam2", "SAM 2.1 (GPU)"],
                         ["mediapipe", "MediaPipe skin + landmarks (CPU)"], ["landmarks", "Landmark polygon"]],
             "stage": "display", "help": "Only used for the segmented video shown next to the 3D view."},
            {"key": "display_px", "label": "Video size", "type": "select", "default": 720,
             "options": [[512, "512 x 512"], [720, "720 x 720"], [960, "960 x 960"]], "stage": "display"},
            {"key": "draw_axes", "label": "Draw the wrist frame", "type": "bool", "default": True, "stage": "display",
             "help": "Projects the wrist (palm) frame axes onto the video: x red, y green, z blue."},
        ],
    },
    {
        "group": "Compute",
        "fields": [
            {"key": "device", "label": "Device", "type": "select", "default": "auto",
             "options": [["auto", "Auto (GPU if available)"], ["gpu", "GPU (CUDA)"], ["cpu", "CPU"]],
             "stage": "runtime"},
        ],
    },
]

FIELDS3D = {f["key"]: f for g in SCHEMA3D for f in g["fields"]}
PROCESS3D = [k for k, f in FIELDS3D.items() if f["stage"] == "process"]
DISPLAY3D = [k for k, f in FIELDS3D.items() if f["stage"] == "display"]
RUNTIME3D = [k for k, f in FIELDS3D.items() if f["stage"] == "runtime"]
DEFAULTS3D = {k: f["default"] for k, f in FIELDS3D.items()}


def max_band_hz(fps: float) -> float:
    return round(0.45 * float(fps) * 2) / 2


def normalize3d(settings: dict | None, fps: float | None = None) -> dict:
    out = dict(DEFAULTS3D)
    given = settings or {}
    for k, f in FIELDS3D.items():
        v = given.get(k)
        if v is None:
            v = f["default"]
        try:
            if f["type"] == "number":
                v = min(max(float(v), f["min"]), f["max"])
                v = int(round(v)) if float(f.get("step", 1)).is_integer() and float(f["default"]).is_integer() \
                    else round(v, 6)
            elif f["type"] == "bool":
                v = bool(v)
            elif f["type"] == "select":
                allowed = [o[0] for o in f["options"]]
                if v not in allowed:
                    match = [a for a in allowed if str(a) == str(v)]
                    if not match:
                        continue
                    v = match[0]
        except (TypeError, ValueError):
            continue
        out[k] = v
    if fps:
        top = max_band_hz(fps)
        out["freq_hi"] = min(out["freq_hi"], top)
        out["freq_lo"] = min(out["freq_lo"], top - 1.0)
    if out["freq_hi"] <= out["freq_lo"] + 0.5:
        out["freq_hi"] = out["freq_lo"] + 1.0
    if not out["noise_floor"]:
        out["n_aug"] = DEFAULTS3D["n_aug"]
    return out


def split3d(settings: dict, fps: float | None = None) -> tuple[dict, dict, dict]:
    s = normalize3d(settings, fps)
    return ({k: s[k] for k in PROCESS3D}, {k: s[k] for k in DISPLAY3D}, {k: s[k] for k in RUNTIME3D})
